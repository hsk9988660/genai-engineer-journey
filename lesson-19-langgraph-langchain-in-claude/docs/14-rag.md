# Stage 14: retrieval (RAG) with LangChain

**Code:** [stage14_rag.py](../stage14_rag.py) | **Tests:** [test_stage14_rag.py](../test_stage14_rag.py) | **Run:** `python stage14_rag.py`

## The idea

A model does not know your help articles. **Retrieval-augmented generation** (RAG) looks up relevant text first and puts it in the prompt, so the model answers from that text and not from memory. The LangChain pieces, in the order the data flows:

```
Document ─► text splitter ─► chunks ─► Embeddings ─► vector store ─► search ─► prompt ─► model ─► checked answer
```

| Piece | Job |
|---|---|
| `Document` | `page_content` (text) plus `metadata` (a dict: id, audience, source, ...) |
| `RecursiveCharacterTextSplitter` | cuts long text into chunks, preferring paragraph and sentence breaks. Chunks keep the metadata |
| `Embeddings` | turns text into a vector. Similar meaning should give a close vector |
| `InMemoryVectorStore` | stores the vectors; `similarity_search_with_score(query, k=, filter=)` finds the nearest |
| retriever | a vector store as a runnable (`store.as_retriever(...)`), so it can sit in a `|` chain (stage 12) |

Lesson 05 built retrieval from scratch. This stage uses LangChain's parts and keeps lesson 05's three rules, because the parts do not enforce them for you.

## The knowledge base

`ARTICLES` has five short articles: shipping, returns, warranty, care, and one **staff-only** article about margins and refund limits. `split` cuts them into 11 chunks and gives each an id like `shipping-1`, so a citation can point at an exact chunk:

```
shipping-1 'Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days and costs extra. Orders placed after 2 pm ship the'
shipping-2 'after 2 pm ship the next business day. You receive a tracking link by email when the parcel leaves our warehouse.'
```

Notice the overlap: "after 2 pm ship the" is the end of chunk 1 and the start of chunk 2. Overlap stops a fact that sits on a boundary from being cut in half (`test_neighbouring_chunks_overlap`). Every chunk keeps the article's metadata, ids are unique, and no word is lost (`TestSplitting`).

### The embeddings in this lesson

`HashingEmbeddings` is a stand-in, so the lesson needs no model and no key. Each word adds weight to one of 4,096 slots of a vector, after dropping common words and cutting a plural "s". Texts that **share words** get close vectors (`test_shared_words_mean_a_closer_vector`).

It matches *words*, not *meanings*. A real embedding model also links "watertight parka" to an article about waterproof jackets; this one does not, and the test says so (`test_it_matches_words_not_meanings_so_a_synonym_is_missed`). It is the honest limit of the stand-in. In production, `Embeddings` is an interface: swap in a real provider's embeddings and nothing else changes.

Two things I learned building it:

- LangChain's `DeterministicFakeEmbedding` needs `numpy`, and gives **random** vectors per text, so it cannot show meaningful retrieval. I wrote the hashing version instead.
- `InMemoryVectorStore` itself needs `numpy` for cosine similarity, and `langchain-core` does not list it as a dependency. If you get `ImportError: cosine_similarity requires numpy`, install it (it is in this lesson's `requirements.txt`).
- My first version used 512 slots. A kayak question then matched a shipping chunk at score 0.29 purely by a **hash collision**. At 4,096 slots it correctly finds nothing. A crude embedding makes the "nothing relevant" case unreliable, which is a reason to test it (`test_an_unrelated_question_finds_nothing`).

## Rule 1: filter before you rank

```python
allowed = VISIBLE.get(audience, set())
hits = self.store.similarity_search_with_score(query, k=k, filter=lambda doc: doc.metadata["audience"] in allowed)
```

The `filter` runs **before** ranking, so a staff-only chunk cannot take one of the top-k places. If you filtered *after*, a customer asking about margins would get the staff chunk as its top result, have it removed, and be left with fewer results than asked for; worse, a bug in the post-filter would leak it. `test_the_filter_runs_before_ranking_so_a_hidden_chunk_cannot_take_a_slot` shows that the best match overall for a question is the staff chunk, and the customer search never returns it.

- Four different margin questions, with `k=50`, never return the staff article to a customer (`test_a_customer_never_gets_a_staff_article_even_with_a_huge_k`).
- An **unknown** audience sees nothing (`test_an_unknown_audience_sees_nothing`). Access control should default to deny.
- The audience comes from `DeskContext` (verified context), never from the model. A tool call that passes `audience="staff"` is ignored (`test_an_audience_argument_from_the_model_is_ignored`).

## Rule 2: say "nothing relevant"

A vector store always returns its nearest chunks, however far away. `min_score` turns "nearest" into "relevant":

```python
return [(doc, score) for doc, score in sorted(hits, key=lambda pair: -pair[1]) if score >= self.min_score]
```

`test_without_a_threshold_the_nearest_chunks_come_back_however_far` shows the danger: with no threshold, a question about kayaks still returns three chunks, none of them relevant. When the result is empty, `ask_knowledge_base` **does not call the model** and returns a fixed message that hands the question to a person (`test_when_nothing_is_relevant_the_model_is_not_called`). A model with no sources tends to answer anyway.

The threshold is a number you must tune on real questions. 0.2 here is a guess that works for this tiny corpus.

## Rule 3: verify the citations in code

The prompt asks the model to cite chunk ids in square brackets. Code then checks them:

```python
cited = list(dict.fromkeys(CITATION.findall(text)))
valid = [c for c in cited if c in sources]
invalid = [c for c in cited if c not in sources]
return RagAnswer(text, sources, valid, invalid, grounded=bool(valid) and not invalid)
```

| Model's answer | `grounded` | Why |
|---|---|---|
| `... [shipping-1]` | True | cited a chunk that was in the prompt |
| `... [shipping-9]` | False | cited an id that does not exist |
| `... [shipping-1] ... [promo-1]` | False | one real, one invented |
| `... [warranty-1]` (exists, but was not retrieved) | False | cited something it was never shown |
| no citation | False | cannot be checked |

`grounded` is a **gate** for your code, for example "do not send an ungrounded answer to a customer; escalate". Be clear about what it proves. It proves the model *said* where it got something and named only sources it was given. It does **not** prove the source supports the claim. Checking that needs a second, independent check (lesson 05).

## As an agent tool

`make_search_tool(kb)` wraps `kb.search` as a tool. The model sees one argument, `query`. The audience is read from `runtime.context.audience`. Run it as a customer and the staff passage is never returned; run it with `DeskContext("S-1", audience="staff")` and it is. Nothing relevant returns `{"passages": []}`, not an error, so the agent can say it does not know.

## Withdrawing an article

`kb.forget("shipping")` deletes every chunk of an article by id (`TestWithdrawingAnArticle`). Knowledge goes stale: a withdrawn policy must disappear from the index, not only from the website. Re-indexing and deletion are part of the system, not an afterthought.

## What can go wrong

- **Filtering after ranking** (above).
- **No threshold**, so the model always gets *something* and always answers.
- **Chunks that are too small or too big.** Too small loses context. Too big drowns the answer in noise and costs tokens. Tune on real questions.
- **Treating a citation as proof.**
- **Prompt injection through documents.** Retrieved text goes into the prompt. A document that says "ignore your rules" is an attack (lesson 04). Only index text you trust.
- **A stale index.** See withdrawing.

## Try it

1. Add a new public article and a question about it. Does it need a different `min_score`?
2. Replace the retrieval in `ask_knowledge_base` with `kb.store.as_retriever(...)` inside a `|` chain, as in stage 12. What do you lose (hint: the score threshold and the audience filter)?
3. Add a second `audience`, `"partner"`, that sees public and partner articles but not staff. Which tests do you need to add first?

## Check yourself

- Why does the filter have to run before ranking?
- What does an empty result mean, and why is the model not called?
- A model cites `[returns-1]`, which exists but was not in the prompt. What does `ask_knowledge_base` report?
- What does a verified citation *not* prove?
