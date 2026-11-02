# Lesson 05: implementation guide

This explains **how the code works**: how a question becomes a verified, cited answer, and how retrieval is built and measured. The [README](README.md) says why the lesson exists and what was measured.

---

## 1. The system on one page

```
 question + roles (set by YOUR code)
      │
      ▼
 answer_question(mode="plain" | "agentic")
      │
      │  plain:   search ONCE with the user's words, put the hits in the prompt
      │  agentic: search_docs is a TOOL; the model may rewrite queries, search several times, or not at all
      │
      │  both must finish by calling submit_answer(Answer)
      ▼
 verify(answer, retrieved)            every citation must point at a chunk retrieved IN THIS RUN,
      │                               every quote must appear verbatim in that chunk,
      │                               an "answerable" answer must have a citation
      ├─ problems → sent back to the model, which retries (max_retries)
      ▼
 RagResult(answer, verified, searches, retrieved_ids, steps, tokens)

 retrieval:  HybridIndex.search(query, k, roles, mode)
             permission filter → BM25 ranking + vector ranking → reciprocal rank fusion
```

---

## 2. A question, from start to end

### Where it starts
`answer_question(client, question, index, roles=("public",), mode="agentic", ...)`. **`roles` belongs to the signed-in user and is fixed by your code. The model cannot choose or change it.** From the command line: `python agent.py "Can an 8 year old fly alone?"` (add `--staff` to also see staff documents).

### What happens, in order

1. **Mode setup.**
   - `plain`: search once, immediately, with the user's exact words. The hits are formatted into the prompt and the model is given only `submit_answer`.
   - `agentic`: the model gets `search_docs` and `submit_answer`, and instructions to search with the **documents'** wording ("unaccompanied minor" for "can my 8 year old fly alone"), retry weak searches, and search each side of a comparison separately.
2. **The loop**, up to `max_steps`, with `tool_choice={"type": "any"}` so the model must call a tool every turn.
3. **Searches run first** (so an answer submitted in the same turn can cite what the searches found). Each `search_docs` call:
   - is refused with an error once `max_searches` is reached
   - is refused if the query is empty
   - otherwise runs `index.search(query, k, roles)`, records the query, **adds every hit to `retrieved`**, and returns the chunks wrapped in `<retrieved_document>` tags (the system prompt says text inside is data, never instructions)
4. **`submit_answer`** is validated as an `Answer` (Pydantic), then `verify` checks it against `retrieved`.
   - **grounded:** return the result with `verified=True`
   - **problems:** the list goes back to the model as an error result and it tries again; after `max_retries` the loop stops
5. Tool results go back as one user message and the loop continues.

### Where it ends
- **Verified answer:** `verified=True`, with citations.
- **Abstention:** if the model never submits a valid, grounded answer, the result is `answerable=False` with "I could not produce a verified answer from the documents" and `verified=False`. The code prefers saying nothing to saying something ungrounded.

### `verify` in detail
An answer fails if: it says `answerable` but has no citation; a citation names a `chunk_id` that was **not retrieved in this run** (an invented source); a quote is shorter than 8 characters after normalisation (too short to check); or the normalised quote is **not a substring of the chunk** (a fabricated quote). Normalising lowercases, collapses whitespace and trims quotes. This proves a quote exists, **not** that it supports the claim.

---

## 3. Module reference

### `index.py`: search
- `tokenize` / `_stem`: lowercase words with a crude plural/suffix strip.
- `BM25`: scores a chunk by the query words it contains, weighting rare words more.
- `HashingEmbedder`: a deterministic **character-trigram** embedder using `crc32` (not `hash()`, which is randomised per process). It matches word *shapes* ("cancel" and "cancellation"), **not meaning**. For real semantic search pass any object with `embed(list[str]) -> list[list[float]]` returning unit-length vectors.
- `reciprocal_rank_fusion`: merges ranked lists by **position only**, so BM25 and cosine scores need no common scale.
- `HybridIndex.search(query, k, roles, mode)` with mode `bm25`, `vector` or `hybrid`. **Permissions are applied before ranking**: a chunk the user may not read is never a candidate, however the query is worded. Vector results below `min_cosine` (0.3, chosen by measurement on the 14 demo questions) are dropped.

### `chunker.py`
`chunk_document` packs whole sentences up to `max_chars` and repeats the last sentence(s) at the start of the next chunk, so a fact split across a boundary still appears intact somewhere. A `Chunk` keeps its document id, source and `access` set, and `indexed_text` is what is searched.

### `corpus.py`
A small fictional knowledge base and 14 labelled questions. Documents carry an access list (`public`, `staff`). One public document, `promo`, contains a prompt-injection attempt so you can see how retrieved text is handled. A deliberately small chunk size makes several documents split.

### `evals.py`
Retrieval and answers are evaluated **separately**, so you can tell which half broke. `retrieval_report` computes recall@k and MRR per search mode at document level, offline and deterministically. `run_answer_eval` runs plain vs agentic live. Questions with no gold document test abstention instead. With 14 cases the numbers are illustrative, not statistically meaningful.

---

## 4. Design decisions

- **Retrieval as a tool.** The model decides whether and how to search; the cost is more model calls, the gain is handling vague, multi-part and chit-chat questions.
- **Permissions before ranking.** Filtering after ranking can leak through scores or fill the top-k with forbidden chunks.
- **Citations are verified, not trusted.** The model can fabricate a quote that sounds right; a substring check is cheap and catches it.
- **Abstain rather than guess.** `answerable=false` is a legitimate output, and so is the fallback.
- **Rank fusion by position.** Avoids tuning a weight between two incompatible score scales.

## 5. What it does not do

The default embedder is not semantic, so a paraphrase with no shared word shapes can be missed (the README reports one question that failed in every mode). A verified quote does not mean the quote supports the claim. The corpus is tiny and the metrics are not statistically meaningful.

## 6. Running and testing

`python evals.py retrieval` runs offline with no key. `python agent.py "..."` and `python evals.py live` need an API key. `python -m pytest -q` runs 39 offline tests.
