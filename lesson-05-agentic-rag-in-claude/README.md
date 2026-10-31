# Lesson 05: agentic RAG with verified citations

Lesson 05 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) describes agentic RAG but its notebook uses keyword matching over four strings. This folder builds the real mechanics and measures them.

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q                         # 39 tests, offline, no API key
python evals.py retrieval                   # offline retrieval metrics per search mode
set ANTHROPIC_API_KEY=sk-ant-...            # macOS/Linux: export ANTHROPIC_API_KEY=...
python agent.py "Can an 8 year old fly alone?"
python agent.py --staff "What does FARE-204 mean?"
python evals.py live                        # plain RAG vs agentic RAG, scored on the same questions
```

## Files

| File | Role |
|---|---|
| `chunker.py` | Sentence-packing chunker with overlap and metadata |
| `index.py` | BM25, vector, and hybrid (reciprocal rank fusion) search, with permission filtering before ranking |
| `corpus.py` | A fictional knowledge base (public, staff-only, and one poisoned document) plus 14 labelled questions |
| `agent.py` | `plain` and `agentic` RAG. The answer is a schema'd tool call whose citations are verified in code. |
| `evals.py` | Retrieval metrics (offline) and answer scoring (live) |

## What the code enforces (not the prompt)
1. **Citations must point at chunks actually retrieved in this run.** No invented sources.
2. **Every quote must appear verbatim in its chunk.** No fabricated quotes. Failures go back to the model for a retry. After `max_retries` the agent abstains.
3. **An "answerable" answer needs a citation.** "I don't know" (`answerable=false`) is a first-class, valid outcome.
4. **Permissions are applied before ranking.** `roles` is passed by your code. The model can't choose or change it, so a public user's agent can never see staff chunks, however it words its query.
5. **Retrieved text is delimited as data**, and the system prompt says never to follow it. This is a weak defense on its own. The strong ones are that this agent has no side-effect tools and that permissions are enforced in code.

## Measured retrieval results (offline, 11 answerable questions, top 3 chunks)

| Mode | recall@3 | MRR | precision |
|---|---|---|---|
| bm25 | 0.91 | 0.91 | 0.61 |
| vector (trigram) | 0.55 | 0.55 | 0.55 |
| hybrid | 0.91 | 0.91 | 0.61 |

What this shows, honestly:
- **Hybrid only ties BM25.** The default `HashingEmbedder` matches word *shapes*, not *meaning*, so it adds nothing here. It exists so the vector and fusion code paths are real and tested. For genuine semantic search, plug in a real embedding model through the `embedder` argument (`embed(list[str]) -> list[list[float]]`, unit-length vectors), then re-run the table.
- **A tuning lesson:** at `min_cosine=0.2` the weak embedder let noise in and hybrid precision (0.48) fell *below* BM25 (0.59). Raising the threshold to 0.3 fixed that. This was tuned on 14 questions, so treat it as an example of the method and not a result you can rely on.
- **One question fails in every mode:** "Can an 8 year old fly alone?" The documents say "unaccompanied minors aged 5 to 11", so no word overlaps. A test pins this gap and shows that the rewritten query "unaccompanied minor" does find the answer. That is what the agentic loop is for.
- **Precision of 0.61 means about 40% of returned chunks are noise.** The model must ignore it, which is why the live eval includes unanswerable questions.

## Live eval (not yet run)
`python evals.py live` scores both modes on all 14 questions, including two unanswerable ones and one permission trap (asking a public user for a staff-only code). It reports pass rate, average searches, and average tokens. I have not run it, because it needs an API key. My expectation, which the eval may disprove, is that agentic RAG passes the vocabulary-gap question that plain RAG fails, at the cost of more searches and tokens.

## Limits
- 14 questions is far too few to draw statistical conclusions. Grow the set before trusting any number.
- Retrieval is scored at document level, not chunk level.
- Quote checking proves a quote exists in the chunk. It does not prove the quote supports the claim. Checking that needs an LLM judge or a human.
- `max_searches` and `max_steps` bound cost, but there is no budget in dollars.

## Exercises
1. Plug in a real embedding model and re-run `evals.py retrieval`. Does hybrid beat BM25 now? Re-tune `min_cosine`.
2. Add 15 more questions, including paraphrases and multi-document comparisons. See which mode degrades first.
3. Make `max_chars` larger (e.g. 600) and watch recall and precision change.
4. Add a document that contradicts another (an old and a new refund policy). How should the agent behave? Add a date field and handle it.
5. Write an adversarial question that tries to extract the staff code as a public user. Confirm it fails at retrieval, before the model is involved.
