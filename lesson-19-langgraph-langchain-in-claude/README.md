# Lesson 19: LangChain and LangGraph, by building a support desk

Lessons 01 to 18 build agents with no framework, so you can see every moving part. Lesson 02 asks "what do frameworks do?" and answers with a 120-line mini framework. This lesson closes the loop: it uses the **real** ones, [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph), to build one project (a customer-support desk for an outdoor-gear shop) and then goes deep on each library. It keeps the standard of the other lessons: rules in code, every claim tested, no API key needed.

> **LangChain gives you the parts and a ready-made agent. LangGraph is the engine underneath: state, nodes, edges, saved checkpoints and pauses. Use the agent when the loop fits. Draw the graph yourself when code, not the model, must decide what runs next.**

## Start here

This lesson is written for someone new to both libraries. Every stage has a short explanation, the code, and tests that prove the explanation.

1. Read [docs/00-start-here.md](docs/00-start-here.md): what you will build, the twelve words you need, how to run it.
2. Do **Part 1** (stages 1 to 7) in order. It builds the desk.
3. Then pick from **Part 2** (stages 8 to 15). These go deeper, are independent of each other, and each solves one problem. Stages 8 to 11 are LangGraph. Stages 12 to 15 are LangChain.
4. Finish with [docs/16-cheat-sheet.md](docs/16-cheat-sheet.md), the list of main functions.

### Part 1: the desk

| Stage | You learn | Explanation | Code | Tests |
|---|---|---|---|---|
| 0 | Business rules with no model | [00](docs/00-start-here.md) | `desk.py`, `tools.py` | `test_desk.py` (43) |
| 1 | `create_agent`, `@tool`, verified context | [01](docs/01-first-agent.md) | `stage1_first_agent.py` | `test_stage1_agent.py` (24) |
| 2 | The same agent as a hand-built graph; state, reducers, edges | [02](docs/02-graph-by-hand.md) | `stage2_graph_by_hand.py` | `test_stage2_graph.py` (17) |
| 3 | Structured output, checked against the truth | [03](docs/03-structured-triage.md) | `stage3_triage.py` | `test_stage3_triage.py` (21) |
| 4 | Thread memory (checkpointer) and long-term memory (store) | [04](docs/04-memory.md) | `stage4_memory.py` | `test_stage4_memory.py` (14) |
| 5 | Human approval with interrupts | [05](docs/05-human-approval.md) | `stage5_approval.py` | `test_stage5_approval.py` (28) |
| 6 | Middleware: audit, errors, retry, limits, dynamic prompt | [06](docs/06-middleware.md) | `stage6_middleware.py` | `test_stage6_middleware.py` (30) |
| 7 | Everything as one workflow graph | [07](docs/07-support-desk.md) | `stage7_support_desk.py` | `test_stage7_support_desk.py` (34) |
| live | The same desk on the real Claude API | | `run_live.py` | `test_live_wiring.py` (7) |

### Part 2: deeper into LangGraph and LangChain

| Stage | You learn | Explanation | Code | Tests |
|---|---|---|---|---|
| 8 | **LangGraph:** parallel work with `Send`, reducers for fan-in, joins | [08](docs/08-parallel.md) | `stage8_parallel.py` | `test_stage8_parallel.py` (18) |
| 9 | **LangGraph:** `Command`, `interrupt()` in your own node, input and output schemas | [09](docs/09-commands.md) | `stage9_commands.py` | `test_stage9_commands.py` (18) |
| 10 | **LangGraph:** SQLite persistence across a restart, time travel, node retries | [10](docs/10-durability.md) | `stage10_durability.py` | `test_stage10_durability.py` (20) |
| 11 | **LangGraph:** every streaming mode, subgraphs, async | [11](docs/11-streaming.md) | `stage11_streaming.py` | `test_stage11_streaming.py` (19) |
| 12 | **LangChain:** runnables, the `\|` pipe, prompts, parsers, retry and fallbacks | [12](docs/12-runnables.md) | `stage12_runnables.py` | `test_stage12_runnables.py` (33) |
| 13 | **LangChain:** messages, safe trimming, tools with schemas, artifacts, state-writing tools | [13](docs/13-messages-and-tools.md) | `stage13_messages_tools.py` | `test_stage13_messages_tools.py` (32) |
| 14 | **LangChain:** retrieval (RAG): splitting, embeddings, vector store, checked citations | [14](docs/14-rag.md) | `stage14_rag.py` | `test_stage14_rag.py` (46) |
| 15 | **LangChain:** callbacks, cache, rate limits, retry and fallback, summarising | [15](docs/15-model-ops.md) | `stage15_model_ops.py` | `test_stage15_model_ops.py` (32) |

For a map of how the final desk fits together, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Run

```bash
pip install -r requirements.txt     # langchain, langgraph (+ sqlite saver, text splitters), numpy, pytest, langchain-anthropic
python -m pytest -q                 # 436 offline tests, no API key, about 30 seconds
python stage7_support_desk.py       # the whole desk, three tickets, offline
python run_live.py                  # optional: the real Claude API. Needs ANTHROPIC_API_KEY and costs money
```

Every stage file has a `demo()` that prints what happened. Tested with Python 3.14.2, `langchain` 1.4.9, `langchain-core` 1.7.6, `langgraph` 1.2.20, `langgraph-checkpoint-sqlite` 3.1.3, `langchain-text-splitters` 1.1.3, `langchain-anthropic` 1.7.10.

## The final system, in one picture

```
START ─► triage ─┬─► refund_flow ─┐       triage       a schema, then CODE checks the claims and picks the route
                 ├─► answer_faq  ─┼─► finish           refund_flow  an agent; pauses for a person above $25
                 └─► escalate    ─┘       │            answer_faq   answers only from articles that code found
                                          ▼            escalate     opens a ticket, no model call
                                       resolution      finish       writes the result from the refund ledger
```

## The rules the code enforces

### Part 1

1. **Rules about money live in plain Python** (`desk.py`), not in prompts or tools. They are tested with no model at all.
2. **Identity comes from verified context, never from the model.** The tools read `runtime.context.customer_id`. There is no `customer_id` argument for a model to fill in, and an invented one is ignored.
3. **Another customer's order looks exactly like a missing order**, so answers cannot be used to probe.
4. **A schema checks shape; code checks facts.** A triage claim is verified against the customer's message and the order records. The retry is capped at one, and the fallback is an honest "unverified".
5. **A person approves a refund above $25** and sees facts that code looked up, not only what the model said.
6. **Approval is not authorization.** The desk re-checks every refund when it runs: window, remaining amount, agent limit of $200, owner.
7. **The `when` predicate for approval depends only on the call's own arguments**, and anything it cannot read goes to a person.
8. **A refund pays once.** Its key is `thread:tool_call_id`.
9. **Retry covers temporary failures only.** Business refusals are data and are never retried. A bug in a tool becomes an honest message that names the exception type, not its text.
10. **Middleware order is a choice with a test.**
11. **Code chooses the route.** `decide_route` is a plain function.
12. **The resolution comes from the ledger,** not from what the model said it did.
13. **Every tool call gets an answer,** even when a cap stops the run.

### Part 2

14. **Every fan-out is capped in code,** and every parallel write has a reducer (stage 8).
15. **Nothing irreversible happens before an `interrupt()`,** because the node runs again from its first line on resume. A resume is refused unless a question is really pending (stage 9).
16. **A caller can send only what the input schema allows,** and gets back only what the output schema allows (stage 9).
17. **Persisted state is proven to survive a restart,** with fresh graph and model objects (stage 10).
18. **Retry policies list their errors,** so a business refusal is never retried (stage 10).
19. **Personal data is removed by a function before the prompt,** not by asking the model to ignore it (stage 12).
20. **A fallback names the errors it handles,** so it cannot hide bugs (stage 12).
21. **History is cut only at valid places,** so no tool result is left without its request (stage 13).
22. **Schema limits are enforced before the function runs,** and are a separate layer from the desk's rules (stage 13).
23. **Filter by audience before ranking, threshold the score, and verify every citation.** An unknown audience sees nothing, and a question with no relevant source never reaches the model (stage 14).
24. **Cost, cache hits and limiter use are counted, not assumed** (stage 15).

## Findings from building it

These came from running the libraries, not from reading about them.

### Part 1

1. **The default step limit is huge.** In `langgraph` 1.2.20 the default `recursion_limit` is **10,007**. Many tutorials say 25. Set a small limit or use `ToolCallLimitMiddleware`.
2. **A bug in a tool ends the whole run.** `ToolErrorMiddleware` is opt-in and lets you choose what the model hears.
3. **A subgraph that shares an additive list with its parent double counts it.** Messages are safe because `add_messages` deduplicates by id.
4. **A checkpointer makes `thread_id` mandatory.**
5. **My first attribution of refunds to tickets was wrong.** `finish` credited "everything added to the ledger since this ticket began", so with two tickets paused at once, ticket 1 was credited with ticket 2's $50. The same test exposed a tool-call id collision. Both are fixed by putting the thread id into the key.
6. **A weak test survived a mutation check.** My retry test used a business refusal, which is never an exception.
7. **Argument validation happens before your code.** `amount="a lot"` never reaches `issue_refund`.
8. **An edit decision tells the model a human changed the call.**
9. **A module called `trace.py` shadows the standard library** (lesson 14 found the same). I named it `show.py`.

### Part 2

10. **Two nodes writing one key in the same round raise `InvalidUpdateError`** unless the key has a reducer. And `add_edge("a", "join")` plus `add_edge("b2", "join")` runs `join` **twice**; `add_edge(["a", "b2"], "join")` runs it once. Nothing warns you about the second. (Stage 8)
11. **Code before `interrupt()` runs twice.** A side effect placed above it happened 2 times; placed below, once. (Stage 9)
12. **Resuming a thread that is not waiting does not fail clearly.** On an empty thread it died with `KeyError: 'text'`, because the resume value was treated as the graph's input. (Stage 9)
13. **An extra key in the input is ignored** when you declare an `input_schema`, so a caller cannot pre-seed working state. (Stage 9)
14. **A fork writes onto the same thread.** The old checkpoints stay, but the thread's latest state becomes the fork. Replaying re-executes real nodes. (Stage 10)
15. **A SQLite file stays locked on Windows until every connection is closed.** My first demo crashed deleting its own temp folder. (Stage 10)
16. **`"messages"` stream mode also emits whole messages that a node returns,** not only model tokens. A message written by plain code arrives as one event. My first test assumed otherwise and was wrong. (Stage 11)
17. **A `RunnableLambda` at the end of a chain stops the stream** from being word by word, and a token stream ends with an empty marker chunk. (Stage 12)
18. **My redaction function had two bugs:** the email pattern swallowed the full stop at the end of a sentence, and the phone pattern left the opening bracket of `(415) 555-0100`. The first came from running the demo, the second from writing the tests. A date such as `2026-10-20` is still treated as a phone number; that is pinned as a known limit. (Stage 12)
19. **`trim_messages` avoids an orphan tool message that a plain slice leaves behind.** (Stage 13)
20. **`@tool` does not accept `handle_tool_error`.** It is a field you set on the tool object. (Stage 13)
21. **`InMemoryVectorStore` needs `numpy`, and `langchain-core` does not list it.** LangChain's `DeterministicFakeEmbedding` needs it too, and gives random vectors, so it cannot show meaningful retrieval. (Stage 14)
22. **My first hashing embedding used 512 slots, and a kayak question matched a shipping chunk at 0.29 by a hash collision.** At 4,096 slots it correctly finds nothing. (Stage 14)
23. **A cached reply is stamped `usage_metadata={"total_cost": 0}`,** which changes the next prompt in an agent loop. A repeated agent run replays only its first model call (4 lookups, 1 hit). My first cache test passed for the wrong reason; counting hits showed it. (Stage 15)
24. **A new rate-limit bucket starts empty and its clock starts at its first use,** not at creation. (Stage 15)
25. **Model middleware order decides whether a retry ever fires.** With retry outside fallback, the primary was tried once; with fallback outside retry, twice. My first demo had it the wrong way round. (Stage 15)
26. **Python's bytecode cache can hide a mutation test's result.** See the next section. (Testing)

## How I checked the tests can fail

I mutation-checked behaviors in two rounds: deliberately breaking the code and confirming a test fails.

- **Part 1 (17 behaviors):** customer scoping, idempotency, the agent limit, the 30-day boundary, partial refunds, the thread-scoped key, the step cap, the in-message id check, long-term memory scoping, approval, odd amounts, retrying every error, the error layer, leaking exception text, urgency, refund attribution, and an unmatched help question. **One of the first 15 survived** (finding 6). After adding a test, all 17 were caught.
- **Part 2 (23 behaviors):** each stage's key guard, for example the parallel width cap, the sort of results, the question cap, the check that a resume has a pending question, node retries of unlisted errors, the checkpoint lookup, empty stream chunks, nested stream events, redaction, the signature, the fallback's error list, safe trimming, the exception tool flag, the audience filter, the relevance threshold, citation checks, the unknown-audience default, call counting, middleware order and the price table. **All 23 were caught**, each by a test about that behaviour.
- **A flaw in my own harness.** My first mutation run for Part 2 reported all mutants killed, but from the tenth on, every one was "killed" by the same unrelated stage 10 test. The cause: Python reuses compiled bytecode when a file's size and modification second match, and the `==` to `!=` mutant has the same size as the original, so a stale mutated copy survived the restore. I disabled bytecode caching in the harness (`-B`, `PYTHONDONTWRITEBYTECODE`, clearing `__pycache__`) and **re-ran all three rounds**. The results above are from the clean re-run. I also started one run in the background by mistake while another could still be running; I stopped it, restored the two files it had left mutated from its backups, and checked the suite was green again before re-running.

I do not take this as proof the suite is complete, only that these 40 changes are detected.

## What is NOT known

- **Nothing here ran against a live model.** `run_live.py` is checked only for wiring, with a dummy key. I do not know how often real Claude triages wrongly, calls a tool with bad arguments, or ignores the system prompt. The prompts are first drafts.
- **The scripted model is simpler than a real one.** It never sends malformed JSON, never reasons aloud, and never calls several tools in a surprising order. Its streaming is word by word; a real provider's token and tool-call chunks may differ.
- **The embeddings are a word-matching stand-in.** I did not run a real embedding model, so I do not know how the thresholds (0.2 here) or chunk sizes behave on real text. Retrieval quality is unmeasured.
- **The summariser is a scripted double.** I do not know how good a real model's summary is, or whether it keeps the facts the next step needs.
- **Batched approval with several paused calls in one turn is untested.** The resume payload needs one decision per paused call; my helpers handle one.
- **The claim that the `when` predicate runs again on resume** comes from the LangChain docstring, not from a test of mine.
- **Only SQLite was run.** I did not run the Postgres saver, so I have not shown a pause surviving a restart there, nor a process killed mid-step (the restart in stage 10 is a clean close, not a crash).
- **Async is covered by one stream test.** I did not test async tools, async middleware, or cancellation.
- **No concurrency control.** `Desk` is not thread-safe; the tests pause many tickets but never use threads.
- **Not covered at all:** LangGraph's functional API (`@entrypoint`, `@task`), the LangGraph server and SDK, `langchain.mcp`, LangSmith tracing and evaluation, and LangGraph's cache policies.
- **Version drift.** These libraries change quickly. Everything was checked on the versions listed above and nowhere else.

## Limits of the design

- **The customer id and audience are values you pass in.** Authenticating them is your system's job.
- **Approvals are not authenticated.** Nothing checks who is resuming a thread.
- **`triage` sees only the last message.** In a long thread you would pass a summary, and then think about what a hostile earlier message could do.
- **A paused ticket waits forever.** There is no queue, owner or timeout.
- **The ledger is a Python list.** In production it is a database with a unique constraint on the idempotency key.
- **FAQ search and RAG are word matching,** the simplest thing that can say "nothing relevant".
- **A $25 auto-approval threshold invites splitting a big refund into small ones.** The per-order total and the call limit are what stop it.
- **`events` and the message list grow without bound** on a long-lived thread.
- **Streaming tokens to a customer shows text before you can check it.** The verify-before-showing rules of the earlier stages and token streaming pull in opposite directions; this lesson does not resolve that.
- **Redaction is a regex.** It misses names and addresses and flags dates.

## Exercises

1. Run `run_live.py` with a real key. Which triage mistakes do you see? Turn each into a test case.
2. Kill the stage 10 program with `os._exit(1)` between `handle` and `resume`, in a child process. Does the ticket survive?
3. Make the desk's ledger a SQLite table with a unique key. Resume the same ticket twice from two processes. How many refunds are there?
4. Replace `HashingEmbeddings` with a real embedding model. Which stage 14 tests still pass? Which should change?
5. Build the batched case: one turn asks for two refunds on two orders. Resume with two decisions, one approve and one reject.
6. Rebuild stage 8's report with LangGraph's functional API (`@task` per order) and compare the code.
7. Add `ModelCallLimitMiddleware` to `build_ops_agent` and a `UsageTracker` budget check. What should happen when the budget is spent?
