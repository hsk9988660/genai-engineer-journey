# Lesson 08: multi-agent, only where it earns its cost

Lesson 08 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) shows three prompts in a row (planner, concierge, budget reviewer) and calls it a multi-agent system. This folder builds the patterns for the **reasons that actually justify more than one agent**, and a harness to measure whether they pay off.

> Default to one agent. Add another only for a reason you can name and measure.

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q                         # 54 tests, offline, no API key
set ANTHROPIC_API_KEY=sk-ant-...            # macOS/Linux: export ANTHROPIC_API_KEY=...
python research.py "Compare AcmeCloud and BetaCloud on price, security and support"
python router.py "I was charged twice this month"
python compare.py                           # single agent vs multi-agent, same questions (costs real money; not yet run)
```

## The five real reasons, and where each lives in the code

| Reason | Where |
|---|---|
| **Context isolation** | `Runtime.run_agent` starts every agent with a fresh message list. The orchestrator sees each reader's *answer and evidence quotes*, never the raw documents (`research._merge_input`). |
| **Permission separation** | Readers get exactly one tool (`fetch_document`), restricted in code to the documents in their task. The planner, merger and checker have **no tools**. Publishing is plain code behind a human approval and receives only the verified report. |
| **Parallelism** | Independent readers run in a thread pool. |
| **Independent review** | A separate checker agent verifies each claim against the cited evidence quotes. |
| **One budget** | A single thread-safe `Budget` caps API calls and dollars across every agent. |

Only the steps that need judgment are agents (plan, read, merge, check). Validation, fan-out, evidence verification, report formatting, approval and publishing are ordinary code.

## Files

| File | Role |
|---|---|
| `schemas.py` | Structured handoffs and results (`Handoff`, `WorkerResult`, `Report`, `Verdict`, `AgentTurn`) |
| `runtime.py` | `Runtime.run_agent` (isolated run + tools + retry on validation problems), `Tracer` (span tree), `Budget` |
| `research.py` | Orchestrator-worker pipeline: plan, parallel readers, merge, check, revise once, approve, publish |
| `router.py` | Customer-support hand-off routing with ping-pong and hop-cap protection |
| `compare.py` | Harness: same questions through a single agent and the multi-agent pipeline; quality, cost, calls, time |
| `corpus.py` | Demo documents, one containing a prompt injection |
| `fakes.py` | Scripted client routed by agent role, so parallel agents stay deterministic in tests |

## What the code enforces
- **Evidence is verified in code.** A reader's quote must appear verbatim in a document it actually fetched. Fabricated quotes and quotes from unfetched documents are rejected and sent back.
- **A claim must cite a successful task.** Citing a missing or failed task is rejected.
- **A report the checker rejects is revised once, then never published**, and the human is never even asked to approve it.
- **A reader that obeys an injected instruction has nothing dangerous to call.** The injected document tells it to call `publish_report`. Readers have no such tool, and a test confirms the attempt just returns an error.
- **Handoffs cannot loop.** An agent that already handled a conversation can never receive it again, and a hop cap sends long chains to a human.
- **A failed worker does not sink the others.** It becomes a `failed` result the merger can see.

## What the numbers do and do not tell you
`python -m pytest` shows the **structure**: on the happy path this design makes at least `n + 3` agent runs and 7 API calls for 2 sub-tasks, against 2 calls for the single-agent baseline in the same scripted scenario. Those counts come from scripted replies. They say nothing about real token usage.

**I do not know whether the multi-agent design is better than a single agent on real questions.** That is exactly what `python compare.py` measures, and I have not run it. Anthropic's own write-up of its multi-agent research system reported usage several times a normal chat's, so expect a cost multiplier. Whether quality improves enough to justify it is an empirical question for your tasks.

## Limits you should know
- **Live comparison not run.** No API key here. Every test uses a scripted model, so they verify mechanics (isolation, caps, tracing, routing, validation), not model behavior.
- **The quality score is crude:** the fraction of expected facts appearing in the answer. It is easy to game and blind to wrong extra claims. Use it to compare designs on the same cases, not as an absolute measure.
- **The checker is an LLM.** It can be lenient or wrong, and from the same model family it may share the author's blind spots. Evidence-based checking beats opinion, but it is not proof.
- **Injection can still launder through a worker's `answer`.** A reader's summary reaches the merger as text. The merger has no tools, the checker verifies claims against quotes, and a human approves, but a convincing injected claim that also has a real supporting quote could pass. Verbatim evidence checking does not stop a document from quoting itself.
- **Dollar cap is approximate.** The call cap is exact (a call slot is reserved atomically), but a call's cost is known only after it returns, so in-flight parallel calls can push spend slightly past `max_cost_usd`.
- **Documents are fetched whole.** No chunking or retrieval, so one huge document would flood a reader's context.
- **The support router is minimal:** one turn per agent, no tools, no real back-and-forth with the customer, a model-written handoff note, and "escalation" is only a message.
- **Tracing is in memory.** There is no export to a tracing system.
- **Threads, not async.**

## Notes from building it
- **I found a race in my own budget by re-reading it.** `check()` and `charge()` were separate steps, so two parallel agents could both pass the check on the last slot and exceed the cap. The call cap is now reserved atomically (`begin_call`), with a test where 20 threads make 1,000 attempts for 100 slots and exactly 100 succeed. My first concurrency test only covered `charge()`, which is why it did not catch this.
- One test failed because my scripted merger cited a task that had failed. The validator correctly rejected it. The test data was wrong.

## Exercises
1. Run `python compare.py`. Does multi-agent score higher? By how much per dollar? Try a single-agent prompt that is better than mine and see if the gap closes.
2. Add a question where one document is large. Which design copes better?
3. Replace the crude score with an LLM judge, then compare the two scores. Which one can you game more easily?
4. Make the checker use a smaller, cheaper model than the merger. Does quality hold?
5. Write an injection that quotes itself: the document contains a false claim as an exact sentence. Does the pipeline publish it? What would stop it?
