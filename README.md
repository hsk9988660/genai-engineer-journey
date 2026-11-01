# GenAI Engineer Journey

**Thirteen lessons on building AI agents, each one a small, tested, working codebase on the Claude API, with no agent framework and no cloud account.**

This repo follows the first thirteen lessons of Microsoft's [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) course and rebuilds each one from first principles. For every lesson you get three things:

1. **The concept, argued like a senior engineer would.** Not "what is a planner" but "when does a planner beat a loop, and how does it fail?"
2. **Working code you can read in an evening or two.** From about 150 lines (lesson 01) to about 1,450 (lesson 11), comments included, using only `anthropic`, `pydantic` and the standard library.
3. **Tests you can run for free.** 777 offline tests with a scripted fake model. No API key and no cost. (Lesson 11's tests also launch real child processes and a localhost-only HTTP server.)

Every lesson README also has a section called **"What is NOT known"**. That is deliberate. See [Honest status](#honest-status-read-this-before-you-trust-anything).

## Who this is for

- You can write Python and want to understand how agents work **underneath** the frameworks.
- You want to know what production agents need beyond a demo: **safety, evaluation, cost, and failure handling.**
- You learn better from small code you can break than from slides.

It is probably not for you if you want a ready-made library to import. These are teaching codebases, not a framework.

## The big ideas

Eleven ideas come up across the lessons. If you only read this section, read these.

1. **An agent is an LLM in a loop with tools, and the loop is about 25 lines.** Everything else is refinement. (Lesson 01)
2. **The model proposes, your code disposes.** The model only *requests* tool calls. Your code decides whether they run. (04, 06)
3. **Put rules in code, not in prompts.** A prompt is a request. Validation, permissions, budgets and approvals must be enforced by code the model cannot talk its way past. (03, 04, 06, 07)
4. **Never trust the model's claim when you have the ground truth.** Re-check availability, quotes, totals, and dates in code. A schema guarantees the *shape* of an answer, not that it is *true*. (03, 05, 10)
5. **Assume the model will be fooled, and make that not matter.** Prompt injection will sometimes succeed. Limit what a fooled agent can do. (04, 06, 08, 11)
6. **Reflection works when it is grounded in something checkable.** "Review your answer" is weak. "Here is exactly which test failed" is strong. (09)
7. **Add complexity only when you can measure it paying off.** More agents, planners and reflection loops all cost tokens and add failure modes. Compare against a simpler baseline. (07, 08, 09)
8. **You cannot improve what you cannot measure, and small samples lie.** Fifty test cases cannot tell 80% from 86%. Use confidence intervals and a release gate. (10)
9. **A protocol makes a server connectable, never trustworthy.** Tool descriptions are prompt text, tool definitions can change after approval, and results are untrusted. Pin, namespace, and re-verify. (11)
10. **Context is a budget you manage in code, and a summary must be verified before it replaces anything.** Facts supersede explicitly, untrusted facts are quarantined, and the raw transcript is never lost. (12)
11. **Memory is a governed database, not a model feature.** Identity is bound in code (the model never supplies a user id), writes need a trusted source and a verbatim quote, sensitive facts need consent, and deletion is verified against the raw file. (13)

## Quick start

```bash
git clone https://github.com/hsk9988660/genai-engineer-journey.git
cd genai-engineer-journey
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all_tests.py           # all 777 offline tests, one summary table
```

Run one lesson, or a few:

```bash
python run_all_tests.py 04 09     # only lessons whose folder name contains "04" or "09"
cd lesson-04-tool-use-in-claude && python -m pytest -q
```

> **Run tests per lesson, not from the repo root.** Each lesson is self-contained and several share file names (`fakes.py`, `test_agent.py`), so a single `pytest` from the root fails with collection errors. `run_all_tests.py` handles this by giving each lesson its own process.

**Requirements:** Python 3.10 or newer (the code uses `X | None` types). Developed and tested on Python 3.14.2 with `anthropic` 2.3.0, `pydantic` 2.14.2 and `pytest` 9.2.0. Other versions may work, but I have not tested them.

## The lessons

Read them in order the first time. Later lessons build on earlier ones.

| # | Lesson | What you build | Headline finding | Tests |
|---|---|---|---|---|
| [01](lesson-01-agent-loop/) | **The agent loop** | A framework-free tool-using agent with a step trace, step cap, and token counts | The core of every agent is a short loop. Tool errors go back to the model as data, and a step cap prevents runaway cost. | 5 |
| [02](lesson-02-frameworks-in-claude/) | **What frameworks do** | A 120-line mini framework (`@tool`, `Agent`, approval, streaming, conversation threads) | A framework is mostly convenience around the loop. The original demo disabled approval on a tool that spends money. | 11 |
| [03](lesson-03-patterns-in-claude/) | **Design patterns** | Clear instructions, structured output with validation retries, a two-agent pipeline | Code re-verifies the model's claims against ground truth. A valid shape is not a correct answer. | 10 |
| [04](lesson-04-tool-use-in-claude/) | **Tool use** | A safe read-only SQL tool, parallel calls, idempotent booking, a prompt-injection guard, a tool-selection eval | A read-only database role alone is not enough. Layered checks are. Building it exposed a bug where valid recursive queries were wrongly rejected. | 45 |
| [05](lesson-05-agentic-rag-in-claude/) | **Agentic RAG** | A chunker, BM25 / vector / hybrid search, permission filtering, citations verified in code | Permissions must be applied *before* ranking. Hybrid search only tied keywords with the offline embedder, and one question failed in every mode because of a vocabulary gap. | 39 |
| [06](lesson-06-trust-in-claude/) | **Trustworthy agents** | Policy from tool metadata, approvals bound to exact arguments, a PII vault, a tamper-evident audit log, budgets, a red-team scorecard | A keyword risk classifier (my re-typing of the original's logic) auto-ran every reworded dangerous action in the scorecard, while declared tool metadata did not. The comparison is partly stacked, and the lesson README says how. Metadata is only as good as its declarations. | 70 |
| [07](lesson-07-planning-in-claude/) | **Planning** | A plan as a validated graph, a parallel scheduler, replanning, approval of the whole plan | Missing facts should become questions, never invented. Quote-grounding checks the quote exists, not that the normalized value is right. | 54 |
| [08](lesson-08-multi-agent-in-claude/) | **Multi-agent** | Orchestrator and workers with isolated contexts, permission separation, an evidence-based checker, hand-off routing | Multiple agents are justified by isolation, permissions, parallelism and independent review, not by "specialization". Whether it beats one agent is untested here. | 54 |
| [09](lesson-09-metacognition-in-claude/) | **Metacognition** | Grounded reflection loops, recovery in code, stuck detection, calibration metrics, a memory that cannot be poisoned | Ungrounded self-review *can* turn a correct answer wrong. A scripted test shows the harness catches that, and the live rate is unmeasured. It reports fix rate *and* break rate. | 66 |
| [10](lesson-10-production-in-claude/) | **Production** | Telemetry, sampling, metrics, alerts, a statistical release gate, versioned eval sets, retries, a scope-safe cache | Cost per call is the wrong metric. Cost per successful outcome can reverse the ranking (shown with a hypothetical example, not real prices). | 106 |
| [11](lesson-11-protocols-in-claude/) | **Agent protocols (MCP, A2A)** | A real MCP server and client written from scratch over stdio, a security gateway (allowlist, pinned tool definitions, namespacing, taint), hostile demo servers, and an A2A-inspired task protocol | The lesson's notebooks contained no actual MCP or A2A. Building it exposed a startup-timeout flaw and a server that destroyed its own rejections. Not tested against any official SDK or host. | 135 |
| [12](lesson-12-context-engineering-in-claude/) | **Context engineering** | A budgeted context assembler with an inspection record, a versioned fact store, verified compaction, tool-result offloading, and measured tool selection | The lesson's notebook never reduced any history. Total tokens sent grow ~4x each time conversation length doubles. A failing summarizer retried on every turn until a cooldown was added. Not run against a live model. | 90 |
| [13](lesson-13-agent-memory-in-claude/) | **Agent memory** | A SQLite memory store with a typed key registry, supersession, quarantine, consent, verified deletion and a content-free audit log; a budgeted read path; a quote-verified write path; model tools with no user id | SQLite leaves "deleted" text in the file unless told otherwise, and tests now scan the raw file. A generic query missed a stored budget (kept as a labelled row). The notebook design leaks another user's data through a model-supplied `user_id`. Not run against a live model. | 92 |

## Find it by concept

Jumping in for one idea? Start here.

| I want to see... | Look at |
|---|---|
| The agent loop with no framework | [01 `agent.py`](lesson-01-agent-loop/agent.py) |
| A JSON schema generated from a Python function | [02 `mini_framework.py`](lesson-02-frameworks-in-claude/mini_framework.py) |
| Structured output with validation retries | [03 `patterns.py`](lesson-03-patterns-in-claude/patterns.py) |
| Parallel tool calls with timeouts | [04 `toolkit.py`](lesson-04-tool-use-in-claude/toolkit.py) |
| A read-only SQL tool an LLM cannot abuse | [04 `safe_sql.py`](lesson-04-tool-use-in-claude/safe_sql.py) |
| Defense against indirect prompt injection | [04 `toolkit.py`](lesson-04-tool-use-in-claude/toolkit.py), [06 `policy.py`](lesson-06-trust-in-claude/policy.py) |
| Hybrid search (BM25 + vectors + fusion) | [05 `index.py`](lesson-05-agentic-rag-in-claude/index.py) |
| Citations that cannot be fabricated | [05 `agent.py`](lesson-05-agentic-rag-in-claude/agent.py) |
| Human approval bound to exact arguments | [06 `approvals.py`](lesson-06-trust-in-claude/approvals.py) |
| Keeping PII away from the model | [06 `redact.py`](lesson-06-trust-in-claude/redact.py) |
| A tamper-evident audit log | [06 `audit.py`](lesson-06-trust-in-claude/audit.py) |
| Task graphs, cycle detection, parallel scheduling | [07 `plan.py`](lesson-07-planning-in-claude/plan.py), [07 `executor.py`](lesson-07-planning-in-claude/executor.py) |
| Context isolation between agents | [08 `runtime.py`](lesson-08-multi-agent-in-claude/runtime.py) |
| Hand-offs without infinite loops | [08 `router.py`](lesson-08-multi-agent-in-claude/router.py) |
| Retry, fallback and circuit breaker | [09 `resilience.py`](lesson-09-metacognition-in-claude/resilience.py) |
| Detecting an agent going in circles | [09 `stuck.py`](lesson-09-metacognition-in-claude/stuck.py) |
| Measuring whether reflection helps or hurts | [09 `harness.py`](lesson-09-metacognition-in-claude/harness.py) |
| Cost per outcome, percentiles, alerts | [10 `metrics.py`](lesson-10-production-in-claude/metrics.py) |
| Confidence intervals and a ship/block gate | [10 `stats.py`](lesson-10-production-in-claude/stats.py), [10 `regression.py`](lesson-10-production-in-claude/regression.py) |
| A cache that cannot leak across users | [10 `reliability.py`](lesson-10-production-in-claude/reliability.py) |
| Handling money data correctly | [10 `expense.py`](lesson-10-production-in-claude/expense.py) |
| The MCP wire protocol (JSON-RPC over stdio) | [11 `mcp_server.py`](lesson-11-protocols-in-claude/mcp_server.py), [11 `mcp_client.py`](lesson-11-protocols-in-claude/mcp_client.py) |
| Defense against tool poisoning and rug pulls | [11 `gateway.py`](lesson-11-protocols-in-claude/gateway.py) |
| Hostile and broken MCP servers to test against | [11 `demo_servers.py`](lesson-11-protocols-in-claude/demo_servers.py) |
| An agent whose tools are discovered at runtime | [11 `agent_loop.py`](lesson-11-protocols-in-claude/agent_loop.py) |
| Agent-to-agent tasks with a lifecycle (A2A-inspired) | [11 `a2a_lite.py`](lesson-11-protocols-in-claude/a2a_lite.py) |
| Building each model call under a token budget | [12 `assembler.py`](lesson-12-context-engineering-in-claude/assembler.py) |
| A fact that supersedes an older one (economy, then business) | [12 `facts.py`](lesson-12-context-engineering-in-claude/facts.py) |
| Summarizing without silently losing or inventing facts | [12 `compactor.py`](lesson-12-context-engineering-in-claude/compactor.py) |
| Offloading huge tool results behind a handle | [12 `tool_results.py`](lesson-12-context-engineering-in-claude/tool_results.py) |
| Choosing which tools to offer, and measuring recall | [12 `tool_loadout.py`](lesson-12-context-engineering-in-claude/tool_loadout.py) |
| Storing memories per user in a database | [13 `store.py`](lesson-13-agent-memory-in-claude/store.py) |
| Deleting a user's data so it is really gone from the file | [13 `store.py`](lesson-13-agent-memory-in-claude/store.py), [13 `test_deletion.py`](lesson-13-agent-memory-in-claude/test_deletion.py) |
| Choosing which memories to show, under a budget | [13 `retrieval.py`](lesson-13-agent-memory-in-claude/retrieval.py) |
| Saving only what the user really said | [13 `extract.py`](lesson-13-agent-memory-in-claude/extract.py) |
| Memory tools that cannot reach another user | [13 `manager.py`](lesson-13-agent-memory-in-claude/manager.py) |

## How each lesson is organized

Every lesson folder has the same shape:

```
lesson-NN-topic-in-claude/
  README.md        what the lesson teaches, how to run it, what is NOT known, exercises
  *.py             the implementation (small modules, one idea each)
  fakes.py         a scripted stand-in for the model, used by the tests
  test_*.py        offline tests: no API key, no network
  requirements.txt (lesson 01 only; the root one covers all lessons)
```

**A good way to study a lesson:**
1. Read its README first, especially "What is NOT known".
2. Run its tests and read the test *names*. They read like a list of guarantees.
3. Read the code the tests exercise.
4. **Break something on purpose and watch a test fail.** Delete a validation rule and see which test catches it. This teaches more than reading.
5. Do the exercises at the bottom of the README.

**Some code is deliberately duplicated across lessons** (a small PII redactor, a validated-model-call helper). Each lesson is self-contained so you can read or copy one folder without needing the others.

## Running things live

The tests never touch the network. Each lesson also has live entry points that call the real API:

```bash
set ANTHROPIC_API_KEY=sk-ant-...      # macOS/Linux: export ANTHROPIC_API_KEY=...
cd lesson-01-agent-loop && python agent.py "I want a warm beach holiday on a low budget"
```

| Lesson | Live command | Offline command (no key) |
|---|---|---|
| 01 | `python agent.py "..."` | |
| 02 | `python examples.py travel` | |
| 03 | `python patterns.py` | |
| 04 | `python app.py "..."`, `python evals.py` | |
| 05 | `python agent.py "..."`, `python evals.py live` | `python evals.py retrieval` |
| 06 | `python agent.py "..."` | `python redteam.py` |
| 07 | `python orchestrator.py "..."`, `python evals.py live` | `python evals.py lint` |
| 08 | `python research.py "..."`, `python router.py "..."`, `python compare.py` | |
| 09 | `python reflect.py "..."`, `python harness.py` | |
| 10 | `python expense.py receipt.jpg "Your Name"` | |
| 11 | `python agent_loop.py "..."` | `python server_main.py flights` (runs a demo MCP server over stdio) |
| 12 | `python harness.py live` | `python harness.py` (the strategy comparison, no model needed) |
| 13 | | `python harness.py` (naive vs managed memory, no model needed) |

- **Live runs cost real money.** The `evals`, `compare` and `harness` commands make many calls. Check your usage before running them.
- Most lessons read the model name from the `ANTHROPIC_MODEL` environment variable, falling back to `claude-sonnet-5-5`. Set it to the model you want. A few helpers have the default written in directly, so search a lesson for `claude-sonnet-5-5` if you change models.
- Never commit your key. `.env` is already in `.gitignore`.
- Commands that take actions (booking, refunding, publishing, sending) are stubs, and the ones with side effects ask for approval in the terminal.

## Honest status: read this before you trust anything

**Verified:** all 777 tests pass (`python run_all_tests.py`). Lessons 09, 10, 11, 12 and 13 were also *mutation-checked*: key behaviors were deliberately broken in a scratch copy to confirm a test fails each time, and each of those READMEs reports the result. Lessons 01 to 08 were not mutation-checked, so their tests have had less scrutiny.

**Not verified, and the lesson READMEs say so:**
- **Nothing was run against a live model.** Every model response in every test is scripted. The tests prove the *mechanics* (validation, caps, isolation, routing, statistics). They do **not** prove how a real model behaves.
- **The live evaluations have never been run.** `evals.py live`, `compare.py` and `harness.py` are how you would find out, for example, whether multi-agent beats a single agent, or whether reflection helps or hurts. The repo does not know the answers, and any such claim in a README is labelled a hypothesis.
- **Numbers in tests are scripted.** Token counts, prices (placeholders for made-up model names), and the "$2 per failure" example are arithmetic, not measurements.
- **Lesson 13's extractor is a regex test double.** Its comparison shows what the mechanisms do. How well a real model extracts and quotes facts is unmeasured, retrieval weights are untuned, retrieval is lexical only, and SQLite deletion was tested on one machine. There is no live harness.
- **Lesson 12's summarizers are labelled test doubles.** Its comparison shows what the *mechanisms* do. It says nothing about how well a real model summarizes, and `harness.py live` has not been run.
- **Lesson 11 was never tested against an official MCP or A2A implementation.** Its server and client agree with each other, and were written from my recollection of the specs, which differ between versions. Its A2A code is explicitly *inspired by* A2A and is not wire-compatible.
- **Several known limits are documented per lesson**, such as: regex PII detection misses names, a verbatim quote check does not prove a quote supports a claim, and a policy engine is only as good as the metadata developers declare.

Treat this as **learning material, not audited production code.** If you use any of it for something that matters, run the live evaluations on your own data first.

## FAQ

**Do I need an API key?** Not to run the tests or read the code. You need one only for the live commands.

**Why no framework (LangChain, AutoGen, the Microsoft Agent Framework)?** To see what they do for you. The original course uses Microsoft's framework and Azure services. Rebuilding on the raw API shows that the core is small, and it makes it easier to judge whether a framework is worth its cost. Lesson 02 builds a tiny framework to make that concrete.

**Will it work with other models?** The code uses Anthropic's API shapes (tool use blocks, forced `tool_choice`, `input_schema`). I have not tested other providers. The ideas transfer, but the code would need adapting.

**Why so many lessons about what can go wrong?** Because demos hide failure and production does not. Most of the value in lessons 04 to 13 is in the guards, checks and measurements, not in the happy path.

**Where are lessons 14 to 18?** Not built yet. The source course has more lessons (the Microsoft framework, browser use, deployment, local agents, security). This repo covers the first thirteen.

**I found a bug or disagree with something.** Please open an issue. I found several bugs in my own code while building this (and wrote them up in the lesson READMEs), so there are probably more.

## How this was made, and how it relates to the course

- The lessons follow the structure of Microsoft's [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners). This repo is **not affiliated with or endorsed by Microsoft or Anthropic**. It does not copy the course's code. It rebuilds the ideas on a different stack and points out weaknesses in the originals.
- The code, tests and lesson READMEs were written with **Claude (Anthropic's AI coding assistant)**, and the commit history carries a co-author line to that effect. The tests pass and many were mutation-checked, but this is not a substitute for your own review.
- Each lesson's critique of the original course material reflects what was read at the time. The course may have changed since.

## License

No license file has been added yet. Until one is, the code is shared for reading and learning, but others do not have an explicit legal right to reuse it. If you want to reuse something, please open an issue.

## Repository layout

```
genai-engineer-journey/
  README.md                          this file
  requirements.txt                   anthropic, pydantic, pytest
  run_all_tests.py                   run every lesson's tests, print one table
  lesson-01-agent-loop/              the loop
  lesson-02-frameworks-in-claude/    what frameworks do
  lesson-03-patterns-in-claude/      instructions, structured output, pipelines
  lesson-04-tool-use-in-claude/      safe tools, parallelism, injection defense
  lesson-05-agentic-rag-in-claude/   retrieval, permissions, verified citations
  lesson-06-trust-in-claude/         policy, approvals, PII, audit, budgets
  lesson-07-planning-in-claude/      validated plans, scheduling, replanning
  lesson-08-multi-agent-in-claude/   orchestration, isolation, routing
  lesson-09-metacognition-in-claude/ grounded reflection, recovery, calibration
  lesson-10-production-in-claude/    telemetry, evaluation, release gates
  lesson-11-protocols-in-claude/     MCP, A2A, and the trust layer between them
  lesson-12-context-engineering-in-claude/  budgeted context, facts, verified compaction
  lesson-13-agent-memory-in-claude/  governed memory store, verified deletion
```
