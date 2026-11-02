# Lesson 08: implementation guide

This explains **how the code works**: how a multi-agent research pipeline and a support router are built, and how they are kept isolated, bounded and observable. The [README](README.md) says why the lesson exists and what the comparison harness is for.

---

## 1. The system on one page

```
 run_research(rt, question, corpus, approver, publish_fn)

   planner ─► SubTasks            (validated: unique ids, known doc ids, at most max_subtasks)
      │
      ▼  fan out, IN PARALLEL            each reader has a FRESH context and ONE tool (fetch_document)
   reader × N ─► WorkerResult    (answer + verbatim evidence quotes; failures are contained)
      │
      ▼
   merger ─► Report              sees answers and quotes only, never the raw documents; NO tools
      │
      ▼
   checker ─► Verdict            did not write the report; checks each claim against the cited evidence; NO tools
      │   rejected → merger revises (max_revisions)
      ▼
   format_body(report)           plain CODE builds the published text from the VERIFIED report
      ▼
   human approval ─► publish_fn  only the verified report can ever be published

 Shared by every agent:  Runtime = Tracer (a span tree) + Budget (one thread-safe cap on calls and dollars)
```

Only the steps that need judgment are agents (plan, read, merge, check). Everything deterministic is code: validation, fan-out, evidence verification, formatting, approval, publishing.

---

## 2. A research run, from start to end

### Where it starts
`run_research(rt, question, corpus, approver=None, publish_fn=None, max_workers=4, max_subtasks=5, max_revisions=1)`. From the command line: `python research.py "Compare AcmeCloud and BetaCloud on price, security and support"`.

### What happens, in order

1. **A root span** is opened in the tracer.
2. **Plan.** The `planner` agent is asked for `SubTasks`; `validate_subtasks` rejects an empty list, too many tasks, duplicate ids, a task with no documents, or a document id not in the corpus. Problems are fed back.
3. **Read, in parallel.** A `ThreadPoolExecutor` runs one `run_reader` per sub-task. Each reader:
   - gets a `Handoff` (its task and the document ids it may read, "everything the worker is allowed to know")
   - has exactly one tool, `fetch_document`, which **refuses any document not in its handoff**. The permission is enforced in code, not by the prompt
   - must return a `WorkerResult`; `validate` requires the right task id and, for an `ok` result, evidence whose `doc_id` was actually fetched and whose quote (at least 8 characters) appears **verbatim** in that document
   - a worker that cannot produce a valid result becomes a `failed` result, so **one bad worker does not sink the others** (a budget overrun is deliberately not swallowed)
4. **Merge.** The `merger` sees only `_merge_input`: each worker's answer and evidence **quotes**, wrapped in `<worker_result>` tags. It never sees a raw document, which is the point of context isolation. `_validate_report` requires at least one claim, each citing sources, and every cited task must exist and have succeeded.
5. **Check.** The `checker`, which did not write the report, receives each claim next to the evidence quotes it cites (`_check_input`) and returns a `Verdict`. If rejected, the merger gets the checker's problems and writes a corrected report, up to `max_revisions` times.
6. **Format.** `format_body` builds the publishable text **in code** from the verified report. A model never writes what gets published.
7. **Approval and publishing.**
   - checker still rejects: `unverified`. It is never published or even offered for approval
   - `approver is None`: `verified`, nothing published
   - otherwise the approver sees title and body; anything but `"approve"` (an exception included) is `declined`; approval calls `publish_fn` and the status is `published`

### Where it ends
`ResearchOutcome.status` is `published`, `verified`, `declined`, `unverified`, `over_budget` or `failed`. The span tree shows who called whom, with tokens, cost, time and outcome (`runtime.tracer.render()`).

---

## 3. The runtime: shared plumbing

`Runtime.run_agent(agent, system, user, output_model, parent, label, tools, tool_impls, validate, max_retries, max_steps)`:
1. opens a span; builds the tool list from **only** the tools passed to this agent plus `submit_result`
2. starts a **fresh message list** (isolation: nothing is shared with any other agent)
3. for each step: `budget.begin_call()`, then the model call with `tool_choice="any"`, then charge tokens to the budget and the span
4. runs the agent's own tools; on `submit_result` validates the schema and then the caller's `validate`; problems go back for a retry up to `max_retries`, then `AgentError`
5. always closes the span with a status and a short note

`Budget`:
- `begin_call` **atomically checks the limits and reserves one call**, so two parallel agents can never both take the last slot (a race found while building). The **call cap is exact**; the **dollar cap is not**, because a call's cost is known only after it returns, so calls already in flight can overshoot a little.
- The default token prices are placeholders; set real ones.

`Tracer`: spans with parent ids, token and cost totals, `by_agent()` summaries and an indented `render()`.

---

## 4. The support router

`run_support(rt, message, agents, start="triage", max_hops=3)`:
1. the current agent receives the customer's message and a short **note** from the previous agent, never the previous agent's system prompt or reasoning
2. it returns an `AgentTurn`: answer, or hand off to another agent
3. `validate` refuses a handoff to any agent in `allowed`'s complement: **an agent that already handled the conversation can never get it back**, and the prompt tells the model which agents are still allowed
4. more than `max_hops` hand-offs, or an agent that cannot produce a valid turn, **escalates to a human** ("A human agent will follow up with you.") instead of looping
5. statuses: `answered`, `escalated`, `over_budget`

---

## 5. Module reference

- `schemas.py`: the messages agents exchange: `Evidence`, `Handoff`, `SubTasks`, `WorkerResult`, `Claim`, `Report`, `Verdict`, `AgentTurn`, `SingleAnswer`. All structured, so every handoff can be validated in code.
- `runtime.py`: `Runtime`, `Tracer`, `Span`, `Budget`, `AgentError`, `BudgetExceeded`.
- `research.py`, `router.py`: as above.
- `corpus.py`: a tiny document set; `support_a` contains a prompt-injection attempt, which a reader that obeys it cannot act on because it has no dangerous tool.
- `compare.py`: runs the same questions through a **single agent** (all documents, one context) and the **multi-agent pipeline**, and reports quality, cost, API calls and wall time, plus quality per dollar. "Quality" is deliberately crude (the fraction of expected facts present in the answer). `min_agent_runs(n)` is the fewest runs the multi-agent design can use: planner, n readers, merger, checker. I have not run it live.

---

## 6. Design decisions

- **Multi-agent only for reasons that apply:** context isolation, permission separation, parallelism, independent review, one budget.
- **Least privilege per agent.** Readers can read; orchestrator and checker have no tools; publishing is plain code.
- **Review by someone who did not write it,** against the cited evidence, not against its own opinion.
- **Measure before you believe.** `compare.py` exists because "more agents" costs more calls and is not automatically better.
- **A trace tree.** Without it a multi-agent system is a black box.

## 7. What it does not do

The checker is another model, so it can be wrong. Evidence verification proves a quote exists, not that it supports the claim. The dollar cap can overshoot. The comparison metric is a crude proxy.

## 8. Running and testing

`python research.py "..."`, `python router.py "..."` and `python compare.py` need an API key. `python -m pytest -q` runs 54 offline tests with a scripted client.
