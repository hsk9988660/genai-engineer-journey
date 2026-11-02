# Lesson 14: implementation guide

This explains **how the code works**: a small workflow engine (steps and edges as data), what makes a run safe to pause, crash and resume, how approvals and side effects are controlled, and how agents hand work to each other. The [README](README.md) says why the lesson exists and what the harness measured.

The one idea: **control flow is data you can check before you run it, log while it runs, and resume after it stops.** The model proposes work inside steps; the graph, the limits and the side-effect rules are code.

---

## 1. The system on one page

```
 Workflow (workflow.py): nodes + edges, validated BEFORE running
   step(id, fn, kind="code"|"model", side_effect, max_visits)
   parallel(id, branches, join, timeout_s, require)
   edge(src, dst, when=None)   otherwise(src, dst)

 Workflow.run(payload, run_id, identity, checkpoints=, ledger=, responses=, limits=, tools=, middleware=, resume=)
      │
      ▼  loop:  pick next node → check caps → run the step → JSON-check the output → choose the next edge
      │            ├─ Paused (approval needed) → save, return status "paused" + the pending request
      │            ├─ any failure             → save status "failed", raise
      │            └─ success                 → update state, save checkpoint (atomic file)
      ▼
 RunResult(status "completed" | "paused", output, path, steps, tokens, pending)

 Inside a step, through ctx:
   ctx.call_tool(name, **args)     tools run through MIDDLEWARE (middleware.py)
   ctx.once(fn)                    a side effect at most once, via the EffectLedger (SQLite)
   ctx.request_approval(summary)   pause until a human approves exactly this
   ctx.charge(tokens)              count against the token cap

 Separate: HandoffRouter (handoff.py): agent-to-agent handoff with allow-lists, hop and ping-pong limits
```

---

## 2. A run, from start to end

### Where it starts
`Workflow.run(payload, run_id, identity, ...)`. `run_id` must be 1 to 64 letters, digits, `-` or `_` (it names a file, so it cannot contain `/` or `..`). `identity` says **who the run is for**; your auth code builds it and it is never a global.

### What happens, in order

1. **Validate the graph** (`validate`). Before anything runs it raises `GraphError` for: no start step; an edge to a step that does not exist; steps that can **never run** (unreachable from the start); a step that mixes a direct edge with conditional ones, or has several direct edges, or more than one `otherwise`; a cycle (unless you pass `allow_cycles=True`); and **a model step with a conditional edge** unless that edge says `trust_model_output=True`. That last rule is the heart of it: **a model step cannot choose the route.** Put a code step that checks the model's output in between.
2. **Safety preconditions.** If any step is marked `side_effect` there must be an `EffectLedger`.
3. **Load or create state.**
   - no checkpoint: build the state (start node, payload, visit counts, path, tokens, pending approval, approvals used, and a **fingerprint** of the graph)
   - a checkpoint exists and you did not pass `resume=True`: `RunExists`. You are never allowed to silently continue or overwrite
   - resuming: the stored fingerprint must equal the current graph's, or `CheckpointMismatch` ("the workflow changed since this run was checkpointed"). A completed run just returns its stored result.
4. **Approval responses** (`responses`) must match the pending request, or `UnknownApproval`.
5. **The loop**, for each node:
   - step cap reached: save `failed`, raise `LimitExceeded`
   - visit count: only **successful** steps count, so a retry keeps the same visit number and therefore the same effect key. More than `max_visits` raises `LoopLimit`, so a loop cannot run away.
   - run the step: a `code` or `model` step calls `fn(payload, ctx)`; a `parallel` step runs its branches (below)
   - the output must be JSON-serialisable (so it can be checkpointed)
   - `_route` chooses the next step: a single direct edge; otherwise the **one** condition that matches (none matching uses the `otherwise` edge, and none at all is `NoRoute`; two matching is `AmbiguousRoute`; a condition that raises is `RouteError`)
   - `Paused` (raised by `request_approval`): save the run as `paused` with the pending request and return
   - any `WorkflowError`: save `failed` and re-raise; any other exception: save `failed` and raise `StepFailed`
   - success: update payload, next step, visit count and path; record the approvals consumed; trace; **save the checkpoint**
6. When there is no next step, the run is `completed`.

### Where it ends
A `RunResult`: `completed` with the final payload, or `paused` with `pending` (request id, the step, the summary and its hash). A paused run is resumed by calling `run` again with the same `run_id`, `resume=True` and `responses={request_id: Approval(...)}`.

---

## 3. Module reference

### `workflow.py`
- **Limits** (`Limits`): `max_steps` (default 50) and `max_tokens` (default 100,000). `ctx.charge(tokens)` raises `LimitExceeded` over the cap. Each step also has `max_visits`.
- **Approvals** (`ctx.request_approval(summary)`):
  - the request id is `node:visit`; the hash binds the **run id, the request id and the exact summary**, so an approval cannot be replayed in another run or for a different thing
  - no response yet: raise `Paused`
  - a response whose `summary_hash` differs from the current hash: `StaleApproval` ("what was approved is not what is being asked now")
  - one already used: `ApprovalError`; a decision other than `approve` or `reject`: `ApprovalError`
  - the step **re-runs from its start** when resumed, so everything before the approval call must be free of side effects
- **Parallel steps**: each branch runs in a thread with a **timeout**. `require` says how many must finish: `"all"`, `"any"` or a number; fewer raises `BranchesFailed` naming what is missing. Otherwise `join(ok, missing)` combines the results, so the join policy decides whether a missing branch is fatal.
- `Ctx.once(fn)` runs a side effect at most once per `(run, step, visit)` using the ledger; `fn` receives the key.
- `fingerprint()` hashes the graph structure (start, nodes with kind and side-effect flag, edges).

### `checkpoint.py`: two stores, because they fail differently
- `CheckpointStore`: one JSON file per run, written to a temporary file, flushed, `fsync`ed, then moved into place with `os.replace`. A crash leaves the **previous complete file**, never a half-written one.
- `EffectLedger`: a SQLite table `key → result`, with `once(key, fn)` that runs `fn(key)` the first time and records its JSON result; later calls return the record without calling `fn`. Keys are `run:node:visit`.
- **The window no local file can close**: a crash *inside* the effect, after the outside world acted but before the ledger row exists. The fix is to pass the key to the remote system (`fn(key)`) so it can de-duplicate. This lesson does not model remote systems.

### `middleware.py`: tool-call middleware that may not quietly lie
The lesson's version ran the hotel tool, saw "no rooms", and for a priority member **rewrote** the result to "rooms available", so the agent booked a room that did not exist. Here `invoke(tools, middleware, name, args, identity)` runs the middleware chain around the real tool, and a middleware may change the result only through a **declared door**:
- `call.redact(new_value, reason)`: a trusted transformation of real data (hiding an email); the result stays real
- `call.override(new_value, reason)`: invents or replaces data; the result is marked **synthetic**, and `ToolResult.require_real()` refuses it, so call that before an irreversible step
- returning a different value without either door raises `UndeclaredOverride`

Ready-made middleware: `deny(names)`, `rate_limit(max_calls, per_seconds)` (counted **per identity**), `log_calls(trace)` (tool name, argument **names** and a hash of the user id, never values or results), `redact(patterns)` (applied to every string in the result, including nested ones).

### `handoff.py`: agent-to-agent handoff with guards
`HandoffRouter(agents, allowed, start, max_hops=4)`. Each agent is a function of an `AgentView` that returns a `Reply` (done) or a `Handoff(to, reason, summary)`. After each handoff, in order:
1. the target must be on the **calling agent's allow-list**, else `HandoffRefused` (the model proposes, code checks)
2. the same `(from, to)` pair twice is a ping-pong: `HandoffLoop`
3. more than `max_hops` hops: `HandoffLimit`
4. the next agent receives the user's text, the identity, and a **short capped summary and reason**, never the earlier agents' full transcript (`AgentView` has no field for it)

### `agents.py`
`claude_agent(client, instructions, targets, model)` returns an agent function whose `handoff_to` tool offers only the targets that agent is allowed to use. Tested with scripted replies only; I have not run it against a real model.

### `runtrace.py`
`Trace.emit(type, **fields)` for logging: event types, node ids, counts, durations and **hashes, never payload content**, so a trace is safe to log. `digest` is the stable hash used throughout.

### `harness.py`
Compares the lesson's patterns with the guarded versions on five scenarios: **routing** (a liar model that says every city has rooms; a code check against real availability decides), **middleware** (a priority override and a silent rewrite), **crash** (a simulated process death after the booking and before the checkpoint), **slow branch** (a timeout) and **handoff** (loops). The model is a labelled test double, so it shows what the mechanisms do when a model is wrong, not how often a real model is.

---

## 4. Design decisions

- **Validate the graph first.** Unreachable steps, ambiguous edges and loops are build-time errors, not production surprises.
- **A model step never routes.** Routing reads checked data in a code step.
- **Every run is bounded:** steps, tokens and visits.
- **Checkpoint after every step, atomically,** and refuse to resume against a changed graph.
- **Side effects go through a ledger,** keyed so a resumed step gets the recorded result instead of acting twice.
- **Approvals are bound to the run, the step and the exact summary, are single use,** and a stale one is refused.
- **Middleware cannot lie:** every change to a result is declared, and invented results are marked.
- **Handoffs are allow-listed and bounded,** and carry a short summary, not a transcript.

## 5. What it does not do

It is not distributed: one process, one checkpoint directory, no worker leasing. The crash window inside an effect is documented, not closed. Parallel branches that time out are abandoned, not killed (Python threads cannot be). There is no UI for approvals and no versioned migration of an in-flight run to a changed graph (it refuses instead). The Claude-backed agent is tested with scripted replies only.

## 6. Running and testing

`python harness.py` runs the five scenarios offline (scenario 4 uses small real sleeps). `python -m pytest -q` runs 99 offline tests.
