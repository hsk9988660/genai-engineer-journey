# Lesson 14: workflows, middleware and handoff, as checked code

Lesson 14 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) tours Microsoft Agent Framework (MAF): agents, threads, middleware, and *workflows* (executors joined by edges, with pause-for-human and handoff). This folder rebuilds the useful ideas as a small engine on the Claude API, with no framework, and puts guards where the lesson's notebooks have none.

> **A workflow turns an agent's control flow into data: a graph you can check before it runs, log while it runs, pause, and resume. The model works *inside* the boxes. Code owns the arrows.**

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q          # 99 offline tests, no API key, a few seconds
python harness.py            # the five scenarios below (offline; labelled test doubles; small real sleeps in scenario 4)
```

There is no `harness.py live`. Nothing in this lesson has touched a real model.

## What is built

| File | What it is |
|---|---|
| `workflow.py` | **The core.** Steps, conditional/otherwise edges, parallel steps with timeouts, graph validation, step/token/visit caps, checkpoint and resume, pause for approval. |
| `checkpoint.py` | An atomic JSON checkpoint per run, and a SQLite **effect ledger** so a side effect happens once even if a step is re-run after a crash. |
| `middleware.py` | Tool-call middleware (deny, rate limit, log, redact) where a result can only be changed through a *declared* door, and an invented result is marked **synthetic**. |
| `handoff.py` | Agent-to-agent handoff with an allow-list, hop cap, ping-pong detection, and a capped summary instead of the whole transcript. |
| `agents.py` | One Claude-backed agent that answers or hands off. Scripted-reply tests only. |
| `runtrace.py` | A trace that holds event types, node ids, counts and hashes, never payload content. |
| `harness.py` | The lesson's patterns versus the guarded ones, on five scenarios. |

## The rules the code enforces
1. **The graph is validated before anything runs.** Missing or dangling steps, steps that can never run, mixed direct and conditional edges, two `otherwise` edges, and cycles (unless you ask for them) are all errors.
2. **A model step cannot choose the route.** A conditional edge may only leave a *code* step, which reads checked data. The escape hatch (`trust_model_output=True`) is explicit and visible in the graph.
3. **Routes are never ambiguous.** Two matching conditions raise `AmbiguousRoute` rather than silently taking the first. No match and no `otherwise` raises `NoRoute`.
4. **Every run is bounded.** A step cap, a token cap, and a visit cap per step, so a loop cannot run away.
5. **State is checkpointed after every step**, atomically, and a run resumes from it. Resuming ignores the payload argument and uses the saved one. A checkpoint from a changed graph is refused.
6. **Side effects go through `ctx.once`.** The ledger key is `run:step:visit`, and a retry after a failure keeps the same visit number, so the effect is recorded once.
7. **An approval is bound to the exact thing shown.** The hash covers the run id, the request id and the summary. It is single-use. A stale approval, one from another run, one for a request that is not pending, and an invalid decision are all refused.
8. **Parallel steps have timeouts and a join policy** (`all`, `any`, or a number). A missing branch is reported by name, and whether it is fatal is your choice, not an accident.
9. **Middleware cannot lie quietly.** Changing a result without `redact()` or `override()` raises `UndeclaredOverride`. An `override()` result is synthetic, and `require_real()` refuses it before an irreversible step.
10. **Identity is a value you pass in** (`Identity`), never a module global.
11. **A handoff target must be on the caller's allow-list.** Hops are capped, the same (from, to) pair twice is refused, and the next agent gets the user's request plus a short summary, not the transcript.
12. **The trace and the logs contain no content:** tool names, argument *names*, hashed ids, output hashes.

## Measured results

`python harness.py`. The "model" is a test double that claims every city has rooms.

| Scenario | The lesson's pattern | Guarded |
|---|---|---|
| Route on whether rooms exist (Paris has none) | routes on the model's claim: **books Paris** (1 booking with no rooms) | a code step checks inventory first: books only Stockholm |
| Middleware rewrites "no rooms" to "rooms" for a priority user | a declared override that the step ignores: **books** (1). A *silent* rewrite is refused (`UndeclaredOverride`) | the step calls `require_real()`: `SyntheticFact`, 0 bookings |
| The process dies after booking, before the checkpoint | restart from the top: **2 bookings** | checkpoint plus ledger: **1 booking**, 1 email, completed |
| One of three parallel branches is slow (0.6 s) | waits for all three | timeout 0.15 s, 2 of 3 required: returns without `weather`, names it as missing |
| Two agents hand a request back and forth | the loop only stops at my manual cap of 50 hops | `HandoffLoop` after 2 handoffs |

**How to read this honestly.**
- The *"declared override that the step ignores"* row matters: **the synthetic flag only protects you if the step checks it.** Nothing forces a step to call `require_real()`. It is a door that exists, not a wall.
- The handoff "unguarded" baseline is a hand-written loop I wrote to show the absence of a brake. It is not what MAF does, and it is a weaker comparison than the others.
- The crash is *simulated* (a `BaseException` raised while saving). I did not kill a real process.

## Findings from building it
1. **A module called `trace.py` shadows a standard-library module.** I renamed it `runtrace.py` before it caused a confusing import error.
2. **A wrong test premise showed the safer design.** I wrote a test where the amount changes between approval and resume. It *did not* raise, because resume uses the saved payload and ignores the argument. That is the behavior I want, so I rewrote the test to assert it, and added a separate test for "the step now asks something different".
3. **The validator caught my own test fixtures.** My first branching helper left unused steps behind, and `validate()` refused them as unreachable. That is the check working.
4. **The first approval hash did not include the run id**, so an approval issued for one run could be replayed on another run with an identical request. I bound it to the run, and a test covers it.
5. **All 58 mutants were caught on the first run**, which made me suspicious, so I checked that every mutant at least compiles. All 58 do, so the catches are not syntax errors.
6. **The crash window inside an effect cannot be closed locally.** If the process dies after the outside world acted and before the ledger row is written, the effect repeats. The honest fix is to pass the key to the remote system. See below.

## How I checked the tests can fail
I mutation-checked 58 behaviors in a scratch copy: each graph-validation rule, route ambiguity, the step/visit/token caps, resume safety (fingerprint, completed-run shortcut, existing-run refusal, ignoring the payload on resume), approval binding (run id, hash, decision, single-use, persistence, unknown request), parallel requirement and timeout, atomic saving, run-id validation, the ledger, undeclared and synthetic overrides, per-user rate limiting, logging without content, redaction, and every handoff guard. **All 58 were caught.** I do not take that as proof the suite is complete, only that these 58 changes are detected.

## What is NOT known
- **Nothing ran against a live model.** `claude_agent` is tested with scripted replies, and I do not know how often a real model would hand off wrongly or loop.
- **I did not run Microsoft Agent Framework.** My critique comes from reading its README and notebooks. MAF may well provide checkpointing, durability and observability that those notebooks do not demonstrate, and its features may be stronger than what I describe.
- **No real crash test.** Resume was tested by simulating a crash during a checkpoint write, not by killing a process or pulling a plug.
- **Atomic replace was tested on Windows only.** I did not `fsync` the directory after the rename, so on a power loss the rename itself may not be durable on some filesystems.
- **Latency and throughput are unmeasured.** Every step writes a file.

## Limits of the design
- **`kind="model"` is a label the author sets.** A code step that just forwards a model's output unchecked defeats rule 2. The engine cannot tell.
- **The synthetic flag is advisory** (see above), and `redact()` is trusted code: nothing verifies that a redaction did not change the meaning.
- **A timed-out parallel branch is not killed.** Python threads cannot be stopped, so a slow branch keeps running and its side effects can still happen. The engine stops *waiting* for it.
- **Approvals cannot be requested inside a parallel branch.** The pause would be recorded as a missing branch.
- **A step that pauses re-runs from its start on resume.** Everything before `request_approval` must be free of side effects.
- **One process only.** There is no lock on a run id, so two processes resuming the same run would race.
- **Synchronous.** There is no async, streaming, or timer-based "wait until tomorrow".
- **Checkpoints hold the payload in plain JSON.** That can include user data. They are not encrypted and there is no deletion API, so see lesson 13 for what proper deletion needs.
- **Approvals are not authenticated.** The `approver` string is recorded but not verified. Who may approve is your system's job.
- **The visit cap is per step, not per branch.**

## Exercises
1. Wire `claude_agent` to a real model and run a three-agent handoff. How often does it hit `HandoffRefused` or `HandoffLoop`?
2. Replace the simulated crash with a real `os._exit(1)` in a child process, then resume. Does the ledger still hold?
3. Make `require_real()` mandatory: have `ctx.once` refuse to run if any tool result used in the step was synthetic.
4. Add an async version of `Workflow.run`. What happens to the parallel timeout?
5. Make a timed-out branch cancellable by running branches in processes. What do you give up?
