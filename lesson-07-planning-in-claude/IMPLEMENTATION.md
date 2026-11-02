# Lesson 07: implementation guide

This explains **how the code works**: how a user's request becomes a validated plan, a human-approved plan, an executed plan, and (if something fails) a repaired plan. The [README](README.md) says why the lesson exists and what was measured.

---

## 1. The system on one page

```
 request
   │
   ▼  1 extract_requirements   what the USER said; every fact carries a verbatim quote
   │     missing facts → QUESTIONS (status "needs_info"), never invented
   ▼  2 make_plan              the model writes a plan; validate_plan checks it in CODE;
   │                           problems go straight back to the model for a capped retry
   ▼  3 human approval         ONE approval for the whole plan, bound to plan_hash(plan)
   ▼  4 PlanExecutor.run       layer by layer, parallel inside a layer, from CODE
   │       per task: dependencies' outputs in, postcondition check, retry, skip dependents of a failure
   │       budget checked after every layer
   ├─ all succeeded → "completed"
   └─ something failed → make_replacement (keeps what already succeeded), approve the new plan, run again
                         at most max_replans times
```

The model proposes (facts, plan, replacements). **Code validates, schedules, runs and limits.**

---

## 2. A request, from start to end

### Where it starts
`plan_and_execute(client, request, approver, workers, today=None, max_replans=2, budget_usd=None)` in `orchestrator.py`. From the command line: `python orchestrator.py "Plan a trip to Paris from May 2 to May 9, budget 3000 dollars"`.

### What happens, in order

1. **Extract requirements.** `extract_requirements` makes one forced structured call. The model must fill a field **only if the user said it**, copy the supporting words into `quote`, write dates as `YYYY-MM-DD`, and set anything unsaid to `null`. `verify_requirements` then checks every quote actually appears in the request, dates parse, the budget is a number, and the return date is after the departure. Problems go back to the model for a retry.
2. **Ask, never invent.** `missing_questions` turns any missing required fact (destination, depart, return date) into a question. If there are any, the result is `status="needs_info"` and nothing is planned.
3. **Budget.** If no budget was passed but the user stated one, it is used as the spending limit.
4. **Make a plan.** `make_plan` asks the model for a `Plan` (a goal and a list of `Task`s with ids, an agent, dependencies and parameters). `validate_plan` runs as the **validator** inside `call_structured`, so a bad plan is rejected with the exact problems and retried. Failure after the retries is `invalid_plan`.
5. **Approval.** The loop starts. `_approved` shows a human the **whole plan** (hash, goal, every task, and which agents will "reserve or buy") through the approver. A deep copy is shown; anything but the exact word `"approve"`, an exception included, means **declined**.
6. **Execute.** `PlanExecutor(workers, budget_usd).run(plan, approved_hash=digest, prior=prior)`.
   - If any pending task is a side-effect agent (flights, hotels, activities) and `approved_hash` is not the hash of the plan **about to run**, it raises `ApprovalError`: the plan that runs must be the plan that was approved. It re-hashes what really runs.
7. **Branch on the outcome.**
   - over budget: `over_budget`
   - everything ran: `completed`
   - failures remain and the replan limit is reached: `failed`
   - otherwise `make_replacement` asks for replacement tasks for the unfinished part and returns a **combined** plan in which succeeded tasks are kept unchanged. That combined plan is approved again, and the executor is given `prior=` the earlier results so successful tasks are **not run again**.

### Where it ends
`Outcome.status` is one of `needs_info`, `declined`, `completed`, `failed`, `over_budget`, `invalid_plan`, `approval_mismatch`, with the questions, final plan, execution report, replan count and model usage.

---

## 3. The executor in detail

`PlanExecutor.run`:
1. Start the report from `prior` results; skip them.
2. For each **layer** of `layers(plan)` (tasks in a layer do not depend on each other):
   - a task whose dependency **failed or was skipped** is **skipped**, with the reason
   - the rest run **in parallel** (`ThreadPoolExecutor`)
   - a result goes into `results` and its cost is added; an error goes into `failed`
3. **After each layer**, if total cost exceeds the budget, stop with `halted_reason`. Money spent inside a layer cannot be taken back, so the budget can be overshot by up to **one layer**. The honest fix is to make workers quote-only and put the real purchase in a final step.

`_run_step`: look up the worker for the task's agent; build `inputs` from the outputs of the dependencies (**data flows through the graph**); try up to `max_attempts` times; after each result run the agent's **postcondition**: a function returning `None` if acceptable or a message if not (a flight needs an id and a positive price, a hotel needs a name, nights and a price, activities need at least one item, the itinerary needs a non-empty summary). A postcondition failure counts as a failed attempt.

---

## 4. Module reference

### `plan.py`: the plan as validated data
- Models: `AgentName` (flights, hotels, activities, itinerary), `Task`, `Plan`, `Fact`, `Requirements`.
- `validate_plan(plan, facts)` returns a list of problems, written so they can be fed back to the model: duplicate task ids; a task depending on itself or a task that does not exist; missing required parameters for the task's agent; **a parameter that carries something the user said must equal what they said** (so the planner cannot change the destination or dates); dependency cycles; more than one itinerary task; and an itinerary task that does not depend (directly or indirectly) on every other task.
- `layers(plan)` groups a valid plan for parallel execution. `plan_hash(plan)` is the SHA-256 of the canonical JSON.

### `planner.py`: the model-facing half
`call_structured` is a forced `submit_result` tool call with Pydantic validation plus a caller-supplied validator, retried with the problems up to `max_retries`, and counting model calls, tokens and validation failures in `Usage`. `extract_requirements`, `make_plan` and `make_replacement` build on it.

### `executor.py` and `orchestrator.py`
As above. `POSTCONDITIONS` is the per-agent result check; `SIDE_EFFECT_AGENTS` is flights, hotels and activities.

### `workers.py`
Deterministic stand-ins for real services. The "standard" hotel tier is **sold out in Lisbon**, which is what makes the replanning path real.

### `evals.py`
`python evals.py lint` runs offline: hand-written bad plans checked against the validator (it measures the **validator**, not how often a real model writes bad plans). `python evals.py live` measures the real planner; I have not run it.

---

## 5. Design decisions

- **A plan is a graph, not prose.** The original notebook flattened the plan into text and asked one agent to "work through the subtasks in order". A graph can be validated, scheduled, parallelised and resumed.
- **Facts are grounded.** A quote check stops the model inventing a destination, a date or a budget.
- **Missing information becomes a question.** The cheapest fix for a hallucinated value is not to ask the model for it.
- **One approval for the whole plan, bound to its hash.** A person reviews the plan once, and the executor refuses to run anything different.
- **Replan, but cap it, and keep what worked.** Replacement tasks are only for the unfinished part.

## 6. What it does not do

The workers are fakes. The spend check is after each layer, not within it. It does not roll back the effects of tasks that succeeded before a later task failed. The model-side quality (how often a real planner writes valid plans) is unmeasured.

## 7. Running and testing

`python evals.py lint` runs offline. `python orchestrator.py "..."` and `python evals.py live` need an API key. `python -m pytest -q` runs 54 offline tests.
