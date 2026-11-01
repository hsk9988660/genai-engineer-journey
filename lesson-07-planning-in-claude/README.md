# Lesson 07: planning, with a plan you can trust

Lesson 07 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) has a model write a plan and then hands it to one agent as prose. This folder treats the plan as **validated data** and executes it **from code**.

Division of labor: the **model plans** (three structured calls). **Code validates, schedules, checks, and enforces.**

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q                 # 54 tests, offline, no API key
python evals.py lint                # does the plan validator catch 12 kinds of bad plan? (offline)
set ANTHROPIC_API_KEY=sk-ant-...    # macOS/Linux: export ANTHROPIC_API_KEY=...
python orchestrator.py "Plan a trip to Paris from May 2 to May 9, budget 3000 dollars"
python evals.py live                # real-model behaviour (not yet run)
```

## The pipeline

1. **Extract** what the user said. Every fact must carry a quote that appears **verbatim** in the request. A date with no supporting quote is rejected, and the model is told to set it to `null` instead of guessing.
2. **Ask, don't invent.** Missing destination or dates become questions to the user. No plan is written, nothing is approved, nothing runs.
3. **Plan.** The model returns a `Plan`: tasks with an `Enum` agent, `depends_on`, and `params`. Code validates it (below). A rejected plan goes back with the exact problems, up to 2 retries.
4. **Approve once.** A human sees the whole plan, which agents will reserve or buy, and a hash. The approval is bound to that hash, and the executor re-hashes the plan it is actually about to run.
5. **Execute from code.** Layer by layer, independent tasks in parallel, each task given its dependencies' outputs, a postcondition checked on every result, retries, dependents of a failure skipped.
6. **Replan on failure.** The model writes replacement tasks for the unfinished part only. Succeeded tasks are kept and not re-run. The new combined plan needs fresh approval. At most `max_replans` times.

## What the validator checks (`plan.py`)
Duplicate ids, dependencies on tasks that don't exist, self-dependency, **cycles**, missing required params per agent, and **params that carry something the user said must equal what they said** (so a planner can't quietly change a date or destination). Also: exactly one `itinerary` task, and it must depend on every other task. An unknown agent is rejected by the schema (`Enum`), not by free text.

## Files

| File | Role |
|---|---|
| `plan.py` | Schema, validation, fact grounding, layering, plan hash |
| `planner.py` | The three model calls, each validated with feedback and retry |
| `executor.py` | Scheduler: parallel layers, data flow, postconditions, retries, budget, approval check |
| `orchestrator.py` | Ties it together: ask, plan, approve, execute, replan |
| `workers.py` | Deterministic stub workers (Lisbon's standard hotel tier is "sold out" so replanning is exercised) |
| `evals.py` | `lint` (offline validator check) and `live` (real model) |

## What the tests prove
A scripted model covers: the happy path; questions instead of invented dates; an invented date rejected then corrected; a plan that changes the user's dates rejected then fixed; a cyclic plan fixed; an unknown agent rejected; a declined plan running nothing; a crashing approver meaning no; the user's budget enforced; a sold-out hotel causing a replan that keeps finished work and needs re-approval; replacement tasks refused if they reuse ids; the replan cap; and a plan changed after approval being refused by the executor.

## Limits you should know
- **The live eval has not been run.** I have no API key here. `python evals.py lint` only tests the validator on hand-written bad plans. It says nothing about how often a real model writes them. `live` measures that, and I don't know the result.
- **Plan *quality* is not measured at all**, only plan *validity*. A plan can pass every check and still be a poor plan (missing a sensible step, wrong order of preference). That needs golden plans or a judge model.
- **Quote grounding checks that the quote exists, not that the normalized value is right.** If a user says "May 2" and the model writes `2027-05-02`, the quote passes but the year may be wrong. Nothing here verifies that.
- **Budget can be overshot by one layer.** It is checked after each layer, because tasks in a layer run together and reservations already made can't be undone. A test shows this explicitly. If it matters, make workers quote-only and buy in a final step.
- **Workers are stubs.** No real services, no per-step timeouts, and retries are blind (the same call again).
- **Postconditions are simple** (a price is positive, nights > 0). They catch broken results, not subtly wrong ones.
- **Threads, not async.** Fine for a few tasks, not for hundreds.
- **The approver only fails closed on exceptions, not on time.** Unlike lesson 06 there is no approval timeout here.

## Notes from building it
- One test failed because of my own test data: it included a budget quote that wasn't in that test's request. The quote verifier correctly rejected it, which is the feature working.
- One full-suite run took over 3 minutes of wall time while pytest itself reported 13.8s. The next run took 2.1s. I could not reproduce it or find a cause. It looks like transient machine load, but I haven't confirmed that.

## Exercises
1. Add a year check: verify the month and day of a normalized date against the quote text. Does it catch `2027` vs `2026`?
2. Make the executor retry with the previous error message passed to the worker instead of a blind retry.
3. Add a `plan_quality` check: every request keyword like "museum" must appear in some task description. How easy is it to fool?
4. Make the budget check predictive: have workers return a price estimate first and stop before reserving anything that would exceed the budget.
5. Run `python evals.py live`, then read which cases failed and whether the planner's first answers were accepted.
