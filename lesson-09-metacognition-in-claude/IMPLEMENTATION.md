# Lesson 09: implementation guide

This explains **how the code works**: how an agent can check itself, recover from errors, notice it is going in circles, and know what it does not know, all with signals computed in code rather than the model's opinion of itself. The [README](README.md) says why the lesson exists and what was measured.

The module names are short, so here is the map first.

| Question | Module |
|---|---|
| Is this answer actually right? | `checks.py` (deterministic checks) and `reflect.py` (a grounded revise loop) |
| Does reflection help or hurt? | `harness.py` (fix rate, break rate, cost) |
| Should I answer at all? | `calibration.py` (confidence as checkable numbers, an abstention policy) |
| What goes wrong in a source or tool? | `resilience.py` (retry, fallback, circuit breaker) |
| Am I going in circles? | `stuck.py` (repeat, errors, no new information) |
| What should I remember? | `memory.py` (lessons that cannot be poisoned) |

---

## 1. The reflection loop, from start to end

### Where it starts
`solve_itinerary(client, task, constraints, memory=None, max_attempts=4)` in `reflect.py`, or `python reflect.py "Plan 2 days in Lisbon"`.

### What happens, in order
`solve_with_checks(generate, check, max_attempts)` is the generic loop; `ItineraryGenerator` is the model-facing half.

1. **Attempt 1** has no feedback. The generator sends the task and constraints and gets a validated `Itinerary` (a forced structured call, via `structured.call_structured`).
2. **Check it in code.** `check_itinerary` (in `checks.py`) returns a list of `Failure`s, each with a `code` and a message naming the specific thing that is wrong (`days_count`: wrong number of days; `duplicate_day`; `bad_time`: not HH:MM or an end not after its start; `overlap`: two activities in one day overlap, where touching is allowed; `too_many`: more activities in a day than allowed; `no_dinner`: no meal starting at 18:00 or later when dinner is required; `budget`: total cost over budget; `total_mismatch`: the stated total differs from the sum of the activity costs). These are plain code, so they **cannot be talked out of a failure**.
3. **Remember the best.** The loop keeps the candidate with the **fewest failures**, not necessarily the latest. A revision that made things worse is never shipped over a better earlier attempt.
4. **Stop sensibly:**
   - no failures: `passed`
   - **stalled:** this attempt's *fingerprint* (the set of `(failure code, day)` pairs, ignoring wording) equals the previous attempt's: the same failures twice in a row means more tries will not help
   - the cap is reached: `capped`
5. **Feed back the specifics.** The next attempt is given the previous itinerary, the exact failure messages, and "Fix every listed failure. Change as little else as possible."
6. **Learn from validated failures.** After the loop, every failure that a deterministic check confirmed is recorded in the memory (if one was supplied).

### Where it ends
A `LoopResult` with `status` (`passed`, `capped` or `stalled`), the best candidate, its failures, and every attempt, plus token usage.

**Why this and not "review your answer".** The notebook's style of self-evaluation asks the model to re-read its own answer with no information about what is wrong, which can turn a correct answer wrong. Here the feedback is concrete, and `harness.py` measures the difference.

---

## 2. Module reference

### `harness.py`: does reflection pay for itself?
Three designs run on the same cases and are judged by the **same deterministic checker**:
- `baseline`: one generation
- `ungrounded`: generate, then one "review your work" pass with **no failure information** (the notebook's style)
- `grounded`: the loop above

It reports **fix rate** (of the cases wrong at first, the share that ended right), **break rate** (of the cases right at first, the share that ended wrong), **net gain**, and cost. A reflection step that does not clearly beat its own break rate and its token cost should be deleted. I have not run it live.

### `memory.py`: a reflection memory that cannot be poisoned
Reflexion-style memory stores a lesson after a failure. The danger: a lesson distilled from untrusted text becomes a persistent instruction. So **the model never writes a lesson**. Lessons come from **templates keyed by the code of a failure a deterministic checker confirmed**. `record_failures(failures, trusted)` ignores untrusted failures entirely and ignores unknown failure codes (they are never turned into free text). Lessons are counted, expire after a TTL and are evicted by count, and `render` produces advice labelled as advice that must not override the task. The cost of the safety: it only learns failure types you already wrote a check and a template for.

### `calibration.py`: what you do not know, as numbers
- `coverage_and_accuracy(preds, threshold)`: the share answered, and accuracy on only those (**selective accuracy**)
- `risk_coverage`: that trade-off at several thresholds
- `expected_calibration_error`: the average gap between stated confidence and actual accuracy, over bins
- `AbstentionPolicy.decide(confidence, grounded)`: **grounding is checked first and cannot be outvoted by confidence** ("no verified supporting evidence"), then the confidence threshold

Verbalised confidence from a model is not reliably calibrated, so measure it on labelled data before trusting any threshold.

### `resilience.py`: recovery in code, not in the prompt
The notebook told the model in its prompt to try the primary system and use the backup on a 404. That is deterministic logic. Here the model calls **one tool** and `with_fallback(primary, backup, retries, breaker)` does the recovery:
- `NotFoundError` is **permanent**: no retry, and it does **not** count against the breaker (the service is healthy; the item just is not there). The backup is used.
- a **transient** error (timeout, connection error) is retried and **does** count against the breaker
- an unexpected exception is not retried blindly; the backup is used and the breaker is told
- `CircuitBreaker`: after `failure_threshold` consecutive transient failures the primary is **skipped entirely** until `reset_after_s` passes, then **one** trial call (`half_open`) decides whether to close it
- if the backup also fails, `AllSourcesFailed` lists every error
- the result is a `Sourced` value with `source` (`primary` or `backup`) and `notes`, so the final answer can be transparent about what happened

### `stuck.py`: noticing circles
`StuckDetector.observe(tool, args, ok, result)` uses only what actually happened:
- `repeat`: the exact same call more than `max_repeat` times
- `errors`: `max_consecutive_errors` failures in a row
- `no_new_info`: the last `window` results were byte-for-byte identical

`run_monitored_agent` adds the policy: the **first** time it happens the agent gets a corrective note and a chance to change approach (the detector is reset); the **second** time the run **stops** and asks the user for help. An agent that has twice been told it is stuck and still is should not keep spending.

### `structured.py`
One validated, retrying model call that returns a Pydantic object, with token accounting (`Usage`).

---

## 3. Design decisions

- **The signal comes from code.** Self-correction works when the failure is concrete and the model is told exactly what it is, and fails when it is "try again".
- **Return the best, not the last.** Revisions can regress.
- **Stop on a stall,** not just on a cap.
- **Memory fed only by validated facts.** Anything a model read could otherwise become a standing instruction.
- **Put recovery logic in code.** A prompt that says "if you get a 404 try the backup" works only when the model reads the error correctly.
- **Measure.** Both reflection (break rate) and confidence (calibration) are claims to test, not features to assume.

## 4. What it does not do

The checks cover only what you wrote a check for. The calibration numbers are only as good as your labelled data. The live evaluations (`harness.py`) have not been run, so whether reflection helps for a real model is an open question.

## 5. Running and testing

`python reflect.py "..."` and `python harness.py` need an API key. `python -m pytest -q` runs 66 offline tests.
