# Lesson 09: metacognition that is grounded in a signal

Lesson 09 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) asks agents to "reflect". Its hotel example computes a reflection and then ignores it, its "user feedback" is a rule hard-coded in the agent, and its fallback logic lives in a prompt. This folder builds the version that holds up, and a harness that measures whether reflection helps or hurts.

> **Reflect against a signal you can check, not a feeling.** Re-reading your own answer is the weakest form of self-correction. Failing a test and seeing exactly how is the strong form.

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q                      # 66 tests, offline, no API key
set ANTHROPIC_API_KEY=sk-ant-...         # macOS/Linux: export ANTHROPIC_API_KEY=...
python reflect.py "Plan 2 days in Lisbon"
python harness.py                        # baseline vs ungrounded vs grounded reflection (costs real money; not yet run)
```

## What each file does

| File | The idea |
|---|---|
| `checks.py` | **The signal.** Deterministic itinerary checks (budget, overlaps, per-day limit, dinner, total mismatch, times). Each failure message names the specific problem. |
| `reflect.py` | **Grounded loop.** Generate, check, feed back the exact failures, revise. Stops on success, on a cap, or when it **stalls** (the same failures twice). Returns the **best** candidate, not the last. |
| `resilience.py` | **Recovery in code.** Retry, fallback, and a circuit breaker, replacing the notebook's prompt-driven "if 404, try the backup". The model calls one tool. |
| `stuck.py` | **Stuck detection.** Repeated identical calls, consecutive errors, or identical results trigger a corrective note, then a stop that asks the user. |
| `calibration.py` | **Knowing what you don't know, as numbers.** Coverage, selective accuracy, calibration error, and an abstention rule where grounding can't be outvoted by confidence. |
| `memory.py` | **A reflection memory that can't be poisoned.** The model never writes a lesson. Lessons come from templates keyed by failure codes a checker confirmed. |
| `harness.py` | **Does it actually help?** Fix rate, break rate, net gain and cost multiplier across `baseline`, `ungrounded` and `grounded` designs. |

## The key measurement: fix rate AND break rate
- **fix rate**: of cases wrong on the first try, the share that ended right.
- **break rate**: of cases right on the first try, the share that ended wrong.

The `ungrounded` design is the notebook's style: generate, then "review and fix any problems" with no failure information. Because it always ships the revision, it **can turn a correct answer into a wrong one**, and the harness counts that. The `grounded` loop stops the moment the checks pass, so it cannot break a correct answer. A test demonstrates both, using scripted replies.

## Resilience details worth understanding
- **`NotFoundError` is permanent.** No retries, straight to the backup, and it does **not** count against the circuit breaker, because the service answered correctly and the item simply isn't there.
- **Timeouts and connection errors are transient.** They are retried with backoff and they do count against the breaker.
- **After `failure_threshold` consecutive transient failures the circuit opens** and the primary is skipped. After `reset_after_s`, one trial call decides whether to close it.
- The wrapper reports `source` and `notes`, so the model can still be transparent about a fallback.

## How I checked the tests can fail
I mutation-checked six behaviors in a scratch copy (my real files untouched). Each broken version was caught by a test: stall detection removed, returning the last candidate instead of the best, the untrusted-lesson guard removed, `NotFound` counted against the breaker, touching activities counted as an overlap, and a stuck agent that never stops.

## What is NOT known
- **Nothing here was run against a real model.** I have no API key here. All 66 tests use scripted replies, so they verify mechanics (loops, caps, detectors, arithmetic), not whether reflection helps a real model. `python harness.py` is how you find out, and I don't know the result.
- **The harness judges with the same checker the grounded loop optimizes against.** Passing the checks means the constraints were met. It does not mean the itinerary is good, realistic, or interesting. Constraint satisfaction is not quality.
- **Three cases is far too few** for any conclusion.
- **Calibration functions need labelled data from a real model.** I never measured whether Claude's stated confidence is calibrated, and I would not assume it is.

## Limits of the design
- **It needs a checker.** This works because itinerary rules are checkable in code. Many tasks have no cheap, reliable check, and then grounded reflection is not available. Do not pretend an LLM judge is equivalent.
- **Stall detection is a trade-off.** Stopping after two identical failure sets saves tokens, but a model that would have fixed it on a third try with the same feedback is cut off.
- **The stuck detector uses heuristics.** Legitimate repeated polling can look like "no new information". Tune `window` and `max_repeat` for your tools.
- **Memory only learns known failure types.** Safety comes from templates, so it cannot learn anything you didn't write a check and a template for.
- **The `ungrounded` design is my construction of the notebook's approach**, not the notebook's code.
- Threads, retries and breakers here are minimal. The breaker is per-process, not shared.

## Notes from building it
- I first wrote a broken test file (an import line that was nonsense) and rewrote it before running anything.
- After the first full run passed, I mutation-checked instead of trusting it. All six caught.

## Exercises
1. Run `python harness.py`. What are fix rate and break rate for `ungrounded`? Is net gain positive?
2. Add a new constraint (no more than 4 hours of walking per day). Write the check, the lesson template, and a test.
3. Make the grounded loop stop only after three identical failure sets. Does the fix rate change?
4. Collect 30 real answers with the model's stated confidence and the truth. Compute `expected_calibration_error`. Is it calibrated?
5. Make `ItineraryGenerator` pass `memory.render()` and run a batch twice. Do the lessons from run 1 reduce failures in run 2?
