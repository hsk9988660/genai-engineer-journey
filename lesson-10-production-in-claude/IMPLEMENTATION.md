# Lesson 10: implementation guide

This explains **how the code works**: how an agent is observed, evaluated, gated before release, kept reliable and kept cheap in production. The [README](README.md) says why the lesson exists and what was measured.

This lesson is a toolkit rather than one pipeline, so the map comes first, then the one end-to-end flow (the expense-receipt agent), then each module.

| Question | Module |
|---|---|
| What happened in this run, and what did it cost? | `telemetry.py` (run record, spans, dated prices), `redact.py` |
| Which traces do I keep? | `sampling.py` |
| Is the product healthy right now? | `metrics.py` (percentiles, rates, alerts) |
| What do I test against? | `evalset.py` (versioned dataset, held-out split) |
| Did this one run do the right thing? | `graders.py` (code graders, pairwise judge) |
| Is the new version safe to ship? | `regression.py` + `stats.py` (the release gate) |
| What if the provider is flaky? | `reliability.py` (retry, idempotency, scoped cache) |
| A real agent to apply it to | `expense.py` (receipt to approved claim email) |

---

## 1. A receipt, from start to end

### Where it starts
`process_receipt(client, image_bytes, media_type, employee_name, approver, send_fn, policy=None)` in `expense.py`. From the command line: `python expense.py receipt.jpg "Your Name"` (it prints the email and always answers "deny", so nothing is sent).

### What happens, in order

1. **Extract.** One forced structured call (`structured.call_structured`) sends the image and asks for a `ReceiptExtraction`: merchant, line items (date, description, amount, category), tax, tip and the **printed total**. The prompt says to copy the printed total and never compute it. Money is `Decimal`, never float. If the call cannot produce a valid object the result is `extraction_failed`, reported, never swallowed.
2. **Check in code.** `build_claim` runs two groups of checks and collects every problem as text:
   - `reconcile`: items + tax + tip must equal the printed total (a mismatch usually means a line item was missed or misread), no duplicate lines (same date, description and amount), no non-positive amounts, at least one item
   - `policy_flags`: a single item over `Policy.max_single_item`, or a day's Meals above `meal_cap_per_day`
3. **Flagged means stopped.** Any problem gives `needs_review` and the claim is **never sent automatically**.
4. **Render the email in code.** `render_email` builds subject and body from the verified numbers. There is no model-written text to improvise or leave a placeholder in.
5. **Approval, fail closed.** The approver sees subject, body and total. Only the exact string `"approve"` counts; anything else, or an exception, is `declined`.
6. **Send.** `send_fn(subject, body)` is called and the status is `sent`.

### Where it ends
A `ClaimOutcome` with `status` (`sent`, `declined`, `needs_review` or `extraction_failed`), the `Claim` and a detail string.

The point of this example is not the receipt. The model does the reading (the one thing code cannot do); **code reconciles, applies the policy, writes the email and gates the send**. The rest of the lesson is how you would watch, test and release something like it.

---

## 2. Module reference

### `telemetry.py`: observability that cannot take the product down
- **One run record per request** (`RunRecord`): versions, tokens, cost, latency, outcome, tools, feedback. **One span per model or tool call.** `summary()` returns the record without spans, cheap enough for every run.
- `Telemetry.start_run(user_id, versions)` returns a `Run`. `record_model_call`, `record_tool_call`, `note_retry`, `note_fallback`, `add_feedback`, `finish(outcome)`.
- **Never crash the request.** Exporter failures go through `_safely` and are recorded as a problem. An unknown model price becomes a recorded problem (`UnknownModel` internally), not an exception in your product.
- **Redact before export.** `_clip` serialises, redacts, **then** truncates every payload that enters a span, so a new code path cannot skip redaction. The user id is replaced by a salted HMAC (`pseudonymize`).
- **Cost is computed from tokens and a dated `PriceTable`** (`cost(model, input_tokens, output_tokens, cached_tokens, on)`), so an old run keeps its old cost after prices change. `input_tokens` means the non-cached input. The prices in `EXAMPLE_PRICES` are placeholders for made-up model names; set your real ones.
- `TelemetryClient` wraps any client with `messages.create` so each model call becomes a span automatically. `MemoryExporter` and `JsonlExporter` are the two sinks.

### `redact.py`
Irreversible regex redaction (emails, phones, card-like numbers and similar) used on everything that enters a trace. A heuristic floor, not a guarantee: it misses names and free-form identifiers.

### `sampling.py`: which traces to keep
Summaries are kept for every run; full traces are expensive and sensitive, so they are sampled. `SamplingPolicy.decide(rec)` returns `(keep, reason)`:
- **always keep** runs you will want to debug: error, abandoned or escalated outcomes, thumbs-down, fallbacks, and runs above a cost or latency threshold
- keep a fixed share (`rate`, default 10%) of the rest
- the share is decided by `bucket(run_id)`, a hash of the run id mapped into [0, 1). It is deterministic, so a whole trace is kept or dropped together and replaying the same id gives the same answer

### `metrics.py`: is it healthy now?
- `percentile` and `summarize(records, failure_cost_usd, success_outcomes)`: success and error rate, p50/p95/p99 latency, mean/p95/total cost, cost per success, an **effective cost per run** that adds `failure_cost_usd` for each failure, steps, tool failure rate, fallback rate, thumbs-down rate and unpriced calls.
- `check_alerts(current, baseline, min_runs=20, ...)`: compares the current window with a baseline: **relative** limits for p95 latency (+50%), mean cost (+30%) and mean steps (+30%, a hint of loops), an **absolute** limit for error rate (+5 points), and says nothing when there are fewer than `min_runs` runs, so a tiny window cannot page anyone.

### `evalset.py`: the thing you test against
- `Dataset` is a list of `EvalCase`s with a **version that is a hash of its contents**, so a score always names the exact data it came from.
- `dev()` and `heldout()` split by a hash of the case id, so the split is stable when cases are added. `heldout(confirm_final=True)` must be called explicitly: the held-out set is for a final check, not for tuning.
- `add_from_failures` turns production failures into new cases, **unlabelled and redacted**: a human supplies the expected answer with `label(case_id, expected, tags)`. A failure is not a test until someone says what right looks like.

### `graders.py`: scoring one run
- Code graders return a `GradeResult`: `required_tools`, `forbidden_tools`, `max_steps`, `max_cost`, `schema_valid`, `custom`. `grade_all` runs a list of them. Prefer these: they are exact, free and repeatable.
- For subjective quality, `pairwise_judge` asks a judge function to compare two answers **in both orders**; if the two verdicts disagree about which answer is better the result is `inconsistent` instead of an invented winner. `position_bias_rate` is the share of inconsistent results. `judge_agreement` reports accuracy and Cohen's kappa against human labels. `claude_judge(client)` is a real judge built on a forced structured call.
- A model judge is a measurement instrument; check its bias and agreement before trusting it.

### `stats.py` and `regression.py`: the release gate
Each case is run several times per version, so a case has a **pass rate**, not just pass/fail (`per_case_pass_rates`). `compare(results_a, results_b, tags, gate, seed)` takes the current version `a` and the candidate `b`, scored on the same cases, and returns a `Decision`:

1. no cases in common: `inconclusive`
2. a **safety-tagged** case whose pass rate fell: `block`, whatever the average did
3. `paired_bootstrap` confidence interval for the mean difference: if the whole interval is below zero and the mean drop exceeds `tolerance`: `block` ("significantly worse")
4. fewer than `min_cases` cases in common: `inconclusive` (not enough data to trust a comparison)
5. lower bound of the interval is at least `-tolerance`: `ship` (not worse than the tolerance with 95% confidence, a **non-inferiority** test)
6. otherwise `inconclusive`: "cannot rule out a drop larger than the tolerance; add cases or runs"

The default `Gate` is `tolerance=0.02`, `safety_tags={"safety"}`, `min_cases=30`. `consistency_pass_all_k` is the **pass^k** number: the share of cases that pass on all of their first k runs, because an agent that passes a case 8 of 10 times is not reliable whatever its average says. `pass_rate_interval` gives an overall pass rate with a **Wilson** interval (`stats.wilson_interval`), which behaves properly at small counts and near 0 and 100%.

### `reliability.py`: surviving a flaky provider
- `is_transient(exc)`: judged by status code (408, 409, 429 and 5xx are transient) or, with no status, by exception class name, so it works across SDK versions. A bad request, auth error or content problem is permanent: retrying only wastes money.
- **Provider fallback is deliberately not built in**: a different model is a different system and needs its own evaluation before it serves users.
- `retry_with_backoff(fn, retries, base_s, cap_s, sleep, rng, on_retry)`: **full-jitter** exponential backoff, a random delay between 0 and `min(cap, base * 2^attempt)`, so many clients do not retry in lockstep. A provider-supplied `retry_after` is honoured. Non-transient errors and the final attempt re-raise. `sleep` and `rng` are injectable, which is how the tests run instantly.
- `idempotency_key(run_id, step, tool, args_repr)`: the same logical action always gets the same key, so a retried side effect can be recognised and not repeated.
- `ScopedCache`: an LRU cache with expiry, keyed by **(scope, prompt version, normalised input)**. `scope` is a required keyword with no default, so one user's cached answer cannot be served to another by omission, and changing the prompt version invalidates old entries.

### `structured.py` and `fakes.py`
`structured.py` is the same validated, retrying model call used in earlier lessons. `fakes.py` holds the scripted test doubles.

---

## 3. Design decisions

- **Observe everything cheaply, trace selectively.** Summaries for all runs, full traces only for the ones you will want.
- **Redaction at record time,** not as a later cleanup step.
- **Telemetry is not allowed to fail the product.**
- **A release decision is a statistical decision.** Averages from a few noisy runs mislead; the gate asks "can I rule out a meaningful drop?" and says "inconclusive" instead of guessing.
- **Safety cases are not averaged away.**
- **The held-out set is guarded** so it stays an honest estimate.
- **Do the work you can do in code in code** (reconcile, policy, email text, approval), and let the model do only the reading.

## 4. What it does not do

The prices are placeholders. Regex redaction is a floor. The statistics assume the cases are a fair sample of real traffic, which only you can ensure. The model judge needs its own validation. The expense agent has not been run against real receipt photos in this repository.

## 5. Running and testing

`python expense.py receipt.jpg "Name"` needs an API key. Everything else is a library. `python -m pytest -q` runs 106 offline tests.
