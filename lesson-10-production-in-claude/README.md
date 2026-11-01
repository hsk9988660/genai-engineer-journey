# Lesson 10: agents in production, built as machinery you can test

Lesson 10 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) says what to monitor and evaluate, but its "observability" is `time.time()` around one call, its evaluator is an uncalibrated 1-to-5 LLM score, and its expense demo contains neither. This folder builds the actual machinery.

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q                      # 106 tests, offline, no API key
set ANTHROPIC_API_KEY=sk-ant-...         # macOS/Linux: export ANTHROPIC_API_KEY=...
python expense.py receipt.jpg "Ana Ruiz" # live: reads a receipt, prints the claim, sends nothing
```

## What each file does

| File | Job |
|---|---|
| `telemetry.py` | Spans and one run record per request: versions, tokens, cached tokens, cost from a **dated** price table, latency, tools, errors, feedback. `TelemetryClient` wraps any client so model calls become spans automatically. |
| `sampling.py` | Which full traces to keep: always errors, escalations, thumbs-down, fallbacks, expensive and slow runs, plus a deterministic share of the rest. |
| `metrics.py` | p50/p95/p99 latency, cost per run and **per success**, effective cost including what failures cost, steps, tool failure rate, and alerts that compare a recent window with a baseline. |
| `stats.py` | Wilson intervals, paired bootstrap, Cohen's kappa. The arithmetic behind "86% vs 80% on 50 cases is noise". |
| `graders.py` | Code graders (tools, steps, cost, schema), plus a pairwise LLM judge that runs in both orders and reports position bias. |
| `evalset.py` | Versioned dataset (content hash), a stable dev/held-out split, and a redacting, unlabeled path from production failures to new cases. |
| `regression.py` | The release gate: ship, block, or inconclusive, with a hard block on any safety regression. Also pass-all-k consistency. |
| `reliability.py` | Retries with jittered exponential backoff for transient errors only (obeying Retry-After), idempotency keys, and a cache that cannot leak across users. |
| `expense.py` | The lesson's expense demo done properly: structured JSON, `Decimal`, reconciliation against the receipt total, policy in code, approval before sending. |
| `redact.py`, `structured.py`, `fakes.py` | PII redaction, one validated model call, and test support. |

## Design rules the code enforces
1. **Telemetry never takes the product down.** An unpriced model is flagged (`unpriced_calls`), not raised. A failing exporter is recorded in `export_errors`, not raised.
2. **Redact before storing.** Every payload that enters a span passes through redaction, and the user id is a salted HMAC.
3. **Summaries for every run, traces for sampled ones.** Metrics are never sampled, because a sampled average lies.
4. **Cost per outcome, not per call.** A test reproduces the lesson-10 arithmetic: a model 3x cheaper per call can be 5x dearer per outcome once failures cost money.
5. **The gate refuses to guess.** If a drop can't be ruled out it says `inconclusive`. A safety regression blocks even when the average goes up.
6. **The held-out set needs explicit confirmation**, and a case never moves between dev and held-out as the dataset grows.
7. **A cache key must include the caller's scope.** `scope` has no default, so forgetting it is a `TypeError`, not a data leak.
8. **A flagged expense claim is never sent automatically.** Sending needs an exact `"approve"`, and a crashing approver means no.

## The statistics, in one example
With 50 cases and a true rate near 80%, the 95% interval is about 67% to 89%. `wilson_interval(40, 50)` gives it, and a test checks that 80% and 86% both sit inside it. So `compare()` uses paired comparisons on the same cases, multiple runs per case, and a confidence interval. It only says `ship` when it is not worse than the tolerance with 95% confidence, and needs at least `min_cases` cases before it will say anything.

## How I checked the tests can fail
I mutation-checked ten behaviors in a scratch copy: unknown-model handling, errors always sampled, the safety gate, shipping on an inconclusive result, the cache scope, retrying permanent errors, sending a flagged claim, the held-out guard, the judge's swapped order, and redaction. All ten were caught.

One mutant first **survived**: removing a `redact_obj` call on tool arguments left all 106 tests passing. I checked rather than assumed it was a test gap. It was an **equivalent mutant**: `_clip` already redacts the serialized payload, so the extra call was dead weight. Breaking `_clip`'s redaction was caught immediately. I removed the redundant call.

## What is NOT known
- **Nothing here ran against a real model, real receipts, or real traffic.** I have no API key. Every model response in the tests is scripted.
- **The prices are placeholders** for made-up model names (`example-large`, `example-small`). The cost figures in the tests are arithmetic on those placeholders, and the "$2 per failure" figure is hypothetical. Set real prices for your models.
- **Cached-token handling is an assumption.** The code treats `input_tokens` as the non-cached input and reads cache reads from `usage.cache_read_input_tokens` when present. I believe Anthropic reports cache reads separately from `input_tokens`, but verify your provider's usage fields before trusting the cost.
- **`claude_judge` is unrun.** I have not measured how often it agrees with humans, which is exactly what `judge_agreement` is for.
- **OCR quality is unknown.** The expense tests script the extraction. Whether the model reads real receipts accurately (handwriting, crumpled paper, other currencies) needs your own receipts.

## Limits of the design
- **No OpenTelemetry integration.** This is its own small data model. Mapping it to OTel GenAI attributes and exporting to a real backend is the natural next step.
- **Metrics are computed in memory** over lists of records, not in a time-series database. Windows and retention are your job.
- **Reconciliation can be fooled by compensating errors:** one missing item and one extra item that cancel out would pass.
- **The duplicate-line check has false positives** (two identical coffees are legitimate). It sends those to a human, which is the safe direction.
- **The paired bootstrap treats cases as independent.** If your cases are clustered (many variations of one scenario), the interval is too narrow.
- **Gate defaults are arbitrary.** `tolerance=0.02` and `min_cases=30` are examples, not recommendations. Choose them from your risk.
- **The held-out guard is soft.** It stops accidents, not someone determined to peek.
- **PII redaction is regex.** It misses names and addresses. Use a real service for anything regulated.
- Currency handling is absent: `Decimal` fixes float noise but not multi-currency receipts or tax rules.

## Notes from building it
- All 106 tests passed on the first run, so I mutation-checked instead of trusting that. Nine of ten caught at once, and the tenth turned out to be redundant code (above).

## Exercises
1. Map `RunRecord` and spans to OpenTelemetry GenAI attributes and export to a local collector.
2. Run 50 real requests through `TelemetryClient` and compute `summarize()`. What is your cost per success?
3. Write 30 eval cases, run two prompt versions three times each, and call `compare()`. Does it say ship, block, or inconclusive? How many more cases would you need for a clear answer?
4. Label 30 pairs by hand, run `claude_judge`, and compute `judge_agreement` and `position_bias_rate`. Would you trust it?
5. Feed `expense.py` ten real receipts. How often does reconciliation catch a misread?
