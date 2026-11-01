"""Aggregate run summaries into the numbers a team actually watches, and alert on changes.

Use percentiles, not averages: a few 40-second runs vanish inside a 3-second mean but they are what users remember.
Judge cost per SUCCESSFUL task, including what failures cost you (a human taking over, a retry, a refund), because cost
per call alone rewards a cheap model that fails often.
"""

import math
from dataclasses import dataclass


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (the same convention as numpy's default)."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize(records, failure_cost_usd: float = 0.0, success_outcomes=("success",)) -> dict:
    """`records` are RunRecord objects. `failure_cost_usd` is what one failed run costs the business downstream."""
    n = len(records)
    if n == 0:
        return {"runs": 0}
    ok = [r for r in records if r.outcome in success_outcomes]
    costs = [r.cost_usd for r in records]
    rated = [r for r in records if r.feedback in ("up", "down")]
    total_cost = sum(costs)
    success_rate = len(ok) / n
    return {
        "runs": n,
        "success_rate": success_rate,
        "error_rate": sum(r.outcome == "error" for r in records) / n,
        "latency_p50_ms": percentile([r.latency_ms for r in records], 50),
        "latency_p95_ms": percentile([r.latency_ms for r in records], 95),
        "latency_p99_ms": percentile([r.latency_ms for r in records], 99),
        "cost_mean_usd": total_cost / n,
        "cost_p95_usd": percentile(costs, 95),
        "cost_total_usd": total_cost,
        "cost_per_success_usd": (total_cost / len(ok)) if ok else None,
        "effective_cost_per_run_usd": total_cost / n + (1 - success_rate) * failure_cost_usd,
        "steps_mean": sum(r.steps for r in records) / n,
        "steps_p95": percentile([r.steps for r in records], 95),
        "tool_failure_rate": (sum(r.tool_failures for r in records) / max(1, sum(r.tool_calls for r in records))),
        "fallback_rate": sum(r.fallbacks > 0 for r in records) / n,
        "rated_fraction": len(rated) / n,
        "thumbs_down_rate": (sum(r.feedback == "down" for r in rated) / len(rated)) if rated else None,
        "unpriced_calls": sum(r.unpriced_calls for r in records),
    }


@dataclass
class Alert:
    metric: str
    message: str
    current: float
    baseline: float


def check_alerts(current: dict, baseline: dict, min_runs: int = 20, latency_p95_increase: float = 0.5,
                 cost_mean_increase: float = 0.3, steps_mean_increase: float = 0.3,
                 error_rate_increase_abs: float = 0.05) -> list[Alert]:
    """Compare a recent window with a baseline window. Below `min_runs` in either, say nothing: small samples are noise."""
    if current.get("runs", 0) < min_runs or baseline.get("runs", 0) < min_runs:
        return []
    alerts = []

    def relative(metric, label, limit):
        cur, base = current.get(metric), baseline.get(metric)
        if cur is not None and base and cur > base * (1 + limit):
            alerts.append(Alert(metric, f"{label} rose {cur / base - 1:.0%} (limit {limit:.0%}): {base:.4g} to {cur:.4g}", cur, base))

    relative("latency_p95_ms", "p95 latency", latency_p95_increase)
    relative("cost_mean_usd", "mean cost per run", cost_mean_increase)
    relative("steps_mean", "mean steps per run (possible loops)", steps_mean_increase)
    if current["error_rate"] - baseline["error_rate"] > error_rate_increase_abs:
        alerts.append(Alert("error_rate", f"error rate rose from {baseline['error_rate']:.1%} to {current['error_rate']:.1%}",
                            current["error_rate"], baseline["error_rate"]))
    return alerts
