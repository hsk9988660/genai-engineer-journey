import pytest

from fakes import rec
from metrics import check_alerts, percentile, summarize
from sampling import SamplingPolicy


# --- sampling -----------------------------------------------------------------------------------
def test_the_decision_is_deterministic_per_run_id():
    p = SamplingPolicy(rate=0.5)
    assert [p.decide(rec(run_id="abc"))[0] for _ in range(5)] == [p.decide(rec(run_id="abc"))[0]] * 5


def test_the_random_share_matches_the_rate_across_many_runs():
    p = SamplingPolicy(rate=0.1)
    kept = sum(p.decide(rec(run_id=f"run-{i}"))[0] for i in range(10_000))
    assert 900 < kept < 1100


def test_rate_zero_keeps_nothing_random_and_rate_one_keeps_everything():
    assert not any(SamplingPolicy(rate=0.0).decide(rec(run_id=f"r{i}"))[0] for i in range(200))
    assert all(SamplingPolicy(rate=1.0).decide(rec(run_id=f"r{i}"))[0] for i in range(200))


@pytest.mark.parametrize("kwargs,expected_reason", [
    (dict(outcome="error"), "outcome=error"), (dict(outcome="escalated"), "outcome=escalated"),
    (dict(feedback="down"), "negative feedback"), (dict(fallbacks=1), "fallback used"),
    (dict(cost=0.50), "expensive run"), (dict(latency=30_000.0), "slow run")])
def test_runs_you_will_want_to_debug_are_always_kept(kwargs, expected_reason):
    policy = SamplingPolicy(rate=0.0, keep_cost_usd_above=0.1, keep_latency_ms_above=10_000)
    keep, reason = policy.decide(rec(**kwargs))
    assert keep and expected_reason in reason


def test_a_boring_successful_run_is_dropped_when_the_rate_is_zero():
    assert SamplingPolicy(rate=0.0).decide(rec())[0] is False


# --- percentiles ----------------------------------------------------------------------------------
def test_percentiles_interpolate_and_handle_edges():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)
    assert percentile([5], 99) == 5 and percentile([], 50) is None
    assert percentile([3, 1, 2], 0) == 1 and percentile([3, 1, 2], 100) == 3


def test_a_few_slow_runs_hide_in_the_mean_but_show_in_p95_and_p99():
    runs = [rec(latency=1000.0) for _ in range(95)] + [rec(latency=40_000.0) for _ in range(5)]
    s = summarize(runs)
    mean = sum(r.latency_ms for r in runs) / len(runs)
    assert mean < 3000 and s["latency_p50_ms"] == 1000 and s["latency_p99_ms"] == 40_000


# --- summaries --------------------------------------------------------------------------------------
def test_the_core_numbers():
    runs = [rec("success", 0.02, 1000, 3, tool_calls=4, tool_failures=1), rec("success", 0.04, 2000, 5, tool_calls=4),
            rec("error", 0.01, 500, 1, fallbacks=1), rec("success", 0.03, 1500, 4, feedback="up"),
            rec("success", 0.02, 1000, 3, feedback="down")]
    s = summarize(runs)
    assert s["runs"] == 5 and s["success_rate"] == 0.8 and s["error_rate"] == 0.2
    assert s["cost_total_usd"] == pytest.approx(0.12) and s["cost_per_success_usd"] == pytest.approx(0.03)
    assert s["tool_failure_rate"] == pytest.approx(1 / 8) and s["fallback_rate"] == 0.2
    assert s["rated_fraction"] == 0.4 and s["thumbs_down_rate"] == 0.5 and s["steps_mean"] == pytest.approx(3.2)


def test_empty_input_and_no_successes_are_handled():
    assert summarize([]) == {"runs": 0}
    assert summarize([rec("error")])["cost_per_success_usd"] is None


def test_cost_per_call_is_the_wrong_metric_cost_per_outcome_reverses_the_ranking():
    """Reproduces the lesson-10 explanation: the cheap model is 3x cheaper per call and 5x dearer per outcome."""
    small = [rec("success" if i < 70 else "error", cost=0.004) for i in range(100)]
    large = [rec("success" if i < 95 else "error", cost=0.012) for i in range(100)]
    a, b = summarize(small, failure_cost_usd=2.0), summarize(large, failure_cost_usd=2.0)
    assert a["cost_mean_usd"] < b["cost_mean_usd"]  # small wins on the per-call number
    assert a["effective_cost_per_run_usd"] == pytest.approx(0.604) and b["effective_cost_per_run_usd"] == pytest.approx(0.112)
    assert a["effective_cost_per_run_usd"] > 5 * b["effective_cost_per_run_usd"] - 0.01


# --- alerts -----------------------------------------------------------------------------------------
def window(n=50, **kw):
    return summarize([rec(run_id=f"r{i}", **kw) for i in range(n)])


def test_no_alerts_when_nothing_changed():
    assert check_alerts(window(), window()) == []


def test_each_kind_of_degradation_raises_its_own_alert():
    base = window()
    assert [a.metric for a in check_alerts(window(latency=2000.0), base)] == ["latency_p95_ms"]
    assert [a.metric for a in check_alerts(window(cost=0.02), base)] == ["cost_mean_usd"]
    assert [a.metric for a in check_alerts(window(steps=6), base)] == ["steps_mean"]
    erroring = summarize([rec("error" if i % 5 == 0 else "success", run_id=f"r{i}") for i in range(50)])
    assert "error_rate" in [a.metric for a in check_alerts(erroring, base)]


def test_small_samples_never_alert_because_they_are_noise():
    assert check_alerts(window(n=5, latency=99_999.0), window()) == []
    assert check_alerts(window(), window(n=5)) == []


def test_changes_inside_the_thresholds_are_tolerated():
    assert check_alerts(window(latency=1300.0), window(latency=1000.0)) == []  # +30% is under the 50% limit


def test_the_alert_message_says_what_changed_and_by_how_much():
    (alert,) = check_alerts(window(cost=0.02), window())
    assert "mean cost" in alert.message and "100%" in alert.message and alert.baseline == pytest.approx(0.01)
