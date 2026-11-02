import json
from datetime import date
from decimal import Decimal

import pytest

import observability
from agentdef import AgentResult, Tier
from cache import ResponseCache, cache_key, normalize, shareable
from cost import BudgetExceeded, CostLedger, request_cost
from harness import FakeClock, make_version
from observability import AlertRule, Metrics, RequestLog, evaluate_alerts, new_trace_id

V = make_version()


# --- cache keys and what may be shared --------------------------------------------------------------------------------------------------
def test_questions_that_differ_only_in_case_spacing_or_trailing_punctuation_share_a_key():
    assert normalize("  What is   your Return window?? ") == "what is your return window"
    assert cache_key("v1", "What is your return window?") == cache_key("v1", "what is  your return window")


def test_the_agent_version_is_part_of_the_key_so_a_new_prompt_never_serves_an_old_answer():
    assert cache_key("v1", "hello") != cache_key("v2", "hello")


@pytest.mark.parametrize("tools,ctx,ok,text,expected", [
    ((), False, True, "answer", True),
    (("search_policies",), False, True, "answer", True),
    (("get_order_status",), False, True, "answer", False),
    (("search_policies", "get_order_status"), False, True, "answer", False),
    (("issue_refund",), False, True, "answer", False),
    ((), True, True, "answer", False),
    ((), False, False, "answer", False),
    ((), False, True, "   ", False),
])
def test_only_answers_that_are_identical_for_everyone_and_changed_nothing_are_shareable(tools, ctx, ok, text, expected):
    assert shareable(V, AgentResult(text, tools, 1, 1, ok), ctx) is expected


# --- the cache itself -------------------------------------------------------------------------------------------------------------------
def test_a_stored_value_comes_back_and_a_missing_one_is_a_miss():
    c = ResponseCache()
    c.put("k", "v")
    assert c.get("k") == "v" and c.get("other") is None and (c.hits, c.misses) == (1, 1)


def test_entries_expire_after_the_ttl():
    clock = FakeClock()
    c = ResponseCache(ttl_s=60, clock=clock)
    c.put("k", "v")
    clock.advance(59)
    assert c.get("k") == "v"
    clock.advance(2)
    assert c.get("k") is None and c.expired == 1 and len(c) == 0


def test_the_cache_is_bounded_and_the_least_recently_used_entry_goes_first():
    c = ResponseCache(max_entries=2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")  # a is now more recent than b
    c.put("c", 3)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3 and c.evictions == 1 and len(c) == 2


def test_putting_the_same_key_again_replaces_the_value_and_refreshes_its_age():
    clock = FakeClock()
    c = ResponseCache(ttl_s=60, clock=clock)
    c.put("k", "old")
    clock.advance(50)
    c.put("k", "new")
    clock.advance(50)
    assert c.get("k") == "new"


# --- cost ----------------------------------------------------------------------------------------------------------------------------------
TIER = Tier("t", "m", (), Decimal("0.002"), Decimal("0.010"))


def test_cost_is_decimal_arithmetic_per_thousand_tokens():
    assert request_cost(TIER, 1000, 500) == Decimal("0.007") and isinstance(request_cost(TIER, 1, 1), Decimal)
    assert request_cost(TIER, 0, 0) == Decimal(0)


def test_the_ledger_tracks_spend_per_user_per_tier_and_in_total():
    led = CostLedger()
    led.record("a", TIER, 1000, 0)
    led.record("a", TIER, 1000, 0)
    led.record("b", TIER, 1000, 0)
    assert led.spent("a") == Decimal("0.004") and led.spent("b") == Decimal("0.002") and led.spent_total() == Decimal("0.006")
    assert led.report()["requests_by_tier"] == {"t": 3} and led.report()["total_cost"] == "0.006"


def test_a_user_over_budget_is_refused_but_other_users_are_not():
    led = CostLedger(user_daily_budget=Decimal("0.003"), global_daily_budget=Decimal("100"))
    led.record("a", TIER, 2000, 0)
    with pytest.raises(BudgetExceeded) as e:
        led.check("a")
    assert e.value.who == "user"
    led.check("b")


def test_the_global_budget_stops_everyone():
    led = CostLedger(user_daily_budget=Decimal("100"), global_daily_budget=Decimal("0.003"))
    led.record("a", TIER, 2000, 0)
    with pytest.raises(BudgetExceeded) as e:
        led.check("b")
    assert e.value.who == "global"


def test_budgets_reset_when_the_day_changes():
    day = ["2026-11-02"]
    led = CostLedger(user_daily_budget=Decimal("0.003"), today=lambda: day[0])
    led.record("a", TIER, 2000, 0)
    with pytest.raises(BudgetExceeded):
        led.check("a")
    day[0] = "2026-11-03"
    led.check("a")
    assert led.spent("a") == 0


def test_cache_hits_are_counted_and_cost_nothing():
    led = CostLedger()
    led.record_cache_hit()
    led.record_cache_hit()
    assert led.report()["served_from_cache"] == 2 and led.spent_total() == 0


def test_the_default_day_is_todays_iso_date():
    assert CostLedger().today() == date.today().isoformat()


# --- metrics -----------------------------------------------------------------------------------------------------------------------------
def test_counters_are_keyed_by_name_and_labels_in_any_order():
    m = Metrics()
    m.inc("requests", status="ok", tier="small")
    m.inc("requests", tier="small", status="ok")
    m.inc("requests", status="error")
    assert m.count("requests", status="ok", tier="small") == 2 and m.total("requests") == 3 and m.count("requests", status="nope") == 0


def test_totals_do_not_mix_up_metrics_whose_names_share_a_prefix():
    m = Metrics()
    m.inc("request", 5)
    m.inc("requests", 2, status="ok")
    assert m.total("request") == 5 and m.total("requests") == 2


def test_percentiles_use_nearest_rank_and_are_none_when_empty():
    m = Metrics()
    assert m.percentile("latency", 95) is None
    for v in range(1, 101):
        m.observe("latency", v)
    assert m.percentile("latency", 50) == 50 and m.percentile("latency", 95) == 95 and m.percentile("latency", 100) == 100 and m.percentile("latency", 1) == 1


def test_only_the_most_recent_samples_are_kept(monkeypatch):
    monkeypatch.setattr(observability, "MAX_SAMPLES", 5)
    m = Metrics()
    for v in range(1, 11):
        m.observe("x", v)
    assert m.percentile("x", 0.001) == 6  # the oldest five were dropped


def test_the_metrics_text_is_sorted_and_one_line_per_counter():
    m = Metrics()
    m.inc("b")
    m.inc("a", 2, k="v")
    assert m.lines().splitlines() == ["a{k=v} 2", "b 1"]


# --- the request log ----------------------------------------------------------------------------------------------------------------------
def test_the_log_accepts_ids_counts_and_codes_and_can_be_read_back_by_trace():
    log = RequestLog()
    log.emit("t1", "request", status="ok", ms=12, user="abc")
    log.emit("t2", "request", status="error")
    assert log.for_trace("t1") == [{"trace": "t1", "event": "request", "status": "ok", "ms": 12, "user": "abc"}]


@pytest.mark.parametrize("field", ["text", "Text", "QUERY", "prompt", "content", "messages", "answer", "response", "body"])
def test_the_log_refuses_fields_that_would_hold_what_a_user_typed(field):
    with pytest.raises(ValueError, match="not allowed"):
        RequestLog().emit("t", "e", **{field: "hi"})


def test_the_log_refuses_long_strings_even_under_an_innocent_name():
    with pytest.raises(ValueError, match="long string"):
        RequestLog().emit("t", "e", note="x" * 101)
    RequestLog().emit("t", "e", note="x" * 100)


def test_trace_ids_are_16_hex_characters_and_unique():
    ids = {new_trace_id() for _ in range(200)}
    assert len(ids) == 200 and all(len(i) == 16 and int(i, 16) >= 0 for i in ids)


# --- alerts ---------------------------------------------------------------------------------------------------------------------------------
def test_a_latency_alert_fires_when_the_p95_is_over_the_threshold():
    m = Metrics()
    for v in [1] * 90 + [10] * 10:
        m.observe("latency_s", v)
    assert evaluate_alerts(m, [AlertRule("slow", "p95", 5, metric="latency_s"), AlertRule("fine", "p95", 50, metric="latency_s")]) == ["slow"]


def test_an_error_rate_alert_needs_enough_traffic_to_mean_anything():
    m = Metrics()
    rule = AlertRule("errors", "ratio", 0.05, numerator={"status": ["error", "unavailable"]}, min_events=20)
    for _ in range(3):
        m.inc("requests", status="error")
    assert evaluate_alerts(m, [rule]) == []  # 3 of 3 failed, but 3 events prove nothing
    for _ in range(17):
        m.inc("requests", status="ok")
    assert evaluate_alerts(m, [rule]) == ["errors"]  # 3 of 20 = 15%


def test_a_counter_alert_and_an_unknown_kind():
    m = Metrics()
    m.inc("persist_failed", 3)
    assert evaluate_alerts(m, [AlertRule("lost", "counter", 2, metric="persist_failed")]) == ["lost"]
    with pytest.raises(ValueError):
        evaluate_alerts(m, [AlertRule("x", "magic", 1)])
