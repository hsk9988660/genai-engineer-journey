import json

import pytest

from fakes import FakeClient, turn, use
from resilience import (AllSourcesFailed, CircuitBreaker, NotFoundError, make_flight_tool, with_fallback)
from stuck import StuckDetector, run_monitored_agent


# --- fallback ------------------------------------------------------------------------------------
class Counter:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def __call__(self, *a, **k):
        self.calls += 1
        o = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(o, Exception):
            raise o
        return o


def test_a_healthy_primary_is_used_and_the_backup_untouched():
    p, b = Counter("P"), Counter("B")
    r = with_fallback(p, b)("x")
    assert (r.value, r.source, r.notes) == ("P", "primary", []) and b.calls == 0


def test_not_found_goes_straight_to_the_backup_without_retrying():
    p, b = Counter(NotFoundError("no such item")), Counter("B")
    r = with_fallback(p, b, retries=3)("x")
    assert r.source == "backup" and p.calls == 1 and "not found" in r.notes[0]


def test_a_transient_error_is_retried_and_then_succeeds():
    p, b = Counter(TimeoutError("slow"), "P"), Counter("B")
    r = with_fallback(p, b, retries=1)("x")
    assert r.value == "P" and r.source == "primary" and p.calls == 2 and b.calls == 0


def test_a_transient_error_that_keeps_happening_falls_back_after_the_retries():
    p, b = Counter(TimeoutError("slow")), Counter("B")
    r = with_fallback(p, b, retries=2)("x")
    assert r.source == "backup" and p.calls == 3 and "unavailable" in r.notes[0]


def test_an_unexpected_error_is_not_retried_blindly():
    p, b = Counter(ValueError("bug")), Counter("B")
    r = with_fallback(p, b, retries=5)("x")
    assert r.source == "backup" and p.calls == 1


def test_when_everything_fails_both_errors_are_reported():
    with pytest.raises(AllSourcesFailed) as e:
        with_fallback(Counter(NotFoundError("a")), Counter(NotFoundError("b")))("x")
    assert len(e.value.errors) == 2 and "primary: a" in e.value.errors[0] and "backup: " in e.value.errors[1]


def test_backoff_sleeps_between_retries():
    slept = []
    with_fallback(Counter(TimeoutError("x")), Counter("B"), retries=2, sleep=slept.append)("x")
    assert slept == [0.1, 0.2]


# --- circuit breaker --------------------------------------------------------------------------------
def test_the_breaker_opens_after_repeated_transient_failures_and_skips_the_primary():
    now = [0.0]
    br = CircuitBreaker(failure_threshold=3, reset_after_s=30, clock=lambda: now[0])
    p, b = Counter(TimeoutError("down")), Counter("B")
    call = with_fallback(p, b, retries=0, breaker=br)
    for _ in range(3):
        call("x")
    assert br.state == "open" and p.calls == 3
    r = call("x")
    assert p.calls == 3 and r.source == "backup" and "circuit open" in r.notes[0]  # primary not even tried


def test_after_the_reset_time_one_trial_decides_whether_to_close_the_circuit():
    now = [0.0]
    br = CircuitBreaker(failure_threshold=2, reset_after_s=30, clock=lambda: now[0])
    p, b = Counter(TimeoutError("down"), TimeoutError("down"), "P"), Counter("B")
    call = with_fallback(p, b, retries=0, breaker=br)
    call("x"), call("x")
    assert br.state == "open"
    now[0] = 31.0
    r = call("x")
    assert r.source == "primary" and br.state == "closed"


def test_a_failed_trial_reopens_the_circuit():
    now = [0.0]
    br = CircuitBreaker(failure_threshold=1, reset_after_s=10, clock=lambda: now[0])
    call = with_fallback(Counter(TimeoutError("down")), Counter("B"), retries=0, breaker=br)
    call("x")
    now[0] = 11.0
    call("x")  # trial fails
    assert br.state == "open" and br.opened_at == 11.0


def test_not_found_does_not_count_against_the_breaker_because_the_service_is_healthy():
    br = CircuitBreaker(failure_threshold=2)
    call = with_fallback(Counter(NotFoundError("no")), Counter("B"), breaker=br)
    for _ in range(5):
        call("x")
    assert br.state == "closed"


def test_the_model_sees_one_tool_and_recovery_is_reported_in_the_result():
    tool = make_flight_tool()
    assert tool("Paris")["source"] == "primary"
    berlin = tool("Berlin")
    assert berlin["source"] == "backup" and "not found" in berlin["notes"][0] and "09:00" in berlin["flights"]
    assert "error" in tool("Atlantis")


# --- stuck detection --------------------------------------------------------------------------------
def test_varied_calls_are_not_stuck():
    d = StuckDetector()
    assert not any(d.observe("search", {"q": q}, True, {"r": q}).stuck for q in "abcde")


def test_the_same_call_repeated_is_flagged():
    d = StuckDetector(max_repeat=2)
    verdicts = [d.observe("search", {"q": "x"}, True, {"r": i}) for i in range(3)]
    assert [v.stuck for v in verdicts] == [False, False, True] and verdicts[2].kind == "repeat"


def test_consecutive_errors_are_flagged_but_a_success_resets_the_count():
    d = StuckDetector(max_consecutive_errors=3)
    assert not d.observe("a", {"i": 1}, False, "e").stuck
    assert not d.observe("a", {"i": 2}, False, "e").stuck
    assert not d.observe("a", {"i": 3}, True, "ok").stuck  # success resets
    assert not d.observe("a", {"i": 4}, False, "e").stuck
    d2 = StuckDetector(max_consecutive_errors=3)
    v = [d2.observe("a", {"i": i}, False, f"e{i}") for i in range(3)]
    assert v[2].stuck and v[2].kind == "errors"


def test_identical_results_across_different_calls_mean_no_new_information():
    d = StuckDetector(window=3)
    v = [d.observe("search", {"q": q}, True, {"results": []}) for q in "abc"]
    assert v[2].stuck and v[2].kind == "no_new_info"


# --- the monitored agent loop -------------------------------------------------------------------------
SEARCH = ({"name": "search", "description": "s", "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}},
          lambda q: {"results": []})


def agent(replies, **kw):
    client = FakeClient(replies)
    return run_monitored_agent(client, "sys", "find it", {"search": SEARCH}, **kw), client


def call_search(q, id_="t"):
    return turn(use(id_, "search", q=q))


def finish(text="done"):
    return turn(use("fin", "submit_answer", answer=text))


def test_an_agent_that_answers_normally_is_untouched():
    run, client = agent([call_search("a"), finish("found")])
    assert run.status == "answered" and run.answer == "found" and run.stuck_events == []


def test_repeating_the_same_call_triggers_guidance_and_the_agent_recovers():
    run, client = agent([call_search("x", "1"), call_search("x", "2"), call_search("x", "3"), call_search("y", "4"), finish("ok")])
    assert run.status == "answered" and len(run.stuck_events) == 1 and run.stuck_events[0].kind == "repeat"
    guided = [b for b in client.calls[3]["messages"][-1]["content"] if b["type"] == "text"]
    assert guided and "[MONITOR]" in guided[0]["text"] and "different approach" in guided[0]["text"]


def test_an_agent_that_stays_stuck_after_guidance_stops_and_asks_the_user():
    replies = [call_search("x", str(i)) for i in range(10)]
    run, client = agent(replies, max_steps=10)
    assert run.status == "stuck" and "Could you clarify" in run.answer and len(run.stuck_events) >= 2
    assert len(client.calls) < 10  # it stopped early instead of burning every step


def test_the_step_cap_still_applies_to_an_agent_that_is_not_flagged():
    replies = [call_search(f"q{i}", str(i)) for i in range(6)]
    tools = {"search": ({"name": "search", "description": "s", "input_schema": {"type": "object"}}, lambda q: {"n": q})}
    client = FakeClient(replies)
    run = run_monitored_agent(client, "s", "u", tools, max_steps=4)
    assert run.status == "capped" and run.steps == 4


def test_unknown_tools_and_tool_exceptions_become_error_results():
    def boom(q):
        raise RuntimeError("kaput")

    tools = {"search": ({"name": "search", "description": "s", "input_schema": {"type": "object"}}, boom)}
    client = FakeClient([turn(use("1", "search", q="a"), use("2", "nope")), finish()])
    run = run_monitored_agent(client, "s", "u", tools)
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["is_error"] for r in results if r["type"] == "tool_result"] == [True, True]
    assert "kaput" in results[0]["content"] and "No such tool" in results[1]["content"]
    assert run.status == "answered"
