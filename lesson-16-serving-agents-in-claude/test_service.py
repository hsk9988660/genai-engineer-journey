import json
import threading
import time
from decimal import Decimal

import pytest

import support_tools
from admission import Admission
from agentdef import AgentResult
from cache import ResponseCache
from cost import CostLedger
from harness import FakeClock, Provider, make_service, make_version, rule_based_agent
from release import Rollout
from resilience import CircuitBreaker, Permanent, RetryPolicy, Transient
from router import ToolNotAllowed
from service import AgentService, Request, Response
from threads import ThreadStore


def answer(text="the answer", tools=(), tin=100, tout=20):
    return lambda call: AgentResult(text, tuple(tools), tin, tout)


def svc(responder=None, **kw):
    provider = Provider(responder=responder or answer())
    kw.setdefault("sleep", lambda s: None)
    return make_service(provider, **kw), provider


def ask(s, text="What is your return window?", user="alice", **kw):
    return s.handle(Request(user, text, **kw))


# --- the happy path -------------------------------------------------------------------------------------------------------------------
def test_a_question_gets_an_answer_a_thread_a_version_a_tier_and_a_trace_id():
    s, p = svc()
    r = ask(s)
    assert (r.status, r.http, r.text, r.tier, r.cached) == ("ok", 200, "the answer", "small", False)
    assert r.version == make_version().version_id and len(r.trace_id) == 16 and r.thread_id and p.calls == 1


def test_a_trace_id_supplied_by_the_caller_is_kept_and_appears_in_the_log():
    s, _ = svc()
    r = ask(s, trace_id="abc123")
    assert r.trace_id == "abc123" and s.log.for_trace("abc123")[0]["status"] == "ok"


def test_the_first_turn_is_saved_to_the_users_thread_and_a_follow_up_sees_the_history():
    s, p = svc()
    r1 = ask(s, "Where is my order?")
    seen = []
    p.responder = lambda call: seen.append([m["content"] for m in call.messages]) or AgentResult("second", (), 10, 5)
    r2 = ask(s, "And when will it arrive?", thread_id=r1.thread_id)
    assert seen == [["Where is my order?", "the answer", "And when will it arrive?"]] and r2.thread_id == r1.thread_id
    assert s.store.count(__import__("threads").Scope("alice"), r1.thread_id) == 4


def test_cost_and_metrics_are_recorded_for_every_request():
    s, _ = svc()
    ask(s)
    assert s.ledger.report()["requests_by_tier"] == {"small": 1} and s.ledger.spent("alice") > 0
    assert s.metrics.count("requests", status="ok") == 1 and s.metrics.count("tier_requests", tier="small") == 1 and s.metrics.percentile("latency_s", 50) is not None


def test_the_log_records_status_tier_and_a_hashed_user_but_nothing_the_user_said():
    s, _ = svc(answer("SECRET-ANSWER"))
    ask(s, "SECRET-QUESTION", user="bob-7731")
    blob = json.dumps(s.log.events)
    assert "SECRET" not in blob and "bob-7731" not in blob and s.log.events[0]["status"] == "ok" and len(s.log.events[0]["user"]) == 12


# --- bad input ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{"user_id": ""}, {"user_id": None}, {"text": ""}, {"text": "   "}, {"text": "x" * 4001}, {"thread_id": ""}, {"thread_id": "t" * 65},
                                {"idempotency_key": "k" * 65}, {"deadline_s": 0}, {"deadline_s": 301}, {"deadline_s": "soon"}, {"text": 5}])
def test_invalid_requests_are_rejected_before_any_work_is_done(kw):
    s, p = svc()
    args = {"user_id": "alice", "text": "hello"} | kw
    r = s.handle(Request(**args))
    assert r.status == "invalid" and r.http == 400 and p.calls == 0 and s.ledger.spent_total() == 0


# --- threads and ownership -------------------------------------------------------------------------------------------------------------
def test_another_users_thread_id_is_a_404_and_the_model_is_never_called():
    s, p = svc()
    r1 = ask(s, user="alice")
    before = p.calls
    r2 = ask(s, "peek", user="bob", thread_id=r1.thread_id)
    assert r2.status == "not_found" and r2.http == 404 and p.calls == before


def test_a_full_thread_answers_409_before_spending_a_model_call():
    s = AgentService({make_version().version_id: make_version()}, make_version().version_id, Provider(responder=answer()), ThreadStore(max_messages=4), sleep=lambda s: None)
    r = ask(s)
    r = ask(s, "two", thread_id=r.thread_id)
    calls = s.agent_fn.calls
    r3 = ask(s, "three", thread_id=r.thread_id)
    assert r3.status == "thread_full" and r3.http == 409 and s.agent_fn.calls == calls


def test_failing_to_save_the_thread_does_not_throw_away_the_answer():
    s, _ = svc(answer("x" * 9000))  # longer than a thread message may be
    r = ask(s)
    assert r.status == "ok" and r.text == "x" * 9000 and s.metrics.count("persist_failed") == 1


# --- the cache ---------------------------------------------------------------------------------------------------------------------------
def test_a_repeated_public_question_is_answered_from_the_cache_without_the_model_or_a_charge():
    s, p = svc()
    first = ask(s, "What is your return window?", user="alice")
    second = ask(s, "what is your return window", user="bob")
    assert second.cached and second.tier == "cache" and second.text == first.text and p.calls == 1
    assert s.ledger.report()["served_from_cache"] == 1 and s.ledger.spent("bob") == 0 and s.metrics.count("cache", result="hit") == 1


def test_a_cached_answer_still_gets_its_own_thread_for_the_user_who_asked():
    s, _ = svc()
    ask(s, "hello there", user="alice")
    r = ask(s, "hello there", user="bob")
    assert r.cached and r.thread_id and s.store.count(__import__("threads").Scope("bob"), r.thread_id) == 2


def test_an_answer_that_looked_up_the_users_own_order_is_never_shared():
    s, p = svc(rule_based_agent(support_tools.build_tools(support_tools.SupportData())))
    a = ask(s, "Where is my order A1002?", user="cust-42")
    b = ask(s, "where is my order a1002", user="cust-99")
    assert "processing" in a.text and "processing" not in b.text and not b.cached


def test_an_answer_from_a_public_tool_is_shared():
    s, _ = svc(rule_based_agent(support_tools.build_tools(support_tools.SupportData())))
    ask(s, "What is your return window?", user="a")
    assert ask(s, "What is your return window?", user="b").cached


def test_answers_that_used_a_side_effect_tool_or_personal_context_are_not_cached():
    s, p = svc(answer("done", tools=("get_order_status",)))
    ask(s, "q1")
    assert not ask(s, "q1").cached
    s2, _ = svc(user_context=lambda u: "Dana, enterprise")
    ask(s2, "q2")
    assert not ask(s2, "q2").cached and len(s2.cache) == 0


def test_a_follow_up_in_a_thread_never_uses_the_shared_cache():
    s, p = svc()
    first = ask(s, "hello")
    ask(s, "hello", thread_id=first.thread_id)
    assert p.calls == 2


def test_errors_are_never_cached():
    state = {"n": 0}

    def flaky(call):
        state["n"] += 1
        if state["n"] == 1:
            raise Permanent("bad")
        return AgentResult("fine", (), 1, 1)
    s, p = svc(flaky)
    assert ask(s, "q").status == "error"
    assert ask(s, "q").text == "fine" and len(s.cache) == 1


def test_a_new_agent_version_does_not_serve_an_old_versions_cached_answer():
    v1, v2 = make_version("prompt one"), make_version("prompt two")
    cache = ResponseCache()
    p = Provider(responder=answer("from v1"))
    s1 = AgentService({v1.version_id: v1}, v1.version_id, p, ThreadStore(), cache=cache, sleep=lambda s: None)
    ask(s1, "hello")
    p.responder = answer("from v2")
    s2 = AgentService({v2.version_id: v2}, v2.version_id, p, ThreadStore(), cache=cache, sleep=lambda s: None)
    assert ask(s2, "hello").text == "from v2"


def test_many_identical_questions_at_the_same_moment_make_one_model_call():
    gate, started = threading.Event(), threading.Event()

    def slow(call):
        started.set()
        gate.wait(3)
        return AgentResult("shared answer", (), 100, 20)
    s, p = svc(slow, admission=Admission(max_concurrent=20, max_queue=20, per_user_max=5, queue_timeout_s=5))
    results = []
    threads = [threading.Thread(target=lambda i=i: results.append(ask(s, "same question", user=f"u{i}"))) for i in range(8)]
    for t in threads:
        t.start()
    started.wait(2)
    time.sleep(0.15)
    gate.set()
    for t in threads:
        t.join(3)
    assert p.calls == 1 and [r.text for r in results] == ["shared answer"] * 8 and s.metrics.count("cache", result="shared") == 7


# --- escalation --------------------------------------------------------------------------------------------------------------------------
def tier_aware(small_text, large_text="large answer"):
    return lambda call: AgentResult(small_text if call.tier.name == "small" else large_text, (), 100, 20)


def test_an_answer_that_fails_the_check_escalates_to_the_bigger_tier_and_both_are_paid_for():
    s, p = svc(tier_aware("ESCALATE"), check=lambda r: "ESCALATE" not in r.text)
    r = ask(s, "a hard question")
    assert (r.status, r.tier, r.text) == ("ok", "large", "large answer") and p.calls == 2
    assert s.ledger.report()["requests_by_tier"] == {"small": 1, "large": 1} and s.metrics.count("escalations") == 1


def test_a_keyword_hint_starts_at_the_top_tier_to_save_a_wasted_cheap_call():
    s, p = svc(tier_aware("small answer"))
    r = ask(s, "I need a refund please")
    assert r.tier == "large" and p.calls == 1


def test_if_even_the_top_tier_fails_the_check_the_user_gets_an_error_not_a_bad_answer():
    s, _ = svc(answer("nope"), check=lambda r: False)
    r = ask(s, "q")
    assert r.status == "error" and r.http == 500 and "acceptable" in r.detail and len(s.cache) == 0


def test_a_cheap_tier_that_calls_a_tool_it_was_not_offered_is_escalated_past():
    def sneaky(call):
        return AgentResult("x", ("issue_refund",) if call.tier.name == "small" else (), 1, 1)
    s, _ = svc(sneaky)
    assert ask(s, "q").tier == "large"


# --- failures --------------------------------------------------------------------------------------------------------------------------
def test_a_transient_failure_is_retried_and_the_user_never_notices():
    state, slept = {"n": 0}, []

    def flaky(call):
        state["n"] += 1
        if state["n"] < 3:
            raise Transient("429")
        return AgentResult("finally", (), 1, 1)
    s, _ = svc(flaky, sleep=slept.append, rng=lambda: 0.5)
    r = ask(s)
    assert r.status == "ok" and r.text == "finally" and state["n"] == 3 and len(slept) == 2 and s.metrics.count("retries") == 2


def test_when_retries_run_out_the_user_gets_503_with_a_retry_hint():
    s, p = svc(lambda call: (_ for _ in ()).throw(Transient("down")), retry=RetryPolicy(attempts=2))
    r = ask(s)
    assert (r.status, r.http, p.calls) == ("unavailable", 503, 2) and r.retry_after_s


def test_a_permanent_provider_error_is_not_retried_and_does_not_leak_its_message():
    s, p = svc(lambda call: (_ for _ in ()).throw(Permanent("secret internal detail")))
    r = ask(s)
    assert (r.status, p.calls) == ("error", 1) and "secret" not in json.dumps(r.__dict__)


def test_once_the_breaker_opens_requests_fail_fast_without_calling_the_provider_and_it_recovers():
    clock, down = FakeClock(), [True]

    def responder(call):
        if down[0]:
            raise Transient("down")
        return AgentResult("back", (), 1, 1)
    s, p = svc(responder, retry=RetryPolicy(attempts=1), breaker=CircuitBreaker(2, 30.0, clock), clock=clock)
    for i in range(2):
        ask(s, f"q{i}")
    calls = p.calls
    r = ask(s, "q-fast")
    assert r.status == "unavailable" and r.retry_after_s >= 1 and p.calls == calls and s.ready() == {"ready": False, "reasons": ["provider circuit open"]}
    down[0] = False
    clock.advance(31)
    assert ask(s, "q-after").status == "ok" and s.ready()["ready"]


def test_a_request_that_runs_out_of_time_answers_504():
    clock = FakeClock()

    def slow(call):
        clock.advance(100)
        raise Transient("slow")
    s, _ = svc(slow, clock=clock, retry=RetryPolicy(attempts=3, base_s=1, cap_s=1))
    r = s.handle(Request("alice", "q", deadline_s=5))
    assert r.status == "deadline" and r.http == 504


# --- budgets ------------------------------------------------------------------------------------------------------------------------------
def test_a_user_who_has_spent_their_budget_is_refused_before_the_model_is_called():
    s, p = svc(ledger=CostLedger(user_daily_budget=Decimal("0.0001"), global_daily_budget=Decimal("100")))
    ask(s, "first", user="alice")
    r = ask(s, "second", user="alice")
    assert r.status == "budget" and r.detail == "user" and p.calls == 1
    assert ask(s, "third", user="bob").status == "ok"


def test_the_global_budget_stops_everyone():
    s, _ = svc(ledger=CostLedger(user_daily_budget=Decimal("100"), global_daily_budget=Decimal("0.0001")))
    ask(s, "first")
    assert ask(s, "second", user="someone-else").detail == "global"


# --- idempotency --------------------------------------------------------------------------------------------------------------------------
def test_repeating_a_request_with_the_same_key_returns_the_first_answer_and_does_not_run_again():
    s, p = svc()
    r1 = ask(s, "book it", idempotency_key="k1")
    r2 = ask(s, "book it", idempotency_key="k1")
    assert p.calls == 1 and r2.text == r1.text and r2.thread_id == r1.thread_id and r2.detail == "replayed" and s.ledger.report()["requests_by_tier"] == {"small": 1}


def test_a_failed_request_with_a_key_can_be_retried_with_the_same_key():
    state = {"n": 0}

    def flaky(call):
        state["n"] += 1
        if state["n"] == 1:
            raise Permanent("bad")
        return AgentResult("ok now", (), 1, 1)
    s, _ = svc(flaky)
    assert ask(s, "q", idempotency_key="k").status == "error"
    assert ask(s, "q", idempotency_key="k").text == "ok now"


def test_the_same_key_from_two_different_users_is_two_different_requests():
    s, p = svc(lambda call: AgentResult(f"for {call.user_id}", (), 1, 1))
    a = ask(s, "q", user="alice", idempotency_key="same")
    b = ask(s, "q2", user="bob", idempotency_key="same")
    assert a.text == "for alice" and b.text == "for bob"


# --- load and shutdown -----------------------------------------------------------------------------------------------------------------
def test_when_the_service_is_full_the_caller_gets_429_and_a_retry_after_and_the_provider_is_not_touched():
    gate, started = threading.Event(), threading.Event()

    def slow(call):
        started.set()
        gate.wait(3)
        return AgentResult("x", (), 1, 1)
    s, p = svc(slow, admission=Admission(max_concurrent=1, max_queue=0, per_user_max=5))
    t = threading.Thread(target=lambda: ask(s, "first", user="u1"))
    t.start()
    started.wait(2)
    r = ask(s, "second", user="u2")
    assert (r.status, r.http, r.detail) == ("overloaded", 429, "queue_full") and r.retry_after_s >= 1 and p.calls == 1
    gate.set()
    t.join(3)


def test_after_drain_new_requests_get_503_and_readiness_goes_false():
    s, _ = svc()
    ask(s)
    assert s.drain(1) is True
    r = ask(s, "late")
    assert (r.status, r.http) == ("shutting_down", 503) and s.ready() == {"ready": False, "reasons": ["draining"]}


def test_drain_lets_the_request_in_flight_finish():
    gate, started, out = threading.Event(), threading.Event(), []

    def slow(call):
        started.set()
        gate.wait(3)
        return AgentResult("finished", (), 1, 1)
    s, _ = svc(slow)
    t = threading.Thread(target=lambda: out.append(ask(s, "in flight")))
    t.start()
    started.wait(2)
    d = threading.Thread(target=lambda: out.append(("drained", s.drain(3))))
    d.start()
    time.sleep(0.1)
    gate.set()
    t.join(3)
    d.join(3)
    assert any(isinstance(o, Response) and o.text == "finished" for o in out) and ("drained", True) in out


def test_readiness_reports_a_store_that_has_stopped_answering():
    s, _ = svc()
    s.store.close()
    r = s.ready()
    assert r["ready"] is False and "thread store not answering" in r["reasons"]


# --- releases -------------------------------------------------------------------------------------------------------------------------
def test_a_stable_slice_of_users_gets_the_candidate_and_failures_are_counted_against_the_right_version():
    stable, cand = make_version("old prompt"), make_version("new prompt")
    rollout = Rollout(stable.version_id, cand.version_id, salt="x", min_samples=5, stages=(50, 100))

    def responder(call):
        if call.version.version_id == cand.version_id:
            raise Permanent("bad candidate")
        return AgentResult("ok", (), 1, 1)
    s = AgentService({stable.version_id: stable, cand.version_id: cand}, stable.version_id, Provider(responder=responder), ThreadStore(), rollout=rollout, sleep=lambda s: None)
    users = [f"user-{i}" for i in range(200)]
    first = {u: s.handle(Request(u, f"q-{u}")).version for u in users}
    again = {u: s.handle(Request(u, f"r-{u}")).version for u in users}
    assert first == again and set(first.values()) == {stable.version_id, cand.version_id}
    assert s.check_rollout() == "rolled_back"
    assert {s.handle(Request(u, f"z-{u}")).version for u in users} == {stable.version_id}


def test_busy_invalid_and_budget_responses_do_not_count_against_a_version():
    stable, cand = make_version("a"), make_version("b")
    rollout = Rollout(stable.version_id, cand.version_id, min_samples=1, stages=(100,))
    s = AgentService({stable.version_id: stable, cand.version_id: cand}, stable.version_id, Provider(responder=answer()), ThreadStore(),
                     rollout=rollout, ledger=CostLedger(user_daily_budget=Decimal("0")), sleep=lambda s: None)
    s.handle(Request("alice", "x"))
    assert rollout.error_rate(cand.version_id) is None and s.check_rollout() == "waiting"


def test_without_a_rollout_there_is_nothing_to_check():
    s, _ = svc()
    assert s.check_rollout() == "no rollout"


def test_the_default_version_must_exist():
    with pytest.raises(ValueError):
        AgentService({}, "missing", answer(), ThreadStore())


# --- identity and hostile input --------------------------------------------------------------------------------------------------------
def test_a_user_id_that_looks_like_sql_is_just_a_name():
    s, _ = svc()
    r = ask(s, "hi", user="x'); DROP TABLE threads; --")
    assert r.status == "ok" and s.store.ping()
    assert ask(s, "follow", user="alice", thread_id=r.thread_id).status == "not_found"


def test_the_model_gets_the_user_id_only_through_the_call_object_for_tools_never_in_the_messages():
    seen = {}

    def spy(call):
        seen["messages"] = json.dumps(call.messages)
        seen["user"] = call.user_id
        return AgentResult("x", (), 1, 1)
    s, _ = svc(spy)
    ask(s, "hello", user="secret-user-id-42")
    assert seen["user"] == "secret-user-id-42" and "secret-user-id-42" not in seen["messages"]


def test_an_unexpected_exception_from_the_provider_becomes_a_clean_500_not_a_crash():
    s, _ = svc(lambda call: (_ for _ in ()).throw(RuntimeError("SECRET internal failure")))
    r = ask(s, "q")
    assert (r.status, r.http, r.detail) == ("error", 500, "RuntimeError") and "SECRET" not in json.dumps(r.__dict__)
    assert s.metrics.count("requests", status="error") == 1 and s.admission.snapshot()["running"] == 0


def test_a_bug_outside_the_provider_call_is_also_caught_and_the_slot_is_released(monkeypatch):
    s, _ = svc()
    monkeypatch.setattr(s, "_pick_version", lambda user: (_ for _ in ()).throw(KeyError("oops")))
    assert ask(s, "q").status == "error" and s.admission.snapshot()["running"] == 0


def test_a_thread_full_response_does_not_count_against_a_canary_version():
    v = make_version()
    rollout = Rollout(v.version_id, v.version_id + "x", min_samples=1, stages=(100,))
    s = AgentService({v.version_id: v}, v.version_id, Provider(responder=answer()), ThreadStore(max_messages=4), sleep=lambda s: None)
    s.rollout = Rollout(v.version_id, "other", min_samples=1, stages=(1,))  # everyone is on the stable version
    r = ask(s)
    r = ask(s, "two", thread_id=r.thread_id)
    before = s.rollout.error_rate(v.version_id)
    assert ask(s, "three", thread_id=r.thread_id).status == "thread_full"
    assert s.rollout.error_rate(v.version_id) == before and s.rollout._n[v.version_id] == 2  # the 409 was not observed at all


def test_the_deadline_is_checked_before_each_tier_so_an_expired_request_does_not_start_another_model_call():
    clock = FakeClock()

    def slow_cheap(call):
        clock.advance(100)
        return AgentResult("ESCALATE", (), 10, 5)
    s, p = svc(slow_cheap, clock=clock, check=lambda r: "ESCALATE" not in r.text)
    r = s.handle(Request("alice", "a hard question", deadline_s=5))
    assert r.status == "deadline" and p.calls == 1  # the big tier was never called
