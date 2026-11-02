import threading
import time

import pytest

from harness import FakeClock
from resilience import (CircuitBreaker, CircuitOpen, DeadlineExceeded, IdempotencyStore, Permanent, RetryPolicy, Singleflight, Transient, backoff_delay, call_with_retry)

P = RetryPolicy(attempts=4, base_s=1.0, cap_s=10.0)


# --- backoff ------------------------------------------------------------------------------------------------------------------------
def test_full_jitter_picks_a_time_between_zero_and_the_exponential_ceiling():
    assert backoff_delay(0, P, rng=lambda: 1.0) == 1.0 and backoff_delay(1, P, rng=lambda: 1.0) == 2.0 and backoff_delay(2, P, rng=lambda: 1.0) == 4.0
    assert backoff_delay(3, P, rng=lambda: 0.5) == 4.0 and backoff_delay(2, P, rng=lambda: 0.0) == 0.0


def test_the_ceiling_stops_growing_at_the_cap():
    assert backoff_delay(10, P, rng=lambda: 1.0) == 10.0 and backoff_delay(30, P, rng=lambda: 1.0) == 10.0


def test_many_clients_that_failed_together_do_not_all_retry_at_the_same_moment():
    import random
    r = random.Random(1)
    delays = {round(backoff_delay(2, P, rng=r.random), 3) for _ in range(100)}
    assert len(delays) > 90


# --- retrying -----------------------------------------------------------------------------------------------------------------------
def flaky(failures, error=Transient):
    state = {"n": 0}

    def fn():
        state["n"] += 1
        if state["n"] <= failures:
            raise error("fail")
        return "ok"
    fn.state = state
    return fn


def test_success_on_the_first_try_does_not_sleep():
    slept = []
    assert call_with_retry(flaky(0), P, sleep=slept.append) == "ok" and slept == []


def test_a_transient_failure_is_retried_with_a_jittered_delay_until_it_succeeds():
    slept, seen = [], []
    fn = flaky(2)
    assert call_with_retry(fn, P, sleep=slept.append, rng=lambda: 0.5, on_retry=lambda n, d: seen.append((n, d))) == "ok"
    assert fn.state["n"] == 3 and slept == [0.5, 1.0] and seen == [(1, 0.5), (2, 1.0)]


def test_after_the_last_attempt_the_error_is_raised():
    fn = flaky(99)
    with pytest.raises(Transient):
        call_with_retry(fn, P, sleep=lambda s: None)
    assert fn.state["n"] == 4


def test_a_permanent_error_is_never_retried():
    fn = flaky(99, Permanent)
    with pytest.raises(Permanent):
        call_with_retry(fn, P, sleep=lambda s: None)
    assert fn.state["n"] == 1


def test_an_error_that_is_neither_is_not_swallowed_or_retried():
    fn = flaky(99, KeyError)
    with pytest.raises(KeyError):
        call_with_retry(fn, P, sleep=lambda s: None)
    assert fn.state["n"] == 1


def test_a_retry_after_hint_from_the_provider_raises_the_delay_but_never_past_the_cap():
    slept = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise Transient("429", retry_after_s=7.0)
        if calls["n"] == 2:
            raise Transient("429", retry_after_s=500.0)
        return "ok"
    call_with_retry(fn, P, sleep=slept.append, rng=lambda: 0.0)
    assert slept == [7.0, 10.0]


def test_a_retry_that_would_pass_the_deadline_is_not_started():
    clock = FakeClock()
    slept = []
    with pytest.raises(DeadlineExceeded):
        call_with_retry(flaky(99), P, sleep=slept.append, rng=lambda: 1.0, clock=clock, deadline=clock() + 0.5)
    assert slept == []


# --- the circuit breaker --------------------------------------------------------------------------------------------------------------
def fail():
    raise Transient("down")


def test_calls_pass_while_the_circuit_is_closed_and_failures_below_the_threshold_do_not_open_it():
    b = CircuitBreaker(3, 30.0, FakeClock())
    for _ in range(2):
        with pytest.raises(Transient):
            b.call(fail)
    assert b.state == "closed" and b.call(lambda: "ok") == "ok"


def test_consecutive_failures_open_it_and_further_calls_fail_fast_without_calling_the_provider():
    clock, calls = FakeClock(), [0]
    b = CircuitBreaker(3, 30.0, clock)
    for _ in range(3):
        with pytest.raises(Transient):
            b.call(fail)
    assert b.state == "open" and b.opened_count == 1

    def counted():
        calls[0] += 1
    with pytest.raises(CircuitOpen) as e:
        b.call(counted)
    assert calls[0] == 0 and 0 < e.value.retry_after_s <= 30


def test_a_success_resets_the_count_so_failures_must_be_consecutive():
    b = CircuitBreaker(3, 30.0, FakeClock())
    for _ in range(2):
        with pytest.raises(Transient):
            b.call(fail)
    b.call(lambda: "ok")
    for _ in range(2):
        with pytest.raises(Transient):
            b.call(fail)
    assert b.state == "closed"


def test_after_the_pause_one_probe_goes_through_and_success_closes_the_circuit():
    clock = FakeClock()
    b = CircuitBreaker(1, 30.0, clock)
    with pytest.raises(Transient):
        b.call(fail)
    clock.advance(31)
    assert b.call(lambda: "ok") == "ok" and b.state == "closed"


def test_a_failed_probe_reopens_the_circuit_for_another_full_pause():
    clock = FakeClock()
    b = CircuitBreaker(1, 30.0, clock)
    with pytest.raises(Transient):
        b.call(fail)
    clock.advance(31)
    with pytest.raises(Transient):
        b.call(fail)
    assert b.state == "open" and b.opened_count == 2
    with pytest.raises(CircuitOpen):
        b.call(lambda: "ok")


def test_only_one_probe_is_allowed_at_a_time_while_half_open():
    clock = FakeClock()
    b = CircuitBreaker(1, 30.0, clock)
    with pytest.raises(Transient):
        b.call(fail)
    clock.advance(31)
    inside, release, results = threading.Event(), threading.Event(), []

    def slow_probe():
        inside.set()
        release.wait(2)
        return "probe"
    t = threading.Thread(target=lambda: results.append(b.call(slow_probe)))
    t.start()
    inside.wait(2)
    with pytest.raises(CircuitOpen):
        b.call(lambda: "second")
    release.set()
    t.join(2)
    assert results == ["probe"] and b.state == "closed"


def test_a_permanent_error_means_the_provider_answered_so_it_does_not_count_as_an_outage():
    b = CircuitBreaker(2, 30.0, FakeClock())
    for _ in range(5):
        with pytest.raises(Permanent):
            b.call(lambda: (_ for _ in ()).throw(Permanent("bad request")))
    assert b.state == "closed"


def test_an_unexpected_error_during_a_probe_does_not_leave_the_circuit_stuck_half_open():
    clock = FakeClock()
    b = CircuitBreaker(1, 30.0, clock)
    with pytest.raises(Transient):
        b.call(fail)
    clock.advance(31)
    with pytest.raises(RuntimeError):
        b.call(lambda: (_ for _ in ()).throw(RuntimeError("bug")))
    assert b.call(lambda: "ok") == "ok"  # a new probe is allowed


# --- singleflight --------------------------------------------------------------------------------------------------------------------
def test_concurrent_callers_with_the_same_key_share_one_execution():
    sf, runs, gate, results = Singleflight(), [0], threading.Event(), []

    def work():
        runs[0] += 1
        gate.wait(2)
        return "value"
    threads = [threading.Thread(target=lambda: results.append(sf.do("k", work))) for _ in range(6)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    gate.set()
    for t in threads:
        t.join(2)
    assert runs[0] == 1 and sorted(r[1] for r in results) == [False] + [True] * 5 and {r[0] for r in results} == {"value"} and sf.shared == 5


def test_sequential_calls_run_separately_and_different_keys_do_not_interfere():
    sf, runs = Singleflight(), [0]
    for _ in range(3):
        sf.do("k", lambda: runs.__setitem__(0, runs[0] + 1))
    assert runs[0] == 3
    assert sf.do("a", lambda: 1)[0] == 1 and sf.do("b", lambda: 2)[0] == 2


def test_an_exception_is_shared_with_the_followers_and_the_key_is_free_afterwards():
    sf, gate, errors = Singleflight(), threading.Event(), []

    def work():
        gate.wait(2)
        raise ValueError("boom")

    def call():
        try:
            sf.do("k", work)
        except ValueError as e:
            errors.append(str(e))
    threads = [threading.Thread(target=call) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    gate.set()
    for t in threads:
        t.join(2)
    assert errors == ["boom"] * 3 and sf.do("k", lambda: "fresh")[0] == "fresh"


# --- idempotency -----------------------------------------------------------------------------------------------------------------------
def test_repeating_a_key_returns_the_first_result_without_running_again():
    store, runs = IdempotencyStore(), [0]

    def work():
        runs[0] += 1
        return {"n": runs[0]}
    first = store.run("k", work)
    second = store.run("k", work)
    assert first == ({"n": 1}, False) and second == ({"n": 1}, True) and runs[0] == 1


def test_a_result_the_caller_rejects_is_not_stored_so_a_failure_can_be_retried():
    store, runs = IdempotencyStore(), [0]

    def work():
        runs[0] += 1
        return "bad" if runs[0] == 1 else "good"
    assert store.run("k", work, store_if=lambda v: v == "good")[0] == "bad"
    assert store.run("k", work, store_if=lambda v: v == "good") == ("good", False) and store.run("k", work)[1] is True


def test_an_exception_is_not_stored_and_propagates():
    store = IdempotencyStore()
    with pytest.raises(KeyError):
        store.run("k", lambda: (_ for _ in ()).throw(KeyError("x")))
    assert store.run("k", lambda: "later") == ("later", False)


def test_a_stored_result_expires_after_its_ttl():
    clock, store, runs = FakeClock(), None, [0]
    store = IdempotencyStore(ttl_s=60, clock=clock)
    store.run("k", lambda: runs.__setitem__(0, runs[0] + 1) or runs[0])
    clock.advance(59)
    assert store.run("k", lambda: 99)[1] is True
    clock.advance(2)
    assert store.run("k", lambda: 99) == (99, False)


def test_two_requests_with_the_same_key_at_the_same_time_run_once():
    store, runs, gate, results = IdempotencyStore(), [0], threading.Event(), []

    def work():
        runs[0] += 1
        gate.wait(2)
        return "done"
    threads = [threading.Thread(target=lambda: results.append(store.run("k", work))) for _ in range(5)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    gate.set()
    for t in threads:
        t.join(2)
    assert runs[0] == 1 and {r[0] for r in results} == {"done"}


def test_the_store_is_bounded_and_forgets_the_oldest_first():
    store = IdempotencyStore(max_entries=2)
    for k in ("a", "b", "c"):
        store.run(k, lambda k=k: k)
    assert store.run("a", lambda: "again") == ("again", False) and store.run("c", lambda: "x")[1] is True
