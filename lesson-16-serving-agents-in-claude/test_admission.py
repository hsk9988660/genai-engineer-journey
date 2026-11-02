import threading
import time

import pytest

from admission import Admission, Overloaded, ShuttingDown


def hold(adm, user, started, release, errors=None):
    """A request that takes a slot and keeps it until `release` is set."""
    def run():
        try:
            with adm.slot(user):
                started.set()
                release.wait(5)
        except Exception as exc:  # noqa
            if errors is not None:
                errors.append(exc)
            started.set()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_bad_configuration_is_rejected():
    for kw in ({"max_concurrent": 0}, {"max_queue": -1}, {"per_user_max": 0}, {"queue_timeout_s": -1}):
        with pytest.raises(ValueError):
            Admission(**kw)


def test_requests_under_the_limit_run_at_once_and_release_their_slot():
    adm = Admission(max_concurrent=2)
    with adm.slot("a"):
        with adm.slot("b"):
            assert adm.snapshot()["running"] == 2
    assert adm.snapshot()["running"] == 0 and adm.snapshot()["peak_running"] == 2


def test_the_slot_is_released_even_when_the_work_raises():
    adm = Admission(max_concurrent=1)
    with pytest.raises(RuntimeError):
        with adm.slot("a"):
            raise RuntimeError("boom")
    with adm.slot("a"):
        pass
    assert adm.snapshot()["running"] == 0


def test_extra_requests_wait_in_a_bounded_queue_and_run_when_a_slot_frees():
    adm = Admission(max_concurrent=1, max_queue=2, per_user_max=5, queue_timeout_s=3)
    started, release, done = threading.Event(), threading.Event(), []
    hold(adm, "u1", started, release)
    started.wait(2)

    def waiter():
        with adm.slot("u2"):
            done.append(1)
    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    assert wait_until(lambda: adm.snapshot()["waiting"] == 1)
    release.set()
    t.join(2)
    assert done == [1] and adm.snapshot()["running"] == 0 and adm.snapshot()["waiting"] == 0


def test_a_full_queue_refuses_immediately_with_a_retry_hint():
    adm = Admission(max_concurrent=1, max_queue=1, per_user_max=5, queue_timeout_s=3)
    started, release = threading.Event(), threading.Event()
    hold(adm, "u1", started, release)
    started.wait(2)
    hold(adm, "u2", threading.Event(), release)
    assert wait_until(lambda: adm.snapshot()["waiting"] == 1)
    t0 = time.monotonic()
    with pytest.raises(Overloaded) as e:
        with adm.slot("u3"):
            pass
    assert time.monotonic() - t0 < 0.5 and e.value.reason == "queue_full" and e.value.retry_after_s >= 1 and adm.snapshot()["rejected"] == {"queue_full": 1}
    release.set()


def test_a_waiter_gives_up_after_the_queue_timeout():
    adm = Admission(max_concurrent=1, max_queue=5, per_user_max=5, queue_timeout_s=0.05)
    started, release = threading.Event(), threading.Event()
    hold(adm, "u1", started, release)
    started.wait(2)
    with pytest.raises(Overloaded) as e:
        with adm.slot("u2"):
            pass
    assert e.value.reason == "queue_timeout" and adm.snapshot()["waiting"] == 0
    release.set()


def test_one_user_cannot_take_more_than_their_share_but_another_user_still_gets_in():
    adm = Admission(max_concurrent=10, per_user_max=2)
    started, release = threading.Event(), threading.Event()
    hold(adm, "noisy", started, release)
    hold(adm, "noisy", threading.Event(), release)
    assert wait_until(lambda: adm.snapshot()["running"] == 2)
    with pytest.raises(Overloaded) as e:
        with adm.slot("noisy"):
            pass
    assert e.value.reason == "per_user_limit"
    with adm.slot("quiet"):
        pass
    release.set()


def test_counters_return_to_zero_after_rejections_and_timeouts():
    adm = Admission(max_concurrent=1, max_queue=0, per_user_max=1, queue_timeout_s=0.01)
    started, release = threading.Event(), threading.Event()
    hold(adm, "u1", started, release)
    started.wait(2)
    for user in ("u1", "u2"):
        with pytest.raises(Overloaded):
            with adm.slot(user):
                pass
    release.set()
    assert wait_until(lambda: adm.snapshot()["running"] == 0)
    with adm.slot("u1"):  # u1's own count was released, so it can come back
        pass
    assert not adm._per_user


def test_the_service_never_runs_more_than_max_concurrent_at_once_under_a_burst():
    adm = Admission(max_concurrent=3, max_queue=50, per_user_max=50, queue_timeout_s=5)
    active, peak, lock = [0], [0], threading.Lock()

    def work(i):
        with adm.slot(f"u{i}"):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.01)
            with lock:
                active[0] -= 1
    threads = [threading.Thread(target=work, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert peak[0] <= 3 and adm.snapshot()["peak_running"] <= 3 and adm.snapshot()["running"] == 0


def test_the_retry_hint_is_at_least_one_second_and_grows_with_the_queue():
    adm = Admission(max_concurrent=1)
    assert adm._retry_after() == 1
    adm.waiting, adm._avg_latency = 10, 2.0
    assert adm._retry_after() == 22


def test_drain_waits_for_work_in_flight_then_new_requests_are_refused():
    adm = Admission(max_concurrent=2)
    started, release = threading.Event(), threading.Event()
    hold(adm, "u1", started, release)
    started.wait(2)
    result = []
    d = threading.Thread(target=lambda: result.append(adm.drain(3)), daemon=True)
    d.start()
    assert wait_until(lambda: adm.closed)
    with pytest.raises(ShuttingDown):
        with adm.slot("u2"):
            pass
    assert result == []  # still waiting for u1
    release.set()
    d.join(2)
    assert result == [True]


def test_drain_gives_up_after_its_timeout_and_says_so():
    adm = Admission(max_concurrent=1)
    started, release = threading.Event(), threading.Event()
    hold(adm, "u1", started, release)
    started.wait(2)
    assert adm.drain(0.05) is False
    release.set()


def test_requests_waiting_in_the_queue_are_turned_away_when_draining_starts():
    adm = Admission(max_concurrent=1, max_queue=5, per_user_max=5, queue_timeout_s=5)
    started, release, errors = threading.Event(), threading.Event(), []
    hold(adm, "u1", started, release)
    started.wait(2)
    w = threading.Event()
    hold(adm, "u2", w, release, errors)
    assert wait_until(lambda: adm.snapshot()["waiting"] == 1)
    threading.Thread(target=adm.drain, args=(2,), daemon=True).start()
    assert wait_until(lambda: errors != [])
    assert isinstance(errors[0], ShuttingDown)
    release.set()
