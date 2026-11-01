import random

import pytest

from reliability import ScopedCache, idempotency_key, is_transient, retry_with_backoff


class RateLimitError(Exception):
    pass


class StatusError(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"status {status}")
        self.status_code = status
        if retry_after is not None:
            self.retry_after = retry_after


# --- what is worth retrying ---------------------------------------------------------------------------
@pytest.mark.parametrize("status,expected", [(429, True), (500, True), (503, True), (529, True), (408, True), (409, True),
                                             (400, False), (401, False), (403, False), (404, False), (422, False)])
def test_status_codes_decide_whether_an_error_is_transient(status, expected):
    assert is_transient(StatusError(status)) is expected


def test_classification_by_class_name_works_across_sdk_versions():
    assert is_transient(RateLimitError("slow down")) and is_transient(TimeoutError("t")) and is_transient(ConnectionError("c"))
    assert not is_transient(ValueError("bad input")) and not is_transient(KeyError("k"))


# --- retries --------------------------------------------------------------------------------------------
def flaky(*outcomes):
    calls = []

    def fn():
        calls.append(1)
        o = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(o, Exception):
            raise o
        return o

    return fn, calls


def test_a_transient_failure_is_retried_until_it_succeeds():
    fn, calls = flaky(StatusError(503), StatusError(429), "ok")
    sleeps = []
    assert retry_with_backoff(fn, retries=3, sleep=sleeps.append, rng=random.Random(1)) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2


def test_a_permanent_error_is_raised_immediately_without_waiting_or_retrying():
    fn, calls = flaky(StatusError(400))
    sleeps = []
    with pytest.raises(StatusError):
        retry_with_backoff(fn, retries=5, sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []


def test_retries_are_capped_and_the_last_error_is_raised():
    fn, calls = flaky(StatusError(503))
    with pytest.raises(StatusError):
        retry_with_backoff(fn, retries=2, sleep=lambda s: None)
    assert len(calls) == 3  # the first try plus two retries


def test_backoff_grows_exponentially_up_to_the_cap_with_jitter_inside_each_bound():
    fn, _ = flaky(StatusError(503))
    sleeps = []
    with pytest.raises(StatusError):
        retry_with_backoff(fn, retries=6, base_s=1.0, cap_s=5.0, sleep=sleeps.append, rng=random.Random(3))
    bounds = [min(5.0, 1.0 * 2 ** i) for i in range(6)]  # 1, 2, 4, 5, 5, 5
    assert len(sleeps) == 6 and all(0 <= s <= b for s, b in zip(sleeps, bounds))
    assert len(set(sleeps)) > 1  # jitter: the waits are not all identical


def test_a_retry_after_header_is_obeyed_over_the_computed_backoff():
    fn, _ = flaky(StatusError(429, retry_after=7), "ok")
    sleeps = []
    retry_with_backoff(fn, sleep=sleeps.append)
    assert sleeps == [7.0]


def test_the_on_retry_hook_sees_each_attempt():
    fn, _ = flaky(StatusError(503), StatusError(503), "ok")
    seen = []
    retry_with_backoff(fn, sleep=lambda s: None, on_retry=lambda attempt, exc, delay: seen.append(attempt))
    assert seen == [1, 2]


def test_idempotency_keys_are_stable_for_the_same_action_and_differ_otherwise():
    k = idempotency_key("run1", 2, "book", "{'id': 7}")
    assert k == idempotency_key("run1", 2, "book", "{'id': 7}")
    assert k != idempotency_key("run1", 3, "book", "{'id': 7}") != idempotency_key("run2", 2, "book", "{'id': 7}")


# --- the scope-safe cache ---------------------------------------------------------------------------------
def test_a_hit_for_one_user_scope_is_never_served_to_another():
    c = ScopedCache()
    c.put("ANSWER FOR ALICE (includes her private data)", scope="alice-org", prompt_version="v1", text="what is my balance?")
    assert c.get(scope="bob-org", prompt_version="v1", text="what is my balance?") is None
    assert c.get(scope="alice-org", prompt_version="v1", text="what is my balance?") == "ANSWER FOR ALICE (includes her private data)"


def test_a_new_prompt_version_does_not_serve_answers_from_the_old_prompt():
    c = ScopedCache()
    c.put("old", scope="s", prompt_version="v1", text="q")
    assert c.get(scope="s", prompt_version="v2", text="q") is None


def test_forgetting_the_scope_is_a_type_error_not_a_data_leak():
    c = ScopedCache()
    with pytest.raises(TypeError):
        c.get(prompt_version="v1", text="q")
    with pytest.raises(TypeError):
        c.put("x", prompt_version="v1", text="q")


def test_keys_ignore_case_and_extra_whitespace_but_not_meaning():
    c = ScopedCache()
    c.put("A", scope="s", prompt_version="v", text="What  is the REFUND policy?")
    assert c.get(scope="s", prompt_version="v", text="what is the refund policy?") == "A"
    assert c.get(scope="s", prompt_version="v", text="what is the baggage policy?") is None


def test_entries_expire_and_the_oldest_is_evicted_when_full():
    now = [0.0]
    c = ScopedCache(ttl_s=10, max_items=2, clock=lambda: now[0])
    c.put("1", scope="s", prompt_version="v", text="a")
    c.put("2", scope="s", prompt_version="v", text="b")
    c.get(scope="s", prompt_version="v", text="a")  # touch a, so b is now the least recently used
    c.put("3", scope="s", prompt_version="v", text="c")
    assert c.get(scope="s", prompt_version="v", text="b") is None and c.get(scope="s", prompt_version="v", text="a") == "1"
    now[0] = 11.0
    assert c.get(scope="s", prompt_version="v", text="a") is None


def test_hit_and_miss_counters():
    c = ScopedCache()
    c.put("x", scope="s", prompt_version="v", text="q")
    c.get(scope="s", prompt_version="v", text="q")
    c.get(scope="s", prompt_version="v", text="other")
    assert (c.hits, c.misses) == (1, 1)
