"""Surviving a provider that fails: backoff with jitter, a circuit breaker, deadlines, and requests that are safe to repeat.

The lesson says "use retries with exponential backoff" in one sentence. Retries are also how a small outage becomes a big one, so the details matter:
  * only TRANSIENT errors are retried (rate limit, timeout, 5xx). A bad request is permanent: retrying it just wastes quota
  * delays use FULL JITTER, so many clients that failed together do not all come back together
  * a retry never starts if it would run past the request's deadline
  * a circuit breaker stops calling a provider that keeps failing, and probes with ONE request after a pause, so a recovering provider is not hit by everything at once
  * a request with side effects carries an idempotency key: running it twice returns the first result, and two at once run once
"""

import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


class Transient(Exception):
    """Worth retrying: rate limited, timed out, provider error."""

    def __init__(self, message: str = "transient error", retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class Permanent(Exception):
    """Not worth retrying: the request itself is wrong, or the answer will not change."""


class CircuitOpen(Exception):
    def __init__(self, retry_after_s: float):
        super().__init__("circuit open")
        self.retry_after_s = retry_after_s


class DeadlineExceeded(Exception):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    base_s: float = 0.2
    cap_s: float = 5.0


def backoff_delay(attempt: int, policy: RetryPolicy, rng=random.random) -> float:
    """Full jitter: a random time between 0 and min(cap, base * 2^attempt). `attempt` counts from 0."""
    return rng() * min(policy.cap_s, policy.base_s * (2 ** attempt))


def call_with_retry(fn, policy: RetryPolicy = RetryPolicy(), sleep=time.sleep, rng=random.random, clock=time.monotonic, deadline: float | None = None, on_retry=None):
    for attempt in range(policy.attempts):
        try:
            return fn()
        except Transient as exc:
            if attempt + 1 >= policy.attempts:
                raise
            delay = backoff_delay(attempt, policy, rng)
            if exc.retry_after_s is not None:
                delay = max(delay, min(exc.retry_after_s, policy.cap_s))
            if deadline is not None and clock() + delay >= deadline:
                raise DeadlineExceeded("no time left to retry") from exc
            if on_retry:
                on_retry(attempt + 1, delay)
            sleep(delay)


class CircuitBreaker:
    """closed: calls pass. After `failure_threshold` CONSECUTIVE transient failures it opens: calls fail fast. After `reset_after_s` one probe is let through (half-open)."""

    def __init__(self, failure_threshold: int = 5, reset_after_s: float = 30.0, clock=time.monotonic):
        self.failure_threshold, self.reset_after_s, self.clock = failure_threshold, reset_after_s, clock
        self._lock = threading.Lock()
        self.state, self._failures, self._opened_at, self._probing = "closed", 0, 0.0, False
        self.opened_count = 0

    def _retry_after(self) -> float:
        return max(0.0, self._opened_at + self.reset_after_s - self.clock())

    def call(self, fn):
        with self._lock:
            if self.state == "open":
                if self.clock() >= self._opened_at + self.reset_after_s:
                    self.state, self._probing = "half_open", False
                else:
                    raise CircuitOpen(self._retry_after())
            if self.state == "half_open":
                if self._probing:
                    raise CircuitOpen(1.0)  # one probe at a time
                self._probing = True
        try:
            out = fn()
        except Transient:
            self._fail()
            raise
        except Permanent:  # the provider answered; the request was bad. That is not an outage
            self._ok()
            raise
        except BaseException:
            self._release_probe()
            raise
        self._ok()
        return out

    def _release_probe(self):
        with self._lock:
            self._probing = False

    def _fail(self):
        with self._lock:
            self._failures += 1
            self._probing = False
            if self.state == "half_open" or self._failures >= self.failure_threshold:
                if self.state != "open":
                    self.opened_count += 1
                self.state, self._opened_at = "open", self.clock()

    def _ok(self):
        with self._lock:
            self.state, self._failures, self._probing = "closed", 0, False


class _Call:
    def __init__(self):
        self.event, self.value, self.error = threading.Event(), None, None


class Singleflight:
    """Concurrent callers with the same key share ONE execution. Without it, 100 identical questions at once make 100 model calls."""

    def __init__(self):
        self._lock, self._calls = threading.Lock(), {}
        self.shared = 0

    def do(self, key, fn):
        """Returns (value, was_shared). Followers get the leader's value, or its exception."""
        with self._lock:
            call = self._calls.get(key)
            leader = call is None
            if leader:
                call = self._calls[key] = _Call()
            else:
                self.shared += 1
        if leader:
            try:
                call.value = fn()
            except BaseException as exc:
                call.error = exc
            finally:
                with self._lock:
                    self._calls.pop(key, None)
                call.event.set()
        else:
            call.event.wait()
        if call.error is not None:
            raise call.error
        return call.value, not leader


class IdempotencyStore:
    """Run `fn` once per key. A repeat returns the stored result, a concurrent duplicate waits for the first. Only results `store_if` accepts are kept, so a failure can be retried."""

    def __init__(self, ttl_s: float = 3600.0, max_entries: int = 10_000, clock=time.monotonic):
        self.ttl_s, self.max_entries, self.clock = ttl_s, max_entries, clock
        self._lock, self._done, self._sf = threading.Lock(), OrderedDict(), Singleflight()

    def run(self, key: str, fn, store_if=lambda value: True):
        """Returns (value, replayed)."""
        with self._lock:
            hit = self._done.get(key)
            if hit is not None:
                if hit[1] > self.clock():
                    return hit[0], True
                del self._done[key]

        def work():
            value = fn()
            if store_if(value):
                with self._lock:
                    self._done[key] = (value, self.clock() + self.ttl_s)
                    while len(self._done) > self.max_entries:
                        self._done.popitem(last=False)
            return value
        value, shared = self._sf.do(key, work)
        return value, shared
