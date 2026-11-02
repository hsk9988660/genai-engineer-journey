"""Admission control: decide, BEFORE doing expensive work, whether there is room for this request.

A model provider has rate limits and every in-flight request holds a thread, a connection and memory. If you accept everything, a burst turns into a pile-up where
every request is slow and many fail. This bounds three things:
  max_concurrent   how many requests run at once (set it below the provider's limit)
  max_queue        how many may WAIT for a slot. Beyond that a request is refused immediately with a retry hint
  per_user_max     how many one user may have running or waiting, so one noisy user cannot take every slot
A waiter also gives up after `queue_timeout_s` instead of waiting forever. `drain()` supports graceful shutdown: stop admitting, let in-flight work finish.
"""

import math
import threading
import time
from collections import Counter
from contextlib import contextmanager


class Overloaded(Exception):
    def __init__(self, reason: str, retry_after_s: int):
        super().__init__(f"overloaded: {reason}")
        self.reason, self.retry_after_s = reason, retry_after_s


class ShuttingDown(Exception):
    pass


class Admission:
    def __init__(self, max_concurrent: int = 8, max_queue: int = 16, per_user_max: int = 4, queue_timeout_s: float = 2.0):
        if max_concurrent < 1 or max_queue < 0 or per_user_max < 1 or queue_timeout_s < 0:
            raise ValueError("max_concurrent and per_user_max must be at least 1; max_queue and queue_timeout_s cannot be negative")
        self.max_concurrent, self.max_queue, self.per_user_max, self.queue_timeout_s = max_concurrent, max_queue, per_user_max, queue_timeout_s
        self._cv = threading.Condition()
        self.running = self.waiting = self.peak_running = 0
        self.rejected = Counter()
        self._per_user = Counter()
        self._closed = False
        self._avg_latency = 1.0  # seconds, a moving average used only for the retry hint

    def _retry_after(self) -> int:
        return max(1, math.ceil((self.waiting + 1) * self._avg_latency / self.max_concurrent))

    def _reject(self, reason: str):
        self.rejected[reason] += 1
        raise Overloaded(reason, self._retry_after())

    @contextmanager
    def slot(self, user: str):
        with self._cv:
            if self._closed:
                raise ShuttingDown()
            if self._per_user[user] >= self.per_user_max:
                self._reject("per_user_limit")
            self._per_user[user] += 1
            try:
                if self.running >= self.max_concurrent:
                    if self.waiting >= self.max_queue:
                        self._reject("queue_full")
                    self.waiting += 1
                    end = time.monotonic() + self.queue_timeout_s
                    try:
                        while self.running >= self.max_concurrent and not self._closed:
                            left = end - time.monotonic()
                            if left <= 0:
                                self._reject("queue_timeout")
                            self._cv.wait(left)
                        if self._closed:
                            raise ShuttingDown()
                    finally:
                        self.waiting -= 1
            except BaseException:
                self._per_user[user] -= 1
                if self._per_user[user] <= 0:
                    del self._per_user[user]
                self._cv.notify_all()
                raise
            self.running += 1
            self.peak_running = max(self.peak_running, self.running)
        t0 = time.monotonic()
        try:
            yield
        finally:
            took = time.monotonic() - t0
            with self._cv:
                self.running -= 1
                self._per_user[user] -= 1
                if self._per_user[user] <= 0:
                    del self._per_user[user]
                self._avg_latency = 0.8 * self._avg_latency + 0.2 * took
                self._cv.notify_all()

    def drain(self, timeout_s: float = 10.0) -> bool:
        """Stop admitting new work and wait for what is running. True if everything finished in time."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
            end = time.monotonic() + timeout_s
            while self.running > 0:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
            return True

    @property
    def closed(self) -> bool:
        return self._closed

    def snapshot(self) -> dict:
        with self._cv:
            return {"running": self.running, "waiting": self.waiting, "peak_running": self.peak_running, "rejected": dict(self.rejected)}
