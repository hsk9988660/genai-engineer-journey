"""Reliability around model calls: retry only what is worth retrying, and cache without leaking across users.

RETRIES. Rate limits (429), server errors (5xx), timeouts and dropped connections are transient: retry with exponential
backoff plus JITTER (random spread, so a thousand clients that failed together do not all retry together). A bad request
(400), an auth error or a content problem is permanent: retrying only wastes money. If the server sends a Retry-After, obey it.

CACHE. A response cache is where privacy bugs hide. The key here MUST include the caller's permission scope and the prompt
version, and `scope` has no default value, so forgetting it is a TypeError instead of a data leak between users.

Provider fallback (switching to another model when one is down) is deliberately not built in: a different model is a
different system with different behavior, so it needs its own evaluation before it is allowed to serve users.
"""

import hashlib
import random
import time
from collections import OrderedDict

TRANSIENT_NAMES = {"RateLimitError", "InternalServerError", "APIConnectionError", "APITimeoutError", "OverloadedError",
                   "TimeoutError", "ConnectionError"}


def is_transient(exc: Exception) -> bool:
    """Judged by class name and status code so it works across SDK versions without importing them."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    return any(c.__name__ in TRANSIENT_NAMES for c in type(exc).__mro__)


def retry_with_backoff(fn, retries: int = 3, base_s: float = 0.5, cap_s: float = 8.0, sleep=time.sleep,
                       rng: random.Random | None = None, on_retry=None):
    """Call `fn()`. Retry transient errors up to `retries` times with full-jitter exponential backoff."""
    rng = rng or random.Random()
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if not is_transient(exc) or attempt >= retries:
                raise
            retry_after = getattr(exc, "retry_after", None)
            delay = float(retry_after) if retry_after is not None else rng.uniform(0, min(cap_s, base_s * 2 ** attempt))
            attempt += 1
            if on_retry:
                on_retry(attempt, exc, delay)
            sleep(delay)


def idempotency_key(run_id: str, step: int, tool: str, args_repr: str) -> str:
    """The same logical action always gets the same key, so a retried side effect can be recognized and not repeated."""
    return hashlib.sha256(f"{run_id}|{step}|{tool}|{args_repr}".encode()).hexdigest()[:32]


class ScopedCache:
    """An LRU cache with expiry, keyed by (scope, prompt_version, normalized input)."""

    def __init__(self, ttl_s: float = 300.0, max_items: int = 1000, clock=time.monotonic):
        self.ttl_s, self.max_items, self.clock = ttl_s, max_items, clock
        self._items: OrderedDict = OrderedDict()
        self.hits = self.misses = 0

    @staticmethod
    def _key(scope: str, prompt_version: str, text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(f"{scope}\x00{prompt_version}\x00{normalized}".encode()).hexdigest()

    def get(self, *, scope: str, prompt_version: str, text: str):
        key = self._key(scope, prompt_version, text)
        item = self._items.get(key)
        if item is None or self.clock() - item[0] > self.ttl_s:
            self._items.pop(key, None)
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return item[1]

    def put(self, value, *, scope: str, prompt_version: str, text: str) -> None:
        self._items[self._key(scope, prompt_version, text)] = (self.clock(), value)
        self._items.move_to_end(self._key(scope, prompt_version, text))
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
