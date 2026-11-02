"""A response cache that cannot leak one user's answer to another.

The lesson's cache key is `normalize(query)` and nothing else, and the cached answer was written with that customer's private context in the prompt. A second customer
typing the same words gets the first customer's answer, the model never runs, and it stays there forever. Here:
  * an answer is SHAREABLE only if it did not use any user context and called only `public` tools (no order lookups, no refunds, nothing personal, no side effects)
  * the key includes the agent VERSION, so a new prompt or model never serves an old answer
  * entries expire (TTL), the cache is bounded (least recently used goes first), and errors are never cached
  * identical questions arriving together share one model call (see resilience.Singleflight)
"""

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict

from agentdef import AgentResult, AgentVersion


def normalize(query: str) -> str:
    return re.sub(r"[\s]+", " ", query.lower().strip()).rstrip("?.! ")


def cache_key(version_id: str, query: str) -> str:
    return hashlib.sha256(json.dumps([version_id, normalize(query)]).encode()).hexdigest()


def shareable(version: AgentVersion, result: AgentResult, had_user_context: bool) -> bool:
    """True only when the answer would be identical for every user and nothing happened in the world because of it."""
    if not result.ok or had_user_context or not result.text.strip():
        return False
    return all(version.tool(name).scope == "public" for name in result.tool_calls)


class ResponseCache:
    def __init__(self, max_entries: int = 1000, ttl_s: float = 300.0, clock=time.monotonic):
        self.max_entries, self.ttl_s, self.clock = max_entries, ttl_s, clock
        self._lock, self._data = threading.Lock(), OrderedDict()
        self.hits = self.misses = self.evictions = self.expired = 0

    def get(self, key: str):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            value, expires = item
            if expires <= self.clock():
                del self._data[key]
                self.expired += 1
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = (value, self.clock() + self.ttl_s)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.evictions += 1

    def __len__(self) -> int:
        return len(self._data)
