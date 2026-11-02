"""A run trace that is safe to log: event types, node ids, counts, durations and hashes. Never payload content."""

import hashlib
import json
import time


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class Trace:
    def __init__(self, clock=time.time):
        self.clock, self.events = clock, []

    def emit(self, type: str, **fields) -> None:
        self.events.append({"seq": len(self.events) + 1, "ts": round(self.clock(), 4), "type": type, **fields})

    def types(self) -> list[str]:
        return [e["type"] for e in self.events]

    def of(self, type: str) -> list[dict]:
        return [e for e in self.events if e["type"] == type]
