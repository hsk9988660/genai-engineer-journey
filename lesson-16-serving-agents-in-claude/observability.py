"""Metrics, alert rules, and a request log that cannot contain what users typed.

You cannot operate what you cannot see, but the easiest way to see is to log everything, and then the log holds every user's questions and answers. So:
  * metrics are counters and latency histograms (p50/p95), keyed by labels such as status and tier
  * the request log REFUSES fields called text, query, prompt, content and similar, and refuses long string values, so a careless `log(text=...)` fails loudly
  * alert rules are plain data evaluated against the metrics
Every request carries a trace id, returned to the caller, so one complaint can be followed through the logs.
"""

import math
import secrets
import threading
from collections import defaultdict, deque

FORBIDDEN_FIELDS = frozenset({"text", "query", "prompt", "content", "message", "messages", "answer", "response", "body", "input", "output"})
MAX_LOG_VALUE = 100
MAX_SAMPLES = 10_000


def new_trace_id() -> str:
    return secrets.token_hex(8)


def _key(name: str, labels: dict) -> str:
    return name + ("{" + ",".join(f"{k}={labels[k]}" for k in sorted(labels)) + "}" if labels else "")


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters = defaultdict(int)
        self._samples = defaultdict(lambda: deque(maxlen=MAX_SAMPLES))

    def inc(self, name: str, n: int = 1, **labels) -> None:
        with self._lock:
            self.counters[_key(name, labels)] += n

    def observe(self, name: str, value: float, **labels) -> None:
        with self._lock:
            self._samples[_key(name, labels)].append(float(value))

    def count(self, name: str, **labels) -> int:
        with self._lock:
            return self.counters.get(_key(name, labels), 0)

    def total(self, name: str) -> int:
        """Sum of a counter across all its label values."""
        with self._lock:
            return sum(v for k, v in self.counters.items() if k == name or k.startswith(name + "{"))

    def percentile(self, name: str, p: float, **labels) -> float | None:
        """Nearest-rank percentile over the most recent samples. None when there are none."""
        with self._lock:
            data = sorted(self._samples.get(_key(name, labels), ()))
        if not data:
            return None
        return data[max(0, math.ceil(p / 100 * len(data)) - 1)]

    def lines(self) -> str:
        with self._lock:
            return "\n".join(f"{k} {v}" for k, v in sorted(self.counters.items()))


class RequestLog:
    def __init__(self):
        self._lock = threading.Lock()
        self.events = []

    def emit(self, trace_id: str, event: str, **fields) -> None:
        for k, v in fields.items():
            if k.lower() in FORBIDDEN_FIELDS:
                raise ValueError(f"log field {k!r} is not allowed: user content must not be logged")
            if isinstance(v, str) and len(v) > MAX_LOG_VALUE:
                raise ValueError(f"log field {k!r} is a long string: log ids, counts and codes, not content")
        with self._lock:
            self.events.append({"trace": trace_id, "event": event, **fields})

    def for_trace(self, trace_id: str) -> list:
        with self._lock:
            return [e for e in self.events if e["trace"] == trace_id]


class AlertRule:
    """`kind`: 'ratio' fires when numerator_total / denominator_total > threshold; 'p95' when the p95 of a histogram > threshold; 'counter' when a counter > threshold."""

    def __init__(self, name: str, kind: str, threshold: float, metric: str = "", numerator: dict | None = None, denominator: str = "requests", min_events: int = 20):
        self.name, self.kind, self.threshold, self.metric, self.numerator, self.denominator, self.min_events = name, kind, threshold, metric, numerator or {}, denominator, min_events

    def firing(self, m: Metrics) -> bool:
        if self.kind == "p95":
            p = m.percentile(self.metric, 95)
            return p is not None and p > self.threshold
        if self.kind == "counter":
            return m.total(self.metric) > self.threshold
        if self.kind == "ratio":
            denom = m.total(self.denominator)
            if denom < self.min_events:  # too few events for a rate to mean anything
                return False
            num = sum(m.count(self.denominator, **{k: v}) for k, vals in self.numerator.items() for v in vals)
            return num / denom > self.threshold
        raise ValueError(f"unknown alert kind {self.kind!r}")


def evaluate_alerts(m: Metrics, rules: list) -> list[str]:
    return [r.name for r in rules if r.firing(m)]
