"""Observability that can never take your product down.

Per request you get ONE run record (summary: versions, tokens, cost, latency, outcome, tools, feedback) and a list of
spans (one per model call or tool call). Design rules that matter in production:

  * Telemetry must never crash the request. An unknown model price, a failing exporter or a bad payload is recorded as
    a problem, not raised.
  * Redact before export. Traces hold user data, so tool arguments, results and errors pass through PII redaction at
    record time, and the user id is replaced by a salted HMAC.
  * Summaries for EVERY run (cheap, drive dashboards and alerts). Full traces only for SAMPLED runs (expensive,
    sensitive). See sampling.py.
  * Cost is computed from tokens and a DATED price table, because prices change and old runs must keep their old cost.

The price numbers in EXAMPLE_PRICES are placeholders for made-up model names. Set the real prices for your models.
"""

import hashlib
import hmac
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

from redact import redact_text

MAX_PAYLOAD_CHARS = 2000


class UnknownModel(Exception):
    pass


@dataclass(frozen=True)
class Price:
    model: str
    effective_from: str  # ISO date; the price applies from this date until a later entry replaces it
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None  # if None, cached tokens are billed at the normal input rate


class PriceTable:
    def __init__(self, prices: list[Price], today=lambda: time.strftime("%Y-%m-%d")):
        self.prices, self.today = list(prices), today

    def cost(self, model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0, on: str | None = None) -> float:
        """`input_tokens` are the NON-cached input tokens; `cached_tokens` were served from the provider's cache."""
        day = on or self.today()
        candidates = [p for p in self.prices if p.model == model and p.effective_from <= day]
        if not candidates:
            raise UnknownModel(f"no price for {model!r} on {day}")
        p = max(candidates, key=lambda x: x.effective_from)
        cached_rate = p.input_per_mtok if p.cached_input_per_mtok is None else p.cached_input_per_mtok
        return (input_tokens * p.input_per_mtok + cached_tokens * cached_rate + output_tokens * p.output_per_mtok) / 1e6


EXAMPLE_PRICES = PriceTable([Price("example-large", "2026-01-01", 3.0, 15.0, 0.3),
                             Price("example-small", "2026-01-01", 1.0, 5.0, 0.1)])


@dataclass
class RunRecord:
    run_id: str = ""
    user: str = ""
    versions: dict = field(default_factory=dict)
    outcome: str = ""
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    latency_ms: float = 0.0
    tools_used: list = field(default_factory=list)
    tool_calls: int = 0
    tool_failures: int = 0
    errors: list = field(default_factory=list)
    fallbacks: int = 0
    retries: int = 0
    feedback: str | None = None
    spans: list = field(default_factory=list)

    def summary(self) -> dict:
        """The record without spans: safe and cheap enough to keep for every run."""
        d = asdict(self)
        d.pop("spans")
        return d


class MemoryExporter:
    def __init__(self):
        self.items: list = []

    def export(self, item: dict) -> None:
        self.items.append(item)


class JsonlExporter:
    def __init__(self, path):
        self.path, self._lock = path, threading.Lock()

    def export(self, item: dict) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, default=str) + "\n")


def _clip(value, limit=MAX_PAYLOAD_CHARS):
    """Serialize, REDACT, then truncate. Every payload that enters a span goes through here, so redaction cannot be skipped."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return redact_text(text)[0][:limit]


class Run:
    def __init__(self, telemetry: "Telemetry", run_id: str, user: str, versions: dict):
        self._t, self._lock = telemetry, threading.Lock()
        self.rec = RunRecord(run_id=run_id, user=user, versions=dict(versions))
        self._started = telemetry.clock()
        self._tools: set = set()

    def record_model_call(self, model: str, input_tokens: int, output_tokens: int, cached: int = 0,
                          latency_ms: float = 0.0, finish_reason: str = "", error: str | None = None) -> None:
        try:
            cost = self._t.prices.cost(model, input_tokens, output_tokens, cached) if self._t.prices else None
        except UnknownModel:
            cost = None  # never crash the request over a missing price: flag it
        with self._lock:
            r = self.rec
            r.steps += 1
            r.input_tokens += input_tokens
            r.output_tokens += output_tokens
            r.cached_tokens += cached
            if cost is None:
                r.unpriced_calls += 1
            else:
                r.cost_usd += cost
            if error:
                r.errors.append(_clip(error, 300))
            r.spans.append({"kind": "model", "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens,
                            "cached_tokens": cached, "latency_ms": round(latency_ms, 1), "finish_reason": finish_reason,
                            "cost_usd": cost, "error": _clip(error, 300) if error else None})

    def record_tool_call(self, name: str, args, result, duration_ms: float = 0.0, ok: bool = True) -> None:
        with self._lock:
            r = self.rec
            r.tool_calls += 1
            r.tool_failures += 0 if ok else 1
            self._tools.add(name)
            r.spans.append({"kind": "tool", "name": name, "args": _clip(args), "result": _clip(result),
                            "duration_ms": round(duration_ms, 1), "ok": ok})

    def note_retry(self) -> None:
        with self._lock:
            self.rec.retries += 1

    def note_fallback(self) -> None:
        with self._lock:
            self.rec.fallbacks += 1

    def add_feedback(self, value: str | None) -> None:
        self.rec.feedback = value  # "up", "down" or None

    def finish(self, outcome: str) -> RunRecord:
        with self._lock:
            r = self.rec
            r.outcome = outcome
            r.latency_ms = round((self._t.clock() - self._started) * 1000, 1)
            r.tools_used = sorted(self._tools)
        self._t._emit(r)
        return r


class Telemetry:
    def __init__(self, prices: PriceTable | None = None, salt: bytes = b"change-me", sampling=None, summary_sinks=None,
                 trace_exporters=None, clock=time.monotonic, id_factory=lambda: uuid.uuid4().hex):
        self.prices, self.salt, self.sampling = prices, salt, sampling
        self.summary_sinks = summary_sinks if summary_sinks is not None else []
        self.trace_exporters = trace_exporters if trace_exporters is not None else []
        self.clock, self.id_factory = clock, id_factory
        self.export_errors: list[str] = []

    def pseudonymize(self, user_id: str) -> str:
        return hmac.new(self.salt, user_id.encode(), hashlib.sha256).hexdigest()[:16]

    def start_run(self, user_id: str, versions: dict | None = None) -> Run:
        return Run(self, self.id_factory(), self.pseudonymize(user_id), versions or {})

    def _emit(self, rec: RunRecord) -> None:
        for sink in self.summary_sinks:  # every run, always
            self._safely(sink.export, rec.summary())
        keep, reason = self.sampling.decide(rec) if self.sampling else (True, "no sampling policy")
        if keep:
            for exporter in self.trace_exporters:
                self._safely(exporter.export, {**asdict(rec), "sample_reason": reason})

    def _safely(self, fn, item) -> None:
        try:
            fn(item)
        except Exception as exc:  # an exporter outage must never break a request
            self.export_errors.append(f"{type(exc).__name__}: {exc}")


class TelemetryClient:
    """Wrap any client that has `messages.create` so every model call becomes a span automatically."""

    def __init__(self, client, run: Run):
        self._client, self._run = client, run
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        start = self._run._t.clock()
        model = str(kwargs.get("model", "unknown"))
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            self._run.record_model_call(model, 0, 0, latency_ms=(self._run._t.clock() - start) * 1000,
                                        error=f"{type(exc).__name__}: {exc}")
            raise
        usage = resp.usage
        self._run.record_model_call(model, usage.input_tokens, usage.output_tokens,
                                    cached=getattr(usage, "cache_read_input_tokens", 0) or 0,
                                    latency_ms=(self._run._t.clock() - start) * 1000,
                                    finish_reason=getattr(resp, "stop_reason", "") or "")
        return resp
