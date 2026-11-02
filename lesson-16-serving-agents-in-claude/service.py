"""The serving layer: one request in, one response out, with every production concern in its place.

Order of work for a request (each step can end it with a precise status):
  1 validate         bad input -> 400
  2 admit            too busy -> 429 + Retry-After; shutting down -> 503
  3 idempotency      a repeated key returns the first answer instead of running again
  4 budget           this user or the day is out of budget -> 429
  5 pick a version   a stable canary slice of users gets the candidate
  6 thread           load the user's own thread (someone else's id is a 404)
  7 cache            only a shareable answer to a first-turn question; identical simultaneous questions share one model call
  8 cascade          cheap tier first, check, escalate; each provider call goes through retries (backoff + jitter, within the deadline) and a circuit breaker
  9 persist, cost, metrics, log
The agent function does the actual model work. It receives the user id so TOOLS can use it; the model never supplies it.
"""

import random
import time
from dataclasses import dataclass, replace

from admission import Admission, Overloaded, ShuttingDown
from agentdef import AgentCall, AgentResult, AgentVersion
from cache import ResponseCache, cache_key, shareable
from cost import BudgetExceeded, CostLedger
from observability import Metrics, RequestLog, new_trace_id
from release import Rollout
from resilience import (CircuitBreaker, CircuitOpen, DeadlineExceeded, IdempotencyStore, Permanent, RetryPolicy, Singleflight, Transient, call_with_retry)
from router import ToolNotAllowed, cascade, default_check, enforce_tools, start_index
from runtrace import digest
from threads import NotFound, Scope, ThreadError, ThreadStore

HTTP = {"ok": 200, "invalid": 400, "not_found": 404, "thread_full": 409, "overloaded": 429, "budget": 429, "unavailable": 503, "shutting_down": 503, "deadline": 504, "error": 500}
FAILURES = frozenset({"unavailable", "error", "deadline"})  # what counts against a version during a canary. Bad input, busy and out-of-budget are not the version's fault
MAX_TEXT = 4000
MAX_ID = 64


@dataclass
class Request:
    user_id: str
    text: str
    thread_id: str | None = None
    idempotency_key: str | None = None
    trace_id: str = ""
    deadline_s: float = 30.0


@dataclass
class Response:
    status: str
    text: str = ""
    thread_id: str | None = None
    version: str = ""
    tier: str = ""
    cached: bool = False
    trace_id: str = ""
    retry_after_s: float | None = None
    detail: str = ""

    @property
    def http(self) -> int:
        return HTTP[self.status]


class AgentService:
    def __init__(self, versions: dict, default_version: str, agent_fn, store: ThreadStore, *, rollout: Rollout | None = None, admission: Admission | None = None,
                 cache: ResponseCache | None = None, ledger: CostLedger | None = None, metrics: Metrics | None = None, log: RequestLog | None = None,
                 breaker: CircuitBreaker | None = None, retry: RetryPolicy = RetryPolicy(), idempotency: IdempotencyStore | None = None, user_context=None,
                 check=default_check, history_messages: int = 20, clock=time.monotonic, sleep=time.sleep, rng=random.random):
        if default_version not in versions:
            raise ValueError("default_version is not one of the versions")
        self.versions, self.default_version, self.agent_fn, self.store = versions, default_version, agent_fn, store
        self.rollout = rollout
        self.admission, self.cache, self.ledger = admission or Admission(), cache or ResponseCache(), ledger or CostLedger()
        self.metrics, self.log, self.breaker = metrics or Metrics(), log or RequestLog(), breaker or CircuitBreaker()
        self.retry, self.idem, self.user_context, self.check = retry, idempotency or IdempotencyStore(), user_context, check
        self.history_messages, self.clock, self.sleep, self.rng = history_messages, clock, sleep, rng
        self._flight = Singleflight()

    # --- the public surface ---------------------------------------------------------------------------------------------------------
    def handle(self, req: Request) -> Response:
        trace, t0 = req.trace_id or new_trace_id(), self.clock()
        try:
            resp = self._handle(req, trace)
        except Exception as exc:  # a serving layer must never throw: whatever went wrong, the caller gets a clean 500 with no message in it
            self.metrics.inc("unexpected_errors")
            resp = Response("error", detail=type(exc).__name__)
        resp = replace(resp, trace_id=trace)
        took = self.clock() - t0
        self.metrics.inc("requests", status=resp.status)
        self.metrics.observe("latency_s", took)
        if resp.tier:
            self.metrics.inc("tier_requests", tier=resp.tier)
        if self.rollout and resp.version and (resp.status == "ok" or resp.status in FAILURES):
            self.rollout.observe(resp.version, resp.status == "ok")
        self.log.emit(trace, "request", status=resp.status, version=resp.version, tier=resp.tier, cached=resp.cached, user=digest(str(req.user_id))[:12], ms=int(took * 1000))
        return resp

    def ready(self) -> dict:
        reasons = []
        if self.admission.closed:
            reasons.append("draining")
        if self.breaker.state == "open":
            reasons.append("provider circuit open")
        try:
            if not self.store.ping():
                reasons.append("thread store not answering")
        except Exception:
            reasons.append("thread store not answering")
        return {"ready": not reasons, "reasons": reasons}

    def drain(self, timeout_s: float = 10.0) -> bool:
        return self.admission.drain(timeout_s)

    def check_rollout(self) -> str:
        return self.rollout.evaluate() if self.rollout else "no rollout"

    # --- internals ------------------------------------------------------------------------------------------------------------------
    def _validate(self, req: Request) -> str | None:
        if not isinstance(req.user_id, str) or not req.user_id or len(req.user_id) > 200:
            return "user_id is required"
        if not isinstance(req.text, str) or not req.text.strip() or len(req.text) > MAX_TEXT:
            return f"text must be 1 to {MAX_TEXT} characters"
        for name in ("thread_id", "idempotency_key"):
            v = getattr(req, name)
            if v is not None and (not isinstance(v, str) or not v or len(v) > MAX_ID):
                return f"{name} must be 1 to {MAX_ID} characters"
        if not isinstance(req.deadline_s, (int, float)) or not 0 < req.deadline_s <= 300:
            return "deadline_s must be between 0 and 300"
        return None

    def _handle(self, req: Request, trace: str) -> Response:
        bad = self._validate(req)
        if bad:
            return Response("invalid", detail=bad)
        t0 = self.clock()
        try:
            with self.admission.slot(req.user_id):
                if req.idempotency_key:
                    resp, replayed = self.idem.run(f"{req.user_id}:{req.idempotency_key}", lambda: self._serve(req, t0), store_if=lambda r: r.status == "ok")
                    return replace(resp, detail="replayed") if replayed else resp
                return self._serve(req, t0)
        except Overloaded as exc:
            return Response("overloaded", retry_after_s=exc.retry_after_s, detail=exc.reason)
        except ShuttingDown:
            return Response("shutting_down", retry_after_s=5, detail="shutting down")

    def _pick_version(self, user: str) -> AgentVersion:
        vid = self.rollout.version_for(user) if self.rollout else self.default_version
        return self.versions[vid]

    def _serve(self, req: Request, t0: float) -> Response:
        user, deadline = req.user_id, t0 + req.deadline_s
        try:
            self.ledger.check(user)
        except BudgetExceeded as exc:
            return Response("budget", detail=exc.who)
        version, scope, history = self._pick_version(user), Scope(user), []
        if req.thread_id:
            try:
                history = self.store.messages(scope, req.thread_id, last=self.history_messages)
                if self.store.count(scope, req.thread_id) >= self.store.max_messages - 1:
                    return Response("thread_full", thread_id=req.thread_id, version=version.version_id)
            except NotFound:
                return Response("not_found", detail="no such thread")
        context = self.user_context(user) if self.user_context else ""
        messages = history + [{"role": "user", "content": req.text}]
        use_cache = req.thread_id is None and not context
        key = cache_key(version.version_id, req.text)
        try:
            if use_cache:
                hit = self.cache.get(key)
                if hit is not None:
                    self.ledger.record_cache_hit()
                    self.metrics.inc("cache", result="hit")
                    return Response("ok", hit, self._persist(scope, None, req.text, hit), version.version_id, "cache", cached=True)
                self.metrics.inc("cache", result="miss")
                (result, tier, passed), shared = self._flight.do(key, lambda: self._compute(version, user, messages, context, deadline, req.text))
                if shared:
                    self.ledger.record_cache_hit()
                    self.metrics.inc("cache", result="shared")
            else:
                (result, tier, passed), shared = self._compute(version, user, messages, context, deadline, req.text), False
        except CircuitOpen as exc:
            return Response("unavailable", version=version.version_id, retry_after_s=max(1, round(exc.retry_after_s)), detail="provider unavailable")
        except Transient:
            return Response("unavailable", version=version.version_id, retry_after_s=5, detail="provider failing")
        except DeadlineExceeded:
            return Response("deadline", version=version.version_id, detail="deadline exceeded")
        except (Permanent, ToolNotAllowed) as exc:
            return Response("error", version=version.version_id, detail=type(exc).__name__)
        if not passed:
            return Response("error", version=version.version_id, tier=tier, detail="no tier produced an acceptable answer")
        if use_cache and shareable(version, result, bool(context)):
            self.cache.put(key, result.text)
        return Response("ok", result.text, self._persist(scope, req.thread_id, req.text, result.text), version.version_id, tier)

    def _compute(self, version: AgentVersion, user: str, messages: list, context: str, deadline: float, text: str):
        def run_tier(tier):
            def attempt():
                if self.clock() >= deadline:
                    raise DeadlineExceeded("deadline passed")
                call = AgentCall(version, tier, messages, context, deadline, user)
                return self.breaker.call(lambda: enforce_tools(tier, self.agent_fn(call)))
            res = call_with_retry(attempt, self.retry, self.sleep, self.rng, self.clock, deadline, on_retry=lambda n, d: self.metrics.inc("retries"))
            self.ledger.record(user, tier, res.input_tokens, res.output_tokens)  # tokens were spent even if the answer is then rejected
            return res
        result, tier, passed, attempts = cascade(version, run_tier, self.check, start_index(version, text))
        if len(attempts) > 1:
            self.metrics.inc("escalations", len(attempts) - 1)
        return result, tier, passed

    def _persist(self, scope: Scope, thread_id: str | None, user_text: str, answer: str) -> str | None:
        """A failure to save the thread must not throw away an answer the user already waited for."""
        try:
            tid = thread_id or self.store.create(scope)
            self.store.append(scope, tid, "user", user_text)
            self.store.append(scope, tid, "assistant", answer)
            return tid
        except ThreadError:
            self.metrics.inc("persist_failed")
            return thread_id
