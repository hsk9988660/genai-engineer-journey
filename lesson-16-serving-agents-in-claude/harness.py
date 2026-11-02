"""The lesson's patterns versus the served versions, on six scenarios. Offline. Threads and small real sleeps are used only where concurrency is the point.

The "model" is a labelled TEST DOUBLE: a provider with a fixed latency and rate limit, a rule-based agent that calls the real support tools, and a seeded failure rate.
So this shows what the MECHANISMS do. It says nothing about real Claude latency, accuracy, rate limits or prices (prices here are placeholders).
"""

import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal

import support_tools
from admission import Admission
from agentdef import AgentCall, AgentResult, AgentVersion, Tier
from cache import normalize
from cost import CostLedger
from release import Rollout, gate, naive_gate_error_rates
from resilience import CircuitBreaker, Permanent, RetryPolicy, Transient
from service import AgentService, Request
from threads import ThreadStore


@dataclass
class Outcome:
    scenario: str
    strategy: str
    metrics: dict = field(default_factory=dict)


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s: float):
        self.t += s


def make_version(prompt: str = "You are Contoso's support agent. Be brief and never invent policy.", top_keywords=("refund", "cancel")) -> AgentVersion:
    small = Tier("small", "claude-haiku-5-5", ("search_policies", "get_order_status"), Decimal("0.001"), Decimal("0.005"))
    large = Tier("large", "claude-sonnet-5-5", ("search_policies", "get_order_status", "issue_refund"), Decimal("0.003"), Decimal("0.015"))
    return AgentVersion("support", prompt, (small, large), support_tools.SPECS, {"top_keywords": top_keywords})


class Provider:
    """TEST DOUBLE for the model provider: fixed latency, an in-flight rate limit that answers 429, and a call counter."""

    def __init__(self, latency: float = 0.0, limit: int | None = None, responder=None, sleep=time.sleep):
        self.latency, self.limit, self.sleep = latency, limit, sleep
        self.responder = responder or (lambda call: AgentResult("ok: " + call.messages[-1]["content"][:20], (), 100, 20))
        self.calls = self.in_flight = self.peak = self.rate_limited = 0
        self._lock = threading.Lock()

    def __call__(self, call: AgentCall) -> AgentResult:
        with self._lock:
            self.calls += 1
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            over = self.limit is not None and self.in_flight > self.limit
        try:
            if self.latency:
                self.sleep(self.latency)
            if over:
                with self._lock:
                    self.rate_limited += 1
                raise Transient("429 rate limited")
            return self.responder(call)
        finally:
            with self._lock:
                self.in_flight -= 1


def make_service(provider, version=None, **kw) -> AgentService:
    version = version or make_version()
    return AgentService({version.version_id: version}, version.version_id, provider, ThreadStore(), **kw)


# --- 1. a burst of traffic ----------------------------------------------------------------------------------------------------------
def scenario_overload(n: int = 40, limit: int = 5) -> list[Outcome]:
    out = []
    configs = [("unbounded", dict(admission=Admission(max_concurrent=1000, max_queue=0, per_user_max=10), retry=RetryPolicy(attempts=1), breaker=CircuitBreaker(10 ** 9))),
               ("bounded_with_queue", dict(admission=Admission(max_concurrent=limit, max_queue=n, per_user_max=4, queue_timeout_s=10))),
               ("bounded_small_queue", dict(admission=Admission(max_concurrent=limit, max_queue=5, per_user_max=4, queue_timeout_s=10)))]
    for name, kw in configs:
        provider = Provider(latency=0.03, limit=limit)
        svc = make_service(provider, **kw)
        barrier = threading.Barrier(n)

        def one(i):
            barrier.wait()
            return svc.handle(Request(f"user-{i}", f"question number {i}")).status
        with ThreadPoolExecutor(n) as pool:
            statuses = list(pool.map(one, range(n)))
        out.append(Outcome("overload", name, {"ok": statuses.count("ok"), "overloaded_429": statuses.count("overloaded"), "failed_503": statuses.count("unavailable"),
                                              "provider_429s": provider.rate_limited, "peak_in_provider": provider.peak}))
    return out


# --- 2. a provider outage ----------------------------------------------------------------------------------------------------------------
def scenario_outage(n: int = 20) -> list[Outcome]:
    out = []
    for name, retry, breaker_threshold in (("retry_5x_no_breaker", RetryPolicy(attempts=5), 10 ** 9), ("retry_3x_with_breaker", RetryPolicy(attempts=3), 5)):
        clock, down = FakeClock(), [True]

        def responder(call):
            if down[0]:
                raise Transient("503 outage")
            return AgentResult("ok", (), 10, 5)
        provider = Provider(responder=responder)
        svc = make_service(provider, retry=retry, breaker=CircuitBreaker(breaker_threshold, 30.0, clock), clock=clock, sleep=lambda s: None, rng=lambda: 0.5)
        statuses = [svc.handle(Request(f"u{i}", f"q{i}")).status for i in range(n)]
        during = provider.calls
        down[0] = False
        clock.advance(31)
        after = svc.handle(Request("u-after", "are you back?")).status
        out.append(Outcome("outage", name, {"requests": n, "provider_calls_during_outage": during, "all_failed": statuses.count("unavailable") == n, "first_request_after_reset": after}))
    return out


# --- 3. one customer's answer shown to another ----------------------------------------------------------------------------------------
def rule_based_agent(impls):
    """TEST DOUBLE for a tool-using model: calls the order tool for order questions, the policy tool otherwise, using the REAL support tools."""
    provider_calls = {"n": 0}

    def fn(call: AgentCall) -> AgentResult:
        provider_calls["n"] += 1
        q = call.messages[-1]["content"].lower()
        m = re.search(r"\b(a\d{4})\b", q)
        if "order" in q and m:
            return AgentResult(impls["get_order_status"](call.user_id, m.group(1)), ("get_order_status",), 120, 30)
        return AgentResult(impls["search_policies"](call.user_id, q), ("search_policies",), 120, 30)
    fn.calls = provider_calls
    return fn


def scenario_cache_leak() -> list[Outcome]:
    data = support_tools.SupportData()
    # the lesson's design: tools that trust the model's order id, and a cache keyed by the normalized query alone
    naive_cache, naive_model_calls = {}, [0]

    def lesson_handle(query: str, customer_id: str) -> tuple[str, bool]:
        key = normalize(query)
        if key in naive_cache:
            return naive_cache[key], True
        naive_model_calls[0] += 1
        m = re.search(r"\b(a\d{4})\b", query.lower())
        order = data.orders.get(m.group(1).upper()) if m else None
        text = f"Order status={order['status']}, total=${order['total']}" if order else "No order found."
        naive_cache[key] = text
        return text, False
    first, _ = lesson_handle("Where is my order A1002?", "cust-42")
    second, from_cache = lesson_handle("where is my order a1002", "cust-99")
    out = [Outcome("cache_leak", "lesson_design", {"other_customer_saw_the_order": "processing" in second, "served_from_cache": from_cache, "model_calls": naive_model_calls[0]})]
    agent = rule_based_agent(support_tools.build_tools(support_tools.SupportData()))
    svc = make_service(Provider(responder=agent))
    a = svc.handle(Request("cust-42", "Where is my order A1002?"))
    b = svc.handle(Request("cust-99", "where is my order a1002"))
    publics = [svc.handle(Request(f"cust-{i}", "What is your return window?")) for i in range(5)]
    out.append(Outcome("cache_leak", "served", {"owner_saw_the_order": "processing" in a.text, "other_customer_saw_the_order": "processing" in b.text, "other_customer_cached": b.cached,
                                                "public_question_model_calls": sum(1 for p in publics if not p.cached), "public_question_cache_hits": sum(1 for p in publics if p.cached)}))
    return out


# --- 4. a bad release --------------------------------------------------------------------------------------------------------------------
def _canary_run(candidate_error: float, stable_error: float = 0.02, total: int = 1000, batch: int = 100, seed: int = 5, until_promoted: bool = False):
    stable, cand = make_version("You are Contoso's support agent. Be brief."), make_version("You are Contoso's support agent. Be brief and very friendly.")
    rng = random.Random(seed)

    def responder(call):
        p = candidate_error if call.version.version_id == cand.version_id else stable_error
        if rng.random() < p:
            raise Permanent("bad answer")
        return AgentResult("ok", (), 50, 10)
    rollout = Rollout(stable.version_id, cand.version_id, salt="harness", min_samples=8)
    svc = AgentService({stable.version_id: stable, cand.version_id: cand}, stable.version_id, Provider(responder=responder), ThreadStore(), rollout=rollout)
    errors = exposed = n = 0
    limit = 30_000 if until_promoted else total
    while n < limit:
        for _ in range(batch):
            r = svc.handle(Request(f"user-{n}", f"question {n}"))
            errors += r.status != "ok"
            exposed += r.version == cand.version_id
            n += 1
        svc.check_rollout()
        if until_promoted and rollout.state != "running":
            break
    return {"errors": errors, "served_candidate": exposed, "requests": n, "final_state": rollout.state}


def scenario_release() -> list[Outcome]:
    rng, bad = random.Random(5), 0.30
    naive_errors = sum(rng.random() < bad for _ in range(1000))  # shipping the bad version to everyone at once
    canary = _canary_run(bad)
    good = _canary_run(0.02, until_promoted=True)
    return [Outcome("release", "ship_to_everyone", {"requests": 1000, "error_responses": naive_errors}),
            Outcome("release", "canary_bad_candidate", {"requests": canary["requests"], "error_responses": canary["errors"], "served_candidate": canary["served_candidate"], "final_state": canary["final_state"]}),
            Outcome("release", "canary_good_candidate", {"requests_until_decision": good["requests"], "final_state": good["final_state"]})]


# --- 5. the evaluation gate --------------------------------------------------------------------------------------------------------------
def scenario_gate(trials: int = 2000, seed: int = 7) -> list[Outcome]:
    exact = naive_gate_error_rates(0.9, 0.6, n=4, need=0.8)
    rng = random.Random(seed)

    def simulate(true_rate: float, n: int) -> dict:
        counts = {"pass": 0, "fail": 0, "inconclusive": 0}
        for _ in range(trials):
            counts[gate(sum(rng.random() < true_rate for _ in range(n)), n).decision] += 1
        return {k: round(v / trials, 3) for k, v in counts.items()}
    return [Outcome("gate", "lesson_rule_4_cases", {"good_agent_(90%)_blocked": round(exact["good_agent_blocked"], 3), "bad_agent_(60%)_shipped": round(exact["bad_agent_shipped"], 3)}),
            Outcome("gate", "interval_gate_100_cases_good_agent", simulate(0.9, 100)),
            Outcome("gate", "interval_gate_100_cases_bad_agent", simulate(0.6, 100)),
            Outcome("gate", "interval_gate_4_cases_good_agent", simulate(0.9, 4))]


# --- 6. the cost report -------------------------------------------------------------------------------------------------------------------
def scenario_cost() -> list[Outcome]:
    def responder(call):
        text = call.messages[-1]["content"].lower()
        if call.tier.name == "small" and ("why" in text or "compare" in text):
            return AgentResult("ESCALATE", (), 200, 5)  # the cheap tier cannot handle it
        return AgentResult("Here is your answer.", (), 200, 40)
    ledger = CostLedger(user_daily_budget=Decimal("5"), global_daily_budget=Decimal("50"))
    svc = make_service(Provider(responder=responder), ledger=ledger, check=lambda r: r.ok and "ESCALATE" not in r.text)
    queries = ["What is your return window?", "How long does shipping take?", "What is your return window?", "Do you offer a warranty?", "What is your return window?",
               "Why was my last order split in two?", "Compare express and standard shipping for me", "How long does shipping take?", "Why is the price different today?", "Do you offer a warranty?"]
    for i, q in enumerate(queries):
        svc.handle(Request(f"cust-{i % 3}", q))
    rep = ledger.report()
    return [Outcome("cost", "ten_mixed_queries", {"small": rep["requests_by_tier"].get("small", 0), "large": rep["requests_by_tier"].get("large", 0), "from_cache": rep["served_from_cache"],
                                                   "total_cost": rep["total_cost"], "escalations": svc.metrics.count("escalations")})]


def evaluate() -> list[Outcome]:
    return scenario_overload() + scenario_outage() + scenario_cache_leak() + scenario_release() + scenario_gate() + scenario_cost()


def print_table(outcomes: list[Outcome]) -> None:
    for o in outcomes:
        print(f"{o.scenario:<11} {o.strategy:<36} " + "  ".join(f"{k}={v}" for k, v in o.metrics.items()))


if __name__ == "__main__":
    print_table(evaluate())
