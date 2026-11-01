"""Compares the lesson's patterns with the guarded versions on five scenarios. Offline and deterministic (apart from small real sleeps in scenario 4).

The "model" is a labelled test double that misreports availability, so this shows what the MECHANISMS do when a model is wrong. It does not measure how often a
real model is wrong.
"""

import os
import tempfile
import time
from dataclasses import dataclass, field

from checkpoint import CheckpointStore, EffectLedger
from handoff import Handoff, HandoffError, HandoffRouter, Reply
from middleware import Identity, SyntheticFact, UndeclaredOverride
from workflow import StepFailed, Workflow

INVENTORY = {"paris": 0, "stockholm": 3}
ALICE = Identity("alice")


@dataclass
class Outcome:
    scenario: str
    strategy: str
    metrics: dict = field(default_factory=dict)


def liar_model(city: str) -> dict:
    """TEST DOUBLE: a 'model' that says every city has rooms. Wrong for Paris."""
    return {"city": city, "available": True}


# --- 1. routing on the model's claim vs on checked data -------------------------------------------------------------------------
def _booking_workflow(bookings: list, checked: bool) -> Workflow:
    wf = Workflow("hotel").step("ask", lambda city, ctx: liar_model(city), kind="model")
    if checked:
        wf.step("verify", lambda claim, ctx: {"city": claim["city"], "has_rooms": INVENTORY.get(claim["city"], 0) > 0})
        wf.edge("ask", "verify")
        source, field_ = "verify", "has_rooms"
    else:
        source, field_ = "ask", "available"
    wf.step("book", lambda d, ctx: bookings.append(d["city"]) or {"booked": d["city"]})
    wf.step("alternative", lambda d, ctx: {"suggested": "stockholm"})
    wf.edge(source, "book", when=lambda d: d[field_], name="has_rooms", trust_model_output=(not checked))
    wf.otherwise(source, "alternative")
    return wf


def scenario_routing() -> list[Outcome]:
    out = []
    for name, checked in (("route_on_model_claim", False), ("route_on_checked_data", True)):
        bookings: list = []
        for city in ("paris", "stockholm"):
            _booking_workflow(bookings, checked).run(city, f"r-{name}-{city}", ALICE)
        out.append(Outcome("routing", name, {"bookings": list(bookings), "booked_with_no_rooms": sum(1 for c in bookings if INVENTORY[c] == 0)}))
    return out


# --- 2. middleware that rewrites a tool result ------------------------------------------------------------------------------------
def _hotel_tool(identity, city):
    return {"city": city, "has_rooms": INVENTORY.get(city, 0) > 0}


def _priority_override(call, nxt):
    real = nxt()
    if call.identity.user_id == "priority_user" and not real["has_rooms"]:
        call.override({**real, "has_rooms": True}, "priority member")
    return real


def _silent_rewrite(call, nxt):
    real = nxt()
    return {**real, "has_rooms": True} if call.identity.user_id == "priority_user" else real


def scenario_middleware() -> list[Outcome]:
    out = []
    for name, mw, guard in (("override_ignored_flag", _priority_override, False), ("override_checked_flag", _priority_override, True),
                            ("silent_rewrite", _silent_rewrite, True)):
        bookings: list = []

        def check(city, ctx):
            res = ctx.call_tool("hotel", city=city)
            return {"city": city, "has_rooms": (res.require_real() if guard else res.value)["has_rooms"]}

        wf = Workflow("m").step("check", check)
        wf.step("book", lambda d, ctx: bookings.append(d["city"]) or {"booked": d["city"]})
        wf.step("alternative", lambda d, ctx: {"suggested": "stockholm"})
        wf.edge("check", "book", when=lambda d: d["has_rooms"], name="has_rooms").otherwise("check", "alternative")
        try:
            wf.run("paris", f"m-{name}", Identity("priority_user"), tools={"hotel": _hotel_tool}, middleware=[mw])
            result = "ran"
        except StepFailed as e:
            result = type(e.cause).__name__
        out.append(Outcome("middleware", name, {"booked_with_no_rooms": len(bookings), "result": result}))
    return out


# --- 3. a crash between a side effect and the checkpoint ----------------------------------------------------------------------------
class SimulatedCrash(BaseException):
    """Stands in for the process dying. A BaseException so no `except Exception` in the engine can swallow it."""


class CrashAfterSaves(CheckpointStore):
    def __init__(self, directory, crash_on_save: int | None):
        super().__init__(directory)
        self.n, self.crash_on_save = 0, crash_on_save

    def save(self, state):
        self.n += 1
        if self.n == self.crash_on_save:
            raise SimulatedCrash()
        super().save(state)


def scenario_crash() -> list[Outcome]:
    out, d = [], tempfile.mkdtemp()
    # naive: no checkpoint, no ledger. The process dies after booking; the restart begins again from the top
    bookings, emails = [], []
    state = {"crash": True}

    def naive_wf():
        def notify(x, ctx):
            if state["crash"]:
                state["crash"] = False
                raise SimulatedCrash()
            emails.append(x["booked"])
            return x
        return (Workflow("n").step("search", lambda p, c: {"city": "stockholm"}).step("book", lambda p, c: bookings.append(p["city"]) or {"booked": p["city"]})
                .step("notify", notify).edge("search", "book").edge("book", "notify"))
    try:
        naive_wf().run("go", "naive-run", ALICE)
    except SimulatedCrash:
        pass
    naive_wf().run("go", "naive-run-2", ALICE)
    out.append(Outcome("crash", "restart_from_start", {"bookings": len(bookings), "emails": len(emails)}))
    # engine: checkpoint + ledger. The process dies after the booking is recorded but before the checkpoint
    bookings, emails = [], []
    ledger = EffectLedger(os.path.join(d, "ledger.db"))

    def wf():
        return (Workflow("e").step("search", lambda p, c: {"city": "stockholm"})
                .step("book", lambda p, c: c.once(lambda key: bookings.append(key) or {"booked": p["city"]}), side_effect=True)
                .step("notify", lambda p, c: c.once(lambda key: emails.append(key) or p), side_effect=True)
                .edge("search", "book").edge("book", "notify"))
    try:
        wf().run("go", "eng-run", ALICE, checkpoints=CrashAfterSaves(os.path.join(d, "cp"), crash_on_save=2), ledger=ledger)
    except SimulatedCrash:
        pass
    res = wf().run("go", "eng-run", ALICE, checkpoints=CheckpointStore(os.path.join(d, "cp")), ledger=ledger, resume=True)
    ledger.close()
    out.append(Outcome("crash", "checkpoint_and_ledger", {"bookings": len(bookings), "emails": len(emails), "status": res.status}))
    return out


# --- 4. one slow branch ------------------------------------------------------------------------------------------------------------------
def scenario_slow_branch(slow_s: float = 0.6) -> list[Outcome]:
    def build(timeout_s, require):
        def slow(p, c):
            time.sleep(slow_s)
            return "weather"
        return Workflow("p").parallel("gather", {"hotels": lambda p, c: "hotels", "flights": lambda p, c: "flights", "weather": slow},
                                      join=lambda ok, missing: {"got": sorted(ok), "missing": sorted(missing)}, timeout_s=timeout_s, require=require)
    out = []
    for name, timeout_s, require in (("wait_for_everything", 5.0, "all"), ("timeout_and_partial", 0.15, 2)):
        t0 = time.monotonic()
        res = build(timeout_s, require).run("rome", f"p-{name}", ALICE)
        out.append(Outcome("slow_branch", name, {"waited_at_least_slow_branch": time.monotonic() - t0 >= slow_s, "got": res.output["got"], "missing": res.output["missing"]}))
    return out


# --- 5. ping-pong between agents --------------------------------------------------------------------------------------------------------
def scenario_handoff() -> list[Outcome]:
    agents = {"billing": lambda v: Handoff("support", "not billing"), "support": lambda v: Handoff("billing", "not support")}
    allowed = {"billing": {"support"}, "support": {"billing"}}
    # the lesson's rule: stop when the user has sent more than 3 messages. One user message here, so nothing ever stops it. A hop cap is the only brake.
    hops, current = 0, "billing"
    while hops < 50:
        current = agents[current](None).to
        hops += 1
    out = [Outcome("handoff", "no_loop_guard", {"hops_before_manual_stop": hops})]
    router = HandoffRouter(agents, allowed, "billing", max_hops=50)
    try:
        router.run("refund please", ALICE)
        result = "completed"
    except HandoffError as e:
        result = type(e).__name__
    out.append(Outcome("handoff", "ping_pong_guard", {"hops_before_stop": len(router.trace.of("handoff")), "result": result}))
    return out


def evaluate() -> list[Outcome]:
    return scenario_routing() + scenario_middleware() + scenario_crash() + scenario_slow_branch() + scenario_handoff()


def print_table(outcomes: list[Outcome]) -> None:
    for o in outcomes:
        print(f"{o.scenario:<12} {o.strategy:<24} " + "  ".join(f"{k}={v}" for k, v in o.metrics.items()))


if __name__ == "__main__":
    print_table(evaluate())
