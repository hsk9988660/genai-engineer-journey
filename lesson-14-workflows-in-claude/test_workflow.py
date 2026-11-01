import json
import time

import pytest

from checkpoint import CheckpointStore, EffectLedger
from middleware import Identity
from runtrace import Trace
from workflow import (AmbiguousRoute, BranchesFailed, GraphError, Limits, LimitExceeded, LoopLimit, NoRoute, RouteError, StepFailed, Workflow)

ME = Identity("alice")


def inc(p, ctx):
    return p + 1


def linear(n=3):
    wf = Workflow("lin")
    for i in range(n):
        wf.step(f"s{i}", inc)
        if i:
            wf.edge(f"s{i - 1}", f"s{i}")
    return wf


# --- running ----------------------------------------------------------------------------------------------------------------
def test_a_linear_workflow_runs_every_step_in_order_and_returns_the_last_output():
    r = linear().run(0, "r1", ME)
    assert (r.status, r.output, r.path, r.steps) == ("completed", 3, ["s0", "s1", "s2"], 3)


def test_the_first_step_added_is_the_start_unless_changed():
    wf = Workflow().step("b", inc).step("a", inc)
    assert wf.start == "b" and wf.start_at("a").start == "a"


def test_steps_receive_the_identity_the_run_was_started_with_and_not_a_global():
    seen = []
    Workflow().step("s", lambda p, ctx: seen.append((ctx.identity.user_id, ctx.run_id, ctx.node, ctx.visit)) or p).run(1, "run-9", Identity("bob"))
    assert seen == [("bob", "run-9", "s", 1)]


def test_a_payload_that_is_not_json_is_refused_up_front_and_so_is_a_non_json_output():
    with pytest.raises(TypeError):
        linear().run({1, 2}, "r2", ME)
    wf = Workflow().step("s", lambda p, c: {1, 2})
    with pytest.raises(StepFailed) as e:
        wf.run(0, "r3", ME)
    assert isinstance(e.value.cause, TypeError)


# --- validating the graph ---------------------------------------------------------------------------------------------------
def test_an_empty_workflow_a_duplicate_id_and_an_empty_id_are_rejected():
    with pytest.raises(GraphError):
        Workflow().validate()
    with pytest.raises(GraphError):
        Workflow().step("a", inc).step("a", inc)
    with pytest.raises(GraphError):
        Workflow().step("", inc)
    with pytest.raises(GraphError):
        Workflow().step("a", inc, kind="robot")


def test_an_edge_to_a_step_that_does_not_exist_is_rejected():
    with pytest.raises(GraphError, match="does not exist"):
        Workflow().step("a", inc).edge("a", "ghost").validate()


def test_a_step_that_can_never_run_is_rejected_by_name():
    with pytest.raises(GraphError, match=r"\['orphan'\]"):
        Workflow().step("a", inc).step("orphan", inc).validate()


def test_mixing_a_direct_edge_with_conditions_or_having_two_direct_edges_is_rejected():
    base = lambda: Workflow().step("a", inc).step("b", inc).step("c", inc)
    with pytest.raises(GraphError, match="mixes"):
        base().edge("a", "b").edge("a", "c", when=lambda p: True).validate()
    with pytest.raises(GraphError, match="mixes"):
        base().edge("a", "b").edge("a", "c").validate()
    with pytest.raises(GraphError, match="mixes"):
        base().edge("a", "b").otherwise("a", "c").validate()


def test_two_otherwise_edges_are_rejected():
    with pytest.raises(GraphError, match="more than one"):
        Workflow().step("a", inc).step("b", inc).step("c", inc).otherwise("a", "b").otherwise("a", "c").validate()


def test_a_model_step_cannot_choose_the_route_unless_the_edge_says_it_trusts_the_output():
    def build(trust):
        return Workflow().step("ask", inc, kind="model").step("x", inc).step("y", inc).edge("ask", "x", when=lambda p: p > 0, trust_model_output=trust).otherwise("ask", "y")
    with pytest.raises(GraphError, match="model step"):
        build(False).validate()
    build(True).validate()


def test_a_model_step_can_feed_a_code_step_that_routes():
    wf = Workflow().step("ask", inc, kind="model").step("check", inc).step("x", inc).step("y", inc)
    wf.edge("ask", "check").edge("check", "x", when=lambda p: p > 100).otherwise("check", "y")
    wf.validate()


def test_cycles_are_rejected_unless_asked_for():
    wf = Workflow().step("a", inc).step("b", inc).edge("a", "b").edge("b", "a")
    with pytest.raises(GraphError, match="cycle"):
        wf.validate()
    wf.validate(allow_cycles=True)


# --- routing ----------------------------------------------------------------------------------------------------------------
def branching(pred_a, pred_b=None, otherwise=True):
    wf = Workflow().step("start", lambda p, c: p).step("a", lambda p, c: "A")
    wf.edge("start", "a", when=pred_a, name="to_a")
    if pred_b:
        wf.step("b", lambda p, c: "B").edge("start", "b", when=pred_b, name="to_b")
    if otherwise:
        wf.step("z", lambda p, c: "Z").otherwise("start", "z")
    return wf


def test_the_matching_condition_decides_and_otherwise_catches_the_rest():
    wf = branching(lambda p: p == 1, lambda p: p == 2)
    assert [wf.run(v, f"r{v}", ME).output for v in (1, 2, 3)] == ["A", "B", "Z"]


def test_two_matching_conditions_are_an_error_not_a_silent_first_match():
    with pytest.raises(AmbiguousRoute, match=r"to_a.*to_b"):
        branching(lambda p: True, lambda p: True).run(1, "r", ME)


def test_no_matching_condition_and_no_otherwise_is_an_error():
    with pytest.raises(NoRoute):
        branching(lambda p: False, otherwise=False).run(1, "r", ME)


def test_a_condition_that_raises_is_reported_as_a_route_error():
    with pytest.raises(RouteError, match="to_a"):
        branching(lambda p: p["missing"]).run(1, "r", ME)


# --- bounds -----------------------------------------------------------------------------------------------------------------
def test_the_step_cap_stops_a_run():
    with pytest.raises(LimitExceeded, match="step cap"):
        linear(5).run(0, "r", ME, limits=Limits(max_steps=3))


def test_a_loop_is_stopped_by_the_per_step_visit_cap():
    wf = Workflow().step("a", inc, max_visits=2).step("b", inc, max_visits=2).edge("a", "b").edge("b", "a")
    with pytest.raises(LoopLimit, match="'a'"):
        wf.run(0, "r", ME, allow_cycles=True)


def test_the_token_cap_stops_a_run_when_a_step_charges_past_it():
    wf = Workflow().step("a", lambda p, c: c.charge(60) or p).step("b", lambda p, c: c.charge(60) or p).edge("a", "b")
    with pytest.raises(LimitExceeded, match="token cap"):
        wf.run(0, "r", ME, limits=Limits(max_tokens=100))
    assert wf.run(0, "r2", ME, limits=Limits(max_tokens=200)).tokens == 120


def test_a_step_that_raises_is_wrapped_with_its_name_and_cause():
    wf = Workflow().step("boom", lambda p, c: 1 / 0)
    with pytest.raises(StepFailed, match="boom") as e:
        wf.run(0, "r", ME)
    assert isinstance(e.value.cause, ZeroDivisionError) and e.value.node == "boom"


def test_a_side_effect_step_with_no_ledger_is_refused_before_anything_runs():
    ran = []
    wf = Workflow().step("a", lambda p, c: ran.append(1) or p).step("pay", inc, side_effect=True).edge("a", "pay")
    with pytest.raises(GraphError, match="EffectLedger"):
        wf.run(0, "r", ME)
    assert ran == []


def test_ctx_once_without_a_ledger_is_refused():
    with pytest.raises(GraphError):
        Workflow().step("a", lambda p, c: c.once(lambda k: 1)).run(0, "r", ME)


# --- parallel steps ---------------------------------------------------------------------------------------------------------
def par(branches, **kw):
    return Workflow().parallel("p", branches, join=lambda ok, missing: {"ok": dict(sorted(ok.items())), "missing": missing}, **kw)


def test_parallel_branches_all_run_and_the_join_sees_every_result():
    r = par({"a": lambda p, c: 1, "b": lambda p, c: 2}).run(0, "r", ME)
    assert r.output == {"ok": {"a": 1, "b": 2}, "missing": {}}


def test_each_branch_gets_its_own_effect_key_so_two_branches_do_not_share_a_ledger_slot(tmp_path):
    ledger = EffectLedger(tmp_path / "l.db")
    wf = par({"a": lambda p, c: c.once(lambda k: k), "b": lambda p, c: c.once(lambda k: k)}, side_effect=True)
    r = wf.run(0, "r", ME, ledger=ledger)
    assert r.output["ok"] == {"a": "r:p.a:1", "b": "r:p.b:1"}
    ledger.close()


def test_a_slow_branch_is_cut_off_at_the_timeout_and_reported_as_missing_when_partial_results_are_allowed():
    t0 = time.monotonic()
    r = par({"fast": lambda p, c: 1, "slow": lambda p, c: time.sleep(0.5) or 2}, timeout_s=0.1, require=1).run(0, "r", ME)
    assert r.output == {"ok": {"fast": 1}, "missing": {"slow": "timeout"}} and time.monotonic() - t0 < 0.4


def test_requiring_all_branches_fails_the_step_when_one_times_out_or_raises():
    with pytest.raises(BranchesFailed, match="timeout"):
        par({"fast": lambda p, c: 1, "slow": lambda p, c: time.sleep(0.4)}, timeout_s=0.05).run(0, "r", ME)
    with pytest.raises(BranchesFailed, match="ZeroDivisionError"):
        par({"ok": lambda p, c: 1, "bad": lambda p, c: 1 / 0}).run(0, "r2", ME)


def test_a_failed_branch_is_recorded_when_enough_others_succeed():
    r = par({"ok": lambda p, c: 1, "bad": lambda p, c: 1 / 0}, require="any").run(0, "r", ME)
    assert r.output["ok"] == {"ok": 1} and r.output["missing"]["bad"].startswith("ZeroDivisionError")


def test_the_numeric_requirement_counts_successes():
    branches = {"a": lambda p, c: 1, "b": lambda p, c: 1 / 0, "c": lambda p, c: 1 / 0}
    with pytest.raises(BranchesFailed):
        par(branches, require=2).run(0, "r", ME)
    assert par(branches, require=1).run(0, "r2", ME).output["ok"] == {"a": 1}


def test_a_parallel_step_needs_branches():
    with pytest.raises(GraphError):
        Workflow().parallel("p", {}, join=lambda a, b: 0)


# --- the trace --------------------------------------------------------------------------------------------------------------
def test_the_trace_records_steps_and_hashes_but_never_payload_content():
    tr = Trace(clock=lambda: 1.0)
    Workflow().step("a", lambda p, c: {"secret": "SECRET-PAYLOAD"}).step("b", lambda p, c: p).edge("a", "b").run({"in": "SECRET-INPUT"}, "r", ME, trace=tr)
    blob = json.dumps(tr.events)
    assert "SECRET" not in blob
    assert tr.types() == ["run_started", "step_started", "step_completed", "step_started", "step_completed", "run_completed"]
    assert all(len(e["output_hash"]) == 16 for e in tr.of("step_completed"))


def test_a_failed_step_is_traced_with_the_error_type_only():
    tr = Trace()
    with pytest.raises(StepFailed):
        Workflow().step("a", lambda p, c: 1 / 0).run(0, "r", ME, trace=tr)
    assert tr.of("step_failed")[0]["error"] == "ZeroDivisionError" and "division" not in json.dumps(tr.events)
