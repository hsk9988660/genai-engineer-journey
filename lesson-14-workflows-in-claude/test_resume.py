import json
import os

import pytest

from checkpoint import CheckpointStore, EffectLedger, check_run_id
from middleware import Identity
from runtrace import Trace
from workflow import (Approval, ApprovalError, CheckpointMismatch, GraphError, RunExists, StaleApproval, StepFailed, UnknownApproval, Workflow)

ME = Identity("alice")


class Crash(BaseException):
    """A stand-in for the process dying: BaseException, so the engine's `except Exception` cannot swallow it."""


class CrashOnSave(CheckpointStore):
    def __init__(self, directory, n):
        super().__init__(directory)
        self.count, self.n = 0, n

    def save(self, state):
        self.count += 1
        if self.count == self.n:
            raise Crash()
        super().save(state)


@pytest.fixture
def cp(tmp_path):
    return CheckpointStore(tmp_path / "cp")


@pytest.fixture
def ledger(tmp_path):
    l = EffectLedger(tmp_path / "ledger.db")
    yield l
    l.close()


def counting(log):
    wf = Workflow("w")
    for name in ("a", "b", "c"):
        wf.step(name, lambda p, ctx, name=name: log.append(name) or p + [name])
    return wf.edge("a", "b").edge("b", "c")


# --- checkpoints -----------------------------------------------------------------------------------------------------------------
def test_state_is_saved_after_every_step(cp):
    counting([]).run([], "r1", ME, checkpoints=cp)
    s = cp.load("r1")
    assert s["status"] == "completed" and s["path"] == ["a", "b", "c"] and s["payload"] == ["a", "b", "c"] and s["next"] is None


def test_a_crash_then_resume_continues_from_the_last_saved_step_without_redoing_earlier_ones(tmp_path):
    log = []
    with pytest.raises(Crash):
        counting(log).run([], "r1", ME, checkpoints=CrashOnSave(tmp_path / "cp", 3))  # dies while saving after step 'c'... saves: a, b, then c (3rd)
    assert log == ["a", "b", "c"]
    log.clear()
    r = counting(log).run([], "r1", ME, checkpoints=CheckpointStore(tmp_path / "cp"), resume=True)
    assert log == ["c"] and r.output == ["a", "b", "c"] and r.status == "completed"


def test_resuming_a_completed_run_returns_the_stored_result_and_runs_nothing(cp):
    log = []
    counting(log).run([], "r1", ME, checkpoints=cp)
    log.clear()
    r = counting(log).run([], "r1", ME, checkpoints=cp, resume=True)
    assert log == [] and r.output == ["a", "b", "c"]


def test_starting_a_run_whose_id_already_has_a_checkpoint_is_refused_unless_resuming(cp):
    counting([]).run([], "r1", ME, checkpoints=cp)
    with pytest.raises(RunExists):
        counting([]).run([], "r1", ME, checkpoints=cp)


def test_resume_with_no_checkpoint_just_starts(cp):
    log = []
    assert counting(log).run([], "fresh", ME, checkpoints=cp, resume=True).status == "completed" and log == ["a", "b", "c"]


def test_a_checkpoint_from_a_different_workflow_is_refused(cp):
    counting([]).run([], "r1", ME, checkpoints=cp)
    other = Workflow("w").step("a", lambda p, c: p).step("x", lambda p, c: p).edge("a", "x")
    with pytest.raises(CheckpointMismatch):
        other.run([], "r1", ME, checkpoints=cp, resume=True)


def test_a_failed_step_leaves_a_resumable_checkpoint_and_the_retry_keeps_the_same_visit_number(cp, ledger):
    keys, fail = [], [True]

    def flaky(p, ctx):
        keys.append(ctx.effect_key)
        if fail[0]:
            raise RuntimeError("down")
        return p

    wf = Workflow("w").step("a", lambda p, c: p).step("b", flaky).edge("a", "b")
    with pytest.raises(StepFailed):
        wf.run(0, "r1", ME, checkpoints=cp)
    assert cp.load("r1")["status"] == "failed" and cp.load("r1")["next"] == "b"
    fail[0] = False
    assert wf.run(0, "r1", ME, checkpoints=cp, resume=True).status == "completed"
    assert keys == ["r1:b:1", "r1:b:1"]


def test_the_step_and_token_counts_survive_a_resume(tmp_path):
    def build():
        return Workflow("w").step("a", lambda p, c: c.charge(10) or p).step("b", lambda p, c: c.charge(5) or p).edge("a", "b")
    with pytest.raises(Crash):
        build().run(0, "r1", ME, checkpoints=CrashOnSave(tmp_path / "cp", 2))
    r = build().run(0, "r1", ME, checkpoints=CheckpointStore(tmp_path / "cp"), resume=True)
    assert r.tokens == 15 and r.steps == 2


# --- the files themselves ----------------------------------------------------------------------------------------------------------
def test_run_ids_that_could_escape_the_checkpoint_directory_are_rejected(cp):
    for bad in ("../x", "a/b", "a\\b", "", "x" * 65, "has space", "dot.dot"):
        with pytest.raises(ValueError):
            cp.load(bad)
    assert check_run_id("ok_run-1") == "ok_run-1"


def test_a_failed_save_leaves_the_previous_complete_file_and_no_temp_files(cp):
    cp.save({"run_id": "r1", "v": 1})
    with pytest.raises(TypeError):
        cp.save({"run_id": "r1", "v": {1, 2}})  # not JSON: fails while writing
    assert cp.load("r1") == {"run_id": "r1", "v": 1}
    assert [f for f in os.listdir(cp.dir) if f.endswith(".tmp")] == []


# --- the effect ledger -------------------------------------------------------------------------------------------------------------
def test_a_side_effect_runs_once_per_key_and_later_calls_get_the_recorded_result(ledger):
    calls = []
    assert ledger.once("k", lambda key: calls.append(key) or {"n": 1}) == {"n": 1}
    assert ledger.once("k", lambda key: calls.append(key) or {"n": 2}) == {"n": 1} and calls == ["k"]
    assert ledger.once("other", lambda key: {"n": 3}) == {"n": 3} and ledger.keys() == ["k", "other"]


def test_the_ledger_survives_closing_and_reopening(tmp_path):
    l = EffectLedger(tmp_path / "l.db")
    l.once("k", lambda key: "first")
    l.close()
    l2 = EffectLedger(tmp_path / "l.db")
    assert l2.once("k", lambda key: "second") == "first"
    l2.close()


def test_a_crash_after_the_effect_but_before_the_checkpoint_does_not_repeat_the_effect(tmp_path, ledger):
    done = []

    def build():
        return (Workflow("w").step("a", lambda p, c: p)
                .step("pay", lambda p, c: c.once(lambda key: done.append(key) or {"paid": 1}), side_effect=True)
                .step("mail", lambda p, c: c.once(lambda key: done.append(key) or p), side_effect=True).edge("a", "pay").edge("pay", "mail"))
    with pytest.raises(Crash):
        build().run(0, "r1", ME, checkpoints=CrashOnSave(tmp_path / "cp", 2), ledger=ledger)  # crashes saving after 'pay', whose effect is already recorded
    assert done == ["r1:pay:1"]
    r = build().run(0, "r1", ME, checkpoints=CheckpointStore(tmp_path / "cp"), ledger=ledger, resume=True)
    assert done == ["r1:pay:1", "r1:mail:1"] and r.status == "completed"


def test_two_different_runs_do_not_share_effects(ledger):
    done = []
    wf = Workflow("w").step("pay", lambda p, c: c.once(lambda key: done.append(key) or 1), side_effect=True)
    wf.run(0, "r1", ME, ledger=ledger)
    wf.run(0, "r2", ME, ledger=ledger)
    assert done == ["r1:pay:1", "r2:pay:1"]


# --- approvals --------------------------------------------------------------------------------------------------------------------
def approval_workflow(log):
    def gate(p, ctx):
        decision = ctx.request_approval({"action": "refund", "amount": p["amount"]})
        log.append(decision)
        return {**p, "decision": decision}
    return Workflow("w").step("prepare", lambda p, c: p).step("gate", gate).step("pay", lambda p, c: {**p, "paid": p["decision"] == "approve"}) \
        .edge("prepare", "gate").edge("gate", "pay")


def pause(cp, run_id="r1", amount=50):
    log = []
    r = approval_workflow(log).run({"amount": amount}, run_id, ME, checkpoints=cp)
    return r, log


def test_a_step_that_needs_approval_pauses_the_run_and_saves_what_is_being_asked(cp):
    r, log = pause(cp)
    assert r.status == "paused" and r.pending["id"] == "gate:1" and r.pending["summary"] == {"action": "refund", "amount": 50} and log == []
    s = cp.load("r1")
    assert s["status"] == "paused" and s["next"] == "gate" and s["pending"]["summary_hash"] == r.pending["summary_hash"]


def test_resuming_with_no_answer_pauses_again_with_the_same_request(cp):
    first, _ = pause(cp)
    again = approval_workflow([]).run({"amount": 50}, "r1", ME, checkpoints=cp, resume=True)
    assert again.status == "paused" and again.pending["summary_hash"] == first.pending["summary_hash"]


def test_an_approval_for_exactly_what_was_shown_lets_the_run_finish(cp):
    r, _ = pause(cp)
    log = []
    done = approval_workflow(log).run({"amount": 50}, "r1", ME, checkpoints=cp, resume=True,
                                      responses={"gate:1": Approval("approve", r.pending["summary_hash"], "manager-7")})
    assert done.status == "completed" and done.output["paid"] is True and log == ["approve"]
    assert cp.load("r1")["approvals_used"] == ["gate:1"] and cp.load("r1")["pending"] is None


def test_a_rejection_reaches_the_step_so_it_can_refuse_to_pay(cp):
    r, _ = pause(cp)
    done = approval_workflow([]).run({"amount": 50}, "r1", ME, checkpoints=cp, resume=True,
                                     responses={"gate:1": Approval("reject", r.pending["summary_hash"])})
    assert done.output["paid"] is False


def test_an_approval_of_something_else_is_refused_as_stale(cp):
    pause(cp)
    with pytest.raises(StaleApproval):
        approval_workflow([]).run({"amount": 50}, "r1", ME, checkpoints=cp, resume=True, responses={"gate:1": Approval("approve", "0" * 64)})


def test_a_payload_passed_on_resume_is_ignored_so_the_approved_amount_is_the_one_paid(cp):
    r, _ = pause(cp, amount=50)
    done = approval_workflow([]).run({"amount": 5000}, "r1", ME, checkpoints=cp, resume=True, responses={"gate:1": Approval("approve", r.pending["summary_hash"])})
    assert done.output["amount"] == 50


def test_if_the_step_now_asks_for_something_different_the_old_approval_is_refused(cp):
    r, _ = pause(cp, amount=50)

    def changed_gate(p, ctx):
        return {**p, "decision": ctx.request_approval({"action": "refund", "amount": p["amount"] * 100})}  # the code now asks about a bigger amount
    wf = Workflow("w").step("prepare", lambda p, c: p).step("gate", changed_gate).step("pay", lambda p, c: p).edge("prepare", "gate").edge("gate", "pay")
    with pytest.raises(StaleApproval):
        wf.run({"amount": 50}, "r1", ME, checkpoints=cp, resume=True, responses={"gate:1": Approval("approve", r.pending["summary_hash"])})


def test_an_approval_issued_for_one_run_cannot_be_used_in_another_run_with_the_same_request(cp):
    a, _ = pause(cp, "run-a")
    pause(cp, "run-b")
    with pytest.raises(StaleApproval):
        approval_workflow([]).run({"amount": 50}, "run-b", ME, checkpoints=cp, resume=True, responses={"gate:1": Approval("approve", a.pending["summary_hash"])})


def test_a_response_for_a_request_that_is_not_pending_is_refused(cp):
    r, _ = pause(cp)
    with pytest.raises(UnknownApproval):
        approval_workflow([]).run({"amount": 50}, "r1", ME, checkpoints=cp, resume=True, responses={"pay:1": Approval("approve", r.pending["summary_hash"])})


def test_a_response_when_nothing_is_pending_is_refused(cp):
    counting([]).run([], "r1", ME, checkpoints=cp)
    wf = counting([])
    cp.save({**cp.load("r1"), "status": "failed", "next": "b"})
    with pytest.raises(UnknownApproval):
        wf.run([], "r1", ME, checkpoints=cp, resume=True, responses={"gate:1": Approval("approve", "x")})


def test_a_decision_must_be_approve_or_reject(cp):
    r, _ = pause(cp)
    with pytest.raises(ApprovalError, match="approve"):
        approval_workflow([]).run({"amount": 50}, "r1", ME, checkpoints=cp, resume=True, responses={"gate:1": Approval("yes please", r.pending["summary_hash"])})


def test_an_approval_that_was_already_used_cannot_be_used_again(cp):
    r, _ = pause(cp)
    resp = {"gate:1": Approval("approve", r.pending["summary_hash"])}
    approval_workflow([]).run({"amount": 50}, "r1", ME, checkpoints=cp, resume=True, responses=resp)
    s = cp.load("r1")
    # force the run back to the gate as if the checkpoint of the following step had been lost, with the response consumed
    cp.save({**s, "status": "paused", "next": "gate", "payload": {"amount": 50}, "pending": {"id": "gate:1", "node": "gate", "summary": {}, "summary_hash": r.pending["summary_hash"]}, "visits": {"prepare": 1}})
    with pytest.raises(ApprovalError, match="already used"):
        approval_workflow([]).run({"amount": 50}, "r1", ME, checkpoints=cp, resume=True, responses=resp)


def test_the_trace_records_who_approved_but_not_the_amount(cp):
    r, _ = pause(cp)
    tr = Trace()
    approval_workflow([]).run({"amount": 50}, "r1", ME, checkpoints=cp, resume=True, trace=tr, responses={"gate:1": Approval("approve", r.pending["summary_hash"], "manager-7")})
    used = tr.of("approval_used")[0]
    assert used["approver"] == "manager-7" and used["decision"] == "approve" and "50" not in json.dumps(tr.events)
