import os
import threading
from types import SimpleNamespace as NS

import pytest

from approval import ApprovalLedger, ApprovalRefused, GuardedExecutor, action_digest, check_intents_against_approvals, issue_approval, verify_approval
from canon import CanonError
from fakes import FakeClient, text_reply
from gateway import ApprovalBroker, run_turn, wrap_impls
from harness import NOW, SimulatedCrash, setup
from keys import KeyRegistry, Signer
from receipt import APPROVAL, INTENT, ALG, receipt_hash, sign
from recorder import Indeterminate, read_chain
from verify import ChainVerifier

ARGS = {"order": "A1002", "amount": 4000}


def approver_env():
    approver = Signer.generate("approver")
    reg = KeyRegistry()
    reg.add_signer(approver)
    return approver, reg


def make_approval(approver, args=ARGS, **over):
    kw = dict(approver_id="supervisor-7", tool="refund", args=args, policy_version=1, expires_at=NOW + 600, now=NOW)
    kw.update(over)
    return issue_approval(approver, **kw)


def verify(ap, reg, **over):
    kw = dict(now=NOW, current_policy_version=1, expected=("refund", ARGS))
    kw.update(over)
    return verify_approval(ap, reg, **kw)


def refusal(fn):
    with pytest.raises(ApprovalRefused) as e:
        fn()
    return e.value.reason


# --- the action digest ------------------------------------------------------------------------------------------------------------------
def test_the_action_digest_ignores_argument_order_and_changes_with_the_tool_or_any_argument():
    assert action_digest("refund", {"a": 1, "b": 2}) == action_digest("refund", {"b": 2, "a": 1})
    assert action_digest("refund", ARGS) != action_digest("refund", {**ARGS, "amount": 4001}) != action_digest("other", ARGS)
    assert action_digest("refund", ARGS).startswith("sha256:")


def test_an_amount_given_as_a_float_cannot_be_approved():
    with pytest.raises(CanonError):
        action_digest("refund", {"amount": 40.0})


# --- issuing and verifying an approval ----------------------------------------------------------------------------------------------
def test_an_approval_names_the_approver_the_exact_action_the_policy_and_an_expiry():
    approver, reg = approver_env()
    ap = make_approval(approver)
    assert ap["type"] == APPROVAL and ap["approver_id"] == "supervisor-7" and ap["tool"] == "refund" and ap["args"] == ARGS and ap["policy_version"] == 1 and ap["expires_at"] == NOW + 600
    assert ap["action_digest"] == action_digest("refund", ARGS) and ap["kid"] == approver.kid and len(ap["nonce"]) == 16
    assert make_approval(approver)["nonce"] != make_approval(approver)["nonce"]


def test_only_an_approver_key_can_issue_an_approval():
    with pytest.raises(ValueError):
        make_approval(Signer.generate("agent"))


def test_a_good_approval_verifies_for_the_action_it_covers():
    approver, reg = approver_env()
    ap = make_approval(approver)
    assert verify(ap, reg) is ap


@pytest.mark.parametrize("bad", ["text", None, {}, {"type": APPROVAL}, 5])
def test_malformed_approvals_are_refused(bad):
    approver, reg = approver_env()
    assert refusal(lambda: verify(bad, reg)) == "malformed"


def test_an_approval_with_a_changed_field_is_a_bad_signature():
    approver, reg = approver_env()
    ap = make_approval(approver)
    for field, value in (("args", {**ARGS, "amount": 400000}), ("expires_at", NOW + 10 ** 9), ("policy_version", 9), ("approver_id", "someone-else")):
        assert refusal(lambda: verify({**ap, field: value}, reg, expected=None, current_policy_version=9)) == "bad_signature"


def test_an_approval_from_an_unregistered_approver_key_is_refused():
    approver, reg = approver_env()
    assert refusal(lambda: verify(make_approval(Signer.generate("approver")), reg)) == "unknown_key"


def test_an_approver_key_registered_in_the_wrong_role_is_refused():
    approver, _ = approver_env()
    wrong = KeyRegistry()
    wrong.add(approver.public_bytes, "agent")
    assert refusal(lambda: verify(make_approval(approver), wrong)) == "wrong_role"


def test_a_key_rotated_out_since_the_approval_was_given_is_stale_authority():
    approver, reg = approver_env()
    ap = make_approval(approver)
    reg.revoke(approver.kid, NOW)
    assert refusal(lambda: verify(ap, reg)) == "key_revoked"
    approver2 = Signer.generate("approver")
    reg2 = KeyRegistry()
    reg2.add_signer(approver2, valid_from=0, valid_to=NOW + 1)
    assert refusal(lambda: verify(make_approval(approver2), reg2, now=NOW + 5)) == "key_expired"


def test_an_approval_is_refused_at_its_expiry_and_not_before():
    approver, reg = approver_env()
    ap = make_approval(approver, expires_at=NOW + 60)
    assert verify(ap, reg, now=NOW + 59)
    assert refusal(lambda: verify(ap, reg, now=NOW + 60)) == "expired" and refusal(lambda: verify(ap, reg, now=NOW + 600)) == "expired"


def test_an_approval_dated_in_the_future_is_refused():
    approver, reg = approver_env()
    assert refusal(lambda: verify(make_approval(approver, now=NOW + 10_000, expires_at=NOW + 20_000), reg)) == "issued_in_future"


def test_an_approval_given_under_another_policy_version_is_stale():
    approver, reg = approver_env()
    assert refusal(lambda: verify(make_approval(approver, policy_version=1), reg, current_policy_version=2)) == "stale_policy"


def test_a_real_approval_for_a_different_action_is_refused_as_digest_substitution():
    approver, reg = approver_env()
    assert refusal(lambda: verify(make_approval(approver), reg, expected=("refund", {**ARGS, "amount": 400000}))) == "action_mismatch"
    assert refusal(lambda: verify(make_approval(approver), reg, expected=("delete_account", ARGS))) == "action_mismatch"
    assert refusal(lambda: verify(make_approval(approver), reg, expected=("refund", {"amount": 1.5}))) == "action_mismatch"


def test_an_approval_whose_digest_does_not_match_its_own_action_is_refused():
    approver, reg = approver_env()
    forged = sign({"type": APPROVAL, "alg": ALG, "kid": approver.kid, "iat": NOW, "approver_id": "s", "tool": "refund", "args": ARGS, "action_digest": "sha256:" + "0" * 64,
                   "policy_version": 1, "expires_at": NOW + 600, "nonce": "n"}, approver)
    assert refusal(lambda: verify(forged, reg)) == "digest_mismatch"


# --- the ledger -------------------------------------------------------------------------------------------------------------------------
def test_an_approval_can_be_reserved_once_and_the_status_follows_its_life():
    led = ApprovalLedger()
    assert led.status("h") is None and led.reserve("h", "req1") is True and led.status("h") == "reserved" and led.reserve("h", "req2") is False
    led.complete("h", "completed")
    assert led.status("h") == "completed" and led.reserve("h", "req3") is False
    led.complete("h", "failed")  # only a RESERVED approval can be completed: a finished one does not change
    assert led.status("h") == "completed"


def test_only_known_final_statuses_are_accepted():
    led = ApprovalLedger()
    led.reserve("h", "r")
    with pytest.raises(ValueError):
        led.complete("h", "banana")


def test_unfinished_reservations_become_indeterminate_at_startup_and_finished_ones_are_left_alone():
    led = ApprovalLedger()
    led.reserve("a", "r")
    led.reserve("b", "r")
    led.complete("b", "completed")
    assert led.mark_unfinished_indeterminate() == 1 and led.status("a") == "indeterminate" and led.status("b") == "completed"


def test_the_ledger_survives_a_restart(tmp_path):
    ApprovalLedger(tmp_path / "l.db").reserve("h", "r")
    assert ApprovalLedger(tmp_path / "l.db").reserve("h", "r2") is False


def test_twenty_threads_reserving_the_same_approval_get_exactly_one_yes():
    led, wins, barrier = ApprovalLedger(), [], threading.Barrier(20)

    def go():
        barrier.wait()
        wins.append(led.reserve("same", "r"))
    threads = [threading.Thread(target=go) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert wins.count(True) == 1 and wins.count(False) == 19


# --- the guarded executor ---------------------------------------------------------------------------------------------------------------
@pytest.fixture
def ex(tmp_path):
    agent, ckpt, reg, creg, writer, rec = setup(str(tmp_path))
    approver, areg = approver_env()
    ledger = ApprovalLedger()
    effects = []
    executor = GuardedExecutor(rec, areg, ledger, {"refund"}, 1, clock=lambda: NOW)
    return NS(executor=executor, ledger=ledger, effects=effects, approver=approver, areg=areg, writer=writer, reg=reg,
              refund=lambda **a: effects.append(a) or "refunded", chain=lambda: read_chain(writer.path) if os.path.exists(writer.path) else [])


def test_a_tool_that_does_not_change_the_world_runs_without_an_approval_and_is_recorded(ex):
    assert ex.executor.execute("lookup", lambda **a: "found", {"q": 1}) == "found" and len(ex.chain()) == 2


def test_a_tool_that_changes_the_world_is_refused_without_an_approval_and_leaves_no_trace(ex):
    assert refusal(lambda: ex.executor.execute("refund", ex.refund, ARGS)) == "approval_required" and ex.effects == [] and ex.chain() == []


def test_a_valid_approval_lets_the_action_run_once_and_the_intent_cites_it(ex):
    ap = make_approval(ex.approver)
    assert ex.executor.execute("refund", ex.refund, ARGS, ap) == "refunded" and len(ex.effects) == 1
    intent = ex.chain()[0]
    assert intent["approval_ref"] == receipt_hash(ap) and intent["action_digest"] == ap["action_digest"] and ex.ledger.status(receipt_hash(ap)) == "completed"


def test_the_same_approval_cannot_be_used_again(ex):
    ap = make_approval(ex.approver)
    ex.executor.execute("refund", ex.refund, ARGS, ap)
    assert refusal(lambda: ex.executor.execute("refund", ex.refund, ARGS, ap)) == "already_used" and len(ex.effects) == 1


def test_an_approval_for_a_smaller_amount_cannot_authorise_a_larger_one(ex):
    ap = make_approval(ex.approver)
    assert refusal(lambda: ex.executor.execute("refund", ex.refund, {**ARGS, "amount": 400000}, ap)) == "action_mismatch" and ex.effects == [] and ex.ledger.status(receipt_hash(ap)) is None


def test_a_refused_approval_is_not_spent(ex):
    ap = make_approval(ex.approver, policy_version=0)
    assert refusal(lambda: ex.executor.execute("refund", ex.refund, ARGS, ap)) == "stale_policy" and ex.ledger.status(receipt_hash(ap)) is None


def test_a_failed_action_still_spends_the_approval_so_a_new_one_is_needed(ex):
    ap = make_approval(ex.approver)

    def boom(**a):
        raise RuntimeError("provider rejected")
    with pytest.raises(RuntimeError):
        ex.executor.execute("refund", boom, ARGS, ap)
    assert ex.ledger.status(receipt_hash(ap)) == "failed" and refusal(lambda: ex.executor.execute("refund", ex.refund, ARGS, ap)) == "already_used" and ex.effects == []


def test_an_indeterminate_outcome_is_never_retried(ex):
    ap = make_approval(ex.approver)

    def unsure(**a):
        raise Indeterminate("timeout after sending")
    with pytest.raises(Indeterminate):
        ex.executor.execute("refund", unsure, ARGS, ap)
    assert ex.ledger.status(receipt_hash(ap)) == "indeterminate" and refusal(lambda: ex.executor.execute("refund", ex.refund, ARGS, ap)) == "indeterminate_outcome" and ex.effects == []


def test_a_crash_leaves_the_approval_reserved_and_a_restart_makes_it_indeterminate(ex):
    ap = make_approval(ex.approver)

    def crash(**a):
        raise SimulatedCrash()
    with pytest.raises(SimulatedCrash):
        ex.executor.execute("refund", crash, ARGS, ap)
    assert ex.ledger.status(receipt_hash(ap)) == "reserved" and refusal(lambda: ex.executor.execute("refund", ex.refund, ARGS, ap)) == "in_progress"
    ex.ledger.mark_unfinished_indeterminate()
    assert refusal(lambda: ex.executor.execute("refund", ex.refund, ARGS, ap)) == "indeterminate_outcome" and ex.effects == []


def test_eight_threads_spending_one_approval_run_the_action_once(ex):
    ap = make_approval(ex.approver)
    barrier, outcomes = threading.Barrier(8), []

    def spend():
        barrier.wait()
        try:
            ex.executor.execute("refund", ex.refund, ARGS, ap)
            outcomes.append("executed")
        except ApprovalRefused as exc:
            outcomes.append(exc.reason)
    threads = [threading.Thread(target=spend) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ex.effects) == 1 and outcomes.count("executed") == 1 and set(outcomes) <= {"executed", "in_progress", "already_used"}


def test_the_chain_of_an_approved_action_verifies_end_to_end(ex):
    ap = make_approval(ex.approver)
    ex.executor.execute("refund", ex.refund, ARGS, ap)
    rep = ChainVerifier(ex.reg, approver_registry=ex.areg).verify(ex.chain(), approvals={receipt_hash(ap): ap}, now=NOW, current_policy_version=1)
    assert rep.valid and rep.complete


def test_checking_intents_against_approvals_ignores_intents_that_cite_nothing():
    assert check_intents_against_approvals(["junk", {"type": INTENT, "approval_ref": None}], {}, KeyRegistry(), now=NOW, current_policy_version=1) == []


# --- the gateway ------------------------------------------------------------------------------------------------------------------------
def test_the_broker_finds_an_approval_only_for_the_exact_action():
    approver, _ = approver_env()
    b = ApprovalBroker()
    ap = make_approval(approver)
    b.add(ap)
    assert b.find("refund", ARGS) is ap and b.find("refund", {**ARGS, "amount": 1}) is None and b.find("other", ARGS) is None and b.find("refund", {"amount": 1.5}) is None


def test_a_wrapped_tool_records_the_user_in_the_action_and_passes_the_user_id_to_the_implementation(ex):
    seen = []
    impls = wrap_impls(ex.executor, {"lookup": lambda user_id, q: seen.append((user_id, q)) or "found"}, ApprovalBroker())
    assert impls["lookup"]("cust-42", q="x") == "found" and seen == [("cust-42", "x")]
    from receipt import open_commit
    intent = ex.chain()[0]
    assert intent["tool"] == "lookup" and ex.executor.recorder.salts.get(intent["receipt_id"], "args") is not None
    assert open_commit({"user_id": "cust-42", "q": "x"}, ex.executor.recorder.salts.get(intent["receipt_id"], "args"), intent["args_commit"])


def test_a_world_changing_tool_without_an_approval_comes_back_to_the_model_as_text_and_does_nothing(ex):
    effects = []
    impls = wrap_impls(ex.executor, {"refund": lambda user_id, **a: effects.append(a) or "done"}, ApprovalBroker())
    assert impls["refund"]("cust-42", order="A1", amount=5) == "Not performed: approval_required" and effects == [] and ex.chain() == []


def test_with_the_right_approval_in_the_broker_the_tool_runs(ex):
    effects, broker = [], ApprovalBroker()
    impls = wrap_impls(ex.executor, {"refund": lambda user_id, **a: effects.append((user_id, a)) or "done"}, broker)
    args = {"user_id": "cust-42", "order": "A1", "amount": 5}
    broker.add(make_approval(ex.approver, args=args))
    assert impls["refund"]("cust-42", order="A1", amount=5) == "done" and effects == [("cust-42", {"order": "A1", "amount": 5})]


def test_an_approval_given_for_one_customer_does_not_cover_another(ex):
    effects, broker = [], ApprovalBroker()
    impls = wrap_impls(ex.executor, {"refund": lambda user_id, **a: effects.append(user_id) or "done"}, broker)
    broker.add(make_approval(ex.approver, args={"user_id": "cust-42", "order": "A1", "amount": 5}))
    assert impls["refund"]("cust-99", order="A1", amount=5) == "Not performed: approval_required" and effects == []


def block(**kw):
    return NS(**kw)


def tool_reply(name, tool_input):
    return NS(content=[NS(type="tool_use", id="t1", name=name, input=tool_input)], stop_reason="tool_use", usage=NS(input_tokens=1, output_tokens=1))


def test_a_claude_turn_leaves_a_verifiable_chain_and_the_model_sees_the_refusal(ex):
    impls = wrap_impls(ex.executor, {"lookup": lambda user_id, q: "found", "refund": lambda user_id, **a: "done"}, ApprovalBroker())
    client = FakeClient([tool_reply("lookup", {"q": "order"}), tool_reply("refund", {"order": "A1", "amount": 5}), text_reply("I could not refund that.")])
    out = run_turn(client, "claude-sonnet-5-5", "system", "refund my order", [], impls, "cust-42")
    assert out == "I could not refund that."
    assert "Not performed: approval_required" in client.calls[2]["messages"][-1]["content"][0]["content"]
    rep = ChainVerifier(ex.reg).verify(ex.chain())
    assert rep.valid and rep.complete and rep.length == 2


def test_the_turn_handles_unknown_tools_bad_arguments_and_runaway_loops(ex):
    impls = wrap_impls(ex.executor, {"lookup": lambda user_id, q: "found"}, ApprovalBroker())
    client = FakeClient([tool_reply("nope", {}), tool_reply("lookup", {"wrong": 1}), text_reply("ok")])
    assert run_turn(client, "m", "s", "hi", [], impls, "u") == "ok"
    assert "Unknown tool" in client.calls[1]["messages"][-1]["content"][0]["content"] and "invalid arguments" in client.calls[2]["messages"][-1]["content"][0]["content"]
    loop = FakeClient([tool_reply("lookup", {"q": "x"}) for _ in range(10)])
    assert run_turn(loop, "m", "s", "hi", [], impls, "u", max_rounds=3).startswith("Stopped")
