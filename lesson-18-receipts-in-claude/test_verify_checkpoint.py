import copy
import os

import pytest

from approval import ApprovalLedger, GuardedExecutor, issue_approval
from checkpoint import make_checkpoint, merkle_root_of, verify_checkpoint_chain
from harness import NOW, setup
from keys import KeyRegistry, Signer
from receipt import APPROVAL, CHECKPOINT, RESULT, INTENT, receipt_hash, sign
from recorder import ChainWriter, ToolRecorder, read_chain
from verify import ChainVerifier


def build(tmp_path, calls=3):
    agent, ckpt, reg, creg, writer, rec = setup(str(tmp_path))
    for i in range(calls):
        rec.call("lookup", lambda **a: "ok", {"i": i})
    return agent, ckpt, reg, creg, writer, rec, read_chain(writer.path)


@pytest.fixture
def chain6(tmp_path):
    return build(tmp_path, 3)


def codes(report):
    return set(report.codes())


# --- a good chain -------------------------------------------------------------------------------------------------------------------------
def test_a_genuine_chain_is_valid_and_complete(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    rep = ChainVerifier(reg).verify(chain)
    assert rep.valid and rep.complete and rep.errors == [] and rep.unresolved == [] and rep.length == 6 and rep.head == receipt_hash(chain[-1]) and rep.statuses == ["ok"] * 6


def test_an_empty_chain_is_valid_with_no_head():
    rep = ChainVerifier(KeyRegistry()).verify([])
    assert rep.valid and rep.complete and rep.length == 0 and rep.head is None


# --- tampering ------------------------------------------------------------------------------------------------------------------------------
def test_changing_a_field_is_a_bad_signature_at_that_receipt_and_it_breaks_the_next_link(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    t = copy.deepcopy(chain)
    t[2]["tool"] = "transfer_money"
    rep = ChainVerifier(reg).verify(t)
    assert not rep.valid and "bad_signature" in codes(rep) and rep.statuses[2] == "invalid" and rep.statuses[3] == "invalid" and rep.statuses[0] == rep.statuses[1] == "ok"
    assert any(f.index == 2 and f.code == "bad_signature" for f in rep.errors) and any(f.index == 3 and f.code == "broken_link" for f in rep.errors)


def test_deleting_a_receipt_from_the_middle_is_a_gap_and_a_broken_link(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    rep = ChainVerifier(reg).verify(chain[:2] + chain[3:])
    assert not rep.valid and {"sequence_gap", "broken_link"} <= codes(rep)


def test_reordering_two_receipts_is_detected(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    t = list(chain)
    t[2], t[3] = t[3], t[2]
    rep = ChainVerifier(reg).verify(t)
    assert not rep.valid and "broken_link" in codes(rep) and ("sequence_gap" in codes(rep) or "duplicate_sequence" in codes(rep))


def test_repeating_a_receipt_is_a_duplicate_sequence(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    rep = ChainVerifier(reg).verify(chain + [chain[-1]])
    assert not rep.valid and "duplicate_sequence" in codes(rep) and "broken_link" in codes(rep)


def test_a_receipt_spliced_in_from_another_run_is_mixed_identity(tmp_path, chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    other = ChainWriter(os.path.join(tmp_path, "other.jsonl"), agent, agent_id="support-bot", run_id="OTHER-RUN", agent_version="v", model_id="m", clock=lambda: NOW, fsync=False)
    ToolRecorder(other, "p", "h").call("t", lambda **a: 1, {})
    rep = ChainVerifier(reg).verify(chain + read_chain(other.path)[:1])
    assert not rep.valid and "mixed_identity" in codes(rep)


# --- keys ----------------------------------------------------------------------------------------------------------------------------------
def test_a_whole_chain_signed_by_an_unregistered_key_is_unknown_key_on_every_receipt(tmp_path, chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    attacker = Signer.generate("agent")
    evil = ChainWriter(os.path.join(tmp_path, "evil.jsonl"), attacker, agent_id="support-bot", run_id="run-1", agent_version="v", model_id="m", clock=lambda: NOW, fsync=False)
    ToolRecorder(evil, "p", "h").call("t", lambda **a: 1, {})
    rep = ChainVerifier(reg).verify(read_chain(evil.path))
    assert not rep.valid and codes(rep) == {"unknown_key"}


def test_claiming_the_real_key_id_without_the_private_key_is_a_bad_signature(tmp_path, chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    t = copy.deepcopy(chain)
    for r in t:
        r["sig"] = chain[0]["sig"]
    assert "bad_signature" in codes(ChainVerifier(reg).verify(t))


def test_keys_that_are_expired_revoked_not_yet_valid_or_in_the_wrong_role_are_named(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    for setup_registry, expected in ((lambda r: r.add_signer(agent, valid_from=0, valid_to=NOW), "key_expired"), (lambda r: r.add_signer(agent, valid_from=NOW + 1), "key_not_yet_valid"),
                                     (lambda r: r.revoke(r.add_signer(agent), NOW), "key_revoked"), (lambda r: r.add(agent.public_bytes, "checkpoint"), "wrong_role")):
        r = KeyRegistry()
        setup_registry(r)
        assert expected in codes(ChainVerifier(r).verify(chain)), expected


def test_a_key_revoked_later_still_verifies_the_earlier_receipts_but_warns_that_the_time_is_only_a_claim(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    r = KeyRegistry()
    r.revoke(r.add_signer(agent), NOW + 1000)
    rep = ChainVerifier(r).verify(chain)
    assert rep.valid and {f.code for f in rep.warnings} == {"signed_by_key_revoked_later"} and len(rep.warnings) == 6


# --- malformed input ------------------------------------------------------------------------------------------------------------------------
def test_garbage_entries_are_reported_and_do_not_stop_the_rest_of_the_chain_being_checked(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    t = [chain[0], "garbage", {"unreadable": "x"}, None] + chain[1:]
    rep = ChainVerifier(reg).verify(t)
    assert not rep.valid and [f.code for f in rep.errors if f.index in (1, 2, 3)] == ["malformed"] * 3 and rep.statuses[0] == "ok" and rep.length == 9


def test_a_receipt_of_the_wrong_kind_does_not_belong_in_a_tool_chain(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    approver = Signer.generate("approver")
    a = issue_approval(approver, approver_id="s", tool="t", args={}, policy_version=1, expires_at=NOW + 9, now=NOW)
    assert "unexpected_type" in codes(ChainVerifier(reg).verify(chain + [a]))


# --- results and intents ---------------------------------------------------------------------------------------------------------------------
def test_a_result_with_no_intent_is_an_orphan(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    assert "orphan_result" in codes(ChainVerifier(reg).verify([chain[1]]))


def test_a_result_for_the_wrong_tool_or_a_second_result_for_one_intent_is_refused(tmp_path):
    agent, ckpt, reg, creg, w, rec = setup(str(tmp_path))
    i = w.append({"type": INTENT, "tool": "lookup", "args_commit": "c", "policy_id": "p", "policy_hash": "h", "approval_ref": None, "action_digest": None})
    w.append({"type": RESULT, "intent_ref": receipt_hash(i), "tool": "DIFFERENT", "outcome": "success", "result_commit": None, "error_type": None})
    w.append({"type": RESULT, "intent_ref": receipt_hash(i), "tool": "lookup", "outcome": "success", "result_commit": None, "error_type": None})
    rep = ChainVerifier(reg).verify(read_chain(w.path))
    assert {"tool_mismatch", "duplicate_result"} <= codes(rep)


def test_an_intent_with_no_result_is_valid_but_not_complete_and_is_listed(tmp_path):
    agent, ckpt, reg, creg, w, rec = setup(str(tmp_path))
    rec.call("lookup", lambda **a: 1, {})
    w.append({"type": INTENT, "tool": "book", "args_commit": "c", "policy_id": "p", "policy_hash": "h", "approval_ref": None, "action_digest": None})
    rep = ChainVerifier(reg).verify(read_chain(w.path))
    assert rep.valid and not rep.complete and rep.unresolved == [2]


def test_failures_and_indeterminate_results_resolve_their_intents(tmp_path):
    from recorder import Indeterminate
    agent, ckpt, reg, creg, w, rec = setup(str(tmp_path))
    with pytest.raises(ValueError):
        rec.call("t", lambda **a: (_ for _ in ()).throw(ValueError("x")), {})
    with pytest.raises(Indeterminate):
        rec.call("t", lambda **a: (_ for _ in ()).throw(Indeterminate("x")), {})
    assert ChainVerifier(reg).verify(read_chain(w.path)).complete


# --- time -----------------------------------------------------------------------------------------------------------------------------------
def test_time_that_goes_backwards_or_into_the_future_is_a_warning_not_an_error(tmp_path):
    agent, ckpt, reg, creg, _, _ = setup(str(tmp_path))
    times = iter([1000, 900, 5000])
    w = ChainWriter(os.path.join(tmp_path, "t.jsonl"), agent, agent_id="a", run_id="r", agent_version="v", model_id="m", clock=lambda: next(times), fsync=False)
    for _ in range(3):
        w.append({"type": INTENT, "tool": "t", "args_commit": "c", "policy_id": "p", "policy_hash": "h", "approval_ref": None, "action_digest": None})
    rep = ChainVerifier(reg).verify(read_chain(w.path), now=2000)
    assert rep.valid and {f.code for f in rep.warnings} == {"time_goes_backwards", "future_time"}


# --- truncation ---------------------------------------------------------------------------------------------------------------------------
def test_a_chain_cut_short_at_the_end_still_verifies_which_is_why_an_anchor_is_needed(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    rep = ChainVerifier(reg).verify(chain[:4])
    assert rep.valid and rep.complete


def test_a_pinned_head_or_a_minimum_length_catches_truncation(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    v = ChainVerifier(reg)
    assert "head_mismatch" in codes(v.verify(chain[:4], expected_head=w.head))
    assert v.verify(chain, expected_head=w.head).valid
    assert "truncated" in codes(v.verify(chain[:4], min_length=6)) and v.verify(chain, min_length=6).valid


# --- checkpoints --------------------------------------------------------------------------------------------------------------------------
def checkpoint(ckpt, chain, **kw):
    return make_checkpoint(ckpt, agent_id="support-bot", run_id="run-1", receipts=chain, clock=lambda: NOW, **kw)


def test_a_checkpoint_over_the_full_chain_or_a_prefix_verifies(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    v = ChainVerifier(reg, creg)
    assert v.verify(chain, checkpoints=[checkpoint(ckpt, chain)]).valid and v.verify(chain, checkpoints=[checkpoint(ckpt, chain[:4])]).valid
    assert v.verify([], checkpoints=[checkpoint(ckpt, [])]).valid


def test_a_chain_shorter_than_its_checkpoint_is_truncated(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    assert "truncated_vs_checkpoint" in codes(ChainVerifier(reg, creg).verify(chain[:4], checkpoints=[checkpoint(ckpt, chain)]))


def test_history_rewritten_with_the_agent_key_is_caught_by_the_checkpoint(tmp_path, chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    cp = checkpoint(ckpt, chain)
    rewritten = ChainWriter(os.path.join(tmp_path, "rewrite.jsonl"), agent, agent_id="support-bot", run_id="run-1", agent_version="v-3f9a2c", model_id="claude-sonnet-5-5", clock=lambda: NOW, fsync=False)
    r2 = ToolRecorder(rewritten, "refund-policy", "sha256:policyhash")
    for i in range(3):
        r2.call("lookup", lambda **a: "DIFFERENT", {"i": i})
    rep = ChainVerifier(reg, creg).verify(read_chain(rewritten.path), checkpoints=[cp])
    assert rep.valid is False and codes(rep) & {"forked_vs_checkpoint"}


def test_a_checkpoint_with_the_right_head_but_a_wrong_merkle_root_is_caught(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    good = checkpoint(ckpt, chain)
    payload = {k: v for k, v in good.items() if k not in ("sig",)}
    payload["merkle_root"] = merkle_root_of(["sha256:" + "0" * 64] * 6)
    bad = sign(payload, ckpt)
    assert "merkle_mismatch" in codes(ChainVerifier(reg, creg).verify(chain, checkpoints=[bad]))


def test_a_checkpoint_from_an_unregistered_key_or_a_key_registered_for_another_role_is_refused(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    assert "unknown_key" in codes(ChainVerifier(reg, creg).verify(chain, checkpoints=[checkpoint(Signer.generate("checkpoint"), chain)]))
    wrong = KeyRegistry()
    wrong.add(ckpt.public_bytes, "agent")  # the checkpoint key is known, but only as an agent key
    assert "wrong_role" in codes(ChainVerifier(reg, wrong).verify(chain, checkpoints=[checkpoint(ckpt, chain)]))


def test_a_tampered_checkpoint_is_a_bad_signature(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    cp = {**checkpoint(ckpt, chain), "length": 2}
    assert "bad_signature" in codes(ChainVerifier(reg, creg).verify(chain, checkpoints=[cp]))


def test_a_checkpoint_for_a_different_run_is_ignored_for_this_chain(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    other = make_checkpoint(ckpt, agent_id="support-bot", run_id="ANOTHER", receipts=chain + chain, clock=lambda: NOW)
    assert ChainVerifier(reg, creg).verify(chain, checkpoints=[other]).valid


def test_checkpoints_link_to_each_other_and_an_older_history_than_the_pinned_one_is_a_rollback(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    c0 = checkpoint(ckpt, chain[:2])
    c1 = checkpoint(ckpt, chain[:4], ckpt_seq=1, prev_checkpoint=receipt_hash(c0))
    c2 = checkpoint(ckpt, chain, ckpt_seq=2, prev_checkpoint=receipt_hash(c1))
    assert verify_checkpoint_chain([c0, c1, c2], creg) == [] and verify_checkpoint_chain([c0, c1, c2], creg, pinned_latest=receipt_hash(c2)) == []
    rollback = verify_checkpoint_chain([c0, c1], creg, pinned_latest=receipt_hash(c2))
    assert [f[1] for f in rollback] == ["checkpoint_rollback_or_fork"]
    v = ChainVerifier(reg, creg)
    assert "checkpoint_rollback_or_fork" in codes(v.verify(chain, checkpoints=[c0, c1], pinned_latest_checkpoint=receipt_hash(c2)))
    assert v.verify(chain, checkpoints=[c0, c1, c2], pinned_latest_checkpoint=receipt_hash(c2)).valid


def test_a_broken_checkpoint_sequence_or_link_is_reported(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    c0 = checkpoint(ckpt, chain[:2])
    skipped = checkpoint(ckpt, chain[:4], ckpt_seq=5, prev_checkpoint=receipt_hash(c0))
    unlinked = checkpoint(ckpt, chain[:4], ckpt_seq=1, prev_checkpoint="sha256:" + "0" * 64)
    assert "checkpoint_sequence" in {f[1] for f in verify_checkpoint_chain([c0, skipped], creg)}
    assert "checkpoint_link" in {f[1] for f in verify_checkpoint_chain([c0, unlinked], creg)}
    assert verify_checkpoint_chain(["junk"], creg)[0][1] == "malformed"


def test_checkpoints_need_a_checkpoint_key_and_commit_to_head_length_and_root(chain6):
    agent, ckpt, reg, creg, w, rec, chain = chain6
    with pytest.raises(ValueError):
        make_checkpoint(agent, agent_id="a", run_id="r", receipts=chain)
    cp = checkpoint(ckpt, chain)
    assert cp["length"] == 6 and cp["head"] == receipt_hash(chain[-1]) and cp["merkle_root"] == merkle_root_of([receipt_hash(r) for r in chain]) and cp["type"] == CHECKPOINT


# --- approvals inside a chain ----------------------------------------------------------------------------------------------------------------
def approved_run(tmp_path, amount=4000):
    agent, ckpt, reg, creg, w, rec = setup(str(tmp_path))
    approver = Signer.generate("approver")
    areg = KeyRegistry()
    areg.add_signer(approver)
    ex = GuardedExecutor(rec, areg, ApprovalLedger(), {"refund"}, 1, clock=lambda: NOW)
    args = {"order": "A1", "amount": amount}
    ap = issue_approval(approver, approver_id="sup", tool="refund", args=args, policy_version=1, expires_at=NOW + 600, now=NOW)
    ex.execute("refund", lambda **a: "ok", args, ap)
    return agent, reg, areg, approver, w, rec, ap, read_chain(w.path)


def test_a_chain_with_an_approved_action_verifies_with_the_approval_supplied(tmp_path):
    agent, reg, areg, approver, w, rec, ap, chain = approved_run(tmp_path)
    v = ChainVerifier(reg, approver_registry=areg)
    assert v.verify(chain, approvals={receipt_hash(ap): ap}, now=NOW, current_policy_version=1).valid


def test_an_intent_citing_an_approval_that_was_not_supplied_is_flagged(tmp_path):
    agent, reg, areg, approver, w, rec, ap, chain = approved_run(tmp_path)
    assert "approval_missing" in codes(ChainVerifier(reg, approver_registry=areg).verify(chain, approvals={}, now=NOW, current_policy_version=1))


def test_the_approval_must_still_be_acceptable_when_the_chain_is_audited(tmp_path):
    agent, reg, areg, approver, w, rec, ap, chain = approved_run(tmp_path)
    v = ChainVerifier(reg, approver_registry=areg)
    ap_map = {receipt_hash(ap): ap}
    assert "approval_stale_policy" in codes(v.verify(chain, approvals=ap_map, now=NOW, current_policy_version=2))
    assert "approval_expired" in codes(v.verify(chain, approvals=ap_map, now=NOW + 10_000, current_policy_version=1))
    areg.revoke(approver.kid, NOW)
    assert "approval_key_revoked" in codes(v.verify(chain, approvals=ap_map, now=NOW, current_policy_version=1))


def test_an_intent_whose_action_digest_differs_from_the_approval_is_digest_substitution(tmp_path):
    agent, ckpt, reg, creg, w, rec = setup(str(tmp_path))
    approver = Signer.generate("approver")
    areg = KeyRegistry()
    areg.add_signer(approver)
    ap = issue_approval(approver, approver_id="s", tool="refund", args={"amount": 4000}, policy_version=1, expires_at=NOW + 600, now=NOW)
    # a compromised agent records a DIFFERENT action but cites the real approval
    rec.call("refund", lambda **a: "ok", {"amount": 400000}, approval=receipt_hash(ap), action_digest="sha256:" + "f" * 64)
    rep = ChainVerifier(reg, approver_registry=areg).verify(read_chain(w.path), approvals={receipt_hash(ap): ap}, now=NOW, current_policy_version=1)
    assert "approval_digest_substitution" in codes(rep)


def test_one_approval_cited_by_two_intents_is_reuse_and_a_different_tool_is_flagged(tmp_path):
    agent, ckpt, reg, creg, w, rec = setup(str(tmp_path))
    approver = Signer.generate("approver")
    areg = KeyRegistry()
    areg.add_signer(approver)
    ap = issue_approval(approver, approver_id="s", tool="refund", args={"amount": 5}, policy_version=1, expires_at=NOW + 600, now=NOW)
    h = receipt_hash(ap)
    rec.call("refund", lambda **a: 1, {"amount": 5}, approval=h, action_digest=ap["action_digest"])
    rec.call("delete_account", lambda **a: 1, {"amount": 5}, approval=h, action_digest=ap["action_digest"])
    rep = ChainVerifier(reg, approver_registry=areg).verify(read_chain(w.path), approvals={h: ap}, now=NOW, current_policy_version=1)
    assert {"approval_reused", "approval_tool_mismatch"} <= codes(rep)
