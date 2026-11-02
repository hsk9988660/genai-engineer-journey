import pytest

import harness


@pytest.fixture(scope="module")
def by():
    return {(o.scenario, o.strategy): o.metrics for o in harness.evaluate()}


def test_verifying_with_the_key_inside_the_receipt_accepts_a_forgery_and_a_pinned_key_does_not(by):
    assert by[("forgery", "lesson_verify_with_the_embedded_key")]["forgery_accepted"] is True
    p = by[("forgery", "lesson_verify_with_a_pinned_key")]
    assert p["forgery_accepted"] is False and p["genuine_accepted"] is True


def test_a_whole_chain_signed_by_the_attackers_key_and_one_that_copies_the_key_id_are_both_refused(by):
    m = by[("forgery", "pinned_registry_chain_verifier")]
    assert m["own_key_chain_accepted"] is False and m["own_key_reason"] == ["unknown_key"] and m["genuine_accepted"] is True
    assert m["copied_kid_accepted"] is False and "bad_signature" in m["copied_kid_reason"]


def test_whoever_can_verify_an_hmac_chain_can_also_forge_it_and_a_public_key_verifier_cannot(by):
    h, e = by[("who_can_forge", "hmac_chain (secret shared with the verifier)")], by[("who_can_forge", "ed25519 (verifier holds public keys only)")]
    assert h["genuine_verifies"] and h["insider_forgery_verifies"] is True
    assert e["genuine_verifies"] and e["insider_forgery_verifies"] is False


def test_a_chain_cut_short_still_verifies_until_a_head_or_a_signed_checkpoint_is_pinned(by):
    assert by[("truncation", "lesson_verify_chain")]["cut_chain_verifies"] is True
    alone = by[("truncation", "chain_verifier_alone")]
    assert alone["valid"] and alone["complete"] and alone["truncation_detected"] is False
    assert by[("truncation", "with_a_pinned_head")]["truncation_detected"] is True
    cp = by[("truncation", "with_a_signed_checkpoint")]
    assert cp["truncation_detected"] is True and cp["full_chain_valid"] is True


def test_a_crash_after_the_effect_leaves_no_receipt_in_the_lessons_wrapper_and_a_visible_intent_in_ours(by):
    assert by[("crash_after_effect", "lesson_wrapper (receipt written after the call)")] == {"effects_in_the_world": 1, "receipts": 0}
    m = by[("crash_after_effect", "intent_then_result")]
    assert m["effects_in_the_world"] == 1 and m["receipts"] == 1 and m["valid"] is True and m["complete"] is False and m["unresolved_intents"] == [0]


def test_guessable_arguments_are_recovered_from_a_plain_hash_but_not_from_a_salted_commitment(by):
    p, s = by[("guessable_arguments", "plain_sha256_of_the_arguments")], by[("guessable_arguments", "salted_commitment")]
    assert p["arguments_recovered"] is True and p["guesses_needed"] <= p["search_space"] and p["search_space"] > 10_000
    assert s["arguments_recovered"] is False and s["owner_can_open"] is True and s["wrong_salt_opens"] is False


def test_a_boolean_flag_lets_three_refunds_through_and_signed_one_time_approvals_let_one(by):
    flag = by[("approvals", "an_approved_flag_on_the_call")]
    assert flag["refunds_executed"] == 3 and flag["amounts"] == [4000, 400_000, 4000]
    m = by[("approvals", "signed_approval_and_a_one_time_ledger")]
    assert m["refunds_executed"] == 1
    assert m["results"] == {"no approval": "approval_required", "valid approval": "executed", "same approval again": "already_used", "bigger amount, same approval": "action_mismatch",
                            "approver key not registered": "unknown_key", "old policy version": "stale_policy", "approval expired": "expired", "approver key revoked since": "key_revoked"}


def test_eight_threads_spending_one_approval_run_it_exactly_once(by):
    m = by[("approvals", "eight_threads_spend_one_approval")]
    assert m["refunds_executed"] == 1 and m["outcomes"].get("executed") == 1 and sum(m["outcomes"].values()) == 8


def test_deleting_one_tools_chain_goes_unnoticed_per_tool_and_is_caught_in_a_single_chain(by):
    a, b = by[("one_chain", "a_chain_per_tool (the lesson's ReceiptedTool)")], by[("one_chain", "one_chain_for_the_agent")]
    assert a["remaining_chains_verify"] is True and a["deletion_detected"] is False and a["receipt_names_the_model"] is False
    assert b["deletion_detected"] is True and {"sequence_gap", "broken_link"} <= set(b["codes"]) and b["receipt_names_the_model"] is True


def test_the_table_prints_every_row(capsys):
    outcomes = harness.evaluate()
    harness.print_table(outcomes)
    assert len(capsys.readouterr().out.strip().splitlines()) == len(outcomes)
