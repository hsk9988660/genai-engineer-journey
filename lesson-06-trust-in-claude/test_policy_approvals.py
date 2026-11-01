import time
from dataclasses import replace

import pytest

from approvals import ApprovalGate, args_hash
from demo_tools import AccountArgs, Query, build_tools
from policy import PolicyEngine, RunState, ToolSpec, UserContext

TOOLS = {t.name: t for t in build_tools()}
ALL = UserContext("u", frozenset({"support", "refunds", "admin"}))
engine = PolicyEngine()


def decide(tool, args=None, user=ALL, tainted=False):
    return engine.decide(TOOLS.get(tool), args or {}, user, RunState(tainted=tainted))


# --- policy -------------------------------------------------------------------------------
def test_read_only_tools_are_allowed():
    assert decide("search_flights", {"query": "x"}).verdict == "allow"


def test_side_effect_and_irreversible_tools_need_approval():
    assert decide("send_email").verdict == "ask"
    d = decide("delete_account")
    assert d.verdict == "ask" and "irreversible" in d.reason


def test_cost_computed_from_arguments_is_reported_in_the_reason():
    d = decide("issue_refund", {"booking_id": "BK1001", "amount_usd": 5000})
    assert d.verdict == "ask" and "$5,000.00" in d.reason


def test_a_cheap_read_only_tool_below_the_threshold_stays_allowed_but_an_expensive_one_asks():
    cheap = ToolSpec("c", "d", Query, lambda query: 1, cost_usd=1.0)
    pricey = ToolSpec("p", "d", Query, lambda query: 1, cost_usd=40.0)
    assert engine.decide(cheap, {}, ALL, RunState()).verdict == "allow"
    assert engine.decide(pricey, {}, ALL, RunState()).verdict == "ask"


def test_unknown_tool_and_missing_scope_are_denied():
    assert decide("run_shell").verdict == "deny"
    d = decide("issue_refund", {"booking_id": "BK1001", "amount_usd": 5}, UserContext("u", frozenset({"support"})))
    assert d.verdict == "deny" and "scope" in d.reason


def test_side_effects_are_denied_after_untrusted_content_but_reads_are_not():
    assert decide("send_email", tainted=True).verdict == "deny"
    assert decide("search_flights", {"query": "x"}, tainted=True).verdict == "allow"


def test_scope_is_checked_before_anything_else():
    nobody = UserContext("u", frozenset())
    assert decide("lookup_customer", {"booking_id": "BK1001"}, nobody).verdict == "deny"


def test_known_limit_policy_trusts_the_developers_declarations():
    """If a tool that really sends email is registered WITHOUT side_effects=True, policy cannot know."""
    mislabelled = ToolSpec("notify", "Notify someone.", Query, lambda query: "emailed everyone")  # forgot side_effects
    assert engine.decide(mislabelled, {}, ALL, RunState()).verdict == "allow"


# --- approvals ----------------------------------------------------------------------------
SPEC = TOOLS["issue_refund"]
ARGS = {"booking_id": "BK1001", "amount_usd": 20.0}


def gate(answer="approve", **kw):
    return ApprovalGate(lambda summary: answer, **kw)


def test_exact_action_is_redeemable_once():
    g = gate()
    grant = g.request(SPEC, ARGS)
    assert g.redeem(grant, "issue_refund", ARGS) == (True, "ok")
    assert g.redeem(grant, "issue_refund", ARGS) == (False, "grant already used")


def test_changed_arguments_or_tool_are_refused():
    g = gate()
    assert not g.redeem(g.request(SPEC, ARGS), "issue_refund", {**ARGS, "amount_usd": 5000.0})[0]
    assert not g.redeem(g.request(SPEC, ARGS), "delete_account", ARGS)[0]


def test_grant_expires():
    now = [0.0]
    g = gate(ttl_s=10, clock=lambda: now[0])
    grant = g.request(SPEC, ARGS)
    now[0] = 10.5
    assert g.redeem(grant, "issue_refund", ARGS) == (False, "grant expired")


@pytest.mark.parametrize("answer", ["deny", "yes please", "Approve", "APPROVE ", "", None, True])
def test_only_the_exact_word_approve_approves(answer):
    assert gate(answer).request(SPEC, ARGS) is None


def test_approver_crash_and_timeout_fail_closed():
    assert ApprovalGate(lambda s: 1 / 0).request(SPEC, ARGS) is None

    def slow(summary):
        time.sleep(1)
        return "approve"

    start = time.monotonic()
    assert ApprovalGate(slow, timeout_s=0.05).request(SPEC, ARGS) is None
    assert time.monotonic() - start < 0.8


def test_approver_cannot_change_what_it_is_approving():
    """The hash is taken before the approver runs and the approver only receives a copy."""

    def tampering(summary):
        summary["args"]["amount_usd"] = 9999.0
        return "approve"

    g = ApprovalGate(tampering)
    grant = g.request(SPEC, ARGS)
    assert ARGS["amount_usd"] == 20.0  # the caller's dict is untouched
    assert g.redeem(grant, "issue_refund", ARGS)[0]  # grant still matches the ORIGINAL arguments


def test_arguments_changed_after_approval_are_refused():
    args = dict(ARGS)
    g = gate()
    grant = g.request(SPEC, args)
    args["amount_usd"] = 5000.0  # something mutates the dict between approval and execution
    assert not g.redeem(grant, "issue_refund", args)[0]


def test_summary_carries_what_a_human_needs_to_decide():
    seen = {}
    ApprovalGate(lambda s: seen.update(s) or "deny").request(SPEC, ARGS, {"cost_usd": 20.0, "tainted": False})
    assert seen["tool"] == "issue_refund" and seen["args"] == ARGS
    assert seen["irreversible"] is True and seen["cost_usd"] == 20.0 and seen["tainted"] is False


def test_args_hash_ignores_key_order_but_not_values():
    assert args_hash("t", {"a": 1, "b": 2}) == args_hash("t", {"b": 2, "a": 1})
    assert args_hash("t", {"a": 1}) != args_hash("t", {"a": 2})
    assert args_hash("t", {"a": 1}) != args_hash("u", {"a": 1})


def test_replace_helper_keeps_spec_metadata():
    spy = replace(TOOLS["delete_account"], fn=lambda account_id: "spy")
    assert spy.irreversible and spy.input_model is AccountArgs
