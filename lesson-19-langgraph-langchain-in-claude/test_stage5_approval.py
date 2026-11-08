from datetime import date
from types import SimpleNamespace

import pytest

from conftest import tool_texts
from desk import Desk, Order
from fakes import ScriptedChatModel, call, say
from show import final_text
from stage1_first_agent import ask
from stage5_approval import (AUTO_APPROVE_LIMIT_CENTS, approve, build_approval_agent, edit, needs_approval, pending_approval, reject, resume)

REFUND_89 = {"order_id": "A1001", "amount": 89.0, "reason": "too small"}


def start(desk, refund_args=None, script_tail=None, customer="C-100", text="Refund it."):
    model = ScriptedChatModel(script=[call("issue_refund", refund_args or REFUND_89), *(script_tail or [say("All done.")])])
    agent = build_approval_agent(model, desk)
    return agent, model, ask(agent, text, customer_id=customer, thread_id="t")


class TestThePause:
    def test_the_agent_pauses_before_the_money_moves(self, desk):
        _, _, paused = start(desk)
        assert pending_approval(paused) is not None and desk.refunds == []

    def test_the_reviewer_is_shown_facts_that_code_looked_up(self, desk):
        _, _, paused = start(desk)
        request = pending_approval(paused)
        assert request.tool == "issue_refund" and request.args == REFUND_89
        assert "$89.00" in request.description and "Trail Shoes" in request.description and "still refundable $89.00" in request.description
        assert request.allowed == ["approve", "edit", "reject"]

    def test_the_reviewer_is_told_when_the_order_cannot_be_verified(self, desk):
        _, _, paused = start(desk, {**REFUND_89, "order_id": "A9999"})
        assert "could not be verified" in pending_approval(paused).description

    def test_the_pause_is_saved_so_the_program_could_stop_and_come_back(self, desk):
        agent, _, _ = start(desk)
        snapshot = agent.get_state({"configurable": {"thread_id": "t"}})
        assert snapshot.next and "HumanInTheLoopMiddleware" in snapshot.next[0]

    def test_a_finished_run_has_no_pending_approval(self, desk):
        agent, _, paused = start(desk)
        assert pending_approval(resume(agent, approve(), "t")) is None

    def test_read_only_tools_never_pause(self, desk):
        model = ScriptedChatModel(script=[call("lookup_order", {"order_id": "A1001"}), call("check_refund_policy", {"order_id": "A1001"}), say("ok")])
        result = ask(build_approval_agent(model, desk), "info", thread_id="t")
        assert pending_approval(result) is None and final_text(result["messages"]) == "ok"


class TestTheDecisions:
    def test_approve_runs_the_refund_exactly_once(self, desk):
        agent, _, _ = start(desk)
        done = resume(agent, approve(), "t")
        assert [(r.order_id, r.amount_cents) for r in desk.refunds] == [("A1001", 8900)]
        assert final_text(done["messages"]) == "All done."

    def test_reject_runs_nothing_and_tells_the_model_why(self, desk):
        agent, model, _ = start(desk)
        done = resume(agent, reject("Offer store credit instead."), "t")
        assert desk.refunds == [] and "Offer store credit instead." in tool_texts(done)[0]
        assert "Offer store credit instead." in " ".join(str(m.content) for m in model.seen[-1])

    def test_edit_runs_the_tool_with_the_reviewers_arguments(self, desk):
        agent, _, paused = start(desk)
        resume(agent, edit("issue_refund", {**pending_approval(paused).args, "amount": 30.0}), "t")
        assert [r.amount_cents for r in desk.refunds] == [3000]

    def test_the_model_is_told_that_a_human_changed_the_call(self, desk):
        agent, _, paused = start(desk)
        done = resume(agent, edit("issue_refund", {**REFUND_89, "amount": 30.0}), "t")
        assert "reviewer replaced this tool call" in tool_texts(done)[0] and "30.0" in tool_texts(done)[0]


class TestApprovalIsNotAuthorization:
    def test_an_edit_above_what_the_order_allows_is_refused_by_the_desk(self, desk):
        agent, _, _ = start(desk)
        done = resume(agent, edit("issue_refund", {**REFUND_89, "amount": 500.0}), "t")
        assert desk.refunds == [] and "more than the $89.00" in tool_texts(done)[0]

    def test_an_approved_refund_above_the_agent_limit_is_still_refused(self, desk):
        agent, _, _ = start(desk, {"order_id": "A1002", "amount": 240.0, "reason": "defective"})
        done = resume(agent, approve(), "t")
        assert desk.refunds == [] and "agent limit" in tool_texts(done)[0]

    def test_an_approved_refund_on_an_ineligible_order_is_refused(self, desk):
        agent, _, _ = start(desk, {"order_id": "A1003", "amount": 50.0, "reason": "x"})
        done = resume(agent, approve(), "t")
        assert desk.refunds == [] and "return window" in tool_texts(done)[0]

    def test_an_approved_refund_on_another_customers_order_is_refused(self, desk):
        agent, _, _ = start(desk, {"order_id": "A1004", "amount": 50.0, "reason": "x"}, customer="C-100")
        done = resume(agent, approve(), "t", customer_id="C-100")
        assert desk.refunds == [] and "no order A1004" in tool_texts(done)[0]

    def test_an_unreadable_amount_pauses_and_is_refused_even_if_approved(self, desk):
        agent, _, paused = start(desk, {**REFUND_89, "amount": "a lot"})
        request = pending_approval(paused)
        assert request is not None and "unreadable amount" in request.description
        done = resume(agent, approve(), "t")
        assert desk.refunds == [] and "valid number" in tool_texts(done)[0]


class TestRiskTiers:
    def small_desk(self):
        return Desk([Order("S1", "C-100", "Socks", 5000, "delivered", date(2026, 11, 1))])

    @pytest.mark.parametrize("cents, pauses", [(2400, False), (AUTO_APPROVE_LIMIT_CENTS, False), (AUTO_APPROVE_LIMIT_CENTS + 1, True)])
    def test_only_refunds_above_the_limit_wait_for_a_person(self, cents, pauses):
        desk = self.small_desk()
        _, _, result = start(desk, {"order_id": "S1", "amount": cents / 100, "reason": "x"})
        assert (pending_approval(result) is not None) == pauses
        assert len(desk.refunds) == (0 if pauses else 1)

    @pytest.mark.parametrize("args, expected", [
        ({"amount": 10.0}, False), ({"amount": 25.0}, False), ({"amount": 25.01}, True), ({"amount": "5"}, False),
        ({"amount": "lots"}, True), ({"amount": None}, True), ({}, True), ({"amount": -5}, True), ({"amount": 0}, True),
    ])
    def test_needs_approval_sends_anything_odd_to_a_person(self, args, expected):
        assert needs_approval(SimpleNamespace(tool_call={"args": args})) is expected

    def test_the_decision_depends_only_on_the_call_itself(self):
        request = SimpleNamespace(tool_call={"args": {"amount": 30.0}})
        assert needs_approval(request) == needs_approval(request) is True
