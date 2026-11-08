import pytest
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.tools import tool

from conftest import tool_texts
from desk import Desk, TemporaryOutage
from fakes import ScriptedChatModel, call, calls, say
from show import final_text
from stage1_first_agent import SYSTEM_PROMPT, build_agent
from stage5_approval import approve, pending_approval, resume
from stage6_middleware import MAX_LOGGED_TEXT, AuditTrail, build_hardened_agent
from tools import DeskContext


def build(script, desk, **kwargs):
    model, trail = ScriptedChatModel(script=script), AuditTrail()
    return build_hardened_agent(model, desk, trail, approval=False, **kwargs), model, trail


def run(agent, text="go", customer="C-100", name="Sam", config=None):
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, config, context=DeskContext(customer, name))


def count_lookups(desk):
    seen = []
    original = desk.get_order

    def counting(customer_id, order_id):
        seen.append(order_id)
        return original(customer_id, order_id)

    desk.get_order = counting
    return seen


class TestAudit:
    def test_each_tool_call_becomes_one_numbered_record(self, desk):
        agent, _, trail = build([calls(("lookup_order", {"order_id": "A1001"}, "a"), ("search_faq", {"query": "shipping"}, "b")), say("ok")], desk)
        run(agent)
        assert [(r["seq"], r["tool"], r["outcome"]) for r in trail.records] == [(1, "lookup_order", "ok"), (2, "search_faq", "ok")]

    def test_a_refusal_from_the_business_rules_is_recorded_as_refused(self, desk):
        agent, _, trail = build([call("lookup_order", {"order_id": "A9999"}), say("not found")], desk)
        run(agent)
        assert trail.records[0]["outcome"] == "refused" and "no order A9999" in trail.records[0]["detail"]

    def test_a_schema_error_is_recorded_as_an_error(self, desk):
        agent, _, trail = build([call("issue_refund", {"order_id": "A1001", "amount": "lots", "reason": "x"}), say("sorry")], desk)
        run(agent)
        assert trail.records[0]["outcome"] == "error" and desk.refunds == []

    def test_long_text_is_clipped_in_the_record_but_not_in_the_call(self, desk):
        long_reason = "r" * 300
        agent, _, trail = build([call("issue_refund", {"order_id": "A1001", "amount": 5.0, "reason": long_reason}), say("ok")], desk)
        run(agent)
        assert len(trail.records[0]["args"]["reason"]) == MAX_LOGGED_TEXT and desk.refunds[0].reason == long_reason

    def test_every_conversation_needs_its_own_trail(self, desk):
        agent, _, first = build([call("search_faq", {"query": "x"}), say("ok")], desk)
        run(agent)
        _, _, second = build([say("hi")], Desk())
        assert len(first.records) == 1 and second.records == []

    def test_the_record_does_not_change_what_the_model_receives(self, desk):
        agent, model, _ = build([call("lookup_order", {"order_id": "A1001"}), say("ok")], desk)
        run(agent)
        assert "Trail Shoes" in model.seen[1][-1].content


class TestRetry:
    def test_a_temporary_outage_is_retried_and_the_model_never_notices(self, desk):
        desk.outages = 2
        agent, model, trail = build([call("lookup_order", {"order_id": "A1001"}), say("ok")], desk, max_retries=2)
        result = run(agent)
        assert desk.outages == 0 and "Trail Shoes" in tool_texts(result)[0] and len(model.seen) == 2

    def test_the_audit_outside_retry_sees_one_logical_call(self, desk):
        desk.outages = 2
        agent, _, trail = build([call("lookup_order", {"order_id": "A1001"}), say("ok")], desk, max_retries=2)
        run(agent)
        assert [r["outcome"] for r in trail.records] == ["ok"]

    def test_the_audit_inside_retry_sees_every_attempt(self, desk):
        desk.outages = 2
        trail, model = AuditTrail(), ScriptedChatModel(script=[call("lookup_order", {"order_id": "A1001"}), say("ok")])
        retry = ToolRetryMiddleware(max_retries=2, retry_on=(TemporaryOutage,), initial_delay=0.0, jitter=False)
        agent = build_agent(model, desk, middleware=[retry, trail.middleware])
        run(agent)
        assert [r["outcome"] for r in trail.records] == ["exception", "exception", "ok"]
        assert [r["detail"] for r in trail.records][:2] == ["TemporaryOutage", "TemporaryOutage"]

    def test_when_retries_run_out_the_model_gets_an_error_message_not_a_crash(self, desk):
        desk.outages = 10
        agent, _, trail = build([call("lookup_order", {"order_id": "A1001"}), say("The order system is down, sorry.")], desk, max_retries=2)
        result = run(agent)
        assert desk.outages == 7  # 1 try + 2 retries
        assert final_text(result["messages"]).startswith("The order system is down")
        assert trail.records[0]["outcome"] == "error"

    def test_business_refusals_are_not_retried(self, desk):
        lookups = count_lookups(desk)
        agent, _, _ = build([call("lookup_order", {"order_id": "A9999"}), say("not found")], desk, max_retries=3)
        run(agent)
        assert lookups == ["A9999"]

    def test_zero_retries_means_the_first_failure_is_final(self, desk):
        desk.outages = 1
        agent, _, trail = build([call("lookup_order", {"order_id": "A1001"}), say("sorry")], desk, max_retries=0)
        run(agent)
        assert trail.records[0]["outcome"] == "error"


class TestUnexpectedErrors:
    """A bug in a tool is not a business refusal. By default it ends the whole run."""

    def broken_tool(self):
        calls_made = []

        @tool
        def broken() -> str:
            """Always fails with a bug."""
            calls_made.append(1)
            raise RuntimeError("secret internal detail")

        return broken, calls_made

    def test_by_default_a_bug_in_a_tool_ends_the_run(self, desk):
        broken, calls_made = self.broken_tool()
        agent = build_agent(ScriptedChatModel(script=[call("broken"), say("sorry")]), desk, extra_tools=[broken])
        with pytest.raises(RuntimeError, match="secret internal detail"):
            run(agent)
        assert len(calls_made) == 1

    def test_the_error_layer_turns_it_into_a_message_and_the_run_continues(self, desk):
        broken, _ = self.broken_tool()
        agent, _, _ = build([call("broken"), say("A person will follow up.")], desk, extra_tools=[broken])
        result = run(agent)
        assert final_text(result["messages"]) == "A person will follow up."
        assert "failed with RuntimeError" in tool_texts(result)[0]

    def test_the_model_is_not_shown_the_raw_exception_text(self, desk):
        broken, _ = self.broken_tool()
        agent, model, _ = build([call("broken"), say("sorry")], desk, extra_tools=[broken])
        run(agent)
        assert "secret internal detail" not in " ".join(str(m.content) for m in model.seen[1])

    def test_an_unexpected_exception_is_not_retried(self, desk):
        broken, calls_made = self.broken_tool()
        agent, _, _ = build([call("broken"), say("sorry")], desk, max_retries=3, extra_tools=[broken])
        run(agent)
        assert len(calls_made) == 1

    def test_the_audit_records_it_as_an_error(self, desk):
        broken, _ = self.broken_tool()
        agent, _, trail = build([call("broken"), say("sorry")], desk, extra_tools=[broken])
        run(agent)
        assert [(r["tool"], r["outcome"]) for r in trail.records] == [("broken", "error")]

    def test_an_exhausted_retry_names_the_exception_type_but_not_its_message(self, desk):
        desk.outages = 10
        agent, _, _ = build([call("lookup_order", {"order_id": "A1001"}), say("sorry")], desk, max_retries=1)
        message = tool_texts(run(agent))[0]
        assert "TemporaryOutage" in message and "temporarily unavailable" not in message


class TestCallLimit:
    def runaway(self, desk, limit, turns=4):
        script = [call("lookup_order", {"order_id": "A1001"}, call_id=f"c{i}") for i in range(turns)] + [say("giving up")]
        return build(script, desk, tool_call_limit=limit)

    def test_tools_stop_running_after_the_limit(self, desk):
        lookups = count_lookups(desk)
        agent, _, _ = self.runaway(desk, limit=2)
        run(agent)
        assert len(lookups) == 2

    def test_the_model_is_told_why_the_extra_calls_did_nothing(self, desk):
        agent, _, _ = self.runaway(desk, limit=2)
        result = run(agent)
        assert any("limit exceeded" in t.lower() for t in tool_texts(result))
        assert final_text(result["messages"]) == "giving up"

    def test_every_call_gets_an_answer_even_after_the_limit(self, desk):
        agent, _, _ = self.runaway(desk, limit=1)
        result = run(agent)
        asked = {c["id"] for m in result["messages"] if m.type == "ai" for c in m.tool_calls}
        assert asked == {m.tool_call_id for m in result["messages"] if m.type == "tool"}

    def test_the_budget_is_per_run_so_a_new_run_starts_fresh(self, desk):
        lookups = count_lookups(desk)
        for _ in range(2):
            agent, _, _ = self.runaway(desk, limit=2, turns=2)
            run(agent)
        assert len(lookups) == 4


class TestDynamicPrompt:
    def system_text(self, model):
        return model.seen[0][0].content

    def test_the_prompt_is_built_from_the_verified_context(self, desk):
        agent, model, _ = build([say("hi")], desk)
        run(agent, name="Sam", customer="C-100")
        text = self.system_text(model)
        assert "Sam" in text and "C-100" in text and "2026-11-07" in text and "$200.00" in text and SYSTEM_PROMPT in text

    def test_there_is_exactly_one_system_message(self, desk):
        agent, model, _ = build([say("hi")], desk)
        run(agent)
        assert [m.type for m in model.seen[0]].count("system") == 1

    def test_different_customers_get_different_prompts(self, desk):
        a, model_a, _ = build([say("hi")], desk)
        b, model_b, _ = build([say("hi")], Desk())
        run(a, name="Sam", customer="C-100")
        run(b, name="Alex", customer="C-200")
        assert "Alex" in self.system_text(model_b) and "Sam" not in self.system_text(model_b)


class TestAllLayersTogether:
    def hardened(self, script, desk):
        model, trail = ScriptedChatModel(script=script), AuditTrail()
        return build_hardened_agent(model, desk, trail), trail

    def test_a_big_refund_pauses_and_is_audited_only_when_it_runs(self, desk):
        agent, trail = self.hardened([call("issue_refund", {"order_id": "A1001", "amount": 89.0, "reason": "x"}), say("done")], desk)
        paused = run(agent, config={"configurable": {"thread_id": "t"}})
        assert pending_approval(paused) is not None and trail.records == []
        resume(agent, approve(), "t")
        assert [(r["tool"], r["outcome"]) for r in trail.records] == [("issue_refund", "ok")] and len(desk.refunds) == 1

    def test_an_outage_during_an_approved_refund_check_is_absorbed(self, desk):
        agent, trail = self.hardened([call("check_refund_policy", {"order_id": "A1001"}), call("issue_refund", {"order_id": "A1001", "amount": 89.0, "reason": "x"}), say("done")], desk)
        desk.outages = 1
        config = {"configurable": {"thread_id": "t"}}
        paused = run(agent, config=config)
        assert pending_approval(paused) is not None and desk.outages == 0
        done = resume(agent, approve(), "t")
        assert len(desk.refunds) == 1 and final_text(done["messages"]) == "done"

    def test_the_hardened_agent_without_approval_has_no_checkpointer(self, desk):
        agent, _, _ = build([say("hi")], desk)
        assert agent.checkpointer is None

    def test_the_hardened_agent_with_approval_gets_one(self, desk):
        agent, _ = self.hardened([say("hi")], desk)
        assert agent.checkpointer is not None

    def test_an_explicit_none_leaves_the_checkpointer_to_a_parent_graph(self, desk):
        agent = build_hardened_agent(ScriptedChatModel(script=[say("hi")]), desk, AuditTrail(), approval=True, checkpointer=None)
        assert agent.checkpointer is None
