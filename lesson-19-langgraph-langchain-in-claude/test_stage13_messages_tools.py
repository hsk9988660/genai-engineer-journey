import pytest
from langchain.agents import create_agent
from langchain.messages import RemoveMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from conftest import tool_texts
from fakes import ScriptedChatModel, call, say
from stage1_first_agent import ask, build_agent
from stage13_messages_tools import RefundArgs, build_demo_agent, make_demo_tools, naive_window, run, safe_window, trim_history
from tools import DeskContext


def history():
    return [SystemMessage("rules", id="s"), HumanMessage("q1", id="h1"),
            AIMessage("", id="a1", tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}]),
            ToolMessage("r1", tool_call_id="c1", id="t1"), AIMessage("a2", id="a2"), HumanMessage("q2", id="h2"), AIMessage("a3", id="a3")]


def orphans(messages):
    """Tool results that have no earlier AI message asking for them."""
    asked = set()
    bad = []
    for message in messages:
        if isinstance(message, AIMessage):
            asked |= {c["id"] for c in message.tool_calls}
        if isinstance(message, ToolMessage) and message.tool_call_id not in asked:
            bad.append(message.id)
    return bad


class TestMessages:
    def test_text_and_content_blocks(self):
        message = AIMessage("hello")
        assert message.text == "hello" and message.content_blocks == [{"type": "text", "text": "hello"}]

    def test_text_joins_blocks(self):
        assert AIMessage(content=[{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]).text == "hi there"

    def test_a_tool_call_is_data_on_the_message(self):
        message = call("lookup_order", {"order_id": "A1001"}, call_id="x")
        assert message.tool_calls == [{"name": "lookup_order", "args": {"order_id": "A1001"}, "id": "x", "type": "tool_call"}]

    def test_usage_numbers_travel_on_the_message(self):
        assert say("hi", tokens=(12, 3)).usage_metadata == {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}

    def test_remove_message_deletes_by_id(self):
        assert [m.id for m in add_messages(history(), [RemoveMessage(id="h1"), RemoveMessage(id="a1")])] == ["s", "t1", "a2", "h2", "a3"]


class TestKeepingHistoryValid:
    def test_a_plain_slice_can_start_with_an_orphan_tool_result(self):
        assert naive_window(history(), 4)[0].id == "t1" and orphans(naive_window(history(), 4)) == ["t1"]

    def test_the_safe_window_never_leaves_an_orphan(self):
        for keep in range(2, 8):
            assert orphans(safe_window(history(), keep)) == [], keep

    def test_the_safe_window_keeps_the_system_message_and_starts_on_a_person(self):
        kept = safe_window(history(), 4)
        assert kept[0].type == "system" and kept[1].type == "human"

    def test_a_large_window_keeps_everything(self):
        assert [m.id for m in safe_window(history(), 100)] == [m.id for m in history()]

    def test_the_trimming_middleware_shrinks_what_the_model_sees_and_what_is_saved(self, desk):
        model = ScriptedChatModel(script=[say("ok")])
        agent = build_demo_agent(model, desk, middleware=[trim_history(keep=3)])
        past = [m for m in history() if m.id != "s"]
        result = agent.invoke({"messages": [*past, HumanMessage("q3", id="h3")]}, context=DeskContext("C-100"))
        assert orphans(model.seen[0]) == [] and len(model.seen[0]) <= 3 and model.seen[0][0].type == "human"
        assert len(result["messages"]) < len(past) + 2

    def test_a_short_history_is_left_alone(self, desk):
        model = ScriptedChatModel(script=[say("ok")])
        result = run(build_demo_agent(model, desk, middleware=[trim_history(keep=10)]), "hi")
        assert [m.type for m in result["messages"]] == ["human", "ai"]


class TestSchemaIsEnforcedBeforeTheFunctionRuns:
    ARGS = {"order_id": "A1001", "amount": 50.0, "reason": "too small"}

    def run_refund(self, desk, **changes):
        model = ScriptedChatModel(script=[call("refund_with_exception", {**self.ARGS, **changes}), say("ok")])
        return run(build_demo_agent(model, desk), "refund")

    def test_the_limits_are_part_of_what_the_model_is_shown(self, desk):
        properties = make_demo_tools(desk)[0].tool_call_schema.model_json_schema()["properties"]
        assert properties["amount"]["exclusiveMinimum"] == 0 and properties["amount"]["maximum"] == 200
        assert properties["order_id"]["pattern"] == r"^[A-Z]\d{4}$"

    def test_the_runtime_argument_is_not_in_the_schema(self, desk):
        assert set(make_demo_tools(desk)[0].tool_call_schema.model_json_schema()["properties"]) == {"order_id", "amount", "reason"}

    @pytest.mark.parametrize("change", [{"amount": 500.0}, {"amount": 0}, {"amount": -5}, {"order_id": "a1001"}, {"order_id": "1001"}, {"reason": "x"}])
    def test_a_call_outside_the_limits_never_reaches_the_function(self, desk, change):
        result = self.run_refund(desk, **change)
        tool_message = next(m for m in result["messages"] if m.type == "tool")
        assert tool_message.status == "error" and desk.refunds == []

    def test_the_same_limits_exist_as_a_pydantic_model(self):
        with pytest.raises(ValueError):
            RefundArgs(order_id="A1001", amount=201, reason="ok ok")
        assert RefundArgs(order_id="A1001", amount=200, reason="ok ok").amount == 200


class TestTwoStylesOfSayingNo:
    def test_a_raised_tool_exception_is_flagged_as_an_error(self, desk):
        model = ScriptedChatModel(script=[call("refund_with_exception", {"order_id": "A1003", "amount": 50.0, "reason": "too old"}), say("ok")])
        tool_message = next(m for m in run(build_demo_agent(model, desk), "refund")["messages"] if m.type == "tool")
        assert tool_message.status == "error" and "return window" in tool_message.content

    def test_a_returned_error_dictionary_is_a_normal_result_that_says_no(self, desk):
        model = ScriptedChatModel(script=[call("issue_refund", {"order_id": "A1003", "amount": 50.0, "reason": "too old"}), say("ok")])
        tool_message = next(m for m in ask(build_agent(model, desk), "refund")["messages"] if m.type == "tool")
        assert tool_message.status == "success" and "return window" in tool_message.content

    def test_a_valid_refund_goes_through(self, desk):
        model = ScriptedChatModel(script=[call("refund_with_exception", {"order_id": "A1001", "amount": 50.0, "reason": "too small"}), say("ok")])
        result = run(build_demo_agent(model, desk), "refund")
        assert tool_texts(result) == ["refunded $50.00 on A1001"] and desk.refunds[0].amount_cents == 5000

    def test_the_schema_limit_and_the_desk_limit_are_separate_layers(self, desk):
        model = ScriptedChatModel(script=[call("refund_with_exception", {"order_id": "A1002", "amount": 200.0, "reason": "defect"}), say("ok")])
        run(build_demo_agent(model, desk), "refund")
        assert desk.refunds[0].amount_cents == 20_000


class TestArtifacts:
    def receipt_message(self, desk, order_id="A1001", customer="C-100"):
        model = ScriptedChatModel(script=[call("order_receipt", {"order_id": order_id}), say("ok")])
        result = run(build_demo_agent(model, desk), "receipt", customer)
        return next(m for m in result["messages"] if m.type == "tool")

    def test_the_model_gets_one_line_and_the_app_gets_the_payload(self, desk):
        message = self.receipt_message(desk)
        assert message.content == "Trail Shoes, $89.00"
        assert message.artifact == {"order_id": "A1001", "item": "Trail Shoes", "total_cents": 8900, "refunded_cents": 0}

    def test_the_payload_is_not_in_the_text_the_model_reads(self, desk):
        assert "total_cents" not in self.receipt_message(desk).content

    def test_another_customers_receipt_is_an_error_with_no_details(self, desk):
        message = self.receipt_message(desk, "A1004")
        assert message.status == "error" and "Sleeping Bag" not in message.content and message.artifact is None


class TestToolsThatWriteState:
    def test_a_tool_can_write_a_key_in_the_agent_state(self, desk):
        model = ScriptedChatModel(script=[call("flag_for_followup", {"reason": "wants a call"}), say("ok")])
        result = run(build_demo_agent(model, desk), "please flag")
        assert result["followup"] == "wants a call" and tool_texts(result) == ["A follow-up was requested: wants a call"]

    def test_without_the_tool_the_key_stays_empty(self, desk):
        result = run(build_demo_agent(ScriptedChatModel(script=[say("hi")]), desk), "hello")
        assert "followup" not in result

    def test_a_tool_can_read_the_state_and_the_model_cannot_see_that_argument(self, desk):
        tool = make_demo_tools(desk)[3]
        assert tool.tool_call_schema.model_json_schema()["properties"] == {}
        model = ScriptedChatModel(script=[call("messages_so_far"), say("ok")])
        assert tool_texts(run(build_demo_agent(model, desk), "how long"))[0] == "2 messages"


class TestWhatTheModelIsToldAbout:
    def test_the_bound_tools_are_exactly_the_agents_tools(self, desk):
        model = ScriptedChatModel(script=[say("hi")])
        run(build_demo_agent(model, desk), "hello")
        assert model.bound[0]["tools"] == ["refund_with_exception", "order_receipt", "flag_for_followup", "messages_so_far"]

    def test_a_response_format_adds_one_more_tool_and_forces_a_tool_call(self, desk):
        class Answer(BaseModel):
            ok: bool

        model = ScriptedChatModel(script=[call("Answer", {"ok": True})])
        agent = create_agent(model, make_demo_tools(desk), response_format=Answer, context_schema=DeskContext)
        result = agent.invoke({"messages": [HumanMessage("x")]}, context=DeskContext("C-100"))
        assert model.bound[0]["tools"][-1] == "Answer" and model.bound[0]["kwargs"] == {"tool_choice": "any"} and result["structured_response"].ok
