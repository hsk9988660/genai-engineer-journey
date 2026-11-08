import json

import pytest
from langgraph.errors import GraphRecursionError

from conftest import kinds, tool_texts
from fakes import ScriptedChatModel, ScriptExhausted, call, calls, say
from show import final_text, format_messages
from stage1_first_agent import SYSTEM_PROMPT, ask, build_agent
from tools import DeskContext, make_tools


def agent_for(script, desk):
    model = ScriptedChatModel(script=script)
    return build_agent(model, desk), model


class TestTheLoop:
    def test_one_tool_round_trip(self, desk):
        agent, model = agent_for([call("lookup_order", {"order_id": "A1001"}), say("It is Trail Shoes.")], desk)
        result = ask(agent, "What is in A1001?")
        assert kinds(result) == ["human", "ai", "tool", "ai"]
        assert final_text(result["messages"]) == "It is Trail Shoes."
        assert len(model.seen) == 2

    def test_tool_result_is_sent_back_to_the_model(self, desk):
        agent, model = agent_for([call("lookup_order", {"order_id": "A1001"}), say("ok")], desk)
        ask(agent, "What is in A1001?")
        second_call = model.seen[1]
        assert second_call[-1].type == "tool" and json.loads(second_call[-1].content)["item"] == "Trail Shoes"

    def test_the_model_sees_the_system_prompt_first(self, desk):
        agent, model = agent_for([say("hi")], desk)
        ask(agent, "hello")
        assert (model.seen[0][0].type, model.seen[0][0].content) == ("system", SYSTEM_PROMPT)

    def test_no_tool_call_means_the_run_ends_after_one_model_call(self, desk):
        agent, model = agent_for([say("Hello!")], desk)
        assert kinds(ask(agent, "hi")) == ["human", "ai"] and len(model.seen) == 1

    def test_several_tool_calls_in_one_turn_each_get_their_own_result(self, desk):
        script = [calls(("lookup_order", {"order_id": "A1001"}, "x1"), ("lookup_order", {"order_id": "A1002"}, "x2")), say("both")]
        agent, _ = agent_for(script, desk)
        result = ask(agent, "two orders")
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert [m.tool_call_id for m in tool_messages] == ["x1", "x2"]
        assert [json.loads(m.content)["item"] for m in tool_messages] == ["Trail Shoes", "Camp Stove"]

    def test_the_loop_can_chain_several_tools(self, desk):
        script = [call("lookup_order", {"order_id": "A1003"}), call("check_refund_policy", {"order_id": "A1003"}), say("Too old.")]
        agent, model = agent_for(script, desk)
        result = ask(agent, "refund A1003?")
        assert kinds(result) == ["human", "ai", "tool", "ai", "tool", "ai"] and len(model.seen) == 3

    def test_a_runaway_loop_is_stopped_by_the_recursion_limit(self, desk):
        agent, _ = agent_for([call("search_faq", {"query": "x"}, call_id=f"c{i}") for i in range(30)], desk)
        with pytest.raises(GraphRecursionError):
            agent.invoke({"messages": [{"role": "user", "content": "loop"}]}, {"recursion_limit": 6}, context=DeskContext("C-100"))


    def test_the_default_step_limit_is_far_above_the_old_25(self, desk):
        script = [call("search_faq", {"query": "x"}, call_id=f"c{i}") for i in range(40)] + [say("finally")]
        agent, _ = agent_for(script, desk)
        assert final_text(ask(agent, "loop")["messages"]) == "finally"


class TestToolsAreSafeToExpose:
    def test_the_model_never_sees_the_runtime_argument(self, desk):
        shown = {t.name: set(t.tool_call_schema.model_json_schema()["properties"]) for t in make_tools(desk)}
        assert shown == {"lookup_order": {"order_id"}, "check_refund_policy": {"order_id"},
                         "issue_refund": {"order_id", "amount", "reason"}, "search_faq": {"query"}}

    def test_every_tool_has_a_description_for_the_model(self, desk):
        assert all(len(t.description) > 40 for t in make_tools(desk))

    def test_domain_errors_come_back_as_data_not_crashes(self, desk):
        agent, _ = agent_for([call("lookup_order", {"order_id": "A9999"}), say("I could not find it.")], desk)
        result = ask(agent, "find A9999")
        assert "error" in json.loads(tool_texts(result)[0]) and final_text(result["messages"]) == "I could not find it."

    def test_a_customer_cannot_read_another_customers_order(self, desk):
        agent, _ = agent_for([call("lookup_order", {"order_id": "A1004"}), say("Not found.")], desk)
        result = ask(agent, "show A1004", customer_id="C-100")
        assert "Sleeping Bag" not in tool_texts(result)[0] and "error" in json.loads(tool_texts(result)[0])

    def test_the_same_order_is_visible_to_its_owner(self, desk):
        agent, _ = agent_for([call("lookup_order", {"order_id": "A1004"}), say("ok")], desk)
        assert "Sleeping Bag" in tool_texts(ask(agent, "show A1004", customer_id="C-200"))[0]

    def test_the_model_cannot_pass_a_customer_id_of_its_own(self, desk):
        script = [call("lookup_order", {"order_id": "A1004", "customer_id": "C-200"}), say("done")]
        agent, _ = agent_for(script, desk)
        result = ask(agent, "show A1004", customer_id="C-100")
        assert "Sleeping Bag" not in " ".join(tool_texts(result))

    def test_a_tool_the_model_invented_becomes_an_error_message_and_the_run_continues(self, desk):
        agent, _ = agent_for([call("delete_all_orders"), say("Sorry, I cannot do that.")], desk)
        result = ask(agent, "wipe everything")
        assert "not a valid tool" in tool_texts(result)[0] and final_text(result["messages"]) == "Sorry, I cannot do that."

    def test_issue_refund_uses_the_tool_call_id_as_its_idempotency_key(self, desk):
        script = [call("issue_refund", {"order_id": "A1001", "amount": 10.0, "reason": "x"}, call_id="same-id"), say("done")]
        for _ in range(2):
            agent, _model = agent_for(list(script), desk)
            ask(agent, "refund")
        assert len(desk.refunds) == 1 and desk.refunds[0].key == "same-id"

    def test_inside_a_thread_the_key_also_carries_the_thread_id(self, desk):
        from langgraph.checkpoint.memory import InMemorySaver
        model = ScriptedChatModel(script=[call("issue_refund", {"order_id": "A1001", "amount": 10.0, "reason": "x"}, call_id="same-id"), say("done")])
        ask(build_agent(model, desk, checkpointer=InMemorySaver()), "refund", thread_id="t9")
        assert desk.refunds[0].key == "t9:same-id"

    def test_the_same_tool_call_id_in_two_threads_does_not_collide(self, desk):
        from langgraph.checkpoint.memory import InMemorySaver
        saver = InMemorySaver()
        for thread, order in (("ta", "A1001"), ("tb", "A1002")):
            model = ScriptedChatModel(script=[call("issue_refund", {"order_id": order, "amount": 10.0, "reason": "x"}, call_id="dup"), say("done")])
            ask(build_agent(model, desk, checkpointer=saver), "refund", thread_id=thread)
        assert sorted(r.key for r in desk.refunds) == ["ta:dup", "tb:dup"]

    def test_a_bad_amount_is_refused_by_the_desk_and_reported_as_data(self, desk):
        agent, _ = agent_for([call("issue_refund", {"order_id": "A1001", "amount": 1.005, "reason": "x"}), say("done")], desk)
        result = ask(agent, "refund")
        assert "decimal places" in tool_texts(result)[0] and desk.refunds == []


class TestHelpers:
    def test_ask_with_a_thread_id_builds_a_config(self, desk):
        from langgraph.checkpoint.memory import InMemorySaver
        model = ScriptedChatModel(script=[say("hi")])
        agent = build_agent(model, desk, checkpointer=InMemorySaver())
        ask(agent, "hello", thread_id="t1")
        assert agent.get_state({"configurable": {"thread_id": "t1"}}).values["messages"][-1].content == "hi"

    def test_format_messages_shows_each_step(self, desk):
        agent, _ = agent_for([call("lookup_order", {"order_id": "A1001"}), say("Done.")], desk)
        lines = format_messages(ask(agent, "What is A1001?")["messages"])
        assert lines[0].startswith("customer :") and "asks for lookup_order(order_id='A1001')" in lines[1]
        assert lines[2].startswith("tool     :") and lines[3] == "model    : Done."

    def test_final_text_is_empty_when_the_model_never_spoke(self):
        assert final_text([]) == ""


class TestTheFakeModel:
    def test_running_out_of_script_is_a_loud_error(self, desk):
        agent, _ = agent_for([call("search_faq", {"query": "x"})], desk)
        with pytest.raises(ScriptExhausted):
            ask(agent, "go")

    def test_the_fake_records_everything_it_was_sent(self, desk):
        agent, model = agent_for([say("a")], desk)
        ask(agent, "question")
        assert [m.type for m in model.seen[0]] == ["system", "human"]
