import json
import operator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from conftest import kinds, tool_texts
from desk import Desk
from fakes import ScriptedChatModel, call, say
from show import final_text
from stage1_first_agent import ask, build_agent
from stage2_graph_by_hand import build_graph, run, shared_reducer_gotcha

LOOKUP_THEN_ANSWER = [call("lookup_order", {"order_id": "A1001"}), say("Trail Shoes.")]


def graph_for(script, desk, **kwargs):
    model = ScriptedChatModel(script=script)
    return build_graph(model, desk, **kwargs), model


class TestTheGraphIsTheAgent:
    def test_it_has_the_expected_nodes_and_edges(self, desk):
        graph, _ = graph_for([say("x")], desk)
        drawn = graph.get_graph()
        assert set(drawn.nodes) == {"__start__", "model", "tools", "give_up", "__end__"}
        edges = {(e.source, e.target) for e in drawn.edges}
        assert {("__start__", "model"), ("tools", "model"), ("give_up", "__end__"), ("model", "tools"), ("model", "give_up"), ("model", "__end__")} <= edges

    def test_it_gives_the_same_conversation_as_create_agent(self, desk):
        by_hand, _ = graph_for(list(LOOKUP_THEN_ANSWER), desk)
        packaged = build_agent(ScriptedChatModel(script=list(LOOKUP_THEN_ANSWER)), Desk())
        a, b = run(by_hand, "What is A1001?"), ask(packaged, "What is A1001?")
        assert kinds(a) == kinds(b)
        assert tool_texts(a) == tool_texts(b) and final_text(a["messages"]) == final_text(b["messages"])

    def test_the_tools_node_gets_the_verified_context(self, desk):
        graph, _ = graph_for([call("lookup_order", {"order_id": "A1004"}), say("x")], desk)
        assert "Sleeping Bag" in tool_texts(run(graph, "show", customer_id="C-200"))[0]
        graph, _ = graph_for([call("lookup_order", {"order_id": "A1004"}), say("x")], Desk())
        assert "Sleeping Bag" not in tool_texts(run(graph, "show", customer_id="C-100"))[0]


class TestState:
    def test_a_reducer_key_accumulates_across_nodes(self, desk):
        script = [call("lookup_order", {"order_id": "A1001"}), call("check_refund_policy", {"order_id": "A1001"}), say("done")]
        graph, _ = graph_for(script, desk)
        assert run(graph, "go")["tools_used"] == ["lookup_order", "check_refund_policy"]

    def test_a_plain_key_is_overwritten_with_the_latest_value(self, desk):
        script = [call("lookup_order", {"order_id": "A1001"}), call("search_faq", {"query": "x"}), say("done")]
        graph, _ = graph_for(script, desk)
        assert run(graph, "go")["model_calls"] == 3

    def test_add_messages_appends(self):
        merged = add_messages([HumanMessage("a", id="1")], [AIMessage("b", id="2")])
        assert [m.content for m in merged] == ["a", "b"]

    def test_add_messages_replaces_a_message_with_the_same_id(self):
        merged = add_messages([HumanMessage("old", id="1")], [HumanMessage("new", id="1")])
        assert [m.content for m in merged] == ["new"]

    def test_operator_add_concatenates_lists_and_never_deduplicates(self):
        assert operator.add(["a"], ["a", "b"]) == ["a", "a", "b"]

    def test_state_is_saved_per_thread_when_a_checkpointer_is_given(self, desk):
        graph, _ = graph_for(LOOKUP_THEN_ANSWER, desk, checkpointer=InMemorySaver())
        run(graph, "go", thread_id="t1")
        saved = graph.get_state({"configurable": {"thread_id": "t1"}}).values
        assert saved["tools_used"] == ["lookup_order"] and saved["model_calls"] == 2


class TestTheStepCap:
    def runaway(self, desk, cap):
        script = [call("issue_refund", {"order_id": "A1001", "amount": 5.0, "reason": "x"}, call_id=f"c{i}") for i in range(10)]
        graph, model = graph_for(script, desk, max_model_calls=cap)
        return run(graph, "loop"), model

    def test_the_model_is_called_at_most_the_cap_times(self, desk):
        result, model = self.runaway(desk, cap=3)
        assert len(model.seen) == 3 and result["model_calls"] == 3

    def test_the_run_ends_with_a_stop_message(self, desk):
        result, _ = self.runaway(desk, cap=2)
        assert final_text(result["messages"]).startswith("Stopped:")

    def test_the_unanswered_tool_call_is_closed_and_not_executed(self, desk):
        result, _ = self.runaway(desk, cap=2)
        assert result["messages"][-2].type == "tool" and result["messages"][-2].status == "error"
        assert len(desk.refunds) == 1  # only the first call ran. The one that hit the cap did not.

    def test_every_tool_call_in_the_history_has_a_result(self, desk):
        result, _ = self.runaway(desk, cap=4)
        asked = {c["id"] for m in result["messages"] if m.type == "ai" for c in m.tool_calls}
        answered = {m.tool_call_id for m in result["messages"] if m.type == "tool"}
        assert asked == answered

    def test_a_finishing_run_never_reaches_give_up(self, desk):
        graph, _ = graph_for(LOOKUP_THEN_ANSWER, desk, max_model_calls=2)
        assert not final_text(run(graph, "go")["messages"]).startswith("Stopped:")


class TestTheSharedReducerTrap:
    def test_a_subgraph_that_shares_an_additive_key_double_counts(self):
        naive, _ = shared_reducer_gotcha()
        assert naive == ["before", "before", "sub did work"]

    def test_starting_the_subgraph_empty_fixes_it(self):
        _, fixed = shared_reducer_gotcha()
        assert fixed == ["before", "sub did work"]


def test_tool_results_are_valid_json(desk):
    graph, _ = graph_for(LOOKUP_THEN_ANSWER, desk)
    assert json.loads(tool_texts(run(graph, "go"))[0])["order_id"] == "A1001"
