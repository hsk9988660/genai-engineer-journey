import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from conftest import tool_texts
from fakes import ScriptedChatModel, call, say
from stage1_first_agent import ask, build_agent
from stage4_memory import build_memory_agent, checkpoint_history, recall_preferences, remember_preference, thread_summary


def agent_for(script, desk, **kwargs):
    model = ScriptedChatModel(script=script)
    return build_memory_agent(model, desk, **kwargs), model


def texts(messages):
    return [str(m.content) for m in messages]


class TestShortTermMemory:
    SCRIPT = [call("lookup_order", {"order_id": "A1001"}), say("Trail Shoes."), say("They cost $89.00.")]

    def test_the_same_thread_remembers_the_earlier_turn(self, desk):
        agent, model = agent_for(self.SCRIPT, desk)
        ask(agent, "What is in A1001?", thread_id="t1")
        ask(agent, "How much was that?", thread_id="t1")
        third_call = texts(model.seen[2])
        assert "What is in A1001?" in third_call and "Trail Shoes." in third_call and "How much was that?" in third_call
        assert any("Trail Shoes" in t and "delivered" in t for t in third_call)

    def test_another_thread_starts_fresh(self, desk):
        agent, model = agent_for(self.SCRIPT, desk)
        ask(agent, "What is in A1001?", thread_id="t1")
        ask(agent, "Hello there", thread_id="t2")
        assert [m.type for m in model.seen[2]] == ["system", "human"]
        assert not any("A1001" in t for t in texts(model.seen[2]))

    def test_an_agent_with_a_checkpointer_refuses_to_run_without_a_thread_id(self, desk):
        agent, _ = agent_for([say("one")], desk)
        with pytest.raises(ValueError, match="thread_id"):
            ask(agent, "first")

    def test_an_agent_without_a_checkpointer_remembers_nothing(self, desk):
        model = ScriptedChatModel(script=[say("one"), say("two")])
        agent = build_agent(model, desk)
        ask(agent, "first")
        ask(agent, "second")
        assert not any("first" in t for t in texts(model.seen[1]))

    def test_a_finished_run_has_nothing_next(self, desk):
        agent, _ = agent_for([say("hi")], desk)
        ask(agent, "hello", thread_id="t1")
        assert thread_summary(agent, "t1") == {"messages": 2, "next": ()}

    def test_the_history_lists_every_saved_step_oldest_first(self, desk):
        agent, _ = agent_for([call("lookup_order", {"order_id": "A1001"}), say("done")], desk)
        ask(agent, "go", thread_id="t1")
        counts = [c["messages"] for c in checkpoint_history(agent, "t1")]
        assert counts == sorted(counts) and counts[0] == 0 and counts[-1] == 4

    def test_the_history_shows_which_node_ran_next_at_each_step(self, desk):
        agent, _ = agent_for([call("lookup_order", {"order_id": "A1001"}), say("done")], desk)
        ask(agent, "go", thread_id="t1")
        nexts = [c["next"] for c in checkpoint_history(agent, "t1")]
        assert ("tools",) in nexts and nexts[-1] == ()

    def test_a_shared_checkpointer_keeps_threads_apart_across_agent_objects(self, desk):
        saver = InMemorySaver()
        first, _ = agent_for([say("a")], desk, checkpointer=saver)
        ask(first, "remember me", thread_id="t1")
        second, model = agent_for([say("b")], desk, checkpointer=saver)
        ask(second, "again", thread_id="t1")
        assert "remember me" in texts(model.seen[0])


class TestLongTermMemory:
    def save_then_recall(self, desk, store, customer, note="contact me by email only"):
        saver, _ = agent_for([call("remember_preference", {"preference": note}), say("saved")], desk, store=store)
        ask(saver, "Please remember: " + note, customer_id=customer, thread_id=f"w-{customer}")
        reader, _ = agent_for([call("recall_preferences"), say("ok")], desk, store=store)
        return tool_texts(ask(reader, "What do you know about me?", customer_id=customer, thread_id=f"r-{customer}"))[0]

    def test_a_saved_preference_can_be_recalled_in_another_conversation(self, desk):
        assert "contact me by email only" in self.save_then_recall(desk, InMemoryStore(), "C-100")

    def test_another_customer_sees_nothing(self, desk):
        store = InMemoryStore()
        self.save_then_recall(desk, store, "C-100")
        reader, _ = agent_for([call("recall_preferences"), say("ok")], desk, store=store)
        assert tool_texts(ask(reader, "What do you know about me?", customer_id="C-200", thread_id="r-200"))[0] == "[]"

    def test_the_tools_do_not_take_a_customer_id(self):
        for t in (remember_preference, recall_preferences):
            assert "customer_id" not in t.tool_call_schema.model_json_schema()["properties"]

    def test_a_customer_id_invented_by_the_model_is_ignored(self, desk):
        store = InMemoryStore()
        agent, _ = agent_for([call("remember_preference", {"preference": "x", "customer_id": "C-200"}), say("saved")], desk, store=store)
        ask(agent, "save", customer_id="C-100", thread_id="w")
        assert len(store.search(("preferences", "C-100"))) == 1 and store.search(("preferences", "C-200")) == []

    def test_a_long_preference_is_clipped(self, desk):
        store = InMemoryStore()
        agent, _ = agent_for([call("remember_preference", {"preference": "y" * 500}), say("saved")], desk, store=store)
        ask(agent, "save", thread_id="w")
        assert len(store.search(("preferences", "C-100"))[0].value["text"]) == 200

    def test_several_preferences_are_all_recalled_in_order(self, desk):
        store = InMemoryStore()
        script = [call("remember_preference", {"preference": "b second"}, call_id="1"), call("remember_preference", {"preference": "a first"}, call_id="2"), say("ok")]
        agent, _ = agent_for(script, desk, store=store)
        ask(agent, "save two", thread_id="w")
        reader, _ = agent_for([call("recall_preferences"), say("ok")], desk, store=store)
        assert json.loads(tool_texts(ask(reader, "recall", thread_id="r"))[0]) == ["a first", "b second"]
