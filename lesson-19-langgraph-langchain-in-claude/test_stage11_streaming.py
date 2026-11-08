import asyncio

import pytest
from langchain_core.messages import AIMessageChunk

from desk import Desk
from fakes import ScriptedChatModel, call, say
from stage7_support_desk import DeskModels, build_desk
from stage8_parallel import build_report_graph
from tools import DeskContext
from stage11_streaming import _event, nested_nodes, report_events, spoken_text, ticket_events, ticket_events_async

FAQ_ANSWER = "Standard shipping takes 3 to 5 business days."


def triage(category, order_id=None, urgency=1):
    return call("Triage", {"category": category, "order_id": order_id, "urgency": urgency, "summary": "s"})


def app_for(triage_turn, agent=(), faq=()):
    models = DeskModels(ScriptedChatModel(script=[triage_turn]), ScriptedChatModel(script=list(agent)), ScriptedChatModel(script=list(faq)))
    return build_desk(models, Desk())


class TestTokensAndNodes:
    @pytest.fixture
    def faq_events(self):
        return list(ticket_events(app_for(triage("faq"), faq=[say(FAQ_ANSWER)]), "How long does shipping take?", thread_id="t"))

    def test_the_tokens_spell_the_final_answer(self, faq_events):
        assert spoken_text(faq_events) == FAQ_ANSWER

    def test_the_answer_arrives_in_pieces_not_all_at_once(self, faq_events):
        assert sum(e["kind"] == "token" for e in faq_events) == len(FAQ_ANSWER.split())

    def test_every_token_comes_from_the_node_that_called_the_model(self, faq_events):
        assert {e["node"] for e in faq_events if e["kind"] == "token"} == {"answer_faq"}

    def test_tokens_arrive_before_their_node_is_reported_finished(self, faq_events):
        kinds = [(e["kind"], e.get("node")) for e in faq_events]
        assert max(i for i, k in enumerate(kinds) if k[0] == "token") < kinds.index(("node", "answer_faq"))

    def test_nodes_are_reported_in_the_order_they_finish(self, faq_events):
        assert [e["node"] for e in faq_events if e["kind"] == "node"] == ["triage", "answer_faq", "finish"]

    def test_a_structured_output_call_shows_no_text_tokens(self, faq_events):
        assert not [e for e in faq_events if e["kind"] == "token" and e["node"] == "triage"]

    def test_a_message_written_by_code_arrives_whole_not_word_by_word(self):
        # "messages" mode streams model tokens AND any message a node returns. The escalation note is written by code,
        # so it comes as ONE event from the escalate node.
        events = list(ticket_events(app_for(triage("other", urgency=5)), "I am furious!", thread_id="t"))
        tokens = [e for e in events if e["kind"] == "token"]
        assert len(tokens) == 1 and tokens[0]["node"] == "escalate" and tokens[0]["text"].startswith("I have passed this to a person.")
        assert [e["node"] for e in events if e["kind"] == "node"] == ["triage", "escalate", "finish"]


class TestPauses:
    def test_a_refund_that_needs_a_person_ends_the_stream_with_a_pause(self):
        agent = [call("issue_refund", {"order_id": "A1001", "amount": 89.0, "reason": "x"}), say("Done.")]
        events = list(ticket_events(app_for(triage("refund", "A1001", 2), agent), "Refund order A1001 please.", thread_id="t"))
        assert events[-1] == {"kind": "pause"} and spoken_text(events) == ""


class TestCustomEvents:
    def test_each_worker_reports_progress(self, desk):
        events = list(report_events(desk, "C-100", ["A1001", "A1002", "A1003"]))
        assert sorted(e["checked"] for e in events if e["kind"] == "progress") == ["A1001", "A1002", "A1003"]

    def test_the_collect_step_comes_last(self, desk):
        events = list(report_events(desk, "C-100", ["A1001", "A1002"]))
        assert events[-1] == {"kind": "node", "node": "collect"}

    def test_every_progress_event_happens_before_collect(self, desk):
        events = list(report_events(desk, "C-100", ["A1001", "A1002"]))
        assert all(e["kind"] != "progress" for e in events[events.index({"kind": "node", "node": "collect"}):])


class TestSubgraphs:
    def agent_app(self):
        return app_for(triage("refund", "A1001", 2), [call("issue_refund", {"order_id": "A1001", "amount": 89.0, "reason": "x"}), say("Done.")])

    def test_without_subgraphs_the_inner_agent_is_invisible(self):
        app = self.agent_app()
        seen = []
        for chunk in app.stream({"messages": [("user", "Refund order A1001 please.")]}, {"configurable": {"thread_id": "t"}}, stream_mode="updates",
                                context=DeskContext("C-100")):
            seen.extend(chunk)
        assert "model" not in seen and "triage" in seen

    def test_with_subgraphs_the_inner_nodes_appear_under_their_parent(self):
        pairs = nested_nodes(self.agent_app(), "Refund order A1001 please.", thread_id="t")
        assert ("", "triage") in pairs and ("refund_flow", "model") in pairs


class TestValuesMode:
    def test_the_last_values_chunk_is_the_final_state(self, desk):
        app = build_report_graph(desk)
        payload = {"customer_id": "C-100", "order_ids": ["A1001"], "results": []}
        chunks = list(app.stream(payload, stream_mode="values"))
        assert chunks[-1] == app.invoke(payload) and len(chunks) > 2

    def test_values_chunks_grow_as_the_run_goes_on(self, desk):
        chunks = list(build_report_graph(desk).stream({"customer_id": "C-100", "order_ids": ["A1001"], "results": []}, stream_mode="values"))
        assert "report" not in chunks[0] and "report" in chunks[-1]


class TestAsync:
    def test_the_async_stream_gives_the_same_events(self):
        async def collect():
            app = app_for(triage("faq"), faq=[say(FAQ_ANSWER)])
            return [e async for e in ticket_events_async(app, "How long does shipping take?", thread_id="t")]

        sync_events = list(ticket_events(app_for(triage("faq"), faq=[say(FAQ_ANSWER)]), "How long does shipping take?", thread_id="t"))
        assert asyncio.run(collect()) == sync_events


class TestEventTranslation:
    def test_an_interrupt_update_becomes_a_pause(self):
        assert _event("updates", {"__interrupt__": ()}) == {"kind": "pause"}

    def test_a_chunk_with_no_text_is_not_shown(self):
        assert _event("messages", (AIMessageChunk(content=""), {})) is None

    def test_an_unknown_mode_is_ignored(self):
        assert _event("debug", {"anything": 1}) is None
