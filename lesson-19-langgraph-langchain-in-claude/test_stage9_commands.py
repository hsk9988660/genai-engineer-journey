import pytest

from stage9_commands import (MAX_QUESTIONS, NoPendingQuestion, build_clarify_graph, pending_question, reply, side_effect_runs, start)
from tools import DeskContext


@pytest.fixture
def app(desk):
    return build_clarify_graph(desk)


class TestRoutingWithCommand:
    def test_an_id_in_the_message_skips_the_question(self, app):
        result = start(app, "Can I return the shoes from order A1001?", thread_id="t")
        assert pending_question(result) is None and result["outcome"]["status"] == "quote" and result["outcome"]["refundable"] == "$89.00"

    def test_without_an_id_the_graph_asks(self, app):
        result = start(app, "I want to return my shoes.", thread_id="t")
        assert pending_question(result) == {"question": "Which order do you mean? For example A1001.", "attempt": 1}

    def test_the_pause_is_saved_with_the_waiting_node(self, app):
        start(app, "I want a refund", thread_id="t")
        assert app.get_state({"configurable": {"thread_id": "t"}}).next == ("clarify",)

    def test_the_drawn_graph_shows_every_route_a_command_can_take(self, app):
        edges = {(e.source, e.target) for e in app.get_graph().edges}
        assert {("find_order", "quote"), ("find_order", "clarify"), ("clarify", "quote"), ("clarify", "clarify"), ("clarify", "handoff")} <= edges


class TestTheConversation:
    def test_a_good_answer_is_accepted_whatever_its_case_or_spaces(self, app):
        start(app, "refund please", thread_id="t")
        result = reply(app, "  a1001 ", thread_id="t")
        assert result["outcome"]["order_id"] == "A1001" and pending_question(result) is None

    def test_a_bad_answer_is_asked_again_with_the_attempt_number(self, app):
        start(app, "refund please", thread_id="t")
        second = reply(app, "the blue ones", thread_id="t")
        assert pending_question(second)["attempt"] == 2

    def test_a_bad_answer_then_a_good_one_gets_a_quote(self, app):
        start(app, "refund please", thread_id="t")
        reply(app, "no idea", thread_id="t")
        result = reply(app, "A1002", thread_id="t")
        assert result["outcome"]["status"] == "quote" and "bad answer 1" in result["log"] and "customer chose A1002" in result["log"]

    def test_after_the_last_bad_answer_a_person_takes_over(self, app):
        start(app, "refund please", thread_id="t")
        for _ in range(MAX_QUESTIONS - 1):
            assert pending_question(reply(app, "nope", thread_id="t")) is not None
        result = reply(app, "still nope", thread_id="t")
        assert pending_question(result) is None and result["outcome"]["status"] == "handoff"

    def test_another_customers_order_id_counts_as_a_bad_answer(self, app):
        start(app, "refund please", customer_id="C-100", thread_id="t")
        assert pending_question(reply(app, "A1004", customer_id="C-100", thread_id="t"))["attempt"] == 2

    def test_an_id_in_the_message_that_is_not_theirs_gives_not_found_and_no_details(self, app):
        result = start(app, "refund order A1004", customer_id="C-100", thread_id="t")
        assert result["outcome"]["status"] == "not_found" and "Sleeping Bag" not in str(result)

    def test_two_threads_wait_independently(self, app):
        start(app, "refund", thread_id="one")
        start(app, "refund", thread_id="two")
        second = reply(app, "A1002", thread_id="two")
        first = reply(app, "A1001", thread_id="one")
        assert (first["outcome"]["order_id"], second["outcome"]["order_id"]) == ("A1001", "A1002")


class TestSchemas:
    def test_the_caller_gets_only_the_output_schema(self, app):
        assert set(start(app, "order A1001", thread_id="t")) == {"outcome", "log"}

    def test_a_caller_cannot_pre_seed_a_working_key(self, app):
        result = app.invoke({"text": "I want a refund", "order_id": "A1001"}, {"configurable": {"thread_id": "t"}},
                            context=DeskContext("C-100"))
        assert pending_question(result) is not None


class TestResumingSafely:
    def test_resuming_a_thread_that_never_started_is_a_clear_error(self, app):
        with pytest.raises(NoPendingQuestion):
            reply(app, "A1001", thread_id="never-started")

    def test_resuming_a_finished_thread_is_a_clear_error(self, app):
        start(app, "order A1001", thread_id="done")
        with pytest.raises(NoPendingQuestion):
            reply(app, "A1002", thread_id="done")


class TestTheSharpEdge:
    def test_code_before_an_interrupt_runs_twice(self):
        assert side_effect_runs(before_interrupt=True) == 2

    def test_code_after_an_interrupt_runs_once(self):
        assert side_effect_runs(before_interrupt=False) == 1

    def test_the_clarify_node_has_nothing_irreversible_before_its_interrupt(self, desk):
        start(build_clarify_graph(desk), "refund", thread_id="t")
        assert desk.refunds == [] and desk.tickets == []
