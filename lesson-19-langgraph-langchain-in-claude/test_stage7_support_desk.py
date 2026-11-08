from datetime import date

import pytest

from desk import Desk, Order
from fakes import ScriptedChatModel, call, say
from show import final_text
from stage5_approval import approve, edit, pending_approval, reject, resume
from stage6_middleware import AuditTrail
from stage7_support_desk import URGENT_AT, DeskModels, build_desk, decide_route, handle, path_of

REFUND_TEXT = "Please refund order A1001, the shoes are too small."


def triage(category="refund", order_id="A1001", urgency=2, summary="A summary."):
    return call("Triage", {"category": category, "order_id": order_id, "urgency": urgency, "summary": summary})


def refund_agent(order_id="A1001", amount=89.0, closing="Refund done."):
    return [call("issue_refund", {"order_id": order_id, "amount": amount, "reason": "customer request"}), say(closing)]


def make(triage_script=(), agent_script=(), faq_script=(), desk=None):
    models = DeskModels(ScriptedChatModel(script=list(triage_script)), ScriptedChatModel(script=list(agent_script)), ScriptedChatModel(script=list(faq_script)))
    desk, trail = desk or Desk(), AuditTrail()
    return build_desk(models, desk, trail), models, desk, trail


def verified(**changes):
    return {"verified": True, "problems": [], "urgency": 2, "category": "refund", "faq_hits": 0, **changes}


class TestTheRoutingDecision:
    @pytest.mark.parametrize("triage_record, route", [
        (verified(category="refund"), "refund_flow"),
        (verified(category="faq", faq_hits=2), "answer_faq"),
        (verified(category="faq", faq_hits=0), "escalate"),
        (verified(category="other"), "escalate"),
        (verified(category="refund", urgency=URGENT_AT), "escalate"),
        (verified(category="refund", urgency=URGENT_AT - 1), "refund_flow"),
        (verified(category="faq", faq_hits=1, urgency=5), "escalate"),
        (verified(verified=False, problems=["bad id"]), "escalate"),
        (verified(verified=False, problems=["bad id"], category="faq", faq_hits=3), "escalate"),
    ])
    def test_every_branch(self, triage_record, route):
        assert decide_route(triage_record)[0] == route

    def test_the_reason_names_the_problem(self):
        assert "bad id" in decide_route(verified(verified=False, problems=["bad id"]))[1]

    def test_the_decision_is_the_same_every_time(self):
        record = verified(category="faq", faq_hits=1)
        assert decide_route(record) == decide_route(record)


class TestTheShapeOfTheGraph:
    def test_nodes_and_edges(self):
        app, *_ = make()
        drawn = app.get_graph()
        assert set(drawn.nodes) == {"__start__", "triage", "refund_flow", "answer_faq", "escalate", "finish", "__end__"}
        edges = {(e.source, e.target) for e in drawn.edges}
        assert {("__start__", "triage"), ("triage", "refund_flow"), ("triage", "answer_faq"), ("triage", "escalate"),
                ("refund_flow", "finish"), ("answer_faq", "finish"), ("escalate", "finish"), ("finish", "__end__")} <= edges


class TestRefundPath:
    def test_the_whole_story_with_an_approval(self):
        app, models, desk, trail = make([triage()], refund_agent())
        paused = handle(app, REFUND_TEXT, customer_name="Sam", thread_id="t")
        request = pending_approval(paused)
        assert request is not None and desk.refunds == [] and "resolution" not in paused
        done = resume(app, approve(), "t")
        assert done["resolution"] == {"status": "refunded", "refunded_cents": 8900, "refund_ids": ["R-0001"]}
        assert final_text(done["messages"]) == "Refund done."
        assert done["events"] == ["triage -> refund_flow", "finish"]
        assert [(r["tool"], r["outcome"]) for r in trail.records] == [("issue_refund", "ok")]

    def test_the_pause_is_saved_inside_the_parent_graph(self):
        app, *_ = make([triage()], refund_agent())
        handle(app, REFUND_TEXT, thread_id="t")
        assert app.get_state({"configurable": {"thread_id": "t"}}).next == ("refund_flow",)

    def test_a_rejected_refund_is_declined_and_nothing_is_paid(self):
        app, _, desk, _ = make([triage()], refund_agent(closing="I could not refund that."))
        handle(app, REFUND_TEXT, thread_id="t")
        done = resume(app, reject("No."), "t")
        assert done["resolution"]["status"] == "declined" and done["resolution"]["refunded_cents"] == 0 and desk.refunds == []

    def test_a_reviewer_edit_that_breaks_a_rule_is_declined(self):
        app, _, desk, _ = make([triage()], refund_agent())
        paused = handle(app, REFUND_TEXT, thread_id="t")
        done = resume(app, edit("issue_refund", {**pending_approval(paused).args, "amount": 900.0}), "t")
        assert done["resolution"]["status"] == "declined" and desk.refunds == []

    def test_an_edit_to_a_lower_amount_is_paid_at_the_lower_amount(self):
        app, *_ = make([triage()], refund_agent())
        paused = handle(app, REFUND_TEXT, thread_id="t")
        done = resume(app, edit("issue_refund", {**pending_approval(paused).args, "amount": 40.0}), "t")
        assert done["resolution"]["refunded_cents"] == 4000

    def test_a_small_refund_needs_no_person(self):
        desk = Desk([Order("S1001", "C-100", "Socks", 1500, "delivered", date(2026, 11, 1))])
        app, *_ = make([triage(order_id="S1001")], refund_agent("S1001", 15.0), desk=desk)
        done = handle(app, "Please refund order S1001, thanks.", thread_id="t")
        assert pending_approval(done) is None and done["resolution"]["status"] == "refunded"

    def test_a_resume_does_not_duplicate_messages(self):
        app, *_ = make([triage()], refund_agent())
        handle(app, REFUND_TEXT, thread_id="t")
        done = resume(app, approve(), "t")
        ids = [m.id for m in done["messages"]]
        assert len(ids) == len(set(ids)) and [m.type for m in done["messages"]] == ["human", "ai", "tool", "ai"]

    def test_each_resolution_counts_only_its_own_refund(self):
        app, _, desk, _ = make([triage(), triage(order_id="A1002")], [*refund_agent(closing="one"), *refund_agent("A1002", 50.0, closing="two")])
        handle(app, REFUND_TEXT, thread_id="t1")
        resume(app, approve(), "t1")
        handle(app, "Refund order A1002 please.", thread_id="t2")
        second = resume(app, approve(), "t2")
        assert second["resolution"] == {"status": "refunded", "refunded_cents": 5000, "refund_ids": ["R-0002"]} and len(desk.refunds) == 2

    def test_the_verified_customer_reaches_the_nested_agent(self):
        app, _, desk, _ = make([triage(order_id="A1004")], refund_agent("A1004", 150.0))
        paused = handle(app, "Refund A1004 please.", customer_id="C-200", thread_id="t")
        assert "Sleeping Bag" in pending_approval(paused).description
        resume(app, approve(), "t", customer_id="C-200")
        assert [(r.customer_id, r.order_id) for r in desk.refunds] == [("C-200", "A1004")]


class TestAnotherCustomersOrder:
    def test_it_never_reaches_the_refund_agent(self):
        app, models, desk, _ = make([triage(order_id="A1004")], refund_agent("A1004", 150.0))
        done = handle(app, "Refund A1004 please.", customer_id="C-100", thread_id="t")
        assert models.agent.seen == [] and desk.refunds == []
        assert done["resolution"]["status"] == "escalated" and "no order A1004" in done["reason"]


class TestFaqPath:
    def test_a_help_question_is_answered_from_the_articles(self):
        app, models, _, _ = make([triage("faq", None, 1)], faq_script=[say("3 to 5 days.")])
        done = handle(app, "How long does shipping take?", thread_id="t")
        assert final_text(done["messages"]) == "3 to 5 days." and done["resolution"]["status"] == "answered"
        assert done["events"] == ["triage -> answer_faq", "answer_faq", "finish"]

    def test_the_model_sees_only_the_matching_articles_and_the_rule(self):
        app, models, _, _ = make([triage("faq", None, 1)], faq_script=[say("ok")])
        handle(app, "How long does shipping take?", thread_id="t")
        prompt = models.faq.seen[0][0].content
        assert "ONLY" in prompt and "Standard shipping takes 3 to 5 business days" in prompt and "warranty" not in prompt.lower()

    def test_no_matching_article_means_no_model_call_and_a_person(self):
        app, models, desk, _ = make([triage("faq", None, 1)], faq_script=[say("I made something up.")])
        done = handle(app, "Do you sell kayaks?", thread_id="t")
        assert models.faq.seen == [] and done["resolution"]["status"] == "escalated"
        assert desk.tickets[0]["reason"] == "no help article matches"


class TestEscalationPath:
    def test_an_urgent_message_goes_to_a_person_without_any_agent(self):
        app, models, desk, _ = make([triage("other", None, 5, "Furious about a late parcel.")])
        done = handle(app, "My parcel is late and I am furious!", thread_id="t")
        assert models.agent.seen == [] and models.faq.seen == []
        assert desk.tickets == [{"ticket_id": "T-0001", "customer_id": "C-100", "reason": "urgency 5 of 5", "summary": "Furious about a late parcel."}]
        assert "T-0001" in final_text(done["messages"])
        assert done["resolution"] == {"status": "escalated", "refunded_cents": 0, "refund_ids": [], "ticket_id": "T-0001"}

    def test_an_invented_order_id_is_escalated(self):
        app, models, desk, _ = make([triage(order_id="A9999")], refund_agent())
        done = handle(app, "Please refund my last order.", thread_id="t")
        assert done["route"] == "escalate" and "could not be verified" in done["reason"] and models.agent.seen == []

    def test_a_model_that_keeps_returning_nonsense_is_escalated(self):
        bad = call("Triage", {"category": "refund", "urgency": 99, "summary": "x"})
        app, _, desk, _ = make([bad, bad], refund_agent())
        done = handle(app, REFUND_TEXT, thread_id="t")
        assert done["resolution"]["status"] == "escalated" and "valid Triage" in done["reason"]

    def test_a_message_that_is_neither_a_refund_nor_a_help_question_is_escalated(self):
        app, *_ = make([triage("other", None, 2)])
        assert handle(app, "I want to talk about the weather.", thread_id="t")["route"] == "escalate"

    def test_even_a_confident_refund_is_escalated_when_urgent(self):
        app, models, _, _ = make([triage("refund", "A1001", 5)], refund_agent())
        assert handle(app, REFUND_TEXT, thread_id="t")["route"] == "escalate" and models.agent.seen == []


class TestSeveralTicketsAtOnce:
    def test_two_paused_refunds_resume_independently_in_any_order(self):
        app, _, desk, _ = make([triage(), triage(order_id="A1002")], [*refund_agent("A1001", 89.0, "one done")[:1], *refund_agent("A1002", 50.0)[:1], say("two done"), say("one done")])
        handle(app, REFUND_TEXT, thread_id="t1")
        handle(app, "Refund order A1002 please.", thread_id="t2")
        second = resume(app, approve(), "t2")
        first = resume(app, approve(), "t1")
        assert (final_text(second["messages"]), second["resolution"]["refunded_cents"]) == ("two done", 5000)
        assert (final_text(first["messages"]), first["resolution"]["refunded_cents"]) == ("one done", 8900)
        assert sorted((r.order_id, r.amount_cents) for r in desk.refunds) == [("A1001", 8900), ("A1002", 5000)]


class TestWatchingARun:
    def test_path_of_lists_the_nodes_in_order(self):
        app, *_ = make([triage("faq", None, 1)], faq_script=[say("ok")])
        assert path_of(app, "How long does shipping take?", thread_id="t") == ["triage", "answer_faq", "finish"]

    def test_path_of_a_paused_refund_ends_at_the_pause(self):
        app, *_ = make([triage()], refund_agent())
        nodes = path_of(app, REFUND_TEXT, thread_id="t")
        assert nodes[0] == "triage" and nodes[-1] == "__interrupt__"

    def test_the_final_state_is_saved_for_later_audit(self):
        app, *_ = make([triage("other", None, 5)])
        handle(app, "Very angry!", thread_id="t")
        saved = app.get_state({"configurable": {"thread_id": "t"}}).values
        assert saved["resolution"]["status"] == "escalated" and saved["triage"]["urgency"] == 5
