import pytest
from pydantic import ValidationError

from desk import Desk
from fakes import ScriptedChatModel, call, say
from stage3_triage import Triage, triage_request, verify

GOOD = {"category": "refund", "order_id": "A1001", "urgency": 2, "summary": "Wants a refund."}
REFUND_TEXT = "Please refund order A1001, the shoes are too small."


def run(script, text=REFUND_TEXT, customer="C-100", **kwargs):
    model = ScriptedChatModel(script=script)
    return triage_request(model, text, Desk(), customer, **kwargs), model


class TestTheSchema:
    def test_a_valid_object_is_accepted(self):
        assert Triage(**GOOD).urgency == 2

    @pytest.mark.parametrize("change", [{"urgency": 0}, {"urgency": 6}, {"category": "complaint"}, {"summary": "x" * 301}, {"urgency": "high"}])
    def test_the_wrong_shape_is_rejected(self, change):
        with pytest.raises(ValidationError):
            Triage(**{**GOOD, **change})

    def test_order_id_is_optional(self):
        assert Triage(category="faq", urgency=1, summary="s").order_id is None


class TestVerification:
    def test_a_correct_answer_is_verified(self):
        result, _ = run([call("Triage", GOOD)])
        assert result.verified and result.problems == [] and result.attempts == 1

    def test_an_order_id_that_is_not_in_the_message_is_not_trusted(self):
        result, _ = run([call("Triage", {**GOOD, "order_id": "A9999"})])
        assert not result.verified and "does not appear in the message" in result.problems[0]

    def test_an_order_id_in_the_message_that_does_not_exist_is_not_trusted(self):
        result, _ = run([call("Triage", {**GOOD, "order_id": "A9999"})], text="Refund A9999 please.")
        assert not result.verified and "no order A9999" in result.problems[0]

    def test_another_customers_order_is_not_trusted(self):
        result, _ = run([call("Triage", {**GOOD, "order_id": "A1004"})], text="Refund A1004 please.", customer="C-100")
        assert not result.verified and "no order A1004" in result.problems[0]

    def test_the_owner_of_that_order_is_trusted(self):
        result, _ = run([call("Triage", {**GOOD, "order_id": "A1004"})], text="Refund A1004 please.", customer="C-200")
        assert result.verified

    def test_a_refund_without_an_order_id_is_not_trusted(self):
        result, _ = run([call("Triage", {**GOOD, "order_id": None})], text="Refund my last order.")
        assert not result.verified and "needs an order id" in result.problems[0]

    def test_a_help_question_needs_no_order_id(self):
        result, _ = run([call("Triage", {"category": "faq", "urgency": 1, "summary": "s"})], text="How long does shipping take?")
        assert result.verified

    def test_verify_reports_the_problem_it_finds(self):
        assert verify(Triage(**{**GOOD, "order_id": "B0000"}), "hello", Desk(), "C-100") == ["order id B0000 does not appear in the message"]


class TestRetries:
    BAD = {**GOOD, "urgency": 9}

    def test_an_invalid_shape_gets_one_more_try_with_the_error_as_feedback(self):
        result, model = run([call("Triage", self.BAD), call("Triage", GOOD)])
        assert result.verified and result.attempts == 2 and len(model.seen) == 2
        assert "not valid" in model.seen[1][-1].content and "urgency" in model.seen[1][-1].content

    def test_the_retry_does_not_pile_up_old_feedback(self):
        model = ScriptedChatModel(script=[call("Triage", self.BAD)] * 3)
        triage_request(model, REFUND_TEXT, Desk(), "C-100", max_attempts=3)
        assert [len(m) for m in model.seen] == [2, 3, 3]

    def test_two_invalid_answers_fall_back_to_a_safe_unverified_result(self):
        result, _ = run([call("Triage", self.BAD), call("Triage", self.BAD)])
        assert (result.triage.category, result.verified, result.attempts) == ("other", False, 2)

    def test_the_fallback_keeps_the_customers_words_but_clips_them(self):
        result, _ = run([call("Triage", self.BAD)] * 2, text="x" * 500)
        assert len(result.triage.summary) == 300

    def test_plain_text_instead_of_an_object_counts_as_a_failure(self):
        result, _ = run([say("I think it is a refund."), call("Triage", GOOD)])
        assert result.attempts == 2 and result.verified

    def test_max_attempts_one_means_no_retry(self):
        result, model = run([call("Triage", self.BAD), call("Triage", GOOD)], max_attempts=1)
        assert not result.verified and len(model.seen) == 1
