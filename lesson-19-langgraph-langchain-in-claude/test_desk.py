"""The business core has no model in it, so every rule can be tested directly."""

from datetime import date, timedelta

import pytest

from desk import (AGENT_REFUND_LIMIT_CENTS, Desk, Order, OrderNotFound, RefundRejected, TemporaryOutage, money, to_cents)


class TestMoney:
    @pytest.mark.parametrize("amount, cents", [(89, 8900), (89.0, 8900), (19.99, 1999), ("0.10", 10), (0.01, 1), ("1234.5", 123450)])
    def test_to_cents(self, amount, cents):
        assert to_cents(amount) == cents

    @pytest.mark.parametrize("bad", [0, -5, 1.005, "abc", float("nan"), float("inf"), None, "", 0.001])
    def test_to_cents_refuses_anything_that_is_not_clean_money(self, bad):
        with pytest.raises(RefundRejected):
            to_cents(bad)

    def test_money_formatting(self):
        assert money(8900) == "$89.00"
        assert money(5) == "$0.05"
        assert money(123456) == "$1,234.56"


class TestOrders:
    def test_owner_can_read_own_order(self, desk):
        assert desk.get_order("C-100", "A1001").item == "Trail Shoes"

    def test_another_customers_order_looks_like_a_missing_order(self, desk):
        with pytest.raises(OrderNotFound) as other:
            desk.get_order("C-100", "A1004")
        with pytest.raises(OrderNotFound) as missing:
            desk.get_order("C-100", "A9999")
        assert str(other.value).replace("A1004", "X") == str(missing.value).replace("A9999", "X")

    def test_outage_fails_the_requested_number_of_times_then_recovers(self, desk):
        desk.outages = 2
        for _ in range(2):
            with pytest.raises(TemporaryOutage):
                desk.get_order("C-100", "A1001")
        assert desk.get_order("C-100", "A1001").order_id == "A1001"
        assert desk.outages == 0


class TestPolicy:
    def policy(self, desk, order_id):
        return desk.policy_for(desk.orders[order_id])

    def test_delivered_within_the_window_is_refundable_in_full(self, desk):
        policy = self.policy(desk, "A1001")
        assert (policy.eligible, policy.refundable_cents) == (True, 8900)

    def test_outside_the_window_is_not_refundable(self, desk):
        policy = self.policy(desk, "A1003")
        assert not policy.eligible and "return window" in policy.reason and policy.refundable_cents == 0

    def test_shipped_but_not_delivered_is_not_refundable(self, desk):
        policy = self.policy(desk, "A1005")
        assert not policy.eligible and "not been delivered" in policy.reason

    def test_window_boundary_is_day_30_inclusive(self):
        today = date(2026, 11, 7)
        orders = [Order("B1", "C", "x", 1000, "delivered", today - timedelta(days=30)), Order("B2", "C", "x", 1000, "delivered", today - timedelta(days=31))]
        desk = Desk(orders, today=today)
        assert desk.policy_for(desk.orders["B1"]).eligible
        assert not desk.policy_for(desk.orders["B2"]).eligible

    def test_refundable_amount_shrinks_after_a_partial_refund(self, desk):
        desk.issue_refund("C-100", "A1001", 3000, "partial", key="k1")
        assert self.policy(desk, "A1001").refundable_cents == 5900

    def test_fully_refunded_order_is_closed(self, desk):
        desk.issue_refund("C-100", "A1001", 8900, "all", key="k1")
        policy = self.policy(desk, "A1001")
        assert not policy.eligible and "already fully refunded" in policy.reason


class TestIssueRefund:
    def test_success_is_recorded_in_the_ledger(self, desk):
        record = desk.issue_refund("C-100", "A1001", 8900, " too small ", key="k1")
        assert (record.refund_id, record.amount_cents, record.reason) == ("R-0001", 8900, "too small")
        assert desk.refunds == [record]

    def test_cannot_refund_someone_elses_order(self, desk):
        with pytest.raises(OrderNotFound):
            desk.issue_refund("C-100", "A1004", 1000, "x", key="k1")
        assert desk.refunds == []

    def test_cannot_refund_more_than_remains(self, desk):
        desk.issue_refund("C-100", "A1001", 5000, "first", key="k1")
        with pytest.raises(RefundRejected, match=r"more than the \$39\.00"):
            desk.issue_refund("C-100", "A1001", 4000, "second", key="k2")
        assert len(desk.refunds) == 1

    def test_agent_limit_applies_even_when_the_order_allows_more(self, desk):
        with pytest.raises(RefundRejected, match="agent limit"):
            desk.issue_refund("C-100", "A1002", AGENT_REFUND_LIMIT_CENTS + 1, "x", key="k1")
        assert desk.issue_refund("C-100", "A1002", AGENT_REFUND_LIMIT_CENTS, "x", key="k2").amount_cents == 20_000

    @pytest.mark.parametrize("order_id, fragment", [("A1003", "return window"), ("A1005", "not been delivered")])
    def test_ineligible_orders_are_refused(self, desk, order_id, fragment):
        with pytest.raises(RefundRejected, match=fragment):
            desk.issue_refund("C-100", order_id, 100, "x", key="k1")

    @pytest.mark.parametrize("amount", [0, -100])
    def test_non_positive_amounts_are_refused(self, desk, amount):
        with pytest.raises(RefundRejected):
            desk.issue_refund("C-100", "A1001", amount, "x", key="k1")

    @pytest.mark.parametrize("reason", ["", "   "])
    def test_a_reason_is_required(self, desk, reason):
        with pytest.raises(RefundRejected, match="reason"):
            desk.issue_refund("C-100", "A1001", 100, reason, key="k1")

    def test_same_key_pays_once(self, desk):
        first = desk.issue_refund("C-100", "A1001", 5000, "x", key="same")
        again = desk.issue_refund("C-100", "A1001", 5000, "x", key="same")
        assert again is first and len(desk.refunds) == 1

    def test_same_key_with_different_arguments_is_refused(self, desk):
        desk.issue_refund("C-100", "A1001", 5000, "x", key="same")
        with pytest.raises(RefundRejected, match="already used"):
            desk.issue_refund("C-100", "A1001", 4000, "x", key="same")
        assert len(desk.refunds) == 1

    def test_refund_ids_count_up(self, desk):
        ids = [desk.issue_refund("C-100", "A1001", 1000, "x", key=f"k{i}").refund_id for i in range(3)]
        assert ids == ["R-0001", "R-0002", "R-0003"]


class TestTicketsAndFaq:
    def test_ticket_ids_count_up(self, desk):
        assert [desk.open_ticket("C-100", "r", "s")["ticket_id"] for _ in range(2)] == ["T-0001", "T-0002"]

    def test_faq_finds_the_matching_article(self, desk):
        assert [a["id"] for a in desk.faq_search("How long does shipping take?")] == ["shipping"]

    def test_faq_says_nothing_when_nothing_matches(self, desk):
        assert desk.faq_search("Do you sell kayaks?") == []

    def test_faq_ignores_punctuation_and_case(self, desk):
        assert desk.faq_search("WARRANTY?!")[0]["id"] == "warranty"

    def test_faq_ranks_more_shared_words_first_and_respects_the_limit(self, desk):
        hits = desk.faq_search("refund policy for returns and shipping", limit=2)
        assert hits[0]["id"] == "returns" and len(hits) == 2
