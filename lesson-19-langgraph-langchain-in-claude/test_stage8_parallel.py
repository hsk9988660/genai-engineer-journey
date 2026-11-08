import pytest
from langgraph.errors import InvalidUpdateError

from desk import Desk, TemporaryOutage
from stage8_parallel import MAX_ORDERS, TooManyOrders, build_report_graph, concurrent_writes, join_runs, order_report


def counting(desk):
    seen = []
    original = desk.get_order

    def spy(customer_id, order_id):
        seen.append(order_id)
        return original(customer_id, order_id)

    desk.get_order = spy
    return seen


class TestFanOut:
    def test_every_order_is_checked_once(self, desk):
        seen = counting(desk)
        order_report(desk, "C-100", ["A1001", "A1002", "A1003"])
        assert sorted(seen) == ["A1001", "A1002", "A1003"]

    def test_the_report_adds_up(self, desk):
        report = order_report(desk, "C-100", ["A1001", "A1002", "A1003"])
        assert report["found"] == 3 and report["missing"] == [] and report["refundable_total"] == "$329.00"

    def test_each_entry_comes_from_the_desk_rules(self, desk):
        by_id = {e["order_id"]: e for e in order_report(desk, "C-100", ["A1001", "A1003"])["orders"]}
        assert by_id["A1001"]["eligible"] and not by_id["A1003"]["eligible"] and by_id["A1003"]["refundable_cents"] == 0

    def test_duplicates_are_checked_once(self, desk):
        seen = counting(desk)
        order_report(desk, "C-100", ["A1001", "A1001", "A1001"])
        assert seen == ["A1001"]

    def test_the_report_does_not_depend_on_the_order_the_workers_finish(self, desk):
        forward = order_report(desk, "C-100", ["A1001", "A1002", "A1003"])
        backward = order_report(Desk(), "C-100", ["A1003", "A1002", "A1001"])
        assert forward == backward and [e["order_id"] for e in forward["orders"]] == ["A1001", "A1002", "A1003"]

    def test_another_customers_order_is_reported_as_missing_not_leaked(self, desk):
        report = order_report(desk, "C-100", ["A1001", "A1004"])
        assert report["missing"] == ["A1004"] and "Sleeping Bag" not in str(report)

    def test_a_missing_order_does_not_stop_the_others(self, desk):
        report = order_report(desk, "C-100", ["A9999", "A1001"])
        assert report["found"] == 1 and report["missing"] == ["A9999"]

    def test_no_orders_still_produces_a_report(self, desk):
        report = order_report(desk, "C-100", [])
        assert report == {"orders": [], "found": 0, "missing": [], "refundable_total": "$0.00"}


class TestTheWidthIsCappedByCode:
    def ids(self, n):
        return [f"A{2000 + i}" for i in range(n)]

    def test_the_maximum_is_allowed(self, desk):
        assert len(order_report(desk, "C-100", self.ids(MAX_ORDERS))["orders"]) == MAX_ORDERS

    def test_one_more_is_refused_before_any_worker_starts(self, desk):
        seen = counting(desk)
        with pytest.raises(TooManyOrders):
            order_report(desk, "C-100", self.ids(MAX_ORDERS + 1))
        assert seen == []

    def test_duplicates_do_not_count_toward_the_cap(self, desk):
        assert order_report(desk, "C-100", ["A1001"] * 50)["found"] == 1


class TestTheRulesOfASuperstep:
    def test_two_nodes_writing_one_key_without_a_reducer_is_an_error(self):
        error = concurrent_writes(with_reducer=False)
        assert isinstance(error, InvalidUpdateError) and "Can receive only one value per step" in str(error)

    def test_with_a_reducer_both_writes_survive(self):
        assert concurrent_writes(with_reducer=True) == ["from a", "from b"]

    def test_separate_join_edges_run_the_join_once_per_branch(self):
        assert join_runs(wait_for_all=False) == 2

    def test_a_list_edge_waits_for_every_branch(self):
        assert join_runs(wait_for_all=True) == 1


class TestTheGraph:
    def test_nodes(self, desk):
        assert set(build_report_graph(desk).get_graph().nodes) == {"__start__", "plan", "check_order", "collect", "__end__"}

    def test_workers_stream_a_custom_event_each(self, desk):
        app = build_report_graph(desk)
        events = list(app.stream({"customer_id": "C-100", "order_ids": ["A1001", "A1002"], "results": []}, stream_mode="custom"))
        assert sorted(e["checked"] for e in events) == ["A1001", "A1002"]

    def test_an_outage_in_a_worker_is_not_hidden(self, desk):
        desk.outages = 1
        with pytest.raises(TemporaryOutage):
            order_report(desk, "C-100", ["A1001"])
