import sqlite3

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from desk import Desk, OrderNotFound, TemporaryOutage
from fakes import ScriptedChatModel, call
from stage5_approval import pending_approval
from stage7_support_desk import DeskModels, build_desk, handle
from stage10_durability import (REFUND_TEXT, TRIAGE, build_lookup_graph, build_quote_graph, checkpoint_before, history, open_saver,
                                pause_then_restart, replay, run_quote, thread, what_if)


class TestDurablePersistence:
    def test_a_paused_ticket_is_finished_after_a_restart(self, tmp_path):
        desk = Desk()
        done = pause_then_restart(tmp_path / "t.db", desk)
        assert done["resolution"] == {"status": "refunded", "refunded_cents": 8900, "refund_ids": ["R-0001"]}
        assert len(desk.refunds) == 1

    def test_the_second_process_never_repeats_a_model_turn(self, tmp_path):
        # The second set of models has no triage turn and no tool-call turn. If the saved state were not used, the
        # desk would ask for one and the fake would raise ScriptExhausted. The run finishing proves it was not asked.
        assert pause_then_restart(tmp_path / "t.db")["messages"][-1].content == "Refund done."

    def test_the_checkpoints_are_really_in_the_file(self, tmp_path):
        path = tmp_path / "t.db"
        pause_then_restart(path)
        with sqlite3.connect(path) as connection:
            rows = connection.execute("select count(*) from checkpoints where thread_id = 'monday'").fetchone()[0]
        connection.close()
        assert rows > 3

    def test_an_in_memory_saver_forgets_everything_after_a_restart(self):
        desk = Desk()
        models = DeskModels(ScriptedChatModel(script=[TRIAGE]), ScriptedChatModel(script=[call("issue_refund", {"order_id": "A1001", "amount": 89.0, "reason": "x"})]),
                            ScriptedChatModel(script=[]))
        paused = handle(build_desk(models, desk, checkpointer=InMemorySaver()), REFUND_TEXT, thread_id="monday")
        assert pending_approval(paused) is not None
        reborn = build_desk(models, desk, checkpointer=InMemorySaver())
        snapshot = reborn.get_state(thread("monday"))
        assert snapshot.values == {} and snapshot.next == ()

    def test_two_threads_in_one_file_do_not_mix(self, tmp_path):
        saver = open_saver(tmp_path / "t.db")
        try:
            app = build_quote_graph(Desk(), checkpointer=saver)
            run_quote(app, "A1001", 10, "one")
            run_quote(app, "A1002", 0, "two")
            assert app.get_state(thread("one")).values["refund_cents"] == 8010 and app.get_state(thread("two")).values["refund_cents"] == 24000
        finally:
            saver.conn.close()


class TestTimeTravel:
    @pytest.fixture
    def traced(self):
        counts: dict = {}
        app = build_quote_graph(Desk(), calls_per_node=counts)
        run_quote(app, "A1001", 10, "q")
        counts.clear()
        return app, counts

    def test_history_is_oldest_first_and_names_the_node_that_ran_next(self, traced):
        app, _ = traced
        assert [(h["step"], h["next"]) for h in history(app, "q")] == [(-1, ("__start__",)), (0, ("load",)), (1, ("price",)), (2, ("finish",)), (3, ())]

    def test_each_checkpoint_holds_the_state_at_that_moment(self, traced):
        app, _ = traced
        states = [h["values"] for h in history(app, "q")]
        assert "total_cents" not in states[1] and states[2]["total_cents"] == 8900 and "refund_cents" not in states[2] and states[4]["refund_cents"] == 8010

    def test_checkpoint_before_finds_the_right_moment(self, traced):
        app, _ = traced
        assert app.get_state(checkpoint_before(app, "q", "price")).next == ("price",)

    def test_asking_for_a_node_that_never_ran_is_a_clear_error(self, traced):
        app, _ = traced
        with pytest.raises(LookupError, match="refund"):
            checkpoint_before(app, "q", "refund")

    def test_replay_runs_only_the_nodes_after_the_checkpoint(self, traced):
        app, counts = traced
        replay(app, checkpoint_before(app, "q", "price"))
        assert counts == {"price": 1, "finish": 1}

    def test_replay_gives_the_same_answer(self, traced):
        app, _ = traced
        assert replay(app, checkpoint_before(app, "q", "price"))["refund_cents"] == 8010

    def test_a_what_if_changes_the_answer_and_runs_only_what_it_must(self, traced):
        app, counts = traced
        result = what_if(app, checkpoint_before(app, "q", "price"), {"fee_pct": 0})
        assert result["refund_cents"] == 8900 and counts == {"price": 1, "finish": 1}

    def test_a_fork_adds_checkpoints_and_keeps_the_old_ones(self, traced):
        app, _ = traced
        before = history(app, "q")
        what_if(app, checkpoint_before(app, "q", "price"), {"fee_pct": 50})
        after = history(app, "q")
        assert len(after) > len(before) and [h["config"] for h in after[: len(before)]] == [h["config"] for h in before]

    def test_the_threads_latest_state_becomes_the_fork(self, traced):
        app, _ = traced
        what_if(app, checkpoint_before(app, "q", "price"), {"fee_pct": 50})
        assert app.get_state(thread("q")).values["refund_cents"] == 4450

    def test_another_thread_is_untouched_by_a_fork(self, traced):
        app, _ = traced
        run_quote(app, "A1002", 10, "other")
        what_if(app, checkpoint_before(app, "q", "price"), {"fee_pct": 50})
        assert app.get_state(thread("other")).values["refund_cents"] == 21600

    def test_the_log_shows_the_replayed_steps_again(self, traced):
        app, _ = traced
        assert replay(app, checkpoint_before(app, "q", "price"))["log"].count("fee 10%") == 1


class TestNodeRetries:
    def test_outages_within_the_attempt_budget_are_absorbed(self):
        desk, calls = Desk(), []
        desk.outages = 2
        assert build_lookup_graph(desk, calls=calls).invoke({"order_id": "A1001"})["item"] == "Trail Shoes" and len(calls) == 3

    def test_one_outage_too_many_is_raised_to_the_caller(self):
        desk, calls = Desk(), []
        desk.outages = 3
        with pytest.raises(TemporaryOutage):
            build_lookup_graph(desk, calls=calls).invoke({"order_id": "A1001"})
        assert len(calls) == 3

    def test_an_error_that_is_not_listed_is_not_retried(self):
        calls = []
        with pytest.raises(OrderNotFound):
            build_lookup_graph(Desk(), calls=calls).invoke({"order_id": "A9999"})
        assert len(calls) == 1

    def test_one_attempt_means_no_retry(self):
        desk, calls = Desk(), []
        desk.outages = 1
        with pytest.raises(TemporaryOutage):
            build_lookup_graph(desk, max_attempts=1, calls=calls).invoke({"order_id": "A1001"})
        assert len(calls) == 1
