"""Stage 10: durability, time travel and node retries.

Three things that decide whether a LangGraph system survives the real world.

1. DURABLE PERSISTENCE. `InMemorySaver` (stage 4) forgets everything when the process ends. `SqliteSaver` writes every
   checkpoint to a file. A ticket paused for a human on Monday can be resumed on Tuesday by a different process, after a
   restart, with a graph object that was built fresh. The graph code does not change; only the saver does. (For several
   servers you use the Postgres saver. SQLite is for one machine.)

2. TIME TRAVEL. Every step is a checkpoint, so you can list a thread's history, REPLAY from an earlier checkpoint, or
   FORK it: change the state at that point and run forward to see "what if". Replay and fork write NEW checkpoints on the
   same thread. The old ones stay in the history, but the thread's latest state becomes the new branch.

3. NODE RETRIES. `RetryPolicy` on a node retries that node when it raises a chosen exception. Stage 6 retried TOOL calls
   with middleware. A retry policy covers ANY node, including one that calls no model and no tool.

Run the offline demo:  python stage10_durability.py
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import Annotated
import operator

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from desk import Desk, TemporaryOutage, money
from fakes import ScriptedChatModel, call, say
from stage5_approval import approve, pending_approval, resume
from stage7_support_desk import DeskModels, build_desk, handle
from tools import DeskContext

REFUND_TEXT = "Please refund order A1001, the shoes are too small."
TRIAGE = call("Triage", {"category": "refund", "order_id": "A1001", "urgency": 2, "summary": "Wants a refund for A1001."})


def open_saver(path: str | Path) -> SqliteSaver:
    """A saver backed by a SQLite file. The connection must allow use from the threads that LangGraph uses."""
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def pause_then_restart(db_path: str | Path, desk: Desk | None = None) -> dict:
    """Pause a refund for approval, throw everything away, rebuild from the file, and finish the ticket.

    The second "process" gets new model objects with ONLY the turns still to come, because the model turns that already
    happened are not repeated: the saved state remembers them. That is what durable execution means.
    """
    desk = desk or Desk()
    first_saver = open_saver(db_path)
    first = build_desk(DeskModels(ScriptedChatModel(script=[TRIAGE]),
                                  ScriptedChatModel(script=[call("issue_refund", {"order_id": "A1001", "amount": 89.0, "reason": "too small"})]),
                                  ScriptedChatModel(script=[])), desk, checkpointer=first_saver)
    paused = handle(first, REFUND_TEXT, thread_id="monday")
    assert pending_approval(paused) is not None and desk.refunds == []
    first_saver.conn.close()  # the "process" ends

    second_saver = open_saver(db_path)
    try:
        second = build_desk(DeskModels(ScriptedChatModel(script=[]), ScriptedChatModel(script=[say("Refund done.")]), ScriptedChatModel(script=[])),
                            desk, checkpointer=second_saver)
        return resume(second, approve(), "monday")
    finally:
        second_saver.conn.close()  # a SQLite file stays locked on Windows until every connection is closed


class QuoteState(TypedDict, total=False):
    order_id: str
    fee_pct: int
    total_cents: int
    refund_cents: int
    log: Annotated[list[str], operator.add]


def build_quote_graph(desk: Desk, checkpointer=None, calls_per_node: dict | None = None):
    """load -> price -> finish. A small deterministic graph, so time travel is easy to see."""
    counts = calls_per_node if calls_per_node is not None else {}

    def tick(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    def load(state: QuoteState, runtime) -> dict:
        tick("load")
        order = desk.get_order(runtime.context.customer_id, state["order_id"])
        return {"total_cents": order.total_cents, "log": [f"loaded {order.order_id}"]}

    def price(state: QuoteState) -> dict:
        tick("price")
        fee = state["total_cents"] * state["fee_pct"] // 100
        return {"refund_cents": state["total_cents"] - fee, "log": [f"fee {state['fee_pct']}%"]}

    def finish(state: QuoteState) -> dict:
        tick("finish")
        return {"log": [f"quote {money(state['refund_cents'])}"]}

    graph = StateGraph(QuoteState, context_schema=DeskContext)
    graph.add_node("load", load)
    graph.add_node("price", price)
    graph.add_node("finish", finish)
    graph.add_edge(START, "load")
    graph.add_edge("load", "price")
    graph.add_edge("price", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def run_quote(app, order_id: str, fee_pct: int, thread_id: str, customer_id: str = "C-100") -> dict:
    return app.invoke({"order_id": order_id, "fee_pct": fee_pct, "log": []}, thread(thread_id), context=DeskContext(customer_id))


def history(app, thread_id: str) -> list[dict]:
    """The thread's checkpoints, oldest first."""
    snapshots = list(app.get_state_history(thread(thread_id)))
    return [{"step": s.metadata.get("step"), "next": s.next, "values": s.values, "config": s.config} for s in reversed(snapshots)]


def checkpoint_before(app, thread_id: str, node: str) -> RunnableConfig:
    """The config of the checkpoint taken just before `node` was going to run."""
    for entry in history(app, thread_id):
        if entry["next"] == (node,):
            return entry["config"]
    raise LookupError(f"no checkpoint before {node!r} in thread {thread_id!r}")


def replay(app, config: RunnableConfig, customer_id: str = "C-100") -> dict:
    """Run forward again from an earlier checkpoint. Nodes before it do not run again."""
    return app.invoke(None, config, context=DeskContext(customer_id))


def what_if(app, config: RunnableConfig, changes: dict, customer_id: str = "C-100") -> dict:
    """Change the state at an earlier checkpoint and run forward from there."""
    return app.invoke(None, app.update_state(config, changes), context=DeskContext(customer_id))


class LookupState(TypedDict, total=False):
    order_id: str
    item: str


def build_lookup_graph(desk: Desk, *, max_attempts: int = 3, calls: list | None = None):
    """One node that calls the order system, with a retry policy that retries ONLY an outage."""
    seen = calls if calls is not None else []

    def lookup(state: LookupState) -> dict:
        seen.append(state["order_id"])
        return {"item": desk.get_order("C-100", state["order_id"]).item}

    graph = StateGraph(LookupState)
    graph.add_node("lookup", lookup, retry_policy=RetryPolicy(max_attempts=max_attempts, initial_interval=0.0, jitter=False, retry_on=TemporaryOutage))
    graph.add_edge(START, "lookup")
    graph.add_edge("lookup", END)
    return graph.compile()


def demo() -> None:
    with tempfile.TemporaryDirectory() as folder:
        done = pause_then_restart(Path(folder) / "tickets.db")
        print("after a restart:", done["resolution"])

        counts: dict = {}
        app = build_quote_graph(Desk(), calls_per_node=counts)
        print("\nbase run (fee 10%):", run_quote(app, "A1001", 10, "q")["log"])
        print("history:", [(h["step"], h["next"]) for h in history(app, "q")])
        before_price = checkpoint_before(app, "q", "price")
        counts.clear()
        replay(app, before_price)
        print("replay from before 'price' ran:", counts)
        counts.clear()
        forked = what_if(app, before_price, {"fee_pct": 0})
        print("what if the fee were 0%:", forked["log"][-1], "| nodes run:", counts)
        print("history is now", len(history(app, "q")), "checkpoints long; the old ones are still there")

    desk = Desk()
    desk.outages = 2
    print("\nnode retry, 2 outages, 3 attempts:", build_lookup_graph(desk).invoke({"order_id": "A1001"}))


if __name__ == "__main__":
    demo()
