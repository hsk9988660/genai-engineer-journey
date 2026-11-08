"""Stage 8: parallel work in LangGraph. Fan-out, fan-in, and the rules of a superstep.

A customer writes: "Can you check orders A1001, A1002 and A1003?" Checking them one after another wastes time. In
LangGraph you can run the checks at the same time.

THE SUPERSTEP. LangGraph runs a graph in rounds. In each round every node that is ready runs "at once", and their
updates are merged into the state together when the round ends. The next round starts from the merged state. Three
rules follow from this, and each is shown below with a test:

1. FAN-OUT with `Send`. A conditional edge can return a list of `Send(node, input)` objects. Each one starts one copy of
   that node, with its own private input. The number of copies is decided at run time (here: one per order id).
2. FAN-IN needs a reducer. When several nodes write the same key in one round, LangGraph must know how to merge the
   writes. With no reducer it refuses (`InvalidUpdateError`) instead of silently keeping one of them.
3. A JOIN waits for all of its inputs only if you ask. `add_edge(["a", "b"], "join")` runs `join` once, after both.
   Two separate edges `a -> join` and `b -> join` run `join` once for each, in different rounds if a and b take a
   different number of steps.

The workers here call no model. Parallel work is often plain code. A model can still be a worker, the same way.

Run the offline demo:  python stage8_parallel.py
"""

import operator
from typing import Annotated

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, StreamWriter
from typing_extensions import TypedDict

from desk import Desk, OrderNotFound, money

MAX_ORDERS = 10  # a fan-out width set by the customer must always be capped by code


class TooManyOrders(ValueError):
    pass


class CheckInput(TypedDict):
    """What ONE worker receives. It is private: it is not part of the main state."""

    customer_id: str
    order_id: str


class ReportState(TypedDict, total=False):
    customer_id: str
    order_ids: list[str]
    results: Annotated[list[dict], operator.add]  # the reducer that makes fan-in legal
    report: dict


def build_report_graph(desk: Desk):
    def plan(state: ReportState) -> dict:
        ids = list(dict.fromkeys(state["order_ids"]))  # drop duplicates, keep the order the customer wrote
        if len(ids) > MAX_ORDERS:
            raise TooManyOrders(f"at most {MAX_ORDERS} orders per request, got {len(ids)}")
        return {"order_ids": ids}

    def fan_out(state: ReportState):
        if not state["order_ids"]:
            return "collect"  # with no workers, nothing would ever reach collect, so go there directly
        return [Send("check_order", {"customer_id": state["customer_id"], "order_id": oid}) for oid in state["order_ids"]]

    def check_order(state: CheckInput, writer: StreamWriter) -> dict:
        try:
            order = desk.get_order(state["customer_id"], state["order_id"])
        except OrderNotFound as exc:
            entry = {"order_id": state["order_id"], "error": str(exc)}
        else:
            policy = desk.policy_for(order)
            entry = {"order_id": order.order_id, "item": order.item, "eligible": policy.eligible, "refundable_cents": policy.refundable_cents,
                     "reason": policy.reason}
        writer({"checked": state["order_id"]})  # a custom event for stage 11
        return {"results": [entry]}

    def collect(state: ReportState) -> dict:
        # Workers finish in no guaranteed order, so sort before presenting. Never rely on arrival order.
        results = sorted(state.get("results", []), key=lambda entry: entry["order_id"])
        found = [r for r in results if "error" not in r]
        return {"report": {
            "orders": results,
            "found": len(found),
            "missing": [r["order_id"] for r in results if "error" in r],
            "refundable_total": money(sum(r["refundable_cents"] for r in found)),
        }}

    graph = StateGraph(ReportState)
    graph.add_node("plan", plan)
    graph.add_node("check_order", check_order)
    graph.add_node("collect", collect)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out, ["check_order", "collect"])
    graph.add_edge("check_order", "collect")  # runs once, after ALL the workers of the round
    graph.add_edge("collect", END)
    return graph.compile()


def order_report(desk: Desk, customer_id: str, order_ids: list[str]) -> dict:
    return build_report_graph(desk).invoke({"customer_id": customer_id, "order_ids": order_ids, "results": []})["report"]


def concurrent_writes(with_reducer: bool) -> object:
    """Two nodes run in the same round and write the same key. Returns the final value, or the error LangGraph raised."""

    class State(TypedDict, total=False):
        note: Annotated[list[str], operator.add] if with_reducer else str

    graph = StateGraph(State)
    graph.add_node("a", lambda state: {"note": ["from a"] if with_reducer else "from a"})
    graph.add_node("b", lambda state: {"note": ["from b"] if with_reducer else "from b"})
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    try:
        return sorted(graph.compile().invoke({})["note"]) if with_reducer else graph.compile().invoke({})["note"]
    except InvalidUpdateError as exc:
        return exc


def join_runs(wait_for_all: bool) -> int:
    """Branch `a` takes one step. Branch `b` takes two. How many times does `join` run?"""

    class State(TypedDict, total=False):
        log: Annotated[list[str], operator.add]

    runs = []

    def join(state: State) -> dict:
        runs.append(1)
        return {}

    graph = StateGraph(State)
    for name in ("a", "b1", "b2", "join"):
        graph.add_node(name, join if name == "join" else (lambda state, name=name: {"log": [name]}))
    graph.add_edge(START, "a")
    graph.add_edge(START, "b1")
    graph.add_edge("b1", "b2")
    if wait_for_all:
        graph.add_edge(["a", "b2"], "join")
    else:
        graph.add_edge("a", "join")
        graph.add_edge("b2", "join")
    graph.add_edge("join", END)
    graph.compile().invoke({})
    return len(runs)


def demo() -> None:
    desk = Desk()
    report = order_report(desk, "C-100", ["A1001", "A1002", "A1003", "A1004", "A1001"])
    for entry in report["orders"]:
        print(entry)
    print(f"\nfound={report['found']} missing={report['missing']} refundable_total={report['refundable_total']}")
    print("\nconcurrent writes, no reducer :", type(concurrent_writes(False)).__name__)
    print("concurrent writes, reducer    :", concurrent_writes(True))
    print("join runs, separate edges     :", join_runs(False))
    print("join runs, add_edge([a, b2])  :", join_runs(True))


if __name__ == "__main__":
    demo()
