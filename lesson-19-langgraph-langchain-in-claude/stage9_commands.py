"""Stage 9: `Command` and `interrupt()` in your own nodes. A graph that asks the customer a question.

Stage 5 paused for a human through middleware. Here you write the pause yourself, inside a node, and you route with
`Command`. Two tools:

COMMAND. A node may return `Command(update=..., goto=...)` instead of a plain dict. It does two things in one step:
change the state AND choose the next node. Use it when the choice is made inside the node, from facts the node just
learned. Use a conditional edge (stages 2 and 7) when the choice is a pure function of the state, because then the
route can be drawn and tested without running the node. Annotating the return type as `Command[Literal["a", "b"]]` tells
LangGraph where the node can go, so the graph can still be drawn.

INTERRUPT. `interrupt(value)` stops the run and hands `value` to the caller. When the caller resumes with
`Command(resume=answer)`, `interrupt()` returns `answer`. The node then CONTINUES... but only after LangGraph has run it
again FROM ITS FIRST LINE. That is the sharp edge of this stage:

    code before `interrupt()` runs twice: once before the pause, once again when the run resumes.

So a node must do nothing irreversible before an interrupt. Put side effects after it, or make them idempotent.

The flow here: find an order id in the customer's message. If there is none, ask which order. Check the answer. After
two bad answers, hand over to a person.

Run the offline demo:  python stage9_commands.py
"""

import operator
import re
from typing import Annotated, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from desk import Desk, DeskError, money
from tools import DeskContext

ORDER_ID = re.compile(r"\b[A-Z]\d{4}\b")
MAX_QUESTIONS = 2


class Request(TypedDict):
    """What the caller must provide. Nothing else goes in."""

    text: str


class Answer(TypedDict, total=False):
    """What the caller gets back. The working keys stay inside the graph."""

    outcome: dict
    log: list[str]


class ClarifyState(TypedDict, total=False):
    text: str
    order_id: str
    questions_asked: int
    outcome: dict
    log: Annotated[list[str], operator.add]


def build_clarify_graph(desk: Desk, checkpointer=None):
    def find_order(state: ClarifyState) -> Command[Literal["quote", "clarify"]]:
        found = ORDER_ID.search(state["text"])
        if found:
            return Command(update={"order_id": found.group(), "log": [f"found {found.group()} in the message"]}, goto="quote")
        return Command(update={"log": ["no order id in the message"]}, goto="clarify")

    def clarify(state: ClarifyState, runtime: Runtime[DeskContext]) -> Command[Literal["quote", "clarify", "handoff"]]:
        asked = state.get("questions_asked", 0)
        # Everything above this line runs again on resume. There is nothing here but reading.
        reply = interrupt({"question": "Which order do you mean? For example A1001.", "attempt": asked + 1})
        candidate = str(reply).strip().upper()
        try:
            if not ORDER_ID.fullmatch(candidate):
                raise DeskError("that does not look like an order id")
            desk.get_order(runtime.context.customer_id, candidate)
        except DeskError:
            if asked + 1 >= MAX_QUESTIONS:
                return Command(update={"questions_asked": asked + 1, "log": [f"bad answer {asked + 1}, giving up"]}, goto="handoff")
            return Command(update={"questions_asked": asked + 1, "log": [f"bad answer {asked + 1}"]}, goto="clarify")
        return Command(update={"order_id": candidate, "log": [f"customer chose {candidate}"]}, goto="quote")

    def quote(state: ClarifyState, runtime: Runtime[DeskContext]) -> dict:
        try:
            order = desk.get_order(runtime.context.customer_id, state["order_id"])
        except DeskError as exc:  # the id was in the message but is not this customer's
            return {"outcome": {"status": "not_found", "detail": str(exc)}}
        policy = desk.policy_for(order)
        return {"outcome": {"status": "quote", "order_id": order.order_id, "eligible": policy.eligible, "refundable": money(policy.refundable_cents)}}

    def handoff(state: ClarifyState) -> dict:
        return {"outcome": {"status": "handoff", "detail": f"no valid order id after {state.get('questions_asked', 0)} questions"}}

    graph = StateGraph(ClarifyState, input_schema=Request, output_schema=Answer, context_schema=DeskContext)
    graph.add_node("find_order", find_order)
    graph.add_node("clarify", clarify)
    graph.add_node("quote", quote)
    graph.add_node("handoff", handoff)
    graph.add_edge(START, "find_order")
    graph.add_edge("quote", END)
    graph.add_edge("handoff", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def start(app, text: str, *, customer_id: str = "C-100", thread_id: str) -> dict:
    return app.invoke({"text": text}, {"configurable": {"thread_id": thread_id}}, context=DeskContext(customer_id))


class NoPendingQuestion(RuntimeError):
    pass


def reply(app, answer: str, *, customer_id: str = "C-100", thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    # Resuming a thread that is not waiting does not fail clearly. On an empty thread it raised KeyError('text') here,
    # because the resume value was treated as the graph's input. So check first.
    if not app.get_state(config).interrupts:
        raise NoPendingQuestion(f"thread {thread_id!r} is not waiting for an answer")
    return app.invoke(Command(resume=answer), config, context=DeskContext(customer_id))


def pending_question(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__")
    return interrupts[0].value if interrupts else None


def side_effect_runs(before_interrupt: bool) -> int:
    """How many times does a side effect run if it sits before the interrupt, versus after it?"""

    class State(TypedDict, total=False):
        answer: str

    runs = []

    def node(state: State) -> dict:
        if before_interrupt:
            runs.append("sent an email")
        answer = interrupt("anything else?")
        if not before_interrupt:
            runs.append("sent an email")
        return {"answer": answer}

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    app.invoke({}, config)
    app.invoke(Command(resume="no"), config)
    return len(runs)


def demo() -> None:
    app = build_clarify_graph(Desk())
    print("1. the id is in the message")
    print("  ", start(app, "Can I return the shoes from order A1001?", thread_id="t1"))
    print("\n2. no id: the graph asks, the customer answers")
    paused = start(app, "I want to return my shoes.", thread_id="t2")
    print("   asked:", pending_question(paused))
    print("  ", reply(app, "a1001", thread_id="t2"))
    print("\n3. two bad answers: a person takes over")
    start(app, "I want a refund.", thread_id="t3")
    second = reply(app, "the blue ones", thread_id="t3")
    print("   asked again:", pending_question(second))
    print("  ", reply(app, "I forgot", thread_id="t3"))
    print("\nside effect before the interrupt runs", side_effect_runs(True), "times; after it,", side_effect_runs(False), "time")


if __name__ == "__main__":
    demo()
