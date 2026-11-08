"""Stage 7: the whole support desk, as one workflow graph.

Every earlier stage is one part. Here they are joined:

    START -> triage -> (route) -> refund_flow | answer_faq | escalate -> finish -> END

- `triage` (stage 3) classifies the message with a schema, then CODE checks the claims and decides the route.
- `refund_flow` is the hardened agent (stages 1, 5 and 6) used as a SUBGRAPH. It can pause for a human.
- `answer_faq` answers only from help articles that code found. No match means no model call: it goes to a person.
- `escalate` opens a ticket. It makes no model call at all.
- `finish` writes a Resolution from facts in the ledger, not from the model's words.

The model never chooses a route. `decide_route` is plain Python and it is the only place that decides. That is the
difference between an agent that wanders and a workflow you can draw, test and explain.

One checkpointer, on the parent graph, saves everything, including the pause inside `refund_flow`.

Run the offline demo:  python stage7_support_desk.py
"""

import operator
from dataclasses import dataclass
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from desk import Desk, money
from fakes import ScriptedChatModel, call, say
from show import final_text
from stage3_triage import triage_request
from stage5_approval import approve, pending_approval, resume
from stage6_middleware import AuditTrail, build_hardened_agent
from tools import DeskContext

URGENT_AT = 4

FAQ_INSTRUCTIONS = "Answer the customer using ONLY the help articles below. If they do not answer the question, say that you do not know.\n\n"


@dataclass
class DeskModels:
    """One model for each job. In production they can differ: a small cheap model for triage, a strong one for the agent."""

    triage: BaseChatModel
    agent: BaseChatModel
    faq: BaseChatModel


class DeskState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    triage: dict
    route: str
    reason: str
    ticket: dict
    resolution: dict
    events: Annotated[list[str], operator.add]


def decide_route(triage: dict) -> tuple[str, str]:
    """Where does this request go, and why? Verified facts in, a route out. No model is involved."""
    if not triage["verified"]:
        return "escalate", "the request could not be verified: " + "; ".join(triage["problems"])
    if triage["urgency"] >= URGENT_AT:
        return "escalate", f"urgency {triage['urgency']} of 5"
    if triage["category"] == "refund":
        return "refund_flow", "a verified refund request"
    if triage["category"] == "faq":
        if triage["faq_hits"]:
            return "answer_faq", "help articles match"
        return "escalate", "no help article matches"
    return "escalate", "not a refund or a help question"


def build_desk(models: DeskModels, desk: Desk, trail: AuditTrail | None = None, *, checkpointer=None):
    trail = trail or AuditTrail()
    refund_agent = build_hardened_agent(models.agent, desk, trail, approval=True, checkpointer=None)  # None: inherit the parent's

    def triage(state: DeskState, runtime: Runtime[DeskContext]) -> dict:
        text = state["messages"][-1].content
        result = triage_request(models.triage, text, desk, runtime.context.customer_id)
        record = {**result.triage.model_dump(), "verified": result.verified, "problems": result.problems,
                  "attempts": result.attempts, "faq_hits": len(desk.faq_search(text))}
        route, reason = decide_route(record)
        return {"triage": record, "route": route, "reason": reason, "events": [f"triage -> {route}"]}

    def answer_faq(state: DeskState) -> dict:
        question = state["messages"][-1].content
        articles = "\n".join(f"- {a['title']} {a['text']}" for a in desk.faq_search(question))
        reply = models.faq.invoke([SystemMessage(FAQ_INSTRUCTIONS + articles), HumanMessage(question)])
        return {"messages": [reply], "events": ["answer_faq"]}

    def escalate(state: DeskState, runtime: Runtime[DeskContext]) -> dict:
        ticket = desk.open_ticket(runtime.context.customer_id, state["reason"], state["triage"]["summary"])
        note = f"I have passed this to a person. Your reference is {ticket['ticket_id']}."
        return {"messages": [AIMessage(note)], "ticket": ticket, "events": ["escalate"]}

    def finish(state: DeskState, config: RunnableConfig) -> dict:
        route = state["route"]
        if route == "refund_flow":
            # Refund keys start with the thread id, so this finds exactly the refunds that THIS ticket caused,
            # even when other tickets are running at the same time.
            paid = [r for r in desk.refunds if r.key.startswith(config["configurable"]["thread_id"] + ":")]
            status = "refunded" if paid else "declined"
            resolution = {"status": status, "refunded_cents": sum(r.amount_cents for r in paid), "refund_ids": [r.refund_id for r in paid]}
        elif route == "answer_faq":
            resolution = {"status": "answered", "refunded_cents": 0, "refund_ids": []}
        else:
            resolution = {"status": "escalated", "refunded_cents": 0, "refund_ids": [], "ticket_id": state["ticket"]["ticket_id"]}
        return {"resolution": resolution, "events": ["finish"]}

    graph = StateGraph(DeskState, context_schema=DeskContext)
    graph.add_node("triage", triage)
    graph.add_node("refund_flow", refund_agent)  # a whole compiled agent used as one node
    graph.add_node("answer_faq", answer_faq)
    graph.add_node("escalate", escalate)
    graph.add_node("finish", finish)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", lambda state: state["route"], {"refund_flow": "refund_flow", "answer_faq": "answer_faq", "escalate": "escalate"})
    for node in ("refund_flow", "answer_faq", "escalate"):
        graph.add_edge(node, "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def handle(app, text: str, *, customer_id: str = "C-100", customer_name: str = "the customer", thread_id: str) -> dict:
    """Start a ticket. If a refund needs a person, the result holds a pause: read it with `pending_approval`."""
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"messages": [HumanMessage(text)]}, config, context=DeskContext(customer_id, customer_name))


def path_of(app, text: str, *, customer_id: str = "C-100", thread_id: str) -> list[str]:
    """Run a ticket and list the nodes it visited, in order. Useful for teaching, tests and debugging."""
    config = {"configurable": {"thread_id": thread_id}}
    nodes = []
    for chunk in app.stream({"messages": [HumanMessage(text)]}, config, context=DeskContext(customer_id), stream_mode="updates"):
        nodes.extend(chunk.keys())
    return nodes


def demo() -> None:
    desk = Desk()
    models = DeskModels(
        triage=ScriptedChatModel(script=[
            call("Triage", {"category": "refund", "order_id": "A1001", "urgency": 2, "summary": "Wants a refund for A1001."}),
            call("Triage", {"category": "faq", "urgency": 1, "summary": "Asks how long shipping takes."}),
            call("Triage", {"category": "other", "urgency": 5, "summary": "Furious about a late parcel."}),
        ]),
        agent=ScriptedChatModel(script=[
            call("check_refund_policy", {"order_id": "A1001"}),
            call("issue_refund", {"order_id": "A1001", "amount": 89.0, "reason": "shoes are too small"}),
            say("I have refunded $89.00 for order A1001."),
        ]),
        faq=ScriptedChatModel(script=[say("Standard shipping takes 3 to 5 business days.")]),
    )
    app = build_desk(models, desk)

    print("TICKET 1: a refund")
    paused = handle(app, "Please refund order A1001, the shoes are too small.", thread_id="t1")
    request = pending_approval(paused)
    print(f"  paused for a person: {request.description}")
    done = resume(app, approve(), "t1")
    print(f"  resolution: {done['resolution']}")
    print(f"  customer sees: {final_text(done['messages'])}")

    print("\nTICKET 2: a help question")
    done = handle(app, "How long does shipping take?", thread_id="t2")
    print(f"  events: {done['events']}\n  customer sees: {final_text(done['messages'])}")

    print("\nTICKET 3: urgent")
    done = handle(app, "My parcel is a week late and I am furious!", thread_id="t3")
    print(f"  events: {done['events']}\n  reason: {done['reason']}\n  customer sees: {final_text(done['messages'])}")
    print(f"\nrefunds in the ledger: {[(r.refund_id, money(r.amount_cents)) for r in desk.refunds]}   tickets: {[t['ticket_id'] for t in desk.tickets]}")


if __name__ == "__main__":
    demo()
