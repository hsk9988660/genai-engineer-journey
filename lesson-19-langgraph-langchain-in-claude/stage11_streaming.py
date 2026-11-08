"""Stage 11: streaming. Showing progress before the answer is ready.

A support agent can take many seconds. A user who sees nothing for ten seconds thinks it is broken. Streaming sends
partial results as they happen. LangGraph has several `stream_mode`s. You can ask for one, or a list of them:

| mode       | each chunk is...                                              | use it for                          |
|------------|---------------------------------------------------------------|-------------------------------------|
| "values"   | the WHOLE state after each step                               | a UI that redraws from full state   |
| "updates"  | `{node_name: what_that_node_returned}`                        | "step 3 of 5 finished" progress     |
| "messages" | `(token, metadata)` for each token of any model call          | typing the answer word by word      |
| "custom"   | whatever a node passes to its stream writer                   | your own progress events            |
| "debug"    | detailed task and result records                              | debugging                           |

With a list of modes, every chunk is a pair `(mode, data)`.

Two more things:
- `subgraphs=True` also shows events from graphs nested inside nodes (the refund agent in stage 7). Chunks then become
  `(namespace, mode, data)`.
- Every method has an async twin: `astream` for `stream`, `ainvoke` for `invoke`. A web server uses the async ones.

Run the offline demo:  python stage11_streaming.py
"""

import asyncio
from collections.abc import AsyncIterator, Iterator

from langchain_core.messages import HumanMessage

from desk import Desk
from fakes import ScriptedChatModel, call, say
from stage7_support_desk import DeskModels, build_desk
from stage8_parallel import build_report_graph
from tools import DeskContext

MODES = ["updates", "messages", "custom"]


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _event(mode: str, data) -> dict | None:
    """Turn one raw chunk into a small typed event that a UI could use. Returns None for chunks we do not show."""
    if mode == "updates":
        for node, update in data.items():
            if node == "__interrupt__":
                return {"kind": "pause"}
            return {"kind": "node", "node": node}
    if mode == "messages":
        token, metadata = data
        text = token.content if isinstance(token.content, str) else ""
        if text:  # tool-call chunks and usage chunks have no text, so they are skipped
            return {"kind": "token", "text": text, "node": metadata.get("langgraph_node")}
    if mode == "custom":
        return {"kind": "progress", **data}
    return None


def ticket_events(app, text: str, *, customer_id: str = "C-100", thread_id: str) -> Iterator[dict]:
    """Stream a desk ticket as events: node finished, token, progress, pause."""
    stream = app.stream({"messages": [HumanMessage(text)]}, _config(thread_id), context=DeskContext(customer_id), stream_mode=MODES)
    for mode, data in stream:
        event = _event(mode, data)
        if event:
            yield event


async def ticket_events_async(app, text: str, *, customer_id: str = "C-100", thread_id: str) -> AsyncIterator[dict]:
    """The same stream, for an async server."""
    stream = app.astream({"messages": [HumanMessage(text)]}, _config(thread_id), context=DeskContext(customer_id), stream_mode=MODES)
    async for mode, data in stream:
        event = _event(mode, data)
        if event:
            yield event


def report_events(desk: Desk, customer_id: str, order_ids: list[str]) -> Iterator[dict]:
    """Stream the parallel order report from stage 8: a progress event for each order, then the finished nodes."""
    app = build_report_graph(desk)
    for mode, data in app.stream({"customer_id": customer_id, "order_ids": order_ids, "results": []}, stream_mode=["updates", "custom"]):
        event = _event(mode, data)
        if event:
            yield event


def nested_nodes(app, text: str, *, customer_id: str = "C-100", thread_id: str) -> list[tuple[str, ...]]:
    """With `subgraphs=True`, list (namespace, node) pairs. The namespace is empty for the top graph."""
    seen = []
    stream = app.stream({"messages": [HumanMessage(text)]}, _config(thread_id), context=DeskContext(customer_id), stream_mode="updates", subgraphs=True)
    for namespace, update in stream:
        for node in update:
            seen.append((namespace[0].split(":")[0] if namespace else "", node))
    return seen


def spoken_text(events: list[dict]) -> str:
    return "".join(e["text"] for e in events if e["kind"] == "token")


def demo() -> None:
    desk = Desk()
    models = DeskModels(
        triage=ScriptedChatModel(script=[call("Triage", {"category": "faq", "urgency": 1, "summary": "Asks about shipping."}),
                                         call("Triage", {"category": "refund", "order_id": "A1001", "urgency": 2, "summary": "Refund."})]),
        agent=ScriptedChatModel(script=[call("issue_refund", {"order_id": "A1001", "amount": 89.0, "reason": "x"}), say("Done.")]),
        faq=ScriptedChatModel(script=[say("Standard shipping takes 3 to 5 business days.")]),
    )
    app = build_desk(models, desk)
    print("a help question, streamed:")
    for event in ticket_events(app, "How long does shipping take?", thread_id="t1"):
        print("  ", event)
    print("\nthe parallel report, streamed:")
    for event in report_events(desk, "C-100", ["A1001", "A1002"]):
        print("  ", event)
    print("\na refund, with the nested agent visible:")
    print("  ", nested_nodes(app, "Refund order A1001 please.", thread_id="t2"))
    print("   paused for a person:", bool(app.get_state(_config("t2")).interrupts))
    events = asyncio.run(_collect(ticket_events_async(build_desk(DeskModels(
        ScriptedChatModel(script=[call("Triage", {"category": "faq", "urgency": 1, "summary": "s"})]), ScriptedChatModel(script=[]),
        ScriptedChatModel(script=[say("Async works too.")])), desk), "How long does shipping take?", thread_id="t3")))
    print("\nasync:", spoken_text(events))


async def _collect(stream) -> list[dict]:
    return [event async for event in stream]


if __name__ == "__main__":
    demo()
