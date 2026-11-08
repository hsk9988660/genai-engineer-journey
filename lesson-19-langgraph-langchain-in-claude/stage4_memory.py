"""Stage 4: memory. Two kinds, for two jobs.

SHORT-TERM memory is the conversation so far. A CHECKPOINTER saves the graph's state after every step, under a
`thread_id`. Call the agent again with the same thread id and it continues that conversation. A new thread id starts
fresh. Because every step is saved, you can also look back through the history of a thread.

LONG-TERM memory is a fact that should survive across conversations: "this customer wants email, not phone". A STORE
holds such facts under a namespace. The namespace here includes the customer id, and the customer id comes from the
verified context. The model never supplies it, so the model cannot read or write another customer's notes (compare
lesson 13).

`InMemorySaver` and `InMemoryStore` live in RAM and vanish when the program ends. They are right for learning and
tests. In production you swap in a database-backed saver or store (SQLite, Postgres). The agent code does not change.

Run the offline demo:  python stage4_memory.py
"""

import uuid

from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from desk import Desk
from fakes import ScriptedChatModel, call, say
from show import format_messages
from stage1_first_agent import ask, build_agent
from tools import DeskContext


@tool
def remember_preference(preference: str, runtime: ToolRuntime[DeskContext]) -> str:
    """Save ONE lasting preference of this customer, for example 'contact me by email only'. Do not save order details or anything sensitive."""
    namespace = ("preferences", runtime.context.customer_id)
    runtime.store.put(namespace, uuid.uuid4().hex[:8], {"text": preference.strip()[:200]})
    return "saved"


@tool
def recall_preferences(runtime: ToolRuntime[DeskContext]) -> list[str]:
    """List the lasting preferences that this customer saved in earlier conversations."""
    items = runtime.store.search(("preferences", runtime.context.customer_id))
    return sorted(item.value["text"] for item in items)


def build_memory_agent(model: BaseChatModel, desk: Desk, *, checkpointer: InMemorySaver | None = None, store: InMemoryStore | None = None):
    return build_agent(model, desk, extra_tools=[remember_preference, recall_preferences],
                       checkpointer=checkpointer or InMemorySaver(), store=store or InMemoryStore())


def thread_summary(agent, thread_id: str) -> dict:
    """What is saved for a thread right now: how many messages, and what runs next (empty when the run is finished)."""
    snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})
    return {"messages": len(snapshot.values.get("messages", [])), "next": snapshot.next}


def checkpoint_history(agent, thread_id: str) -> list[dict]:
    """Every saved step of a thread, oldest first. This is what makes replay and 'time travel' possible."""
    history = list(agent.get_state_history({"configurable": {"thread_id": thread_id}}))
    return [{"next": snap.next, "messages": len(snap.values.get("messages", []))} for snap in reversed(history)]


def demo() -> None:
    desk = Desk()
    model = ScriptedChatModel(script=[
        call("lookup_order", {"order_id": "A1001"}), say("A1001 is a pair of Trail Shoes for $89.00."),
        say("The shoes cost $89.00, as I said."),
        call("remember_preference", {"preference": "contact me by email only"}), say("I will remember that."),
    ])
    agent = build_memory_agent(model, desk)
    ask(agent, "What is in order A1001?", thread_id="t1")
    ask(agent, "How much was that?", thread_id="t1")
    print("thread t1 right now:", thread_summary(agent, "t1"))
    print("the model saw, on the second question:")
    print("  " + "\n  ".join(format_messages(model.seen[2])))
    print("\nsaved steps of t1 (messages after each step):", [c["messages"] for c in checkpoint_history(agent, "t1")])
    ask(agent, "Please only contact me by email.", thread_id="t2")
    print("\nthread t2 started fresh. Its first model call saw", len(model.seen[3]) - 1, "message besides the system prompt.")


if __name__ == "__main__":
    demo()
