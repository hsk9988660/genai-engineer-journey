"""Stage 1: your first LangChain agent.

`create_agent` runs the loop you wrote by hand in lesson 01: ask the model, run the tools it requests, send the
results back, repeat until the model answers. You give it a model and a list of tools. It gives you back a runnable
agent.

Run the offline demo:  python stage1_first_agent.py
"""

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from desk import Desk
from fakes import ScriptedChatModel, call, say
from tools import DeskContext, make_tools
from show import format_messages

SYSTEM_PROMPT = (
    "You are a support agent for an outdoor-gear shop. Look up facts with the tools before you answer. "
    "Never guess an order's details. If a tool returns an error, tell the customer plainly."
)


def build_agent(model: BaseChatModel, desk: Desk, *, extra_tools: Sequence = (), middleware: Sequence = (), checkpointer=None, store=None,
                system_prompt: str | None = SYSTEM_PROMPT):
    """The one place where the agent is assembled. Later stages pass in more tools, middleware, a checkpointer or a store."""
    return create_agent(
        model,
        [*make_tools(desk), *extra_tools],
        system_prompt=system_prompt,
        middleware=list(middleware),
        checkpointer=checkpointer,
        store=store,
        context_schema=DeskContext,  # the type of the facts that code passes beside the conversation
    )


def ask(agent, text: str, customer_id: str = "C-100", thread_id: str | None = None) -> dict:
    """Send one customer message and return the final state: {"messages": [...]}."""
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, config, context=DeskContext(customer_id=customer_id))


def demo() -> None:
    model = ScriptedChatModel(script=[
        call("lookup_order", {"order_id": "A1001"}),
        say("Order A1001 is a pair of Trail Shoes for $89.00. It was delivered on 2026-10-20."),
    ])
    result = ask(build_agent(model, Desk()), "What is in order A1001?")
    print("\n".join(format_messages(result["messages"])))
    print(f"\nmodel calls: {len(model.seen)}")


if __name__ == "__main__":
    demo()
