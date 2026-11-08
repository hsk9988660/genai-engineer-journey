"""Stage 2: the same agent, built by hand as a LangGraph graph.

`create_agent` (stage 1) returns a compiled LangGraph graph. This file builds that graph yourself, so nothing is
hidden. Four ideas:

- STATE is a typed dictionary that every node reads and writes.
- A REDUCER says how a node's update is merged into one key. `add_messages` appends. `operator.add` concatenates lists.
  A key with no reducer is simply overwritten.
- A NODE is a function: state in, partial state out.
- An EDGE says what runs next. A CONDITIONAL edge calls a function of the state to choose. That choice is made by
  your code, never by the model.

Run the offline demo:  python stage2_graph_by_hand.py
"""

import operator
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from desk import Desk
from fakes import ScriptedChatModel, call, say
from show import format_messages
from stage1_first_agent import SYSTEM_PROMPT
from tools import DeskContext, make_tools


class DeskState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # reducer: append (and replace a message that has the same id)
    tools_used: Annotated[list[str], operator.add]  # reducer: concatenate lists
    model_calls: int  # no reducer: the latest value wins


def build_graph(model: BaseChatModel, desk: Desk, *, max_model_calls: int = 6, checkpointer=None):
    tools = make_tools(desk)
    model_with_tools = model.bind_tools(tools)  # tells the model which tools exist

    def call_model(state: DeskState) -> dict:
        response = model_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {
            "messages": [response],
            "tools_used": [tool_call["name"] for tool_call in response.tool_calls],
            "model_calls": state.get("model_calls", 0) + 1,
        }

    def route_after_model(state: DeskState) -> Literal["tools", "give_up", "__end__"]:
        last = state["messages"][-1]
        if not last.tool_calls:
            return END
        if state["model_calls"] >= max_model_calls:
            return "give_up"
        return "tools"

    def give_up(state: DeskState) -> dict:
        # Every tool call in the history needs a result, or the next model call is rejected. So close each one.
        pending = state["messages"][-1].tool_calls
        closed = [ToolMessage(content="not run: the step limit was reached", tool_call_id=c["id"], status="error") for c in pending]
        return {"messages": [*closed, AIMessage("Stopped: the step limit was reached before the agent finished.")]}

    graph = StateGraph(DeskState, context_schema=DeskContext)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))  # runs every tool call in the last AI message and returns ToolMessages
    graph.add_node("give_up", give_up)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route_after_model, {"tools": "tools", "give_up": "give_up", END: END})
    graph.add_edge("tools", "model")
    graph.add_edge("give_up", END)
    return graph.compile(checkpointer=checkpointer)


def run(graph, text: str, customer_id: str = "C-100", thread_id: str | None = None) -> dict:
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    return graph.invoke({"messages": [{"role": "user", "content": text}]}, config, context=DeskContext(customer_id=customer_id))


class LogState(TypedDict):
    log: Annotated[list[str], operator.add]


def shared_reducer_gotcha() -> tuple[list[str], list[str]]:
    """Show a trap with subgraphs. Returns (naive, fixed).

    A subgraph returns its WHOLE final state. If the parent and the subgraph share a key that has an additive
    reducer, the parent adds the old items a second time. Messages are safe because `add_messages` skips an id it has
    already seen. A plain `operator.add` list is not safe.
    """

    def build_sub():
        sub = StateGraph(LogState)
        sub.add_node("work", lambda state: {"log": ["sub did work"]})
        sub.add_edge(START, "work")
        sub.add_edge("work", END)
        return sub.compile()

    sub = build_sub()

    naive = StateGraph(LogState)
    naive.add_node("before", lambda state: {"log": ["before"]})
    naive.add_node("sub", sub)  # the subgraph is added directly as a node
    naive.add_edge(START, "before")
    naive.add_edge("before", "sub")
    naive.add_edge("sub", END)

    def call_sub_with_empty_log(state: LogState) -> dict:
        return {"log": sub.invoke({"log": []})["log"]}  # start the subgraph empty, so it returns only what it added

    fixed = StateGraph(LogState)
    fixed.add_node("before", lambda state: {"log": ["before"]})
    fixed.add_node("sub", call_sub_with_empty_log)
    fixed.add_edge(START, "before")
    fixed.add_edge("before", "sub")
    fixed.add_edge("sub", END)

    return naive.compile().invoke({"log": []})["log"], fixed.compile().invoke({"log": []})["log"]


def demo() -> None:
    model = ScriptedChatModel(script=[
        call("check_refund_policy", {"order_id": "A1003"}),
        say("Order A1003 is outside the 30-day return window, so I cannot refund it."),
    ])
    graph = build_graph(model, Desk())
    result = run(graph, "Can I get a refund for A1003?")
    print("\n".join(format_messages(result["messages"])))
    print(f"\ntools_used={result['tools_used']}  model_calls={result['model_calls']}")
    print("\nThe graph, as a Mermaid diagram (paste it into https://mermaid.live):\n")
    print(graph.get_graph().draw_mermaid())
    naive, fixed = shared_reducer_gotcha()
    print(f"subgraph with a shared additive key : {naive}")
    print(f"subgraph started empty              : {fixed}")


if __name__ == "__main__":
    demo()
