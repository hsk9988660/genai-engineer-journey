"""Stage 13: messages and tools in depth.

Two halves. Both are about the exact objects that pass between your code and the model.

MESSAGES are the model's whole world. Four things to know:
- An `AIMessage` can carry text (`.text`), tool calls (`.tool_calls`) and token counts (`.usage_metadata`).
- `.content_blocks` shows any message as a list of typed blocks (text, tool call, image...), whatever the provider.
- History must stay VALID. A `ToolMessage` must follow the `AIMessage` that asked for it. Cut the history in the wrong
  place and a provider rejects the request. `trim_messages` cuts in the right places. A plain slice does not.
- `RemoveMessage(id=...)` deletes a message from saved state through the `add_messages` reducer.

TOOLS can say more than "here is some text":
- `args_schema` with pydantic constraints (`gt=0`, `le=200`) is sent to the model AND enforced before your function runs.
- `ToolException` with `handle_tool_error=True` sends the error back flagged `status="error"`. Returning
  `{"error": ...}` (stages 1 to 7) is a normal result that happens to say no. Choose on purpose; see the test names.
- `response_format="content_and_artifact"` returns TWO things: short text for the model, and a full payload for your
  application that the model never needs to read.
- `Command(update=...)` from a tool writes to the agent's state, not just to the conversation.
- `InjectedState` gives a tool a read-only view of the state, and is hidden from the model like `ToolRuntime`.

Run the offline demo:  python stage13_messages_tools.py
"""

from typing import Annotated

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage, trim_messages
from langchain.tools import InjectedState, InjectedToolCallId, ToolException, ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from pydantic import BaseModel, Field

from desk import Desk, DeskError, TemporaryOutage, money, to_cents
from fakes import ScriptedChatModel, call, say
from tools import DeskContext


def safe_window(messages: list[BaseMessage], keep: int) -> list[BaseMessage]:
    """The system message plus the newest messages, cut so the history still starts on a person's message."""
    return trim_messages(messages, max_tokens=keep, token_counter=len, strategy="last", start_on="human", include_system=True)


def naive_window(messages: list[BaseMessage], keep: int) -> list[BaseMessage]:
    """The tempting version. It can start in the middle of a tool call, which a provider rejects."""
    return messages[-keep:]


def trim_history(keep: int):
    """Middleware that shrinks the SAVED history before each model call. Uses RemoveMessage under the hood."""

    @before_model
    def trimmer(state: AgentState, runtime) -> dict | None:
        messages = state["messages"]
        kept = safe_window(messages, keep)
        if len(kept) == len(messages):
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}  # replace the whole list with the kept part

    return trimmer


class RefundArgs(BaseModel):
    """What the model must send. The limits are shown to the model and enforced before the function runs."""

    order_id: str = Field(pattern=r"^[A-Z]\d{4}$", description="An order id such as A1001")
    amount: float = Field(gt=0, le=200, description="Dollars, at most 200")
    reason: str = Field(min_length=3)


def make_demo_tools(desk: Desk) -> list:
    @tool(args_schema=RefundArgs)
    def refund_with_exception(order_id: str, amount: float, reason: str, runtime: ToolRuntime[DeskContext]) -> str:
        """Refund an order. A refusal is raised as ToolException, so the message comes back flagged as an error."""
        try:
            record = desk.issue_refund(runtime.context.customer_id, order_id, to_cents(amount), reason, key=runtime.tool_call_id)
        except TemporaryOutage:
            raise
        except DeskError as exc:
            raise ToolException(str(exc)) from exc
        return f"refunded {money(record.amount_cents)} on {order_id}"

    refund_with_exception.handle_tool_error = True  # a ToolException becomes a message for the model instead of ending the run

    @tool(response_format="content_and_artifact")
    def order_receipt(order_id: str, runtime: ToolRuntime[DeskContext]) -> tuple[str, dict]:
        """Get a receipt for an order. The model gets one line; the app gets the full receipt."""
        try:
            order = desk.get_order(runtime.context.customer_id, order_id)
        except TemporaryOutage:
            raise
        except DeskError as exc:
            raise ToolException(str(exc)) from exc
        receipt = {"order_id": order.order_id, "item": order.item, "total_cents": order.total_cents, "refunded_cents": desk.refunded_cents(order_id)}
        return f"{order.item}, {money(order.total_cents)}", receipt

    order_receipt.handle_tool_error = True

    @tool
    def flag_for_followup(reason: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Ask a person to follow up on this conversation later. Writes to the agent's state."""
        return Command(update={"followup": reason, "messages": [ToolMessage(f"A follow-up was requested: {reason}", tool_call_id=tool_call_id)]})

    @tool
    def messages_so_far(state: Annotated[dict, InjectedState]) -> str:
        """Say how many messages the conversation has so far."""
        return f"{len(state['messages'])} messages"

    return [refund_with_exception, order_receipt, flag_for_followup, messages_so_far]


class FollowupState(AgentState):
    followup: str


def build_demo_agent(model: BaseChatModel, desk: Desk, *, middleware=()):
    return create_agent(model, make_demo_tools(desk), state_schema=FollowupState, context_schema=DeskContext, middleware=list(middleware))


def run(agent, text: str, customer_id: str = "C-100") -> dict:
    return agent.invoke({"messages": [HumanMessage(text)]}, context=DeskContext(customer_id))


def demo() -> None:
    history = [SystemMessage("rules", id="s"), HumanMessage("q1", id="h1"),
               AIMessage("", id="a1", tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}]),
               ToolMessage("r1", tool_call_id="c1", id="t1"), AIMessage("a2", id="a2"), HumanMessage("q2", id="h2"), AIMessage("a3", id="a3")]
    print("a plain slice of the last 4 starts with:", type(naive_window(history, 4)[0]).__name__, "(an orphan tool result)")
    print("safe_window keeps:", [m.id for m in safe_window(history, 4)])

    model = ScriptedChatModel(script=[
        call("refund_with_exception", {"order_id": "A1001", "amount": 500, "reason": "too small"}),
        call("order_receipt", {"order_id": "A1001"}),
        call("flag_for_followup", {"reason": "customer wants a call"}),
        say("Done."),
    ])
    result = run(build_demo_agent(model, Desk()), "Refund A1001, show the receipt, and flag this.")
    for message in result["messages"]:
        if message.type == "tool":
            print(f"tool result  status={message.status!r:8} content={str(message.content)[:70]!r}  artifact={message.artifact}")
    print("state key written by a tool:", result["followup"])
    print("tools the model was told about:", model.bound[0]["tools"])


if __name__ == "__main__":
    demo()
