"""The agent's tools: thin LangChain wrappers around the rules in desk.py.

Three ideas to notice:

1. `@tool` turns a function into a tool. The name, the docstring and the type hints become the schema that the model
   sees. The docstring is a prompt: write it for the model.
2. `runtime: ToolRuntime[DeskContext]` is injected by LangChain and is NOT part of the schema. The model cannot see it
   or fill it in. This is how the verified customer id reaches the tools without the model ever supplying it.
3. Business errors are returned as data (`{"error": ...}`) so the model can read them and recover. A
   `TemporaryOutage` is different: it is raised, so retry middleware (stage 6) can retry it.
"""

from dataclasses import dataclass

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from desk import Desk, DeskError, TemporaryOutage, money, to_cents


@dataclass(frozen=True)
class DeskContext:
    """Facts about the caller that code has verified. They travel beside the conversation, not inside it."""

    customer_id: str
    customer_name: str = "the customer"
    audience: str = "public"  # "public" for customers, "staff" for employees. Stage 14 uses it to hide internal articles


def request_key(runtime: ToolRuntime) -> str:
    """The idempotency key for a tool call: unique per conversation thread, stable if the same call is replayed.

    A tool-call id alone is not enough. A provider may repeat ids across conversations, and then two different customers
    would collide. Putting the thread id in front also lets us find, later, every refund that one conversation caused.
    """
    thread_id = (runtime.config or {}).get("configurable", {}).get("thread_id")
    return f"{thread_id}:{runtime.tool_call_id}" if thread_id else runtime.tool_call_id


def make_tools(desk: Desk) -> list[BaseTool]:
    """Build the four tools around one Desk. A fresh Desk per test means a fresh ledger per test."""

    @tool
    def lookup_order(order_id: str, runtime: ToolRuntime[DeskContext]) -> dict:
        """Look up ONE of this customer's orders by its id (for example A1001). Returns the item, the total, and the delivery status."""
        try:
            order = desk.get_order(runtime.context.customer_id, order_id)
        except TemporaryOutage:
            raise
        except DeskError as exc:
            return {"error": str(exc)}
        return {"order_id": order.order_id, "item": order.item, "total": money(order.total_cents), "status": order.status,
                "delivered_on": order.delivered_on.isoformat() if order.delivered_on else None}

    @tool
    def check_refund_policy(order_id: str, runtime: ToolRuntime[DeskContext]) -> dict:
        """Check whether an order can be refunded and how much is still refundable. Always call this before issue_refund."""
        try:
            order = desk.get_order(runtime.context.customer_id, order_id)
        except TemporaryOutage:
            raise
        except DeskError as exc:
            return {"error": str(exc)}
        policy = desk.policy_for(order)
        return {"order_id": order_id, "eligible": policy.eligible, "refundable": money(policy.refundable_cents), "reason": policy.reason}

    @tool
    def issue_refund(order_id: str, amount: float, reason: str, runtime: ToolRuntime[DeskContext]) -> dict:
        """Refund money for an order. `amount` is in dollars, for example 89.00. This moves real money, so only call it after check_refund_policy says the order is eligible."""
        try:
            # The same call, replayed, pays once.
            record = desk.issue_refund(runtime.context.customer_id, order_id, to_cents(amount), reason, key=request_key(runtime))
        except TemporaryOutage:
            raise
        except DeskError as exc:
            return {"error": str(exc)}
        return {"refund_id": record.refund_id, "order_id": record.order_id, "refunded": money(record.amount_cents)}

    @tool
    def search_faq(query: str) -> dict:
        """Search the help articles (shipping, returns, warranty). Returns matching articles, or an empty list when nothing matches."""
        return {"articles": desk.faq_search(query)}

    return [lookup_order, check_refund_policy, issue_refund, search_faq]
