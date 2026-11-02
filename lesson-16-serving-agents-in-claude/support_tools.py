"""The support agent's tools, with the lesson's holes closed.

In the lesson's notebook the tools take an `order_id` from the model and know nothing about the customer, so any customer can ask about (or refund) any order. The refund tool
always asks for approval regardless of amount (its `refund_needs_approval` helper is never called), never checks the amount against the order, and is not safe to retry.

Here every tool receives `user_id` from the SERVING CODE as its first argument. It is not in any tool's schema, so the model cannot supply or change it.
  get_order_status   an order that belongs to someone else answers exactly like one that does not exist
  issue_refund       the order must be the caller's; 0 < amount <= what is still refundable; above the threshold a human must approve; repeating the same refund does nothing
  search_policies    public knowledge, the same answer for everyone
"""

import threading
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from agentdef import ToolSpec

POLICIES = {
    "returns": "Contoso accepts returns within 30 days of delivery for a full refund. Items must be unused and in original packaging.",
    "shipping": "Standard shipping takes 3-5 business days. Express shipping (1-2 days) is available at checkout for an extra fee.",
    "warranty": "All Contoso electronics carry a 12-month limited warranty covering manufacturing defects.",
    "refund policy": "Refunds are processed to the original payment method within 5 business days of approval. Refunds over $50 need a supervisor's approval.",
}
NOT_FOUND = "No order found with that ID."


@dataclass
class SupportData:
    orders: dict = field(default_factory=lambda: {
        "A1001": {"owner": "cust-99", "status": "shipped", "total": Decimal("42.00"), "eta": "2 days"},
        "A1002": {"owner": "cust-42", "status": "processing", "total": Decimal("128.50"), "eta": "5 days"},
        "A1003": {"owner": "cust-99", "status": "delivered", "total": Decimal("19.99"), "eta": "delivered"}})
    refunded: dict = field(default_factory=dict)  # order_id -> total refunded so far
    issued: set = field(default_factory=set)  # (order_id, amount) pairs already done: the idempotency record
    pending: list = field(default_factory=list)  # refunds waiting for a human
    lock: threading.Lock = field(default_factory=threading.Lock)


SPECS = (ToolSpec("search_policies", "public", "Look up a Contoso support policy."), ToolSpec("get_order_status", "user", "Look up one of the customer's own orders."),
         ToolSpec("issue_refund", "effect", "Refund part or all of one of the customer's own orders."))
SCHEMAS = {
    "search_policies": {"name": "search_policies", "description": "Look up a Contoso support policy (returns, shipping, warranty, refund policy).",
                        "input_schema": {"type": "object", "additionalProperties": False, "required": ["query"], "properties": {"query": {"type": "string", "maxLength": 200}}}},
    "get_order_status": {"name": "get_order_status", "description": "Look up the status of one of the customer's own orders.",
                         "input_schema": {"type": "object", "additionalProperties": False, "required": ["order_id"], "properties": {"order_id": {"type": "string", "maxLength": 20}}}},
    "issue_refund": {"name": "issue_refund", "description": "Refund part or all of one of the customer's own orders. Larger refunds need a human to approve them.",
                     "input_schema": {"type": "object", "additionalProperties": False, "required": ["order_id", "amount"],
                                      "properties": {"order_id": {"type": "string", "maxLength": 20}, "amount": {"type": "number", "exclusiveMinimum": 0}}}}}


def build_tools(data: SupportData, approver=None, threshold: Decimal = Decimal("50")):
    """Returns {name: impl}. `approver(summary: dict) -> True | False | None`; None means "a person will decide later"."""

    def search_policies(user_id: str, query: str) -> str:
        q = str(query).lower()
        hits = [text for key, text in POLICIES.items() if key in q or any(w in q for w in key.split())]
        return "\n".join(dict.fromkeys(hits)) or "No matching policy found."

    def get_order_status(user_id: str, order_id: str) -> str:
        order = data.orders.get(str(order_id).upper())
        if order is None or order["owner"] != user_id:
            return NOT_FOUND  # the same words whether the order is someone else's or does not exist
        return f"Order {str(order_id).upper()}: status={order['status']}, total=${order['total']:.2f}, eta={order['eta']}."

    def issue_refund(user_id: str, order_id: str, amount) -> str:
        oid = str(order_id).upper()
        try:
            amt = Decimal(str(amount)).quantize(Decimal("0.01"))
        except (InvalidOperation, ValueError):
            return "That is not a valid amount."
        with data.lock:
            order = data.orders.get(oid)
            if order is None or order["owner"] != user_id:
                return NOT_FOUND
            if (oid, amt) in data.issued:
                return f"A refund of ${amt:.2f} for order {oid} was already issued."
            left = order["total"] - data.refunded.get(oid, Decimal(0))
            if amt <= 0 or amt > left:
                return f"The amount must be more than $0.00 and at most ${left:.2f} for order {oid}."
            if amt > threshold:
                summary = {"order": oid, "amount": str(amt), "order_total": str(order["total"]), "customer": user_id}
                verdict = approver(summary) if approver else None
                if verdict is None:
                    data.pending.append(summary)
                    return f"A refund of ${amt:.2f} for order {oid} needs a human to approve it. A request has been sent."
                if verdict is False:
                    return "A human declined that refund."
            data.refunded[oid] = data.refunded.get(oid, Decimal(0)) + amt
            data.issued.add((oid, amt))
        return f"Refund of ${amt:.2f} issued for order {oid}."

    return {"search_policies": search_policies, "get_order_status": get_order_status, "issue_refund": issue_refund}
