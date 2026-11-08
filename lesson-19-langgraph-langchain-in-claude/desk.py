"""The business core of the support desk. There is no LangChain or LangGraph in this file.

This is deliberate. The rules about money (who may be refunded, how much, how often) live here, in plain Python that
you can test without any model. The agent's tools are thin wrappers around these functions. If the model is fooled,
confused or edited by a human, the rules in this file still hold.

All money is stored as whole cents (integers). Floats are for display only.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

TODAY = date(2026, 11, 7)
RETURN_WINDOW_DAYS = 30
AGENT_REFUND_LIMIT_CENTS = 20_000  # an agent may never refund more than $200.00 in one step, whoever approves it


class DeskError(Exception):
    """A business-rule problem. Tools turn it into data that the model can read."""


class OrderNotFound(DeskError):
    pass


class RefundRejected(DeskError):
    pass


class TemporaryOutage(DeskError):
    """The order system is down for a moment. This one is worth retrying."""


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    item: str
    total_cents: int
    status: str  # "delivered" or "shipped"
    delivered_on: date | None = None


@dataclass(frozen=True)
class Policy:
    eligible: bool
    refundable_cents: int
    reason: str


@dataclass(frozen=True)
class RefundRecord:
    refund_id: str
    order_id: str
    customer_id: str
    amount_cents: int
    reason: str
    key: str


SEED_ORDERS = [
    Order("A1001", "C-100", "Trail Shoes", 8_900, "delivered", date(2026, 10, 20)),
    Order("A1002", "C-100", "Camp Stove", 24_000, "delivered", date(2026, 10, 25)),
    Order("A1003", "C-100", "Rain Jacket", 12_000, "delivered", date(2026, 8, 30)),
    Order("A1004", "C-200", "Sleeping Bag", 15_000, "delivered", date(2026, 11, 1)),
    Order("A1005", "C-100", "Headlamp", 3_500, "shipped", None),
]

FAQ = [
    {"id": "shipping", "title": "How long does shipping take?", "keywords": {"shipping", "ship", "delivery", "arrive", "days"},
     "text": "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days."},
    {"id": "returns", "title": "What is the return policy?", "keywords": {"return", "returns", "refund", "window", "policy"},
     "text": "You can return an item within 30 days of delivery. A refund goes back to the original payment method."},
    {"id": "warranty", "title": "Is there a warranty?", "keywords": {"warranty", "broken", "defect", "repair", "guarantee"},
     "text": "Every product has a one-year warranty against manufacturing defects."},
]


def to_cents(amount: float | int | str) -> int:
    """Convert a dollar amount from the model to cents. Refuse anything that is not a clean positive money value."""
    try:
        value = Decimal(str(amount))
    except InvalidOperation:
        raise RefundRejected(f"'{amount}' is not a number") from None
    if not value.is_finite():
        raise RefundRejected(f"'{amount}' is not a finite number")
    cents = value * 100
    if cents != cents.to_integral_value():
        raise RefundRejected("an amount cannot have more than two decimal places")
    if cents <= 0:
        raise RefundRejected("the amount must be more than zero")
    return int(cents)


def money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


class Desk:
    """Orders, the refund ledger, tickets and help articles for one test or one session."""

    def __init__(self, orders: list[Order] | None = None, today: date = TODAY):
        self.orders = {o.order_id: o for o in (SEED_ORDERS if orders is None else orders)}
        self.today = today
        self.refunds: list[RefundRecord] = []
        self.tickets: list[dict] = []
        self.outages = 0  # how many of the next order lookups fail. Tests and demos set this.
        self._refunds_by_key: dict[str, RefundRecord] = {}

    def get_order(self, customer_id: str, order_id: str) -> Order:
        if self.outages > 0:
            self.outages -= 1
            raise TemporaryOutage("the order system is temporarily unavailable")
        order = self.orders.get(order_id)
        # Same message for "does not exist" and "belongs to someone else", so the answer cannot be used to probe.
        if order is None or order.customer_id != customer_id:
            raise OrderNotFound(f"no order {order_id} found for this customer")
        return order

    def refunded_cents(self, order_id: str) -> int:
        return sum(r.amount_cents for r in self.refunds if r.order_id == order_id)

    def policy_for(self, order: Order) -> Policy:
        if order.status != "delivered" or order.delivered_on is None:
            return Policy(False, 0, "the order has not been delivered yet")
        age = (self.today - order.delivered_on).days
        if age > RETURN_WINDOW_DAYS:
            return Policy(False, 0, f"outside the {RETURN_WINDOW_DAYS}-day return window (delivered {age} days ago)")
        remaining = order.total_cents - self.refunded_cents(order.order_id)
        if remaining <= 0:
            return Policy(False, 0, "the order is already fully refunded")
        return Policy(True, remaining, f"within the return window ({age} days since delivery)")

    def issue_refund(self, customer_id: str, order_id: str, amount_cents: int, reason: str, key: str) -> RefundRecord:
        """Pay a refund. Every rule is checked here, so no caller can skip one."""
        existing = self._refunds_by_key.get(key)
        if existing is not None:
            # The same request again (a retry, a replay after a crash) must not pay twice.
            if (existing.order_id, existing.amount_cents) != (order_id, amount_cents):
                raise RefundRejected("this request key was already used for a different refund")
            return existing
        order = self.get_order(customer_id, order_id)
        if not reason.strip():
            raise RefundRejected("a reason is required")
        if amount_cents <= 0:
            raise RefundRejected("the amount must be more than zero")
        policy = self.policy_for(order)
        if not policy.eligible:
            raise RefundRejected(f"not eligible: {policy.reason}")
        if amount_cents > policy.refundable_cents:
            raise RefundRejected(f"the amount is more than the {money(policy.refundable_cents)} that can still be refunded")
        if amount_cents > AGENT_REFUND_LIMIT_CENTS:
            raise RefundRejected(f"above the agent limit of {money(AGENT_REFUND_LIMIT_CENTS)}: hand this to a manager")
        record = RefundRecord(f"R-{len(self.refunds) + 1:04d}", order_id, customer_id, amount_cents, reason.strip(), key)
        self.refunds.append(record)
        self._refunds_by_key[key] = record
        return record

    def open_ticket(self, customer_id: str, reason: str, summary: str) -> dict:
        ticket = {"ticket_id": f"T-{len(self.tickets) + 1:04d}", "customer_id": customer_id, "reason": reason, "summary": summary}
        self.tickets.append(ticket)
        return ticket

    def faq_search(self, query: str, limit: int = 2) -> list[dict]:
        """Rank help articles by shared keywords. Return nothing when nothing matches, so callers can say 'I don't know'."""
        words = {w.strip(".,?!;:").lower() for w in query.split()}
        scored = [(len(words & a["keywords"]), a) for a in FAQ]
        scored = [(score, a) for score, a in scored if score > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [{"id": a["id"], "title": a["title"], "text": a["text"]} for _, a in scored[:limit]]
