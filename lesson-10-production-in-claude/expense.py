"""The lesson's expense-claim demo, done the way money data should be handled.

What the notebook did, and what this does instead:
  semicolon/pipe-delimited text      structured JSON validated by Pydantic (a '|' or ';' in a description can't break it)
  rows that fail to parse are        nothing is dropped: any problem puts the claim in `needs_review` with a reason
    printed and silently dropped
  float amounts                      Decimal, parsed from strings so 0.1 + 0.2 never appears on a claim
  totals ignored                     the sum of the items (+ tax + tip) must RECONCILE with the receipt's printed total
  tips/fees dropped by a sentence    reimbursement policy lives in code (Policy), not buried in a prompt
    buried in a prompt
  "[Your Name]" placeholder, and     the email is built deterministically with a real name, and sending needs an explicit
    nothing stops a send               "approve" (fail-closed). A flagged claim is never sent automatically.

Run:  python expense.py receipt.jpg "Ana Ruiz"        (needs ANTHROPIC_API_KEY; sends nothing, prints the claim)
"""

import base64
import sys
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, field_validator

from structured import StructuredError, call_structured

Category = Literal["Transportation", "Meals", "Accommodation", "Miscellaneous"]
CENT = Decimal("0.01")


def to_money(value) -> Decimal:
    """'12.10', 12.1, '$1,234.50' -> Decimal. Floats go through str() so binary noise never reaches the claim."""
    try:
        return Decimal(str(value).replace("$", "").replace(",", "").strip())
    except InvalidOperation as exc:
        raise ValueError(f"not a money amount: {value!r}") from exc


class LineItem(BaseModel):
    date: date
    description: str
    amount: Decimal
    category: Category

    @field_validator("amount", mode="before")
    @classmethod
    def _amount(cls, v):
        return to_money(v)


class ReceiptExtraction(BaseModel):
    merchant: str
    items: list[LineItem]
    tax: Decimal = Decimal("0")
    tip: Decimal = Decimal("0")
    printed_total: Decimal

    @field_validator("tax", "tip", "printed_total", mode="before")
    @classmethod
    def _money(cls, v):
        return to_money(v)


@dataclass(frozen=True)
class Policy:
    """Reimbursement rules, decided by people and enforced by code."""
    reimburse_tips: bool = True
    meal_cap_per_day: Decimal = Decimal("75")
    max_single_item: Decimal = Decimal("5000")


@dataclass
class Claim:
    status: str  # ready | needs_review
    items: list
    total: Decimal
    problems: list = field(default_factory=list)


def reconcile(x: ReceiptExtraction) -> list[str]:
    problems = []
    computed = sum((i.amount for i in x.items), Decimal("0")) + x.tax + x.tip
    if abs(computed - x.printed_total) > CENT:
        problems.append(f"Items + tax + tip are ${computed:,.2f} but the receipt total is ${x.printed_total:,.2f}: "
                        f"a line item may be missing or misread (difference ${x.printed_total - computed:,.2f}).")
    seen = set()
    for i in x.items:
        key = (i.date, i.description.strip().lower(), i.amount)
        if key in seen:
            problems.append(f"Possible duplicate line: {i.date} {i.description!r} ${i.amount:,.2f}.")
        seen.add(key)
        if i.amount <= 0:
            problems.append(f"Line {i.description!r} has a non-positive amount (${i.amount:,.2f}); check for a refund or misread sign.")
    if not x.items:
        problems.append("No line items were extracted.")
    return problems


def policy_flags(x: ReceiptExtraction, policy: Policy) -> list[str]:
    flags = []
    for i in x.items:
        if i.amount > policy.max_single_item:
            flags.append(f"{i.description!r} (${i.amount:,.2f}) exceeds the single-item limit of ${policy.max_single_item:,.2f}.")
    meals: dict = {}
    for i in x.items:
        if i.category == "Meals":
            meals[i.date] = meals.get(i.date, Decimal("0")) + i.amount
    for day, total in sorted(meals.items()):
        if total > policy.meal_cap_per_day:
            flags.append(f"Meals on {day} total ${total:,.2f}, over the ${policy.meal_cap_per_day:,.2f} daily cap.")
    return flags


def build_claim(x: ReceiptExtraction, policy: Policy | None = None) -> Claim:
    policy = policy or Policy()
    problems = reconcile(x) + policy_flags(x, policy)
    total = sum((i.amount for i in x.items), Decimal("0")) + x.tax + (x.tip if policy.reimburse_tips else Decimal("0"))
    return Claim("needs_review" if problems else "ready", list(x.items), total, problems)


def render_email(claim: Claim, merchant: str, employee_name: str) -> tuple[str, str]:
    """Deterministic text built by code, so there is nothing for a model to improvise or leave as a placeholder."""
    lines = [f"Dear Finance Team,", "", f"Please find my expense claim for {merchant}:", ""]
    lines += [f"- {i.date.isoformat()} | {i.description}: ${i.amount:,.2f} ({i.category})" for i in claim.items]
    lines += ["", f"Total: ${claim.total:,.2f}", "", "Receipt attached.", "", f"Thank you,", employee_name]
    return f"Expense claim: {merchant} (${claim.total:,.2f})", "\n".join(lines)


EXTRACT_SYSTEM = """Extract the receipt into structured data.
- One line item per purchased thing. Dates as YYYY-MM-DD. Amounts as plain decimal strings like "12.50".
- Put tax and tip in their own fields. Do NOT include them as line items, and do NOT drop them.
- printed_total is the grand total PRINTED on the receipt. Copy it exactly. Never compute it yourself.
- If a value is unreadable, still extract everything else and make your best reading: a later check compares the items to the printed total.
- Categories: Transportation, Meals, Accommodation, Miscellaneous."""


@dataclass
class ClaimOutcome:
    status: str  # sent | declined | needs_review | extraction_failed
    claim: Claim | None = None
    detail: str = ""


def process_receipt(client, image_bytes: bytes, media_type: str, employee_name: str, approver, send_fn,
                    policy: Policy | None = None) -> ClaimOutcome:
    image = {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(image_bytes).decode()}}
    try:
        x = call_structured(client, EXTRACT_SYSTEM, [image, {"type": "text", "text": "Extract this receipt."}], ReceiptExtraction)
    except StructuredError as exc:  # reported, never swallowed: a person must look at it
        return ClaimOutcome("extraction_failed", None, str(exc))
    claim = build_claim(x, policy)
    if claim.status == "needs_review":
        return ClaimOutcome("needs_review", claim, "; ".join(claim.problems))  # a flagged claim is never sent automatically
    subject, body = render_email(claim, x.merchant, employee_name)
    try:
        approved = approver({"subject": subject, "body": body, "total": str(claim.total)}) == "approve"
    except Exception:
        approved = False  # fail closed
    if not approved:
        return ClaimOutcome("declined", claim)
    send_fn(subject, body)
    return ClaimOutcome("sent", claim)


if __name__ == "__main__":
    import anthropic

    path, name = (sys.argv[1] if len(sys.argv) > 1 else "receipt.jpg"), (sys.argv[2] if len(sys.argv) > 2 else "Employee Name")
    with open(path, "rb") as f:
        data = f.read()
    result = process_receipt(anthropic.Anthropic(), data, "image/jpeg", name, approver=lambda s: print(s["body"]) or "deny",
                             send_fn=lambda s, b: None)
    print(f"\nstatus: {result.status}  {result.detail}")
