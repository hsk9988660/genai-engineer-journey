"""Stage 3: structured output, and checking it against the truth.

Free text is hard for code to use. A schema makes the model's answer a typed object: `with_structured_output(Triage)`
returns a `Triage` instead of a string. But a schema only guarantees the SHAPE of an answer. It cannot tell you that the
order id is real. So this stage has two layers:

1. The schema (pydantic) rejects a wrong shape, and the model gets one more try with the error as feedback.
2. `verify` checks the claims against ground truth that code owns: does the order id appear in the message, and does
   this customer own that order?

When both layers fail, the result is marked `verified=False`. A later stage sends such a ticket to a person.

Run the offline demo:  python stage3_triage.py
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from desk import Desk, DeskError
from fakes import ScriptedChatModel, call

ORDER_ID = re.compile(r"\b[A-Z]\d{4}\b")

INSTRUCTIONS = (
    "Read the customer's message and classify it. category: 'refund' for money back on an order, 'faq' for a general "
    "question about shipping, returns or warranty, 'other' for anything else. order_id: copy it from the message, or "
    "leave it empty. urgency: 1 can wait, 5 angry or time-critical. summary: one sentence."
)


class Triage(BaseModel):
    category: Literal["refund", "faq", "other"]
    order_id: str | None = None
    urgency: int = Field(ge=1, le=5)
    summary: str = Field(max_length=300)


@dataclass
class TriageResult:
    triage: Triage
    verified: bool
    problems: list[str] = field(default_factory=list)
    attempts: int = 1


def verify(triage: Triage, text: str, desk: Desk, customer_id: str) -> list[str]:
    """Return what is wrong with the model's claims. An empty list means every claim checked out."""
    problems = []
    ids_in_text = set(ORDER_ID.findall(text))
    if triage.order_id is not None:
        if triage.order_id not in ids_in_text:
            problems.append(f"order id {triage.order_id} does not appear in the message")
        else:
            try:
                desk.get_order(customer_id, triage.order_id)
            except DeskError as exc:
                problems.append(str(exc))
    elif triage.category == "refund":
        problems.append("a refund request needs an order id")
    return problems


def triage_request(model: BaseChatModel, text: str, desk: Desk, customer_id: str, *, max_attempts: int = 2) -> TriageResult:
    structured = model.with_structured_output(Triage)
    base: list[BaseMessage] = [SystemMessage(INSTRUCTIONS), HumanMessage(text)]
    messages = list(base)
    for attempt in range(1, max_attempts + 1):
        try:
            parsed = structured.invoke(messages)
            error = "you did not return a Triage object" if parsed is None else None
        except (ValidationError, OutputParserException) as exc:
            parsed, error = None, str(exc)
        if parsed is not None:
            problems = verify(parsed, text, desk, customer_id)
            return TriageResult(parsed, verified=not problems, problems=problems, attempts=attempt)
        # Tell the model what was wrong, once, and ask again. Not forever: every retry costs money.
        messages = [*base, HumanMessage(f"Your last answer was not valid: {error}. Answer again with a valid Triage.")]
    fallback = Triage(category="other", urgency=3, summary=text[:300])
    return TriageResult(fallback, verified=False, problems=["the model did not return a valid Triage"], attempts=max_attempts)


def demo() -> None:
    desk = Desk()
    cases = [
        ("a good answer", "I want my money back for order A1001, the shoes are too small.",
         [call("Triage", {"category": "refund", "order_id": "A1001", "urgency": 2, "summary": "Wants a refund for A1001."})]),
        ("an invented order id", "Please refund my last order.",
         [call("Triage", {"category": "refund", "order_id": "A9999", "urgency": 2, "summary": "Wants a refund."})]),
        ("an invalid shape, fixed on retry", "How long does shipping take?",
         [call("Triage", {"category": "faq", "urgency": 9, "summary": "Asks about shipping."}),
          call("Triage", {"category": "faq", "urgency": 1, "summary": "Asks about shipping."})]),
    ]
    for title, text, script in cases:
        result = triage_request(ScriptedChatModel(script=script), text, desk, "C-100")
        print(f"{title}: category={result.triage.category} verified={result.verified} attempts={result.attempts} problems={result.problems}")


if __name__ == "__main__":
    demo()
