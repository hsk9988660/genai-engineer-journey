"""Talk to the support desk with the real Claude API.

Every test in this lesson uses a scripted fake model. This file is the one place that uses a real one. It needs
`pip install langchain-anthropic` and an `ANTHROPIC_API_KEY`. It costs a little money per message.

    python run_live.py

Type a customer message. If the agent wants to issue a refund over $25, you play the human reviewer in the terminal.
Type an empty line to quit.

HONEST STATUS: the wiring (`build_live_desk`) is checked by a test with a dummy key, but the lesson's author has not run
this file against the live API. Treat the first run as an experiment, and expect to adjust the prompts.
"""

import os
import uuid

from langchain.chat_models import init_chat_model

from desk import Desk, money
from show import final_text
from stage5_approval import approve, edit, pending_approval, reject, resume
from stage6_middleware import AuditTrail
from stage7_support_desk import DeskModels, build_desk, handle

CUSTOMER_ID = "C-100"
CUSTOMER_NAME = "Sam"


def live_model():
    return init_chat_model(f"anthropic:{os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-5-5')}")


def build_live_desk(desk: Desk | None = None, trail: AuditTrail | None = None):
    model = live_model()
    return build_desk(DeskModels(triage=model, agent=model, faq=model), desk or Desk(), trail)


def review(request) -> dict:
    print(f"\n  A person must decide.\n  {request.description}")
    choice = input("  [a]pprove, [r]eject, [e]dit the amount: ").strip().lower()
    if choice == "a":
        return approve()
    if choice == "e":
        return edit(request.tool, {**request.args, "amount": float(input("  new amount in dollars: "))})
    return reject(input("  reason to tell the agent: ").strip() or "A reviewer rejected this refund.")


def main() -> None:
    desk, trail = Desk(), AuditTrail()
    app = build_live_desk(desk, trail)
    print(f"Support desk for {CUSTOMER_NAME} ({CUSTOMER_ID}). Try: 'Please refund order A1001, the shoes are too small.'")
    while text := input("\ncustomer> ").strip():
        thread_id = uuid.uuid4().hex
        result = handle(app, text, customer_id=CUSTOMER_ID, customer_name=CUSTOMER_NAME, thread_id=thread_id)
        while (request := pending_approval(result)) is not None:
            result = resume(app, review(request), thread_id, CUSTOMER_ID)
        print(f"\nagent> {final_text(result['messages'])}")
        resolution = result["resolution"]
        print(f"[{resolution['status']}] refunded {money(resolution['refunded_cents'])}; route reason: {result['reason']}")
    print(f"\nrefunds issued: {[(r.refund_id, money(r.amount_cents)) for r in desk.refunds]}")
    print("audit trail:")
    for record in trail.records:
        print(f"  {record['seq']}. {record['tool']} -> {record['outcome']} {record['detail']}")


if __name__ == "__main__":
    main()
