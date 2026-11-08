"""Stage 5: a human approves before money moves.

`HumanInTheLoopMiddleware` pauses the agent right after the model asks for a risky tool and BEFORE the tool runs.
The pause is a LangGraph INTERRUPT: the state is saved by the checkpointer, `invoke` returns early with the pending
request, and the program can do other things (or even exit). Later you call the agent again with `Command(resume=...)`
and the same thread id, and it continues from the saved state. That is why a checkpointer is required.

The reviewer can answer three ways: approve (run as asked), edit (run with different arguments), or reject (do not run;
the model is told why).

Three lessons from the design:

- Approval is not authorization. The reviewer may type a wrong number. The rules in desk.py re-check every refund when
  it runs, so an approved or edited refund that breaks a rule is still refused.
- The reviewer is shown facts that CODE looked up (the item, what can still be refunded), not only what the model said.
- The `when` predicate runs again on resume, and if it returned a different answer, the reviewer's decision would be
  skipped. So it depends only on the call's own arguments, and any argument it cannot read goes to a human.

Run the offline demo:  python stage5_approval.py
"""

from dataclasses import dataclass

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from desk import Desk, DeskError, money, to_cents
from fakes import ScriptedChatModel, call, say
from show import format_messages
from stage1_first_agent import ask, build_agent
from tools import DeskContext

AUTO_APPROVE_LIMIT_CENTS = 2_500  # refunds up to $25.00 run without a person


def needs_approval(request) -> bool:
    try:
        return to_cents(request.tool_call["args"]["amount"]) > AUTO_APPROVE_LIMIT_CENTS
    except Exception:
        return True  # an amount we cannot even read is exactly the case a person should see


def approval_middleware(desk: Desk) -> HumanInTheLoopMiddleware:
    def describe(tool_call, state, runtime) -> str:
        args = tool_call["args"]
        try:
            order = desk.get_order(runtime.context.customer_id, args["order_id"])
            facts = f"{order.item}, order total {money(order.total_cents)}, still refundable {money(desk.policy_for(order).refundable_cents)}"
        except (DeskError, KeyError):
            facts = "the order could not be verified"
        try:
            amount = money(to_cents(args["amount"]))
        except (DeskError, KeyError):
            amount = f"an unreadable amount ({args.get('amount')!r})"
        return f"Refund {amount} on order {args.get('order_id')} ({facts}). Reason given: {args.get('reason')!r}"

    return HumanInTheLoopMiddleware(interrupt_on={
        "issue_refund": {"allowed_decisions": ["approve", "edit", "reject"], "description": describe, "when": needs_approval},
    })


def build_approval_agent(model: BaseChatModel, desk: Desk, *, checkpointer=None):
    return build_agent(model, desk, middleware=[approval_middleware(desk)], checkpointer=checkpointer or InMemorySaver())


@dataclass(frozen=True)
class ApprovalRequest:
    tool: str
    args: dict
    description: str
    allowed: list[str]


def pending_approval(result: dict) -> ApprovalRequest | None:
    """Read the pause out of an `invoke` result. Returns None when the agent ran to the end."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    value = interrupts[0].value
    action, review = value["action_requests"][0], value["review_configs"][0]
    return ApprovalRequest(action["name"], action["args"], action["description"], review["allowed_decisions"])


def approve() -> dict:
    return {"decisions": [{"type": "approve"}]}


def reject(message: str) -> dict:
    return {"decisions": [{"type": "reject", "message": message}]}


def edit(tool: str, args: dict) -> dict:
    return {"decisions": [{"type": "edit", "edited_action": {"name": tool, "args": args}}]}


def resume(agent, decision: dict, thread_id: str, customer_id: str = "C-100") -> dict:
    """Continue a paused run. The decision list has one entry per paused tool call (here, one)."""
    config = {"configurable": {"thread_id": thread_id}}
    return agent.invoke(Command(resume=decision), config, context=DeskContext(customer_id=customer_id))


def demo() -> None:
    for title, decide in [("approve", lambda r: approve()), ("reject", lambda r: reject("Please offer store credit instead.")),
                          ("edit to $500 (breaks a rule)", lambda r: edit("issue_refund", {**r.args, "amount": 500.0}))]:
        desk = Desk()
        model = ScriptedChatModel(script=[call("issue_refund", {"order_id": "A1001", "amount": 89.0, "reason": "too small"}), say("All done.")])
        agent = build_approval_agent(model, desk)
        paused = ask(agent, "Refund A1001 please.", thread_id="t")
        request = pending_approval(paused)
        print(f"--- {title}\npaused before running. refunds so far: {len(desk.refunds)}\nreviewer sees: {request.description}")
        done = resume(agent, decide(request), "t")
        print("\n".join(format_messages(done["messages"][2:])))
        print(f"refunds issued: {[(r.refund_id, money(r.amount_cents)) for r in desk.refunds]}\n")


if __name__ == "__main__":
    demo()
