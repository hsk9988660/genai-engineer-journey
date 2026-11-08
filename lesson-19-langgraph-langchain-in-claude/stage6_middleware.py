"""Stage 6: middleware, the hooks that make an agent safe to run.

Middleware wraps the two things an agent does: calling the model, and calling a tool. Instead of editing the agent, you
add a layer. This stage adds five:

- DYNAMIC PROMPT: build the system prompt from the verified context (the customer's name, today's date).
- AUDIT: record every tool call and what became of it.
- RETRY: run the tool again when it failed for a temporary reason, and only for that reason.
- ERROR: turn any other exception into an honest message for the model. Without it, a bug in a tool ends the whole run.
- LIMIT: stop an agent that keeps calling tools.

ORDER MATTERS. For `wrap_tool_call`, the first middleware in the list is the OUTERMOST layer. Here AUDIT is outside RETRY,
so a tool call that needed three attempts is one record. Move audit inside retry and you get three records.

Run the offline demo:  python stage6_middleware.py
"""

import json
from collections.abc import Sequence

from langchain.agents.middleware import (ModelRequest, ToolCallLimitMiddleware, ToolErrorMiddleware, ToolRetryMiddleware, dynamic_prompt,
                                         wrap_tool_call)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from desk import AGENT_REFUND_LIMIT_CENTS, Desk, TemporaryOutage, money
from fakes import ScriptedChatModel, calls, say
from stage1_first_agent import SYSTEM_PROMPT, build_agent
from stage5_approval import approval_middleware
from tools import DeskContext

MAX_LOGGED_TEXT = 80
AUTO = object()  # 'choose the checkpointer for me'


class AuditTrail:
    """Collects one record per tool call. Create one per conversation and read `records` afterwards."""

    def __init__(self) -> None:
        self.records: list[dict] = []

        @wrap_tool_call
        def audit(request, handler):
            tool_call = request.tool_call
            record = {"tool": tool_call["name"], "args": _clip(tool_call["args"])}
            try:
                response = handler(request)  # run the next layer, and finally the tool itself
            except Exception as exc:
                self.records.append({**record, "outcome": "exception", "detail": type(exc).__name__, "seq": len(self.records) + 1})
                raise
            outcome, detail = _classify(response)
            self.records.append({**record, "outcome": outcome, "detail": detail, "seq": len(self.records) + 1})
            return response

        self.middleware = audit


def _clip(args: dict) -> dict:
    return {key: (value[:MAX_LOGGED_TEXT] if isinstance(value, str) else value) for key, value in args.items()}


def _classify(response) -> tuple[str, str]:
    """Turn a tool result into 'ok', 'refused' (the tool said no) or 'error' (the tool broke)."""
    if not isinstance(response, ToolMessage):
        return "ok", ""
    if response.status == "error":
        return "error", str(response.content)[:MAX_LOGGED_TEXT]
    try:
        body = json.loads(response.content)
    except (TypeError, ValueError):
        return "ok", ""
    if isinstance(body, dict) and "error" in body:
        return "refused", str(body["error"])[:MAX_LOGGED_TEXT]
    return "ok", ""


def prompt_middleware(desk: Desk):
    @dynamic_prompt
    def desk_prompt(request: ModelRequest) -> str:
        context = request.runtime.context
        return (
            f"{SYSTEM_PROMPT}\nYou are talking to {context.customer_name} (customer id {context.customer_id}). Today is {desk.today.isoformat()}. "
            f"You can never refund more than {money(AGENT_REFUND_LIMIT_CENTS)} in one step. For more, say that a manager will follow up."
        )

    return desk_prompt


def report_tool_failure(exc: Exception, request) -> str:
    """What the model is told when a tool breaks. It names the exception type but not the raw message, which may hold internal detail."""
    return (f"The tool '{request.tool_call['name']}' failed with {type(exc).__name__}. Do not call it again. "
            "Tell the customer that you could not complete this step and that a person will follow up.")


def build_hardened_agent(model: BaseChatModel, desk: Desk, trail: AuditTrail, *, approval: bool = True, tool_call_limit: int = 8,
                         max_retries: int = 2, extra_tools: Sequence = (), checkpointer=AUTO, store=None):
    middleware = [
        prompt_middleware(desk),
        trail.middleware,  # outermost: sees one record per logical call
        ToolErrorMiddleware(on_error=report_tool_failure),  # catches what retry gives up on, and any other exception
        ToolRetryMiddleware(max_retries=max_retries, retry_on=(TemporaryOutage,), on_failure="error", initial_delay=0.0, jitter=False),
        ToolCallLimitMiddleware(run_limit=tool_call_limit),
    ]
    if approval:
        middleware.append(approval_middleware(desk))
    if checkpointer is AUTO:
        checkpointer = InMemorySaver() if approval else None  # approval needs one. Pass None to inherit it from a parent graph
    return build_agent(model, desk, extra_tools=extra_tools, middleware=middleware, system_prompt=None, store=store, checkpointer=checkpointer)


def demo() -> None:
    desk = Desk()
    desk.outages = 2  # the order system fails twice, then works
    trail = AuditTrail()
    model = ScriptedChatModel(script=[
        calls(("lookup_order", {"order_id": "A1001"}, "c1"), ("check_refund_policy", {"order_id": "A1003"}, "c2")),
        say("A1001 is Trail Shoes. A1003 is too old to refund."),
    ])
    agent = build_hardened_agent(model, desk, trail, approval=False)
    agent.invoke({"messages": [{"role": "user", "content": "Tell me about A1001 and A1003."}]}, context=DeskContext("C-100", "Sam"))
    print("system prompt the model saw:\n  " + model.seen[0][0].content.replace("\n", "\n  "))
    print("\naudit trail:")
    for record in trail.records:
        print(f"  {record['seq']}. {record['tool']}({record['args']}) -> {record['outcome']} {record['detail']}")
    print(f"\norder-system failures absorbed by retry: {2 - desk.outages}")


if __name__ == "__main__":
    demo()
