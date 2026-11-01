"""Hand-off routing for customer support, with loop protection enforced in code.

A triage agent answers or hands the conversation to a specialist, who then owns it. Two failure modes need hard limits:
  * ping-pong: A hands to B, B hands back to A, forever. An agent that already handled the conversation can never get
    it back (each agent at most once), and the model is told which agents are still allowed.
  * long chains: a hop cap (`max_hops`).
When either limit is hit, the conversation escalates to a human instead of looping.

Isolation: each agent receives the customer's message and a short handoff NOTE from the previous agent. It does not
receive the previous agent's system prompt or reasoning.

Run:  python router.py "I was charged twice this month"        (needs ANTHROPIC_API_KEY)
"""

import json
import sys
from dataclasses import dataclass, field

from runtime import AgentError, BudgetExceeded, Runtime
from schemas import AgentTurn

SUPPORT_AGENTS = {
    "triage": "You greet the customer and decide who should help. Answer simple questions yourself. For billing problems hand off to 'billing'; for product or technical problems hand off to 'technical'.",
    "billing": "You resolve billing questions: charges, refunds, invoices. Explain what you found and what happens next.",
    "technical": "You resolve technical problems: errors, setup, performance. Give clear steps.",
}


@dataclass
class SupportOutcome:
    status: str  # answered | escalated | over_budget
    answer: str = ""
    path: list = field(default_factory=list)
    hops: int = 0
    reason: str = ""


def run_support(rt: Runtime, message: str, agents: dict, start: str = "triage", max_hops: int = 3) -> SupportOutcome:
    out = SupportOutcome(status="escalated", path=[start])
    root = rt.tracer.start("pipeline", "support")
    current, note = start, ""
    try:
        while True:
            allowed = [a for a in agents if a not in out.path]  # nobody gets the conversation back

            def validate(turn: AgentTurn, allowed=allowed) -> list[str]:
                if turn.action == "handoff" and turn.to not in allowed:
                    return [f"Cannot hand off to {turn.to!r}. Allowed targets: {allowed}. "
                            "An agent that already handled this conversation cannot receive it again."]
                return []

            system = (f"[ROLE: {current}] {agents[current]}\nYou may answer the customer, or hand off to one of {allowed} "
                      "only if another agent is clearly better. When you hand off, put a short note for that agent in `content`.")
            user = json.dumps({"customer_message": message, "note_from_previous_agent": note})
            turn = rt.run_agent(current, system, user, AgentTurn, root, validate=validate)
            if turn.action == "answer":
                out.status, out.answer = "answered", turn.content
                return out
            out.hops += 1
            if out.hops > max_hops:
                out.reason = f"hop limit ({max_hops}) reached"
                out.answer = "A human agent will follow up with you."
                return out
            current, note = turn.to, turn.content
            out.path.append(current)
    except BudgetExceeded as exc:
        out.status, out.reason = "over_budget", str(exc)
        return out
    except AgentError as exc:
        out.reason, out.answer = f"agent could not complete a valid turn: {exc}", "A human agent will follow up with you."
        return out
    finally:
        rt.tracer.finish(root, "ok" if out.status == "answered" else out.status)


if __name__ == "__main__":
    import anthropic

    runtime = Runtime(anthropic.Anthropic())
    result = run_support(runtime, " ".join(sys.argv[1:]) or "I was charged twice this month", SUPPORT_AGENTS)
    print(f"{result.status}: {result.answer}\npath: {' -> '.join(result.path)}  {result.reason}\n\n{runtime.tracer.render()}")
