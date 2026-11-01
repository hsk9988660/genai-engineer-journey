"""Agent-to-agent handoff with guards.

The lesson's handoff example stops when the user has sent more than three messages. Nothing stops two agents passing the same request back and forth,
and the next agent inherits the whole transcript. Here:
  * a handoff target must be on the CALLING agent's allow-list (the model proposes, this code checks)
  * hops are capped, and the same (from, to) pair twice is a ping-pong and is refused
  * the next agent receives the user's request plus a short capped summary, not the whole conversation
"""

from dataclasses import dataclass, field

from middleware import Identity
from runtrace import Trace, digest

MAX_SUMMARY_CHARS = 500


class HandoffError(Exception):
    pass


class HandoffRefused(HandoffError):
    pass


class HandoffLimit(HandoffError):
    pass


class HandoffLoop(HandoffError):
    pass


@dataclass(frozen=True)
class Reply:
    text: str


@dataclass(frozen=True)
class Handoff:
    to: str
    reason: str = ""
    summary: str = ""


@dataclass
class AgentView:
    """All an agent gets. There is deliberately no field for the earlier agents' full transcript."""
    user_text: str
    identity: Identity
    from_agent: str | None = None
    handoff_summary: str = ""
    handoff_reason: str = ""


@dataclass
class HandoffResult:
    text: str
    path: list
    hops: int


class HandoffRouter:
    def __init__(self, agents: dict, allowed: dict[str, set], start: str, max_hops: int = 4, trace: Trace | None = None):
        if start not in agents:
            raise ValueError("unknown start agent")
        for a, targets in allowed.items():
            if a not in agents or not set(targets) <= set(agents):
                raise ValueError(f"allow-list for {a!r} refers to an unknown agent")
        self.agents, self.allowed, self.start, self.max_hops = agents, {a: frozenset(t) for a, t in allowed.items()}, start, max_hops
        self.trace = trace or Trace()

    def run(self, user_text: str, identity: Identity) -> HandoffResult:
        current, view, path, pairs = self.start, AgentView(user_text, identity), [self.start], set()
        while True:
            out = self.agents[current](view)
            if isinstance(out, Reply):
                self.trace.emit("handoff_done", agent=current, hops=len(path) - 1, text_hash=digest(out.text)[:12])
                return HandoffResult(out.text, path, len(path) - 1)
            if not isinstance(out, Handoff):
                raise HandoffError(f"agent {current!r} returned {type(out).__name__}, expected Reply or Handoff")
            if out.to not in self.allowed.get(current, frozenset()):
                self.trace.emit("handoff_refused", frm=current, to=out.to)
                raise HandoffRefused(f"{current!r} may not hand off to {out.to!r}")
            if (current, out.to) in pairs:
                self.trace.emit("handoff_loop", frm=current, to=out.to)
                raise HandoffLoop(f"{current!r} -> {out.to!r} has already happened in this run")
            if len(path) - 1 >= self.max_hops:
                self.trace.emit("handoff_limit", hops=len(path) - 1)
                raise HandoffLimit(f"more than {self.max_hops} hops")
            pairs.add((current, out.to))
            self.trace.emit("handoff", frm=current, to=out.to, reason_hash=digest(out.reason)[:12])
            view = AgentView(user_text, identity, current, " ".join(out.summary.split())[:MAX_SUMMARY_CHARS], " ".join(out.reason.split())[:200])
            current = out.to
            path.append(current)
