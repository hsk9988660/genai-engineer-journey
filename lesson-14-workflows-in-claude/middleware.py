"""Tool-call middleware that may watch, limit, block and redact, but may not quietly lie.

The lesson's example middleware ran the hotel tool, saw "no rooms", and for a priority member REWROTE the result to "rooms available". The agent
then booked a room that did not exist. Here, a middleware can only change a result through a declared door:
  call.redact(new_value, reason)    a trusted transformation of REAL data (hiding an email). The result stays real.
  call.override(value, reason)      invents or replaces data. The result is marked SYNTHETIC, and `ToolResult.require_real()` refuses it.
Returning a different value without using either door raises `UndeclaredOverride`.

Identity is passed in (`call.identity`) and is never read from a global.
"""

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from runtrace import Trace, digest


class ToolBlocked(Exception):
    pass


class RateLimited(Exception):
    pass


class UndeclaredOverride(Exception):
    pass


class SyntheticFact(Exception):
    pass


@dataclass(frozen=True)
class Identity:
    """WHO this run is for. Built by your code from your auth system and passed in; never a module-level variable."""
    user_id: str

    def __post_init__(self):
        if not isinstance(self.user_id, str) or not self.user_id:
            raise ValueError("user_id must be a non-empty string")


@dataclass
class ToolResult:
    value: object
    synthetic: bool = False
    modifications: list = field(default_factory=list)  # [(kind, reason)] for every declared change

    def require_real(self):
        """Call this before an irreversible step. Returns the value, or raises if any middleware invented it."""
        if self.synthetic:
            raise SyntheticFact(f"refusing to act on a synthetic result: {self.modifications}")
        return self.value


class ToolCall:
    def __init__(self, name: str, args: dict, identity: Identity):
        self.name, self.args, self.identity = name, args, identity
        self.real_result, self.has_real = None, False
        self._declared = None  # (kind, value, reason)
        self.modifications: list = []

    def redact(self, new_value, reason: str) -> None:
        self._declared = ("redact", new_value)
        self.modifications.append(("redact", reason))

    def override(self, new_value, reason: str) -> None:
        self._declared = ("override", new_value)
        self.modifications.append(("override", reason))


def invoke(tools: dict, middleware, name: str, args: dict, identity: Identity) -> ToolResult:
    if name not in tools:
        raise KeyError(f"unknown tool {name!r}")
    call = ToolCall(name, dict(args), identity)

    def run(i: int):
        if i == len(middleware):
            call.real_result, call.has_real = tools[name](identity, **call.args), True
            return call.real_result
        return middleware[i](call, lambda: run(i + 1))

    returned = run(0)
    if call._declared is not None:
        kind, value = call._declared
        return ToolResult(value, synthetic=(kind == "override"), modifications=list(call.modifications))
    if not call.has_real or returned != call.real_result:
        raise UndeclaredOverride(f"middleware changed the result of {name!r} without declaring it with redact() or override()")
    return ToolResult(returned)


# --- ready-made middleware ------------------------------------------------------------------------------------------------
def deny(names) -> callable:
    blocked = frozenset(names)

    def mw(call, nxt):
        if call.name in blocked:
            raise ToolBlocked(f"tool {call.name!r} is blocked")
        return nxt()
    return mw


def rate_limit(max_calls: int, per_seconds: float, clock=time.time) -> callable:
    """At most `max_calls` per user in any `per_seconds` window. Counted per identity, so one user cannot use up another's allowance."""
    seen: dict[str, deque] = defaultdict(deque)

    def mw(call, nxt):
        q, now = seen[call.identity.user_id], clock()
        while q and now - q[0] >= per_seconds:
            q.popleft()
        if len(q) >= max_calls:
            raise RateLimited(f"more than {max_calls} calls in {per_seconds}s")
        q.append(now)
        return nxt()
    return mw


def log_calls(trace: Trace) -> callable:
    """Logs the tool name, the argument NAMES and a hash of the user id. Never argument values or results."""
    def mw(call, nxt):
        trace.emit("tool_call", tool=call.name, arg_names=sorted(call.args), user=digest(call.identity.user_id)[:12])
        out = nxt()
        trace.emit("tool_result", tool=call.name)
        return out
    return mw


def redact(patterns: dict[str, str]) -> callable:
    """`patterns` maps a regex to its replacement. Applied to every string in the result (nested dicts and lists included)."""
    compiled = [(re.compile(p), r) for p, r in patterns.items()]

    def scrub(v):
        if isinstance(v, str):
            for rx, rep in compiled:
                v = rx.sub(rep, v)
            return v
        if isinstance(v, dict):
            return {k: scrub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [scrub(x) for x in v]
        return v

    def mw(call, nxt):
        out = nxt()
        cleaned = scrub(out)
        if cleaned != out:
            call.redact(cleaned, "pattern redaction")
        return out
    return mw
