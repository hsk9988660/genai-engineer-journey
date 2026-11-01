"""Policy: decide allow / ask / deny from what the TOOL is, never from words the model wrote.

The lesson 06 notebook classified risk by matching keywords in an action description written by the model.
That fails three ways: substring bugs ("get" inside "target"), incomplete word lists, and the fact that the
model (or an attacker steering it) controls the words. Here the facts come from metadata the DEVELOPER
declares when registering a tool, and from per-user permissions held by your code.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import BaseModel


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str  # the model reads this: write it carefully
    input_model: type[BaseModel]
    fn: Callable
    side_effects: bool = False  # changes the outside world (sends, writes, spends)
    irreversible: bool = False  # cannot be undone (deletes, money out)
    cost_usd: float = 0.0  # flat cost per call...
    cost_fn: Optional[Callable[[dict], float]] = None  # ...or computed from the arguments (e.g. a refund amount)
    required_scope: Optional[str] = None  # the signed-in user must hold this permission
    untrusted_output: bool = False  # result may carry attacker-controlled text (web pages, emails)
    allow_pii_args: bool = False  # real PII may be substituted into this tool's arguments (e.g. an email address)

    @property
    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_model.model_json_schema()}

    def cost(self, args: dict) -> float:
        return float(self.cost_fn(args)) if self.cost_fn else self.cost_usd


@dataclass(frozen=True)
class UserContext:
    """Who is signed in. Set by YOUR code from your auth system. The model can never change it."""
    user_id: str
    scopes: frozenset = frozenset()


@dataclass
class RunState:
    tainted: bool = False  # True once untrusted content has been read in this run
    tool_calls: int = 0


@dataclass(frozen=True)
class Decision:
    verdict: str  # "allow" | "ask" | "deny"
    reason: str


class PolicyEngine:
    def __init__(self, ask_cost_threshold_usd: float = 25.0):
        self.ask_cost_threshold_usd = ask_cost_threshold_usd

    def decide(self, spec: Optional[ToolSpec], args: dict, user: UserContext, state: RunState) -> Decision:
        if spec is None:
            return Decision("deny", "unknown tool")
        if spec.required_scope and spec.required_scope not in user.scopes:
            return Decision("deny", f"user lacks scope '{spec.required_scope}'")
        risky = spec.side_effects or spec.irreversible
        if risky and state.tainted:
            return Decision("deny", "side-effect tool requested after untrusted content was read")
        reasons = []
        if spec.irreversible:
            reasons.append("irreversible")
        elif spec.side_effects:
            reasons.append("has side effects")
        cost = spec.cost(args)
        if cost >= self.ask_cost_threshold_usd:
            reasons.append(f"costs ${cost:,.2f}")
        if reasons:
            return Decision("ask", ", ".join(reasons))
        return Decision("allow", "read-only")
