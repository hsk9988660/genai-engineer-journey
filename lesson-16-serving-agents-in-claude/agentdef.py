"""What an agent IS, as plain data: a prompt, tiers of models, tools, and routing hints. Hashing it gives the version id.

The lesson's notebook has no notion of a version: "promoting the agent" is a print statement. Here the definition is data, so it has a stable id that changes whenever the
prompt, a model, a tool or the routing changes. Evaluation results, cache entries, canary assignments and metrics are all tied to that id.

Two rules are checked when a definition is built, not left to convention:
  * every tool a tier names must exist
  * a tool with side effects ("effect") may only be offered to the TOP tier, so the cheap, less careful model can never be the one that spends money
"""

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal

SCOPES = ("public", "user", "effect")  # public: same answer for everyone. user: depends on who asks. effect: changes the world


@dataclass(frozen=True)
class ToolSpec:
    name: str
    scope: str
    description: str = ""

    def __post_init__(self):
        if self.scope not in SCOPES:
            raise ValueError(f"tool scope must be one of {SCOPES}")


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    tools: tuple = ()
    in_price: Decimal = Decimal("0")  # price per 1,000 input tokens. PLACEHOLDER numbers: look up real prices before trusting any cost figure
    out_price: Decimal = Decimal("0")


@dataclass(frozen=True)
class AgentVersion:
    name: str
    prompt: str
    tiers: tuple  # cheapest first
    tools: tuple = ()
    routing: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tiers:
            raise ValueError("an agent needs at least one tier")
        known = {t.name: t for t in self.tools}
        for i, tier in enumerate(self.tiers):
            for tool in tier.tools:
                if tool not in known:
                    raise ValueError(f"tier {tier.name!r} names unknown tool {tool!r}")
                if known[tool].scope == "effect" and i != len(self.tiers) - 1:
                    raise ValueError(f"tool {tool!r} has side effects and may only be offered to the top tier, not {tier.name!r}")

    @property
    def version_id(self) -> str:
        canon = {"name": self.name, "prompt": self.prompt, "routing": self.routing,
                 "tiers": [(t.name, t.model, list(t.tools), str(t.in_price), str(t.out_price)) for t in self.tiers],
                 "tools": [(t.name, t.scope) for t in self.tools]}
        return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:12]

    def tool(self, name: str) -> ToolSpec:
        return next(t for t in self.tools if t.name == name)


@dataclass
class AgentResult:
    text: str
    tool_calls: tuple = ()
    input_tokens: int = 0
    output_tokens: int = 0
    ok: bool = True


@dataclass(frozen=True)
class AgentCall:
    """Everything the model-calling function gets. `user_id` is for TOOLS to use. The model never sees it and never supplies it."""
    version: AgentVersion
    tier: Tier
    messages: list
    user_context: str
    deadline: float
    user_id: str
