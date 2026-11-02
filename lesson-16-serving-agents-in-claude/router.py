"""Model routing as a cascade: try the cheap tier, check the answer in code, escalate only when the check fails.

The lesson routes by keywords and length ("refund" or more than 20 words goes to the big model) and gives BOTH models every tool. Two problems: a keyword list is a
guess ("I want my money back" contains none of its words), and nothing stops the cheap model from calling the refund tool. Here:
  * what a tier can do is decided by its TOOL LIST, which the agent definition validates (side-effect tools only on the top tier)
  * a result that uses a tool its tier was not offered is refused (`ToolNotAllowed`), however it got there
  * escalation is driven by a CHECK on the answer, not by guessing the question's difficulty. The keyword hint (`routing["top_keywords"]`) only decides where to START
This module does not say how good the cheap tier is. That needs measurement on real traffic, and nothing here supplies it.
"""

from agentdef import AgentResult, AgentVersion
from resilience import Permanent


class ToolNotAllowed(Permanent):
    pass


def default_check(result: AgentResult) -> bool:
    """The simplest possible acceptance check: the call worked and produced text. Real systems put schema checks, grounding checks and policy checks here."""
    return bool(result.ok and result.text and result.text.strip())


def enforce_tools(tier, result: AgentResult) -> AgentResult:
    for name in result.tool_calls:
        if name not in tier.tools:
            raise ToolNotAllowed(f"tier {tier.name!r} used tool {name!r}, which it was not offered")
    return result


def start_index(version: AgentVersion, text: str) -> int:
    """Where to start. A HINT to save a wasted cheap call, never a safety control."""
    words = tuple(w.lower() for w in version.routing.get("top_keywords", ()))
    low = text.lower()
    return len(version.tiers) - 1 if any(w in low for w in words) else 0


def cascade(version: AgentVersion, run_tier, check=default_check, start: int = 0):
    """Returns (result, tier_name, passed, attempts). `attempts` is [(tier_name, result_or_None, passed)]. The top tier's answer is returned even if it fails the check."""
    attempts, last = [], len(version.tiers) - 1
    for i in range(start, last + 1):
        tier = version.tiers[i]
        try:
            res = run_tier(tier)
        except ToolNotAllowed:
            attempts.append((tier.name, None, False))
            if i == last:
                raise
            continue
        ok = check(res)
        attempts.append((tier.name, res, ok))
        if ok or i == last:
            return res, tier.name, ok, attempts
    raise ToolNotAllowed("no tier could be used")  # only reachable if every tier below the top broke the tool rule and start was the top
