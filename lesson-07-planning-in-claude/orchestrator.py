"""Plan, get approval for the whole plan, execute from code, and replan on failure (capped).

  1. Extract what the user said (every fact must be backed by a verbatim quote). Missing facts become QUESTIONS.
  2. Write a plan and validate it in code (rejected plans go back to the model with the exact problems).
  3. A human approves the plan ONCE. The approval is bound to a hash of the plan, which the executor re-checks.
  4. Execute. If something fails, ask for replacement tasks for the unfinished part, keep what already succeeded,
     and get approval for the new combined plan. At most `max_replans` times.

Run:  python orchestrator.py "Plan a trip to Paris from May 2 to May 9, budget 3000 dollars"   (needs ANTHROPIC_API_KEY)
"""

import copy
import json
import sys
from dataclasses import dataclass, field
from datetime import date

from executor import SIDE_EFFECT_AGENTS, ApprovalError, ExecutionReport, PlanExecutor
from plan import Plan, facts_of, missing_questions, plan_hash
from planner import PlanningError, Usage, extract_requirements, make_plan, make_replacement


@dataclass
class Outcome:
    status: str  # needs_info | declined | completed | failed | over_budget | invalid_plan | approval_mismatch
    questions: list = field(default_factory=list)
    plan: Plan | None = None
    report: ExecutionReport | None = None
    replans: int = 0
    usage: Usage = field(default_factory=Usage)
    error: str = ""


def _approved(approver, plan: Plan, digest: str) -> bool:
    """Show the human the whole plan. Fail closed: anything but the exact word 'approve' means no."""
    summary = {"plan_hash": digest, "goal": plan.goal,
               "tasks": [t.model_dump(mode="json") for t in plan.tasks],
               "will_reserve_or_buy": sorted({t.agent.value for t in plan.tasks if t.agent in SIDE_EFFECT_AGENTS})}
    try:
        return approver(copy.deepcopy(summary)) == "approve"
    except Exception:
        return False


def plan_and_execute(client, request: str, approver, workers, today: str | None = None, max_replans: int = 2,
                     budget_usd: float | None = None) -> Outcome:
    out = Outcome(status="failed")
    try:
        req = extract_requirements(client, request, today or date.today().isoformat(), out.usage)
        out.questions = missing_questions(req)
        if out.questions:  # ask, never invent
            out.status = "needs_info"
            return out
        facts = facts_of(req)
        if budget_usd is None and "budget_usd" in facts:
            budget_usd = float(facts["budget_usd"])
        plan = make_plan(client, request, facts, out.usage)
    except PlanningError as exc:
        out.status, out.error = "invalid_plan", str(exc)
        return out

    prior = None
    for attempt in range(max_replans + 1):
        out.plan = plan
        digest = plan_hash(plan)
        if not _approved(approver, plan, digest):
            out.status = "declined"
            return out
        try:
            out.report = PlanExecutor(workers, budget_usd=budget_usd).run(plan, approved_hash=digest, prior=prior)
        except ApprovalError as exc:
            out.status, out.error = "approval_mismatch", str(exc)
            return out
        if out.report.halted_reason:
            out.status, out.error = "over_budget", out.report.halted_reason
            return out
        if out.report.ok:
            out.status = "completed"
            return out
        if attempt == max_replans:
            out.status = "failed"
            out.error = "; ".join(f"task {i}: {e}" for i, e in out.report.failed.items())
            return out
        try:
            plan = make_replacement(client, request, facts, plan, out.report, out.usage)
        except PlanningError as exc:
            out.status, out.error = "invalid_plan", str(exc)
            return out
        prior = out.report.results
        out.replans += 1
    return out


if __name__ == "__main__":
    import anthropic

    from workers import DEMO_WORKERS

    def ask_in_terminal(summary: dict) -> str:
        print("\n--- PLAN FOR APPROVAL ---")
        print(json.dumps(summary, indent=2))
        return "approve" if input("Type 'approve' to run this plan: ").strip() == "approve" else "deny"

    result = plan_and_execute(anthropic.Anthropic(), " ".join(sys.argv[1:]) or "Plan a trip.", ask_in_terminal, DEMO_WORKERS)
    print(f"\nstatus: {result.status}  replans: {result.replans}  model calls: {result.usage.calls}")
    if result.questions:
        print("Questions:", *result.questions, sep="\n  - ")
    if result.report and result.report.results:
        last = max(result.report.results)
        print(result.report.results[last].output.get("summary", ""))
    if result.error:
        print("error:", result.error)
