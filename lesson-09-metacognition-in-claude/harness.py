"""Does reflection actually help? Measure fix rate AND break rate, against a baseline, with the cost.

Three designs run on the same cases and are all judged by the same deterministic checker:
  baseline    one generation, no reflection
  ungrounded  generate, then one "review your work and fix any problems" pass with NO failure information. This is the
              notebook's style of self-evaluation, and it can make a correct answer wrong.
  grounded    the loop in reflect.py: concrete check failures are fed back, with a cap, stall detection and best-of

  fix rate    of the cases that were wrong on the first try, the share that ended up right
  break rate  of the cases that were right on the first try, the share that ended up wrong
  net gain    final pass rate minus first-try pass rate

A reflection step that does not clearly beat its own break rate and its token cost should be deleted.

Run live:  python harness.py       (needs ANTHROPIC_API_KEY; costs real money)
I have NOT run it. Offline tests only verify the arithmetic and the logic of each design.
"""

from dataclasses import dataclass

from checks import Constraints, check_itinerary
from reflect import ItineraryGenerator, solve_itinerary
from structured import Usage


@dataclass
class Case:
    task: str
    constraints: Constraints


@dataclass
class CaseResult:
    first_ok: bool
    final_ok: bool
    attempts: int
    tokens: int


CASES = [
    Case("Plan 2 days in Lisbon for a couple who love food.", Constraints(budget_usd=300, num_days=2, max_activities_per_day=3)),
    Case("Plan 3 days in Kyoto with temples and a day trip.", Constraints(budget_usd=400, num_days=3, max_activities_per_day=3)),
    Case("Plan a tight, cheap 2-day Rome visit.", Constraints(budget_usd=120, num_days=2, max_activities_per_day=2)),
]

GENERIC_REVIEW = ["Review your itinerary carefully and fix any problems you find."]  # note: no specifics


def design_baseline(client, case: Case) -> CaseResult:
    usage = Usage()
    ok = not check_itinerary(ItineraryGenerator(client, case.task, case.constraints, usage=usage)(None, 1), case.constraints)
    return CaseResult(ok, ok, 1, usage.total)


def design_ungrounded(client, case: Case) -> CaseResult:
    usage = Usage()
    gen = ItineraryGenerator(client, case.task, case.constraints, usage=usage)
    first_ok = not check_itinerary(gen(None, 1), case.constraints)
    final_ok = not check_itinerary(gen(GENERIC_REVIEW, 2), case.constraints)  # the revision is what gets shipped, better or not
    return CaseResult(first_ok, final_ok, 2, usage.total)


def design_grounded(client, case: Case) -> CaseResult:
    result, usage = solve_itinerary(client, case.task, case.constraints, max_attempts=4)
    return CaseResult(not result.attempts[0].failures, result.passed, len(result.attempts), usage.total)


DESIGNS = {"baseline": design_baseline, "ungrounded": design_ungrounded, "grounded": design_grounded}


def summarize(results: list[CaseResult]) -> dict:
    n = len(results)
    failing = [r for r in results if not r.first_ok]
    passing = [r for r in results if r.first_ok]
    first_pass, final_pass = sum(r.first_ok for r in results) / n, sum(r.final_ok for r in results) / n
    return {"cases": n, "first_pass_rate": first_pass, "final_pass_rate": final_pass, "net_gain": final_pass - first_pass,
            "fix_rate": (sum(r.final_ok for r in failing) / len(failing)) if failing else None,
            "break_rate": (sum(not r.final_ok for r in passing) / len(passing)) if passing else None,
            "tokens": sum(r.tokens for r in results), "mean_attempts": sum(r.attempts for r in results) / n}


def evaluate(make_client, cases=CASES, designs=None, baseline: str = "baseline") -> dict:
    report = {}
    for name, design in (designs or DESIGNS).items():
        results = []
        for case in cases:
            try:
                results.append(design(make_client(), case))
            except Exception:  # a crash counts as a failed case, never as a quiet skip
                results.append(CaseResult(False, False, 0, 0))
        report[name] = summarize(results)
    base_tokens = report.get(baseline, {}).get("tokens")
    for row in report.values():
        row["cost_multiplier"] = (row["tokens"] / base_tokens) if base_tokens else None
    return report


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    import anthropic

    for name, r in evaluate(anthropic.Anthropic).items():
        print(f"{name:11s} first={fmt(r['first_pass_rate'])} final={fmt(r['final_pass_rate'])} fix={fmt(r['fix_rate'])} "
              f"break={fmt(r['break_rate'])} net={r['net_gain']:+.2f} tokens={r['tokens']} x{fmt(r['cost_multiplier'])}")
