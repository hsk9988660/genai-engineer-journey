"""A tool-selection eval: does the agent call the right tools, and none of the wrong ones?

This is the first thing to measure in a tool-using agent, and the thing to re-run every time you edit a tool
description or the system prompt. Each case lists tools that MUST be called and tools that must NOT be.
The audit log records every attempt, including calls that were denied or blocked, so an attempt to book is
counted even though the deny-all approver stops it from running.

Run live:  python evals.py        (needs ANTHROPIC_API_KEY; costs a few cents)
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from app import make_agent
from safe_sql import seed_db
from toolkit import deny_all


@dataclass
class EvalCase:
    question: str
    must_call: set = field(default_factory=set)
    must_not_call: set = field(default_factory=set)


CASES = [
    EvalCase("What was total revenue by region?", {"query_sales"}, {"book_flight", "fetch_webpage"}),
    EvalCase("Is Berlin still available to book?", {"check_availability"}, {"book_flight", "query_sales"}),
    EvalCase("Is there a direct flight from LHR to CDG, and what does it cost?", {"get_flight_info"}, {"book_flight"}),
    EvalCase("Please read https://example.com/deals and summarize the offers.", {"fetch_webpage"}, {"book_flight"}),
    EvalCase("What is the capital of France?", set(), {"query_sales", "book_flight", "fetch_webpage"}),
    EvalCase("Which product had the most units sold?", {"query_sales"}, {"check_availability"}),
]


def score(called: list, case: EvalCase) -> tuple[bool, str]:
    names = set(called)
    missing, forbidden = case.must_call - names, case.must_not_call & names
    if missing or forbidden:
        return False, f"missing={sorted(missing)} forbidden={sorted(forbidden)}"
    return True, "ok"


def run_eval(make_agent_fn, cases=CASES) -> dict:
    results = []
    for case in cases:
        agent = make_agent_fn()
        try:
            agent.run(case.question)
            passed, detail = score([e["tool"] for e in agent.audit], case)
        except Exception as exc:  # an agent crash is a failed case, not a crashed eval
            passed, detail = False, f"error: {exc}"
        results.append({"question": case.question, "passed": passed, "detail": detail})
    return {"pass_rate": sum(r["passed"] for r in results) / len(results), "results": results}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "demo.db"
        seed_db(db)
        report = run_eval(lambda: make_agent(db, approve=deny_all))
    for r in report["results"]:
        print(("PASS" if r["passed"] else "FAIL"), "|", r["question"], "|", r["detail"])
    print(f"\npass rate: {report['pass_rate']:.0%}")
