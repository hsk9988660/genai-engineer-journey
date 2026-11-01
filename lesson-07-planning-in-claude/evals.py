"""Evaluate plans separately from execution, so you can tell which half broke.

  python evals.py lint    offline: does the plan validator catch each kind of bad plan? (no API key)
  python evals.py live    needs ANTHROPIC_API_KEY: does the real planner behave? (costs a few cents)

`lint` checks the VALIDATOR, using hand-written bad plans. It says nothing about how often a real model writes bad
plans. That is what `live` measures, and I have not run it.
"""

import sys

from plan import Plan, Task, validate_plan

FACTS = {"destination": "Paris", "depart": "2026-05-02", "return_date": "2026-05-09"}


def T(task_id, agent, deps=(), **params):
    return {"task_id": task_id, "agent": agent, "description": f"{agent} step", "depends_on": list(deps), "params": params}


FLIGHTS = T(1, "flights", destination="Paris", depart="2026-05-02", return_date="2026-05-09")
HOTEL = T(2, "hotels", city="Paris", check_in="2026-05-02", check_out="2026-05-09")
ACTS = T(3, "activities", city="Paris")
FINAL = T(4, "itinerary", [1, 2, 3])

# (name, tasks, substring expected in the problems, or None if the plan must be valid)
LINT_CASES = [
    ("valid plan", [FLIGHTS, HOTEL, ACTS, FINAL], None),
    ("valid plan, tasks listed out of order", [FINAL, ACTS, HOTEL, FLIGHTS], None),
    ("no tasks", [], "no tasks"),
    ("duplicate task id", [FLIGHTS, {**HOTEL, "task_id": 1}, FINAL], "more than once"),
    ("dependency on a task that does not exist", [FLIGHTS, HOTEL, ACTS, T(4, "itinerary", [1, 2, 3, 9])], "does not exist"),
    ("task depends on itself", [FLIGHTS, HOTEL, ACTS, T(4, "itinerary", [1, 2, 3, 4])], "depends on itself"),
    ("dependency cycle", [T(1, "flights", [2], destination="Paris", depart="2026-05-02", return_date="2026-05-09"),
                          T(2, "hotels", [1], city="Paris", check_in="2026-05-02", check_out="2026-05-09"),
                          T(3, "itinerary", [1, 2])], "cycle"),
    ("hotel missing check_out", [FLIGHTS, T(2, "hotels", city="Paris", check_in="2026-05-02"), ACTS, FINAL], "missing params"),
    ("invented departure date", [T(1, "flights", destination="Paris", depart="2026-05-03", return_date="2026-05-09"),
                                 HOTEL, ACTS, FINAL], "must equal"),
    ("changed the destination", [FLIGHTS, HOTEL, T(3, "activities", city="Rome"), FINAL], "must equal"),
    ("two itinerary tasks", [FLIGHTS, HOTEL, ACTS, FINAL, T(5, "itinerary", [1, 2, 3, 4])], "exactly one"),
    ("task that feeds nothing", [FLIGHTS, HOTEL, ACTS, T(4, "itinerary", [1, 2])], "Not connected"),
]


def build(tasks) -> Plan:
    return Plan(goal="g", tasks=[Task.model_validate(t) for t in tasks])


def lint_report() -> list[dict]:
    rows = []
    for name, tasks, expected in LINT_CASES:
        problems = validate_plan(build(tasks), FACTS)
        ok = (not problems) if expected is None else any(expected in p for p in problems)
        rows.append({"case": name, "ok": ok, "problems": problems})
    return rows


# --- live (needs a key) -----------------------------------------------------------------------
# (request, expected status, minimum replans). Lisbon's standard hotel tier is sold out in the demo workers.
LIVE_CASES = [
    ("Plan a trip to Paris from 2026-05-02 to 2026-05-09.", "completed", 0),
    ("Plan a week in Paris.", "needs_info", 0),
    ("Book me a trip.", "needs_info", 0),
    ("Hello!", "needs_info", 0),
    ("I want to go to Lisbon from 2026-06-01 to 2026-06-08.", "completed", 1),
    ("Trip to Paris from 2026-05-02 to 2026-05-09 with a budget of 100 dollars.", "over_budget", 0),
]


def run_live_eval(client, today="2026-01-15") -> dict:
    from orchestrator import plan_and_execute
    from workers import DEMO_WORKERS

    rows = []
    for request, expected, min_replans in LIVE_CASES:
        try:
            out = plan_and_execute(client, request, lambda summary: "approve", DEMO_WORKERS, today=today)
            ok = out.status == expected and out.replans >= min_replans
            rows.append({"request": request, "expected": expected, "got": out.status, "replans": out.replans,
                         "calls": out.usage.calls, "rejected_first_tries": out.usage.validation_failures,
                         "tokens": out.usage.input_tokens + out.usage.output_tokens, "ok": ok})
        except Exception as exc:  # a crash is a failed case, not a crashed eval
            rows.append({"request": request, "expected": expected, "got": f"error: {exc}", "replans": 0, "calls": 0,
                         "rejected_first_tries": 0, "tokens": 0, "ok": False})
    n = len(rows)
    return {"pass_rate": sum(r["ok"] for r in rows) / n, "rows": rows,
            "first_try_validity": 1 - sum(r["rejected_first_tries"] for r in rows) / max(1, sum(r["calls"] for r in rows))}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "lint"
    if which == "lint":
        rows = lint_report()
        for r in rows:
            print(("PASS" if r["ok"] else "FAIL"), "|", r["case"], "|", (r["problems"] or ["(valid)"])[0][:90])
        print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} cases handled correctly by the validator")
    elif which == "live":
        import anthropic

        report = run_live_eval(anthropic.Anthropic())
        for r in report["rows"]:
            print(("PASS" if r["ok"] else "FAIL"), "|", r["request"], "| expected", r["expected"], "got", r["got"],
                  "| replans", r["replans"], "| tokens", r["tokens"])
        print(f"\npass rate {report['pass_rate']:.0%}; model answers accepted first try: {report['first_try_validity']:.0%}")
    else:
        print("usage: python evals.py [lint|live]")
