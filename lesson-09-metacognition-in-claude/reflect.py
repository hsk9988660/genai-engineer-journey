"""A grounded reflection loop: generate, run deterministic checks, feed back the SPECIFIC failures, revise, stop sensibly.

Why this and not "review your answer":
  * the feedback is concrete (what is wrong and where), not "try again"
  * it stops on success, on a hard cap, and when it STALLS (the same failures twice in a row)
  * it returns the BEST candidate seen (fewest failures), not necessarily the last, so a revision that made things worse
    is never shipped over a better earlier attempt
  * lessons from validated failures can be saved to a memory that cannot be poisoned (see memory.py)

Run:  python reflect.py "Plan 2 days in Lisbon"      (needs ANTHROPIC_API_KEY)
"""

import json
import sys
from dataclasses import dataclass, field

from checks import Constraints, Failure, Itinerary, check_itinerary, fingerprint
from memory import ReflectionMemory
from structured import Usage, call_structured


@dataclass
class Attempt:
    number: int
    failures: list
    fingerprint: frozenset


@dataclass
class LoopResult:
    status: str  # passed | capped | stalled
    best: object
    best_failures: list
    attempts: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.best_failures


def solve_with_checks(generate, check, max_attempts: int = 4) -> LoopResult:
    """`generate(feedback, attempt_number)` returns a candidate; `check(candidate)` returns a list of Failure."""
    attempts: list[Attempt] = []
    best, best_failures, feedback = None, None, None
    for n in range(1, max_attempts + 1):
        candidate = generate(feedback, n)
        failures = check(candidate)
        attempts.append(Attempt(n, failures, fingerprint(failures)))
        if best is None or len(failures) < len(best_failures):
            best, best_failures = candidate, failures
        if not failures:
            return LoopResult("passed", best, best_failures, attempts)
        if n > 1 and attempts[-1].fingerprint == attempts[-2].fingerprint:
            return LoopResult("stalled", best, best_failures, attempts)  # same failures twice: more tries will not help
        feedback = [f.message for f in failures]
    return LoopResult("capped", best, best_failures, attempts)


GENERATE_SYSTEM = """You plan travel itineraries and return them as structured data.
Obey every constraint exactly. Times are 24-hour HH:MM. total_cost_usd must equal the sum of every activity's cost_usd."""


class ItineraryGenerator:
    """The model-facing half. Attempt 1 has no feedback. Later attempts see the previous itinerary and the exact failures."""

    def __init__(self, client, task: str, constraints: Constraints, memory: ReflectionMemory | None = None,
                 usage: Usage | None = None):
        self.client, self.task, self.constraints = client, task, constraints
        self.memory, self.usage, self.last = memory, usage if usage is not None else Usage(), None

    def __call__(self, feedback, attempt: int) -> Itinerary:
        lessons = self.memory.render() if self.memory else ""
        system = GENERATE_SYSTEM + (f"\n\n{lessons}" if lessons else "")
        payload = {"task": self.task, "constraints": self.constraints.model_dump()}
        if feedback:
            payload["previous_itinerary"] = self.last.model_dump() if self.last else None
            payload["automatic_check_failures"] = feedback
            payload["instruction"] = "Fix every listed failure. Change as little else as possible."
        self.last = call_structured(self.client, system, json.dumps(payload), Itinerary, self.usage)
        return self.last


def solve_itinerary(client, task: str, constraints: Constraints, memory: ReflectionMemory | None = None,
                    max_attempts: int = 4, usage: Usage | None = None):
    usage = usage if usage is not None else Usage()
    generator = ItineraryGenerator(client, task, constraints, memory, usage)
    result = solve_with_checks(generator, lambda it: check_itinerary(it, constraints), max_attempts)
    if memory is not None:  # every failure here was confirmed by a deterministic checker
        for attempt in result.attempts:
            memory.record_failures(attempt.failures, trusted=True)
    return result, usage


if __name__ == "__main__":
    import anthropic

    task = " ".join(sys.argv[1:]) or "Plan 2 days in Lisbon for a couple who love food."
    res, used = solve_itinerary(anthropic.Anthropic(), task, Constraints(budget_usd=300, num_days=2, max_activities_per_day=3))
    print(f"status={res.status} attempts={len(res.attempts)} tokens={used.total}")
    for a in res.attempts:
        print(f"  attempt {a.number}: {len(a.failures)} failure(s)", *[f"\n    - {f.message}" for f in a.failures])
