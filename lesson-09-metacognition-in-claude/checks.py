"""Deterministic checks: the SIGNAL that makes reflection work.

An agent re-reading its own answer and deciding it "looks fine" is the weakest form of self-correction. An agent whose
output failed a concrete check, and which is told exactly how, is the strong form. These checkers are plain code, so
they cannot be talked out of a failure, and every message names the specific thing that is wrong.
"""

import re
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Activity(BaseModel):
    name: str
    start: str = Field(description="24-hour HH:MM")
    end: str = Field(description="24-hour HH:MM")
    cost_usd: float = Field(ge=0)
    kind: Literal["sight", "meal", "transport", "rest"]


class Day(BaseModel):
    day: int
    activities: list[Activity]


class Itinerary(BaseModel):
    days: list[Day]
    total_cost_usd: float = Field(description="Must equal the sum of every activity's cost_usd")


class Constraints(BaseModel):
    budget_usd: float
    num_days: int
    max_activities_per_day: int = 4
    dinner_required: bool = True


@dataclass(frozen=True)
class Failure:
    code: str  # stable identifier, used for stall detection and for lessons
    message: str  # specific enough for the model to act on
    day: Optional[int] = None


_TIME = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
DINNER_FROM = 18 * 60


def minutes(hhmm: str) -> Optional[int]:
    m = _TIME.match(hhmm)
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def check_itinerary(it: Itinerary, c: Constraints) -> list[Failure]:
    fails: list[Failure] = []
    numbers = [d.day for d in it.days]
    if len(it.days) != c.num_days:
        fails.append(Failure("days_count", f"The trip must have exactly {c.num_days} days but the itinerary has {len(it.days)}."))
    for n in sorted({n for n in numbers if numbers.count(n) > 1}):
        fails.append(Failure("duplicate_day", f"Day number {n} appears more than once.", n))

    computed = 0.0
    for day in it.days:
        timed = []
        for a in day.activities:
            computed += a.cost_usd
            s, e = minutes(a.start), minutes(a.end)
            if s is None or e is None or e <= s:
                fails.append(Failure("bad_time", f"Day {day.day}: '{a.name}' has an invalid time range {a.start}-{a.end}. "
                                                 "Use HH:MM and make the end later than the start.", day.day))
            else:
                timed.append((s, e, a))
        timed.sort(key=lambda t: t[0])
        for (s1, e1, a1), (s2, e2, a2) in zip(timed, timed[1:]):
            if s2 < e1:  # touching (end == next start) is allowed
                fails.append(Failure("overlap", f"Day {day.day}: '{a1.name}' ({a1.start}-{a1.end}) overlaps '{a2.name}' "
                                                f"({a2.start}-{a2.end}).", day.day))
        if len(day.activities) > c.max_activities_per_day:
            fails.append(Failure("too_many", f"Day {day.day} has {len(day.activities)} activities but the limit is "
                                             f"{c.max_activities_per_day}.", day.day))
        if c.dinner_required and not any(a.kind == "meal" and (minutes(a.start) or 0) >= DINNER_FROM for a in day.activities):
            fails.append(Failure("no_dinner", f"Day {day.day} has no dinner. Add a meal that starts at 18:00 or later.", day.day))

    if computed > c.budget_usd:
        fails.append(Failure("budget", f"The activities cost ${computed:,.2f} in total, over the ${c.budget_usd:,.2f} budget."))
    if abs(computed - it.total_cost_usd) > 0.01:
        fails.append(Failure("total_mismatch", f"total_cost_usd is ${it.total_cost_usd:,.2f} but the activities sum to ${computed:,.2f}."))
    return fails


def fingerprint(failures: list[Failure]) -> frozenset:
    """Which problems exist, ignoring wording. Used to notice that two attempts failed in the same way."""
    return frozenset((f.code, f.day) for f in failures)
