"""The plan as validated data: schema, graph checks, grounding in what the user actually said, and layering.

The lesson 07 notebook defined Pydantic models but the README never validated against them, let `assigned_agent`
be free text, and never checked dependencies. Here every property a plan needs in order to be executed safely is
checked in code, and the problems are written so they can be fed straight back to the planning model.
"""

import hashlib
import json
import re
from datetime import date
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field


class AgentName(str, Enum):
    flights = "flights"
    hotels = "hotels"
    activities = "activities"
    itinerary = "itinerary"  # compiles everything into the final answer; has no side effects


Scalar = Union[str, int, float]


class Task(BaseModel):
    task_id: int = Field(ge=1)
    agent: AgentName
    description: str
    depends_on: list[int] = Field(default_factory=list, description="task_ids whose results this task needs")
    params: dict[str, Scalar] = Field(default_factory=dict, description="concrete parameters for this agent")


class Plan(BaseModel):
    goal: str
    tasks: list[Task]


# --- what the user actually said -------------------------------------------------------------
class Fact(BaseModel):
    value: str = Field(description="The normalized value (dates as YYYY-MM-DD, budget as a plain number)")
    quote: str = Field(description="Words copied EXACTLY from the user's request that support this value")


class Requirements(BaseModel):
    destination: Optional[Fact] = None
    depart: Optional[Fact] = None
    return_date: Optional[Fact] = None
    budget_usd: Optional[Fact] = None


REQUIRED_FACTS = {"destination": "Where would you like to travel?",
                  "depart": "What date do you want to leave?",
                  "return_date": "What date do you want to return?"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip("\"'").lower()


def verify_requirements(req: Requirements, request: str) -> list[str]:
    """Reject facts the user did not say. Every fact needs a quote that really appears in the request."""
    problems = []
    for name in ("destination", "depart", "return_date", "budget_usd"):
        fact = getattr(req, name)
        if fact is None:
            continue
        if len(_norm(fact.quote)) < 2 or _norm(fact.quote) not in _norm(request):
            problems.append(f"{name}: the quote {fact.quote!r} does not appear in the user's request. "
                            f"Set {name} to null if the user did not say it. Never guess.")
            continue
        if name in ("depart", "return_date"):
            try:
                date.fromisoformat(fact.value)
            except ValueError:
                problems.append(f"{name}: {fact.value!r} is not a YYYY-MM-DD date.")
        if name == "budget_usd":
            try:
                float(fact.value)
            except ValueError:
                problems.append(f"budget_usd: {fact.value!r} is not a plain number.")
    if not problems and req.depart and req.return_date and req.depart.value >= req.return_date.value:
        problems.append("return_date must be after depart.")
    return problems


def missing_questions(req: Requirements) -> list[str]:
    """Questions to ask the user instead of letting the planner invent the answer."""
    return [q for name, q in REQUIRED_FACTS.items() if getattr(req, name) is None]


def facts_of(req: Requirements) -> dict:
    return {name: getattr(req, name).value for name in ("destination", "depart", "return_date", "budget_usd")
            if getattr(req, name) is not None}


# --- plan validation -------------------------------------------------------------------------
REQUIRED_PARAMS = {
    AgentName.flights: {"destination", "depart", "return_date"},
    AgentName.hotels: {"city", "check_in", "check_out"},
    AgentName.activities: {"city"},
    AgentName.itinerary: set(),
}
# A parameter that carries something the USER said must equal what they said.
FACT_FOR_PARAM = {"destination": "destination", "city": "destination", "depart": "depart",
                  "return_date": "return_date", "check_in": "depart", "check_out": "return_date"}


def _ancestors(by_id: dict, task_id: int, seen=None) -> set:
    seen = set() if seen is None else seen
    for d in by_id[task_id].depends_on:
        if d in by_id and d not in seen:
            seen.add(d)
            _ancestors(by_id, d, seen)
    return seen


def validate_plan(plan: Plan, facts: dict) -> list[str]:
    """Return a list of problems. Empty means the plan is safe to schedule."""
    problems = []
    if not plan.tasks:
        return ["The plan has no tasks."]
    ids = [t.task_id for t in plan.tasks]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"task_id {dup} is used more than once.")
    known = set(ids)
    graph_ok = not problems
    for t in plan.tasks:
        for d in t.depends_on:
            if d == t.task_id:
                problems.append(f"Task {t.task_id} depends on itself.")
                graph_ok = False
            elif d not in known:
                problems.append(f"Task {t.task_id} depends on task {d}, which does not exist.")
                graph_ok = False
        missing = REQUIRED_PARAMS[t.agent] - set(t.params)
        if missing:
            problems.append(f"Task {t.task_id} ({t.agent.value}) is missing params: {sorted(missing)}.")
        for key, value in t.params.items():
            fact_name = FACT_FOR_PARAM.get(key)
            if fact_name and str(value) != facts.get(fact_name):
                problems.append(f"Task {t.task_id}: param {key}={value!r} must equal the user's {fact_name} "
                                f"{facts.get(fact_name)!r}. Do not invent or change details the user gave.")
    if graph_ok:
        by_id = {t.task_id: t for t in plan.tasks}
        cyclic = sorted(i for i in by_id if i in _ancestors(by_id, i))
        if cyclic:
            problems.append(f"Dependency cycle involving tasks {cyclic}.")
        else:
            finals = [t for t in plan.tasks if t.agent is AgentName.itinerary]
            if len(finals) > 1:
                problems.append("Use exactly one itinerary task.")
            elif finals:
                feeding = _ancestors(by_id, finals[0].task_id)
                orphans = sorted(known - feeding - {finals[0].task_id})
                if orphans:
                    problems.append(f"The itinerary task must depend (directly or indirectly) on every other task. "
                                    f"Not connected: {orphans}.")
    return problems


def layers(plan: Plan) -> list[list[Task]]:
    """Group a VALID plan into layers. Tasks in one layer have no dependencies on each other, so they can run in parallel."""
    remaining, done, out = {t.task_id: t for t in plan.tasks}, set(), []
    while remaining:
        ready = [t for t in remaining.values() if set(t.depends_on) <= done]
        if not ready:
            raise ValueError("plan has a cycle or a missing dependency; validate_plan first")
        out.append(sorted(ready, key=lambda t: t.task_id))
        done |= {t.task_id for t in ready}
        for t in ready:
            del remaining[t.task_id]
    return out


def plan_hash(plan: Plan) -> str:
    canonical = json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
