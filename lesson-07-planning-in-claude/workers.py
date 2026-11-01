"""Deterministic stand-in workers. In production each would call a real flight, hotel or activity service.

Failure injection: the "standard" hotel tier is sold out in Lisbon, which is what makes the replanning path real.
"""

from datetime import date

from executor import StepResult
from plan import AgentName, Task


class SoldOut(Exception):
    pass


def flights_worker(task: Task, inputs: dict) -> StepResult:
    p = task.params
    price = 480.0
    return StepResult({"flight": "AF1081", "destination": p["destination"], "depart": p["depart"],
                       "return_date": p["return_date"], "price": price}, price)


def hotels_worker(task: Task, inputs: dict) -> StepResult:
    p = task.params
    tier = str(p.get("tier", "standard"))
    if p["city"] == "Lisbon" and tier == "standard":
        raise SoldOut("no standard rooms left in Lisbon for those dates")
    nights = (date.fromisoformat(str(p["check_out"])) - date.fromisoformat(str(p["check_in"]))).days
    price = (140.0 if tier == "standard" else 80.0) * nights
    return StepResult({"hotel": f"Hotel {p['city']} ({tier})", "nights": nights, "price": price}, price)


def activities_worker(task: Task, inputs: dict) -> StepResult:
    return StepResult({"items": [f"{task.params['city']} museum pass", "Food tour"], "price": 120.0}, 120.0)


def itinerary_worker(task: Task, inputs: dict) -> StepResult:
    """Pure code: compile the outputs of the tasks this one depends on."""
    total = sum(o.get("price", 0.0) for o in inputs.values())
    parts = []
    for out in inputs.values():
        if "flight" in out:
            parts.append(f"Flight {out['flight']} ({out['depart']} to {out['return_date']})")
        elif "hotel" in out:
            parts.append(f"{out['hotel']}, {out['nights']} nights")
        elif "items" in out:
            parts.append("Activities: " + ", ".join(out["items"]))
    return StepResult({"summary": "; ".join(parts), "estimated_total_usd": total}, 0.0)


DEMO_WORKERS = {AgentName.flights: flights_worker, AgentName.hotels: hotels_worker,
                AgentName.activities: activities_worker, AgentName.itinerary: itinerary_worker}
