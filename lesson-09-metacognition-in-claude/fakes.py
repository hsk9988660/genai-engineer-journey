"""Scripted stand-in for the Anthropic client plus ready-made itineraries. Calls are sequential, so replies pop in order."""

import copy
from types import SimpleNamespace as NS

from checks import Constraints


def use(id_, tool_name, **inp):
    return NS(type="tool_use", id=id_, name=tool_name, input=inp)


def turn(*blocks, tokens=(100, 50)):
    return NS(content=list(blocks), stop_reason="tool_use", usage=NS(input_tokens=tokens[0], output_tokens=tokens[1]))


def submit(payload: dict, tokens=(100, 50)):
    return turn(use("sub", "submit_result", **payload), tokens=tokens)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw | {"messages": kw["messages"][:]})
        return self.replies.pop(0)


C = Constraints(budget_usd=300, num_days=2, max_activities_per_day=3, dinner_required=True)


def act(name, start, end, cost, kind="sight"):
    return {"name": name, "start": start, "end": end, "cost_usd": cost, "kind": kind}


def good():
    """A valid itinerary for C: 2 days, 3 activities each, dinner each day, $200 total."""
    return {"days": [
        {"day": 1, "activities": [act("Museum", "09:00", "11:00", 30), act("Lunch", "12:00", "13:00", 20, "meal"),
                                  act("Dinner", "19:00", "20:30", 50, "meal")]},
        {"day": 2, "activities": [act("Tour", "10:00", "12:00", 40), act("Walk", "14:00", "15:00", 0),
                                  act("Dinner", "19:00", "20:00", 60, "meal")]}], "total_cost_usd": 200.0}


def broken(*problems):
    """`good()` with specific, known problems introduced."""
    it = copy.deepcopy(good())
    for p in problems:
        if p == "no_dinner":
            it["days"][1]["activities"][2]["start"], it["days"][1]["activities"][2]["end"] = "16:00", "17:00"
        elif p == "overlap":
            it["days"][0]["activities"][1]["start"] = "10:30"
        elif p == "total_mismatch":
            it["total_cost_usd"] = 150.0
        elif p == "budget":
            it["days"][0]["activities"][0]["cost_usd"] = 400.0
            it["total_cost_usd"] = 570.0
        elif p == "too_many":
            it["days"][0]["activities"].append(act("Cafe", "15:00", "16:00", 5))
            it["total_cost_usd"] = 205.0
    return it
