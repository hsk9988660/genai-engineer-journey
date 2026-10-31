"""Lesson 03's three patterns on the Claude API: clear instructions, structured output,
single-responsibility agents.

The key idea of the file: the model's final answer is delivered as a tool call whose
input schema is a Pydantic model. We validate it, and if it is invalid we send the error
back so the model can fix it. Code then makes the decisions that must be right.
"""

import inspect
import json
import os
import typing

import anthropic
from pydantic import BaseModel, ValidationError

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")


# --- Tool decorator (same idea as lesson 02, compact) -------------------------------------
_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def tool(f):
    """Build a tool schema from the function's signature and docstring."""
    hints = typing.get_type_hints(f)
    props = {n: {"type": _JSON.get(typing.get_origin(hints.get(n, str)) or hints.get(n, str), "string")}
             for n in inspect.signature(f).parameters}
    f.schema = {"name": f.__name__, "description": inspect.getdoc(f),
                "input_schema": {"type": "object", "properties": props, "required": list(props)}}
    return f


# --- Pattern 1: clear instructions ---------------------------------------------------------
# Who the agent is, what it does in order, and what it must NOT do. Reasons help the model
# apply a rule to cases you did not list.
VAGUE_INSTRUCTIONS = "You help with travel."

CONCIERGE_INSTRUCTIONS = """You are Alex, a luxury travel concierge. Your role is to:
1. Understand the traveler's preferences (budget, climate, activities).
2. Recommend only destinations you have checked with a tool, because availability changes daily.
3. Give specific, personalized suggestions.
4. Always mention visa requirements and the best travel season.
Be warm and professional. If you cannot find a suitable destination, say so rather than guessing."""


# --- Pattern 2: structured output ----------------------------------------------------------
class StructuredOutputError(Exception):
    """The model never produced output that matches the schema."""


def _to_json(value) -> str:
    return value.model_dump_json() if isinstance(value, BaseModel) else json.dumps(value, default=str)


def run_structured(client, system, user_msg, output_model, tools=(), max_steps=6,
                   max_retries=2, model=MODEL):
    """Run an agent whose final answer must be a valid `output_model`.

    The model may call your tools any number of times, then must call `submit_result`.
    tool_choice="any" forces a tool call on every turn, so it can never reply in free text.
    """
    submit = {
        "name": "submit_result",
        "description": "Submit the final result. Call this once, when you are done.",
        "input_schema": output_model.model_json_schema(),
    }
    registry = {t.__name__: t for t in tools}
    schemas = [t.schema for t in tools] + [submit]
    messages = [{"role": "user", "content": user_msg}]
    retries = 0

    for _ in range(max_steps):
        resp = client.messages.create(model=model, max_tokens=1500, system=system, tools=schemas,
                                      tool_choice={"type": "any"}, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in (b for b in resp.content if b.type == "tool_use"):
            if block.name == "submit_result":
                try:
                    return output_model.model_validate(block.input)  # never trust, always validate
                except ValidationError as exc:
                    retries += 1
                    if retries > max_retries:
                        raise StructuredOutputError(f"invalid output after {max_retries} retries: {exc}") from exc
                    out, is_error = {"error": "Output failed validation. Fix and call submit_result again.",
                                     "details": str(exc)[:800]}, True
            elif block.name in registry:
                try:
                    out, is_error = registry[block.name](**block.input), False
                except Exception as exc:
                    out, is_error = {"error": str(exc)}, True
            else:
                out, is_error = {"error": f"No such tool: {block.name}"}, True
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": _to_json(out), "is_error": is_error})
        if not results:
            raise StructuredOutputError("model returned no tool call")
        messages.append({"role": "user", "content": results})
    raise StructuredOutputError("step limit reached before a valid result was submitted")


# --- Pattern 3: single-responsibility agents -----------------------------------------------
class Destination(BaseModel):
    destination: str
    available: bool
    best_season: str
    highlights: list[str]
    estimated_budget_usd: int


class Shortlist(BaseModel):
    picks: list[Destination]
    reasoning: str


class Itinerary(BaseModel):
    destination: str
    days: list[str]
    estimated_total_usd: int
    visa_note: str


# Ground truth. In a real system this is a database or API.
_DB = {
    "Barcelona": Destination(destination="Barcelona", available=True, best_season="May-Jun",
                             highlights=["Beach", "Architecture", "Food"], estimated_budget_usd=2000),
    "Tokyo": Destination(destination="Tokyo", available=True, best_season="Mar-Apr",
                         highlights=["Culture", "Food", "Technology"], estimated_budget_usd=2500),
    "Cape Town": Destination(destination="Cape Town", available=False, best_season="Nov-Mar",
                             highlights=["Nature", "Wine"], estimated_budget_usd=1800),
    "Lisbon": Destination(destination="Lisbon", available=True, best_season="Apr-Jun",
                          highlights=["History", "Food", "Coast"], estimated_budget_usd=1700),
}


@tool
def get_destination_details(destination: str) -> Destination:
    """Look up availability, best season, highlights and budget for one destination by name."""
    if destination not in _DB:
        raise ValueError(f"unknown destination '{destination}'; known: {sorted(_DB)}")
    return _DB[destination]


EXPERT_INSTRUCTIONS = """You are a destination research specialist. Your only job is to:
1. Evaluate destinations against the traveler's preferences.
2. Check each candidate with get_destination_details.
3. Submit a short ranked shortlist with your reasoning.
Do NOT plan flights, hotels or itineraries; another agent does that."""

PLANNER_INSTRUCTIONS = """You are a travel logistics planner. Your only job is to:
1. Write a day-by-day itinerary for the destination you are given.
2. Estimate the total cost in USD.
3. Note visa requirements.
Do NOT choose or change the destination; another agent already chose it."""


class NoAvailableDestination(Exception):
    pass


def destination_expert(client, request: str) -> Shortlist:
    return run_structured(client, EXPERT_INSTRUCTIONS, request, Shortlist, tools=[get_destination_details])


def logistics_planner(client, choice: Destination, request: str) -> Itinerary:
    brief = f"Traveler request: {request}\nChosen destination (JSON): {choice.model_dump_json()}"
    return run_structured(client, PLANNER_INSTRUCTIONS, brief, Itinerary)


def plan_trip(client, request: str, budget_usd: int) -> Itinerary:
    """Expert -> code checks -> planner. Structured data lets CODE enforce the hard rules."""
    shortlist = destination_expert(client, request)
    # Re-check against ground truth. The model's `available` and budget fields are claims,
    # not facts: it may have misread a tool result or invented a destination.
    verified = []
    for pick in shortlist.picks:
        truth = _DB.get(pick.destination)
        if truth and truth.available and truth.estimated_budget_usd <= budget_usd:
            verified.append(truth)
    if not verified:
        raise NoAvailableDestination(f"no available destination within ${budget_usd}")
    return logistics_planner(client, verified[0], request)  # pass only the verified top pick


if __name__ == "__main__":
    c = anthropic.Anthropic()
    ask = "I want a week of culture and food for under $2500."
    print("--- Pattern 1: vague vs clear instructions ---")
    for label, system in [("vague", VAGUE_INSTRUCTIONS), ("clear", CONCIERGE_INSTRUCTIONS)]:
        r = c.messages.create(model=MODEL, max_tokens=400, system=system, messages=[{"role": "user", "content": ask}])
        print(f"[{label}]\n{r.content[0].text}\n")
    print("--- Patterns 2 + 3: structured, single-responsibility pipeline ---")
    print(plan_trip(c, ask, budget_usd=2500).model_dump_json(indent=2))
