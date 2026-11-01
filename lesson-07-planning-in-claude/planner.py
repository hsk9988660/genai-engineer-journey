"""The model-facing half: extract what the user said, write a plan, and write replacement tasks after a failure.

Every call returns a validated Pydantic object. If validation fails, the exact problems go back to the model for
a retry (capped), so the planner can fix its own mistakes without a human noticing.
"""

import json
import os
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from plan import Plan, Requirements, Task, validate_plan, verify_requirements

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")


class PlanningError(Exception):
    """The model could not produce a valid result within the retry limit."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    validation_failures: int = 0  # how often the model's first answer was rejected


class Replacement(BaseModel):
    explanation: str
    tasks: list[Task]


def call_structured(client, system, user, output_model, validate=None, max_retries=2, model=MODEL, usage=None):
    usage = usage if usage is not None else Usage()
    submit = {"name": "submit_result", "description": "Submit the result. Call once.",
              "input_schema": output_model.model_json_schema()}
    messages = [{"role": "user", "content": user}]
    retries = 0
    while True:
        resp = client.messages.create(model=model, max_tokens=2500, system=system, tools=[submit],
                                      tool_choice={"type": "tool", "name": "submit_result"}, messages=messages)
        usage.calls += 1
        usage.input_tokens += resp.usage.input_tokens
        usage.output_tokens += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})
        block = next((b for b in resp.content if b.type == "tool_use" and b.name == "submit_result"), None)
        if block is None:
            raise PlanningError("the model did not submit a result")
        try:
            obj = output_model.model_validate(block.input)
            problems = validate(obj) if validate else []
        except ValidationError as exc:
            problems = [f"Schema errors: {str(exc)[:600]}"]
        if not problems:
            return obj
        usage.validation_failures += 1
        retries += 1
        if retries > max_retries:
            raise PlanningError("; ".join(problems))
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                                      "content": json.dumps({"error": "Rejected. Fix every problem and resubmit.",
                                                                             "problems": problems})}]})


EXTRACT_SYSTEM = """Extract the trip details the user STATED. Today's date is {today}.
- Fill a field only if the user said it. Otherwise set it to null. NEVER guess or fill in typical values.
- For every field you fill, copy the exact supporting words from the request into `quote`.
- Write dates as YYYY-MM-DD. If the user gives a date without a year, use the next occurrence on or after today.
- budget_usd is a plain number with no currency symbol."""

PLAN_SYSTEM = """You are a trip planner. Turn the request into a plan of tasks that other workers will execute.
Available agents and the params each needs:
- flights: destination, depart, return_date
- hotels: city, check_in, check_out   (optional: tier = "standard" or "budget")
- activities: city
- itinerary: no params. Compiles everything into the final answer.
Rules:
- Use ONLY these agents. Give every task a unique task_id starting at 1.
- Copy destination and dates EXACTLY from the Known facts. Never invent or change them.
- A task may depend only on tasks whose results it needs. Independent tasks must not depend on each other, so they can run in parallel.
- Include exactly one itinerary task, and make it depend on every other task.
- For a simple request keep the plan short. Do not add tasks the user did not need."""

REPLAN_SYSTEM = """A plan partly failed. Write REPLACEMENT tasks for the work that did not finish.
{plan_rules}
Extra rules:
- Do not repeat tasks that already succeeded. Their results are kept and you may depend on them.
- Give new tasks task_ids larger than any existing id. Never reuse an id.
- Fix the cause of the failure (for example, a different hotel tier). Do not just retry the identical task.
- Include one new itinerary task that depends on every task that will contribute, including the succeeded ones."""


def extract_requirements(client, request: str, today: str, usage=None) -> Requirements:
    return call_structured(client, EXTRACT_SYSTEM.format(today=today), request, Requirements,
                           validate=lambda r: verify_requirements(r, request), usage=usage)


def make_plan(client, request: str, facts: dict, usage=None) -> Plan:
    user = f"Request: {request}\nKnown facts (use exactly): {json.dumps(facts)}"
    return call_structured(client, PLAN_SYSTEM, user, Plan, validate=lambda p: validate_plan(p, facts), usage=usage)


def make_replacement(client, request: str, facts: dict, plan: Plan, report, usage=None) -> Plan:
    """Ask for replacement tasks and return the COMBINED plan: succeeded tasks are kept unchanged."""
    kept = [t for t in plan.tasks if t.task_id in report.results]
    max_id = max(t.task_id for t in plan.tasks)

    def combined(repl: Replacement) -> Plan:
        return Plan(goal=plan.goal, tasks=kept + repl.tasks)

    def check(repl: Replacement) -> list[str]:
        problems = [f"Task id {t.task_id} is not new: use ids above {max_id}." for t in repl.tasks if t.task_id <= max_id]
        return problems or validate_plan(combined(repl), facts)

    failures = {tid: {"task": next(t.model_dump(mode="json") for t in plan.tasks if t.task_id == tid), "error": err}
                for tid, err in report.failed.items()}
    user = json.dumps({"request": request, "known_facts": facts, "succeeded_task_ids": sorted(report.results),
                       "kept_tasks": [t.model_dump(mode="json") for t in kept], "failures": failures,
                       "skipped_task_ids": sorted(report.skipped)})
    repl = call_structured(client, REPLAN_SYSTEM.format(plan_rules=PLAN_SYSTEM), user, Replacement, validate=check, usage=usage)
    return combined(repl)
