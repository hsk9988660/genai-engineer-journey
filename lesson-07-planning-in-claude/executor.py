"""Run a validated plan from CODE, not from a model's memory of the plan.

The lesson 07 notebook flattened the plan into prose and asked one agent to "work through the subtasks in order,
respecting dependencies". Here a scheduler runs the graph layer by layer, in parallel where tasks are independent,
passes each task the outputs of the tasks it depends on, checks a postcondition on every result, retries, skips
the tasks that depended on a failure, and halts when the spend limit is passed.

Honest limit on spend: a layer's tasks run together, so the budget is checked AFTER each layer. Money spent inside a
layer cannot be taken back, so a budget can be overshot by up to one layer. If that matters, make the workers
quote-only and put the real purchase in a final step.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from plan import AgentName, Plan, Task, layers, plan_hash

SIDE_EFFECT_AGENTS = {AgentName.flights, AgentName.hotels, AgentName.activities}  # these reserve things


@dataclass
class StepResult:
    output: dict
    cost_usd: float = 0.0


@dataclass
class ExecutionReport:
    results: dict[int, StepResult] = field(default_factory=dict)
    failed: dict[int, str] = field(default_factory=dict)
    skipped: dict[int, str] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    halted_reason: str | None = None

    @property
    def ok(self) -> bool:
        return not self.failed and not self.skipped and self.halted_reason is None


class ApprovalError(Exception):
    """The plan about to run is not the plan a human approved."""


def _positive(r, *keys):
    return all(r.output.get(k) for k in keys)


# A postcondition returns None when the result is acceptable, or a message saying why not.
POSTCONDITIONS: dict[AgentName, Callable[[StepResult], str | None]] = {
    AgentName.flights: lambda r: None if _positive(r, "flight") and r.output.get("price", 0) > 0 else "flight needs a flight id and a positive price",
    AgentName.hotels: lambda r: None if _positive(r, "hotel") and r.output.get("nights", 0) > 0 and r.output.get("price", 0) > 0 else "hotel needs a name, nights > 0 and a positive price",
    AgentName.activities: lambda r: None if r.output.get("items") else "activities needs at least one item",
    AgentName.itinerary: lambda r: None if str(r.output.get("summary", "")).strip() else "itinerary needs a non-empty summary",
}


class PlanExecutor:
    def __init__(self, workers: dict, postconditions=POSTCONDITIONS, max_attempts: int = 2, max_workers: int = 4,
                 budget_usd: float | None = None):
        self.workers, self.postconditions = workers, postconditions
        self.max_attempts, self.max_workers, self.budget_usd = max_attempts, max_workers, budget_usd

    def run(self, plan: Plan, approved_hash: str | None = None, prior: dict | None = None) -> ExecutionReport:
        """`prior` holds results of tasks that already succeeded in an earlier run; they are not run again."""
        report = ExecutionReport(results=dict(prior or {}))
        report.total_cost_usd = sum(r.cost_usd for r in report.results.values())
        pending = [t for t in plan.tasks if t.task_id not in report.results]
        if any(t.agent in SIDE_EFFECT_AGENTS for t in pending) and approved_hash != plan_hash(plan):
            raise ApprovalError("the plan to execute is not the plan that was approved")  # re-hash what really runs

        for layer in layers(plan):
            runnable = []
            for task in (t for t in layer if t.task_id not in report.results):
                blocked = sorted(d for d in task.depends_on if d in report.failed or d in report.skipped)
                if blocked:
                    report.skipped[task.task_id] = f"not run because task(s) {blocked} did not succeed"
                else:
                    runnable.append(task)
            if runnable:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    outcomes = list(pool.map(lambda t: self._run_step(t, report.results), runnable))
                for task, (result, error) in zip(runnable, outcomes):
                    if error:
                        report.failed[task.task_id] = error
                    else:
                        report.results[task.task_id] = result
                        report.total_cost_usd += result.cost_usd
            if self.budget_usd is not None and report.total_cost_usd > self.budget_usd:
                report.halted_reason = f"over budget: ${report.total_cost_usd:,.2f} > ${self.budget_usd:,.2f}"
                break
        return report

    def _run_step(self, task: Task, results: dict) -> tuple[StepResult | None, str | None]:
        worker = self.workers.get(task.agent)
        if worker is None:
            return None, f"no worker registered for {task.agent.value}"
        inputs = {d: results[d].output for d in task.depends_on}  # data flow: a task receives its dependencies' outputs
        error = "never ran"
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = worker(task, inputs)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                continue
            problem = self.postconditions[task.agent](result)
            if problem is None:
                return result, None
            error = f"postcondition failed: {problem}"
        return None, error
