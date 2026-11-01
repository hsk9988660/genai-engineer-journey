import threading
import time

import pytest

from evals import ACTS, FINAL, FLIGHTS, HOTEL, build
from executor import ApprovalError, PlanExecutor, StepResult
from plan import AgentName, plan_hash
from workers import DEMO_WORKERS

FULL = build([FLIGHTS, HOTEL, ACTS, FINAL])


def run(plan=FULL, workers=None, approved=True, **kw):
    return PlanExecutor(workers or DEMO_WORKERS, **kw).run(plan, approved_hash=plan_hash(plan) if approved else None)


def test_full_plan_runs_and_the_itinerary_receives_its_dependencies_outputs():
    rep = run()
    assert rep.ok and set(rep.results) == {1, 2, 3, 4}
    summary = rep.results[4].output["summary"]
    assert "AF1081" in summary and "7 nights" in summary and "museum pass" in summary
    assert rep.results[4].output["estimated_total_usd"] == 480.0 + 980.0 + 120.0
    assert rep.total_cost_usd == 480.0 + 980.0 + 120.0  # itinerary itself costs nothing


def test_independent_tasks_run_in_parallel():
    def slow(task, inputs):
        time.sleep(0.3)
        return DEMO_WORKERS[task.agent](task, inputs)

    workers = {**DEMO_WORKERS, AgentName.flights: slow, AgentName.hotels: slow, AgentName.activities: slow}
    start = time.monotonic()
    assert run(workers=workers).ok
    assert time.monotonic() - start < 0.8  # three 0.3s tasks sequentially would be 0.9s


def test_a_task_never_starts_before_its_dependencies_finish():
    order, lock = [], threading.Lock()

    def tracking(task, inputs):
        with lock:
            order.append(("start", task.task_id))
        time.sleep(0.05)
        out = DEMO_WORKERS[task.agent](task, inputs)
        with lock:
            order.append(("end", task.task_id))
        return out

    run(workers={a: tracking for a in AgentName})
    assert order.index(("start", 4)) > max(order.index(("end", i)) for i in (1, 2, 3))


def test_failed_task_is_retried_then_succeeds():
    calls = []

    def flaky(task, inputs):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("upstream timeout")
        return DEMO_WORKERS[task.agent](task, inputs)

    rep = run(workers={**DEMO_WORKERS, AgentName.flights: flaky}, max_attempts=2)
    assert rep.ok and len(calls) == 2


def test_a_failure_skips_only_its_dependents_and_independent_work_still_finishes():
    def sold_out(task, inputs):
        raise RuntimeError("no rooms")

    rep = run(workers={**DEMO_WORKERS, AgentName.hotels: sold_out})
    assert set(rep.results) == {1, 3}  # flights and activities still succeeded
    assert rep.failed == {2: "RuntimeError: no rooms"}
    assert 4 in rep.skipped and "[2]" in rep.skipped[4] and not rep.ok


def test_a_result_that_fails_its_postcondition_counts_as_a_failure():
    def bad_price(task, inputs):
        return StepResult({"flight": "AF1", "price": 0}, 0.0)

    rep = run(workers={**DEMO_WORKERS, AgentName.flights: bad_price})
    assert "postcondition failed" in rep.failed[1] and 4 in rep.skipped


def test_missing_worker_is_a_failure_not_a_crash():
    workers = {k: v for k, v in DEMO_WORKERS.items() if k is not AgentName.activities}
    assert "no worker registered" in run(workers=workers).failed[3]


def test_spend_limit_halts_before_the_next_layer():
    rep = run(budget_usd=500.0)
    assert rep.halted_reason and "over budget" in rep.halted_reason
    assert 4 not in rep.results and not rep.ok
    assert rep.total_cost_usd > 500.0  # honest: the first layer already spent its money


# --- approval bound to the exact plan -------------------------------------------------------------
def test_a_plan_with_reservations_will_not_run_without_approval():
    with pytest.raises(ApprovalError):
        run(approved=False)


def test_a_plan_changed_after_approval_is_refused():
    digest = plan_hash(FULL)
    tampered = build([FLIGHTS, HOTEL, ACTS, FINAL])
    tampered.tasks[0].params["destination"] = "Rio"
    with pytest.raises(ApprovalError):
        PlanExecutor(DEMO_WORKERS).run(tampered, approved_hash=digest)


def test_a_plan_with_no_side_effects_needs_no_approval():
    only_summary = build([{"task_id": 1, "agent": "itinerary", "description": "x"}])
    only_summary.tasks[0].params = {}
    rep = PlanExecutor({AgentName.itinerary: lambda t, i: StepResult({"summary": "ok"})}).run(only_summary)
    assert rep.ok


# --- resuming after a partial failure ---------------------------------------------------------------
def test_prior_results_are_kept_and_not_run_again():
    first = run(workers={**DEMO_WORKERS, AgentName.hotels: lambda t, i: (_ for _ in ()).throw(RuntimeError("x"))})
    calls = []

    def counting(task, inputs):
        calls.append(task.task_id)
        return DEMO_WORKERS[task.agent](task, inputs)

    rep = PlanExecutor({a: counting for a in AgentName}).run(FULL, approved_hash=plan_hash(FULL), prior=first.results)
    assert rep.ok and sorted(calls) == [2, 4]  # only the failed task and its dependent ran
    assert rep.total_cost_usd == 480.0 + 980.0 + 120.0  # prior spend is counted exactly once
