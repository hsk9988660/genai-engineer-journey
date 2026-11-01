import pytest
from pydantic import ValidationError

from evals import FACTS, LINT_CASES, build, lint_report
from plan import (AgentName, Fact, Plan, Requirements, Task, facts_of, layers, missing_questions, plan_hash,
                  validate_plan, verify_requirements)

REQUEST = "I want to go to Paris from May 2 to May 9, budget 3000 dollars"


def req(**kw):
    base = dict(destination=Fact(value="Paris", quote="Paris"),
                depart=Fact(value="2026-05-02", quote="May 2"),
                return_date=Fact(value="2026-05-09", quote="May 9"),
                budget_usd=Fact(value="3000", quote="3000 dollars"))
    base.update(kw)
    return Requirements(**base)


# --- the plan validator, case by case (the same cases `python evals.py lint` prints) ---------------
@pytest.mark.parametrize("name,tasks,expected", LINT_CASES, ids=[c[0] for c in LINT_CASES])
def test_validator_handles_each_lint_case(name, tasks, expected):
    problems = validate_plan(build(tasks), FACTS)
    if expected is None:
        assert problems == []
    else:
        assert any(expected in p for p in problems), problems


def test_lint_report_is_all_green():
    assert all(r["ok"] for r in lint_report())


def test_an_unknown_agent_is_rejected_by_the_schema_not_by_free_text():
    with pytest.raises(ValidationError):
        Task.model_validate({"task_id": 1, "agent": "transport_agent", "description": "x"})
    assert {a.value for a in AgentName} == {"flights", "hotels", "activities", "itinerary"}


def test_task_ids_must_be_positive():
    with pytest.raises(ValidationError):
        Task.model_validate({"task_id": 0, "agent": "flights", "description": "x"})


# --- layering -----------------------------------------------------------------------------------
def test_independent_tasks_share_a_layer_and_dependents_come_later():
    from evals import ACTS, FINAL, FLIGHTS, HOTEL
    got = [[t.task_id for t in layer] for layer in layers(build([FINAL, ACTS, HOTEL, FLIGHTS]))]
    assert got == [[1, 2, 3], [4]]


def test_a_chain_has_one_task_per_layer():
    chain = build([{"task_id": 1, "agent": "activities", "description": "a", "params": {"city": "Paris"}},
                   {"task_id": 2, "agent": "itinerary", "description": "b", "depends_on": [1]}])
    assert [[t.task_id for t in layer] for layer in layers(chain)] == [[1], [2]]


def test_layers_refuses_a_cyclic_plan():
    cyc = build([{"task_id": 1, "agent": "itinerary", "description": "a", "depends_on": [2]},
                 {"task_id": 2, "agent": "itinerary", "description": "b", "depends_on": [1]}])
    with pytest.raises(ValueError):
        layers(cyc)


def test_plan_hash_changes_with_content_not_with_task_order_of_keys():
    from evals import FLIGHTS, FINAL
    a, b = build([FLIGHTS, FINAL]), build([FLIGHTS, FINAL])
    assert plan_hash(a) == plan_hash(b)
    b.tasks[0].params["depart"] = "2026-05-03"
    assert plan_hash(a) != plan_hash(b)


# --- grounding facts in what the user said --------------------------------------------------------
def test_grounded_requirements_pass():
    assert verify_requirements(req(), REQUEST) == []
    assert facts_of(req()) == {"destination": "Paris", "depart": "2026-05-02", "return_date": "2026-05-09", "budget_usd": "3000"}


def test_a_fact_whose_quote_is_not_in_the_request_is_rejected():
    problems = verify_requirements(req(depart=Fact(value="2026-05-03", quote="May 3")), REQUEST)
    assert any("depart" in p and "does not appear" in p for p in problems)


def test_quote_matching_ignores_case_and_spacing():
    r = req(destination=Fact(value="Paris", quote="  PARIS "))
    assert verify_requirements(r, REQUEST) == []


def test_malformed_values_are_rejected():
    assert any("YYYY-MM-DD" in p for p in verify_requirements(req(depart=Fact(value="May 2", quote="May 2")), REQUEST))
    assert any("plain number" in p for p in verify_requirements(req(budget_usd=Fact(value="$3k", quote="3000 dollars")), REQUEST))


def test_return_must_be_after_departure():
    swapped = req(depart=Fact(value="2026-05-09", quote="May 9"), return_date=Fact(value="2026-05-02", quote="May 2"))
    assert any("after depart" in p for p in verify_requirements(swapped, REQUEST))


def test_missing_required_facts_become_questions_but_budget_is_optional():
    assert missing_questions(req()) == []
    assert missing_questions(req(budget_usd=None)) == []
    qs = missing_questions(Requirements())
    assert len(qs) == 3 and any("travel" in q for q in qs)
    assert missing_questions(req(depart=None)) == ["What date do you want to leave?"]
