"""The full pipeline with a scripted model: every model reply is written out by hand."""

import json

import pytest

from fakes import FakeClient, submit
from orchestrator import plan_and_execute
from workers import DEMO_WORKERS

REQUEST = "I want to go to Paris from May 2 to May 9, budget 3000 dollars"
TODAY = "2026-01-15"


def facts(dest="Paris", depart=("2026-05-02", "May 2"), ret=("2026-05-09", "May 9"), budget=("3000", "3000 dollars")):
    f = lambda t: None if t is None else {"value": t[0], "quote": t[1]}
    return {"destination": {"value": dest, "quote": dest}, "depart": f(depart), "return_date": f(ret), "budget_usd": f(budget)}


def task(i, agent, deps=(), **params):
    return {"task_id": i, "agent": agent, "description": f"{agent} step", "depends_on": list(deps), "params": params}


def good_plan(city="Paris", depart="2026-05-02", ret="2026-05-09", tier=None):
    hotel = dict(city=city, check_in=depart, check_out=ret, **({"tier": tier} if tier else {}))
    return {"goal": "trip", "tasks": [
        task(1, "flights", destination=city, depart=depart, return_date=ret),
        task(2, "hotels", **hotel), task(3, "activities", city=city), task(4, "itinerary", [1, 2, 3])]}


def run(replies, approver=lambda s: "approve", **kw):
    client = FakeClient([submit(r, f"s{i}") for i, r in enumerate(replies)])
    return plan_and_execute(client, kw.pop("request", REQUEST), approver, DEMO_WORKERS, today=TODAY, **kw), client


# --- the happy path ---------------------------------------------------------------------------------
def test_request_to_executed_plan():
    out, client = run([facts(), good_plan()], budget_usd=5000)
    assert out.status == "completed" and out.replans == 0
    assert "AF1081" in out.report.results[4].output["summary"]
    assert out.usage.calls == 2 and out.usage.input_tokens == 200 and out.usage.validation_failures == 0


def test_the_planner_is_given_only_verified_facts():
    out, client = run([facts(), good_plan()])
    planner_input = client.calls[1]["messages"][0]["content"]
    assert "2026-05-02" in planner_input and "Known facts" in planner_input


# --- missing information is asked for, never invented -----------------------------------------------------
def test_missing_dates_produce_questions_and_no_plan_no_approval_no_execution():
    asked = []
    out, client = run([facts(depart=None, ret=None, budget=None)], approver=lambda s: asked.append(s) or "approve",
                      request="Plan a week in Paris.")
    assert out.status == "needs_info" and len(out.questions) == 2
    assert out.plan is None and out.report is None and asked == [] and len(client.calls) == 1


def test_an_invented_date_in_extraction_is_rejected_and_corrected():
    invented = facts(depart=("2026-05-03", "May 3"))  # 'May 3' is nowhere in the request
    out, client = run([invented, facts(), good_plan()])
    assert out.status == "completed" and out.usage.validation_failures == 1
    assert "does not appear" in client.calls[1]["messages"][-1]["content"][0]["content"]


def test_a_model_that_keeps_inventing_facts_ends_as_invalid_plan():
    out, _ = run([facts(depart=("2026-05-03", "May 3"))] * 3)
    assert out.status == "invalid_plan" and "does not appear" in out.error


# --- plan validation feeds back to the planner -------------------------------------------------------------
def test_a_plan_that_changes_the_users_dates_is_rejected_and_fixed():
    out, client = run([facts(), good_plan(depart="2026-05-03"), good_plan()])
    assert out.status == "completed" and out.usage.validation_failures == 1
    feedback = json.loads(client.calls[2]["messages"][-1]["content"][0]["content"])
    assert any("must equal" in p for p in feedback["problems"])


def test_a_cyclic_plan_is_rejected_and_fixed():
    cyclic = good_plan()
    cyclic["tasks"][0]["depends_on"] = [4]
    out, client = run([facts(), cyclic, good_plan()])
    assert out.status == "completed"
    assert "cycle" in client.calls[2]["messages"][-1]["content"][0]["content"].lower()


def test_a_plan_with_an_unknown_agent_is_rejected_by_the_schema():
    bad = good_plan()
    bad["tasks"][0]["agent"] = "teleporter"
    out, client = run([facts(), bad, good_plan()])
    assert out.status == "completed" and "Schema errors" in client.calls[2]["messages"][-1]["content"][0]["content"]


# --- approval --------------------------------------------------------------------------------------------
def test_a_declined_plan_runs_nothing():
    ran = []
    workers = {a: (lambda t, i, w=w: ran.append(t.task_id) or w(t, i)) for a, w in DEMO_WORKERS.items()}
    client = FakeClient([submit(facts()), submit(good_plan())])
    out = plan_and_execute(client, REQUEST, lambda s: "no thanks", workers, today=TODAY)
    assert out.status == "declined" and ran == []


def test_the_approver_sees_the_whole_plan_what_will_be_reserved_and_its_hash():
    seen = []
    out, _ = run([facts(), good_plan()], approver=lambda s: seen.append(s) or "approve")
    s = seen[0]
    assert [t["agent"] for t in s["tasks"]] == ["flights", "hotels", "activities", "itinerary"]
    assert s["will_reserve_or_buy"] == ["activities", "flights", "hotels"] and len(s["plan_hash"]) == 64


def test_an_approver_that_crashes_means_no():
    out, _ = run([facts(), good_plan()], approver=lambda s: 1 / 0)
    assert out.status == "declined"


# --- budget ----------------------------------------------------------------------------------------------
def test_the_users_stated_budget_is_enforced():
    out, _ = run([facts(budget=("100", "3000 dollars")), good_plan()])
    assert out.status == "over_budget" and "over budget" in out.error


# --- replanning ------------------------------------------------------------------------------------------
LISBON_REQ = "I want to go to Lisbon from June 1 to June 8"


def lisbon_facts():
    return facts(dest="Lisbon", depart=("2026-06-01", "June 1"), ret=("2026-06-08", "June 8"), budget=None)


def replacement(tier="budget", new_ids=(5, 6)):
    return {"explanation": "standard rooms are sold out, use the budget tier", "tasks": [
        task(new_ids[0], "hotels", city="Lisbon", check_in="2026-06-01", check_out="2026-06-08", tier=tier),
        task(new_ids[1], "itinerary", [1, 3, new_ids[0]])]}


def test_a_sold_out_hotel_triggers_a_replan_that_keeps_finished_work():
    asked = []
    out, client = run([lisbon_facts(), good_plan("Lisbon", "2026-06-01", "2026-06-08"), replacement()],
                      approver=lambda s: asked.append(s["plan_hash"]) or "approve", request=LISBON_REQ)
    assert out.status == "completed" and out.replans == 1
    assert sorted(out.report.results) == [1, 3, 5, 6]  # flights and activities were kept, not re-run
    assert "(budget)" in out.report.results[6].output["summary"]
    assert len(asked) == 2 and asked[0] != asked[1]  # the changed plan was approved again
    sent = json.loads(client.calls[2]["messages"][0]["content"])  # what the replanner was told
    assert sorted(sent["failures"]) == ["2"] and "sold out" not in json.dumps(sent["kept_tasks"]) and "no standard rooms" in sent["failures"]["2"]["error"]


def test_replacement_tasks_may_not_reuse_existing_ids():
    out, client = run([lisbon_facts(), good_plan("Lisbon", "2026-06-01", "2026-06-08"),
                       replacement(new_ids=(2, 6)), replacement()], request=LISBON_REQ)
    assert out.status == "completed" and out.usage.validation_failures == 1
    assert "not new" in client.calls[3]["messages"][-1]["content"][0]["content"]


def test_replanning_is_capped():
    out, _ = run([lisbon_facts(), good_plan("Lisbon", "2026-06-01", "2026-06-08"),
                  replacement(tier="standard")], request=LISBON_REQ, max_replans=1)
    assert out.status == "failed" and out.replans == 1 and "no standard rooms" in out.error


def test_a_declined_replan_stops_without_executing_it():
    answers = iter(["approve", "no"])
    out, _ = run([lisbon_facts(), good_plan("Lisbon", "2026-06-01", "2026-06-08"), replacement()],
                 approver=lambda s: next(answers), request=LISBON_REQ)
    assert out.status == "declined" and out.replans == 1 and 5 not in out.report.results


def test_the_hotel_is_reserved_in_the_original_plan_only_with_the_users_dates():
    out, _ = run([facts(), good_plan()])
    assert out.report.results[2].output["nights"] == 7
