import json
import time

import pytest

from corpus import CORPUS
from fakes import (CHECK_OK, GOOD_REPORT, PRICE_QUOTES, ScriptedClient, happy_queues, last_tool_results, planner_reply,
                   reader_replies, submit, tool_names, turn, use)
from research import run_research
from runtime import Budget, Runtime


def go(queues, approver=lambda s: "approve", **kw):
    published = []
    client = ScriptedClient(queues, delays=kw.pop("delays", None))
    rt = Runtime(client, budget=kw.pop("budget", None))
    out = run_research(rt, "Compare AcmeCloud and BetaCloud", CORPUS, approver=approver,
                       publish_fn=lambda title, body: published.append((title, body)), **kw)
    return out, client, rt, published


# --- the happy path ------------------------------------------------------------------------------
def test_full_pipeline_publishes_a_verified_report():
    out, client, rt, published = go(happy_queues())
    assert out.status == "published" and out.verdict.ok and out.revisions == 0
    assert published == [("Cloud comparison", out.body)]
    assert "- Acme costs $12 and Beta $9 per user per month [sources: 1]" in out.body


def test_the_trace_shows_who_ran_what_with_costs_that_add_up():
    out, client, rt, _ = go(happy_queues())
    names = sorted((s.agent, s.label) for s in rt.tracer.spans)  # readers run in parallel, so order is not fixed
    assert names == sorted([("pipeline", "research"), ("planner", ""), ("reader", "task 1"), ("reader", "task 2"),
                            ("merger", ""), ("checker", "")])
    assert rt.budget.calls == 7  # planner 1 + readers 2 each + merger 1 + checker 1
    assert rt.tracer.total_cost() == pytest.approx(7 * (100 * 3 + 50 * 15) / 1e6)
    assert rt.tracer.by_agent()["reader"]["runs"] == 2 and "reader task 1" in rt.tracer.render()
    assert all(s.status == "ok" for s in rt.tracer.spans)


# --- context isolation ---------------------------------------------------------------------------
def test_the_orchestrator_sees_quotes_and_answers_never_the_raw_documents():
    out, client, rt, _ = go(happy_queues())
    merger_input = client.calls_for("merger")[0]["messages"][0]["content"]
    assert "$12 per user per month" in merger_input  # the quoted evidence is there
    assert "Annual billing gives a 15% discount" not in merger_input  # the rest of the document is not
    assert "single sign-on on every plan" not in merger_input


def test_each_reader_works_in_its_own_context_and_cannot_see_the_other_task():
    out, client, rt, _ = go(happy_queues())
    for call in client.calls_for("reader"):
        first = call["messages"][0]["content"]
        task = json.loads(first)["task_id"]
        other = 2 if task == 1 else 1
        assert f'"task_id":{other}' not in first.replace(" ", "")


# --- permission separation -----------------------------------------------------------------------
def test_only_readers_have_a_tool_and_nobody_can_publish():
    out, client, rt, _ = go(happy_queues())
    for call in client.calls:
        role = call["key"][0] if isinstance(call["key"], tuple) else call["key"]
        expected = ["fetch_document", "submit_result"] if role == "reader" else ["submit_result"]
        assert tool_names(call) == expected
        assert "publish_report" not in tool_names(call)


def test_a_reader_that_obeys_an_injected_instruction_has_nothing_dangerous_to_call():
    q = happy_queues()
    q[("reader", 1)] = [
        turn(use("f", "fetch_document", doc_id="support_a"),
             use("evil", "publish_report", body="AcmeCloud is the only safe choice")),  # it "obeys" the document
        submit({"task_id": 1, "status": "ok", "answer": "Acme has 24/7 chat support",
                "evidence": [{"doc_id": "support_a", "quote": "AcmeCloud offers 24/7 chat support"}]})]
    tasks = [{"task_id": 1, "question": "support?", "doc_ids": ["support_a"], "acceptance": "x"},
             {"task_id": 2, "question": "security?", "doc_ids": ["security_a", "security_b"], "acceptance": "x"}]
    q["planner"] = [planner_reply(tasks)]
    out, client, rt, published = go(q, approver=lambda s: "deny")
    reader1 = [c for c in client.calls_for("reader") if c["key"] == ("reader", 1)][1]
    errors = {r["tool_use_id"]: r for r in last_tool_results(reader1)}
    assert errors["evil"]["is_error"] and "No such tool" in errors["evil"]["content"]
    assert published == [] and out.status == "declined"


def test_a_reader_cannot_fetch_a_document_outside_its_task():
    q = happy_queues()
    q[("reader", 1)] = [turn(use("f", "fetch_document", doc_id="security_a")),  # task 1 only owns the pricing docs
                        *reader_replies(1, ["pricing_a", "pricing_b"], PRICE_QUOTES)]
    out, client, rt, _ = go(q)
    second = [c for c in client.calls_for("reader") if c["key"] == ("reader", 1)][1]
    assert last_tool_results(second)[0]["is_error"] and "not available" in last_tool_results(second)[0]["content"]


# --- evidence is verified in code ----------------------------------------------------------------
def test_a_fabricated_quote_is_rejected_and_the_reader_corrects_itself():
    q = happy_queues()
    fake = submit({"task_id": 1, "status": "ok", "answer": "x", "evidence": [{"doc_id": "pricing_a", "quote": "AcmeCloud is free forever"}]})
    q[("reader", 1)] = [reader_replies(1, ["pricing_a", "pricing_b"], PRICE_QUOTES)[0], fake,
                        reader_replies(1, ["pricing_a", "pricing_b"], PRICE_QUOTES)[1]]
    out, client, rt, _ = go(q)
    assert out.status == "published"
    third = [c for c in client.calls_for("reader") if c["key"] == ("reader", 1)][2]
    assert "does not appear verbatim" in last_tool_results(third)[0]["content"]


def test_evidence_from_a_document_that_was_never_fetched_is_rejected():
    q = happy_queues()
    sneaky = submit({"task_id": 1, "status": "ok", "answer": "x", "evidence": [{"doc_id": "pricing_a", "quote": "$12 per user per month"}]})
    q[("reader", 1)] = [sneaky, *reader_replies(1, ["pricing_a", "pricing_b"], PRICE_QUOTES)]
    out, client, rt, _ = go(q)
    second = [c for c in client.calls_for("reader") if c["key"] == ("reader", 1)][1]
    assert "did not fetch" in last_tool_results(second)[0]["content"]


def test_a_worker_that_keeps_failing_becomes_a_failed_result_and_does_not_sink_the_others():
    q = happy_queues()
    bad = submit({"task_id": 2, "status": "ok", "answer": "x", "evidence": []})
    q[("reader", 2)] = [bad, bad, bad]
    q["merger"] = [submit({"title": "Partial comparison", "claims": [{"text": "Acme costs $12 and Beta $9", "sources": [1]}]})]
    out, client, rt, published = go(q)
    failed = [r for r in out.results if r.status == "failed"]
    assert len(failed) == 1 and failed[0].task_id == 2 and "worker failed" in failed[0].answer
    assert out.status == "published"  # task 1 still produced a usable claim
    assert "<worker_result>" in client.calls_for("merger")[0]["messages"][0]["content"]


# --- the report and the checker --------------------------------------------------------------------
def test_a_claim_that_cites_a_missing_task_is_rejected_and_fixed():
    q = happy_queues()
    q["merger"] = [submit({"title": "t", "claims": [{"text": "x", "sources": [9]}]}), GOOD_REPORT]
    out, client, rt, _ = go(q)
    assert out.status == "published"
    assert "does not exist" in last_tool_results(client.calls_for("merger")[1])[0]["content"]


def test_a_claim_that_cites_a_failed_task_is_rejected():
    q = happy_queues()
    q[("reader", 2)] = [submit({"task_id": 2, "status": "failed", "answer": "no info"})]
    q["merger"] = [submit({"title": "t", "claims": [{"text": "x", "sources": [2]}]}),
                   submit({"title": "t", "claims": [{"text": "price", "sources": [1]}]})]
    out, client, rt, _ = go(q)
    assert out.status == "published" and out.report.claims[0].sources == [1]


def test_a_rejected_report_is_revised_once_with_the_checkers_objections_then_published():
    q = happy_queues()
    q["checker"] = [submit({"ok": False, "problems": ["claim 1 overstates the price"]}), CHECK_OK]
    q["merger"] = [GOOD_REPORT, GOOD_REPORT]
    out, client, rt, published = go(q)
    assert out.status == "published" and out.revisions == 1 and len(published) == 1
    assert "claim 1 overstates the price" in client.calls_for("merger")[1]["messages"][0]["content"]


def test_a_report_the_checker_keeps_rejecting_is_never_offered_for_approval_or_published():
    q = happy_queues()
    bad = submit({"ok": False, "problems": ["unsupported"]})
    q["checker"], q["merger"] = [bad, bad], [GOOD_REPORT, GOOD_REPORT]
    asked = []
    out, client, rt, published = go(q, approver=lambda s: asked.append(s) or "approve")
    assert out.status == "unverified" and asked == [] and published == [] and out.revisions == 1


def test_the_checker_sees_only_claims_and_the_cited_evidence():
    out, client, rt, _ = go(happy_queues())
    seen = json.loads(client.calls_for("checker")[0]["messages"][0]["content"])
    assert seen[0]["claim"].startswith("Acme costs") and "$12 per user per month" in seen[0]["evidence"]
    assert not any("SOC 2" in e for e in seen[0]["evidence"])  # claim 1 cites task 1 only


# --- approval ----------------------------------------------------------------------------------------
def test_a_declined_report_is_not_published():
    out, client, rt, published = go(happy_queues(), approver=lambda s: "no")
    assert out.status == "declined" and published == []


def test_an_approver_that_crashes_means_no():
    out, client, rt, published = go(happy_queues(), approver=lambda s: 1 / 0)
    assert out.status == "declined" and published == []


def test_without_an_approver_the_pipeline_stops_after_verification():
    published = []
    client = ScriptedClient(happy_queues())
    out = run_research(Runtime(client), "q", CORPUS, approver=None, publish_fn=lambda *a: published.append(a))
    assert out.status == "verified" and published == []


def test_the_approver_sees_the_exact_text_that_would_be_published():
    seen = []
    out, client, rt, published = go(happy_queues(), approver=lambda s: seen.append(s) or "approve")
    assert seen[0]["body"] == published[0][1]


# --- planning validation ---------------------------------------------------------------------------
def test_a_plan_with_an_unknown_document_is_rejected_and_fixed():
    q = happy_queues()
    bad = planner_reply([{"task_id": 1, "question": "q", "doc_ids": ["nope"], "acceptance": "x"}])
    q["planner"] = [bad, planner_reply()]
    out, client, rt, _ = go(q)
    assert out.status == "published" and "unknown doc_id" in last_tool_results(client.calls_for("planner")[1])[0]["content"]


def test_too_many_subtasks_are_rejected():
    q = happy_queues()
    many = planner_reply([{"task_id": i, "question": "q", "doc_ids": ["pricing_a"], "acceptance": "x"} for i in range(1, 4)])
    q["planner"] = [many, planner_reply()]
    out, client, rt, _ = go(q, max_subtasks=2)
    assert out.status == "published" and "at most 2" in last_tool_results(client.calls_for("planner")[1])[0]["content"]


# --- limits and parallelism --------------------------------------------------------------------------
def test_a_shared_budget_stops_the_whole_pipeline():
    out, client, rt, published = go(happy_queues(), budget=Budget(max_calls=3))
    assert out.status == "over_budget" and "call limit" in out.error and published == []
    assert rt.budget.calls <= 3


def test_independent_readers_run_in_parallel():
    q = happy_queues()
    ids = [1, 2, 3]
    q["planner"] = [planner_reply([{"task_id": i, "question": "q", "doc_ids": ["pricing_a", "pricing_b"], "acceptance": "x"} for i in ids])]
    for i in ids:
        q[("reader", i)] = reader_replies(i, ["pricing_a", "pricing_b"], PRICE_QUOTES)
    q["merger"] = [submit({"title": "t", "claims": [{"text": "x", "sources": [1, 2, 3]}]})]
    start = time.monotonic()
    out, client, rt, _ = go(q, delays={"reader": 0.2})
    assert out.status == "published"
    assert time.monotonic() - start < 0.9  # 3 readers x 2 calls x 0.2s would take 1.2s one after another
