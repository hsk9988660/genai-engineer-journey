"""The harness arithmetic and the STRUCTURE of each design. These tests use scripted token counts, so they say nothing
about how much a real multi-agent run costs relative to a single agent. That needs `python compare.py` with a key."""

import pytest

from compare import CASES, Case, compare, min_agent_runs, score, single_agent
from corpus import CORPUS
from fakes import ScriptedClient, happy_queues, submit, turn, use


def make_client():
    q = happy_queues()
    q["single"] = [turn(use("f1", "fetch_document", doc_id="pricing_a")),
                   submit({"answer": "Acme is $12 and Beta is $9. SOC 2 Type II vs Type I."})]
    return ScriptedClient(q)


CASE = Case("Compare them", ["$12", "$9", "SOC 2 Type II", "Type I", "24/7", "1 business day"])


def test_score_is_the_fraction_of_expected_facts_present():
    assert score("Acme $12, Beta $9", CASE) == pytest.approx(2 / 6)
    assert score("$12 $9 soc 2 type ii type i 24/7 1 business day", CASE) == 1.0
    assert score("", CASE) == 0.0


def test_the_harness_reports_calls_cost_and_score_per_design():
    r = compare(make_client, [CASE], CORPUS)
    assert r["single"]["api_calls"] == 2 and r["multi"]["api_calls"] == 7
    per_call = (100 * 3 + 50 * 15) / 1e6  # scripted: every call uses 100 in / 50 out
    assert r["single"]["total_cost_usd"] == pytest.approx(2 * per_call)
    assert r["multi"]["total_cost_usd"] == pytest.approx(7 * per_call)
    assert r["single"]["mean_score"] == pytest.approx(4 / 6)
    assert r["single"]["score_per_dollar"] == pytest.approx(r["single"]["mean_score"] / r["single"]["total_cost_usd"])


def test_a_crashing_design_is_recorded_not_raised():
    def broken(rt, case, corpus):
        raise RuntimeError("model down")

    r = compare(make_client, [CASE], CORPUS, designs={"broken": broken})
    assert r["broken"]["mean_score"] == 0.0 and "model down" in r["broken"]["errors"][0]


def test_the_multi_agent_design_uses_exactly_the_minimum_agent_runs_when_nothing_goes_wrong():
    from research import run_research
    from runtime import Runtime
    rt = Runtime(ScriptedClient(happy_queues()))
    run_research(rt, "q", CORPUS, approver=None)
    runs = [s for s in rt.tracer.spans if s.agent != "pipeline"]
    assert len(runs) == min_agent_runs(2) == 5


def test_each_run_gets_its_own_budget_so_one_case_cannot_starve_the_next():
    r = compare(make_client, [CASE, CASE], CORPUS, designs={"single": single_agent},
                budget_factory=lambda: __import__("runtime").Budget(max_calls=2))
    assert r["single"]["errors"] == [] and r["single"]["api_calls"] == 4


def test_the_default_case_set_is_well_formed():
    assert CASES and all(c.expected for c in CASES)
