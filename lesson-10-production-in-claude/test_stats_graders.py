import pytest
from pydantic import BaseModel

from fakes import rec
from graders import (custom, forbidden_tools, grade_all, judge_agreement, max_cost, max_steps, pairwise_judge,
                     position_bias_rate, required_tools, schema_valid)
from stats import cohens_kappa, normal_half_width, paired_bootstrap, wilson_interval


# --- the arithmetic behind "86% vs 80% on 50 cases is noise" -------------------------------------------
def test_with_50_cases_an_80_percent_rate_is_only_known_to_about_11_points():
    assert normal_half_width(0.8, 50) == pytest.approx(0.1109, abs=0.001)
    low, high = wilson_interval(40, 50)
    assert low == pytest.approx(0.6696, abs=0.002) and high == pytest.approx(0.8876, abs=0.002)
    assert low < 0.80 < 0.86 < high  # 80% and 86% are both comfortably inside the same interval


def test_more_cases_narrow_the_interval():
    narrow = wilson_interval(400, 500)
    wide = wilson_interval(40, 50)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0]) / 2


def test_wilson_stays_inside_zero_and_one_and_handles_no_data():
    assert wilson_interval(0, 50)[0] == 0.0 and wilson_interval(50, 50)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(50, 50)[0] > 0.9  # unlike the normal approximation, it does not claim certainty at 100%


def test_paired_bootstrap_identical_versions_show_no_difference():
    r = paired_bootstrap([1.0, 0.0, 1.0] * 10, [1.0, 0.0, 1.0] * 10)
    assert r["mean_diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0


def test_paired_bootstrap_detects_a_clear_improvement_and_is_reproducible():
    a, b = [0.0] * 30, [1.0] * 20 + [0.0] * 10
    r1, r2 = paired_bootstrap(a, b, seed=7), paired_bootstrap(a, b, seed=7)
    assert r1 == r2 and r1["mean_diff"] == pytest.approx(2 / 3) and r1["ci_low"] > 0 and r1["p_b_better"] == 1.0


def test_paired_bootstrap_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def test_kappa_measures_agreement_beyond_chance():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)  # 75% raw agreement, but only 0.5 beyond chance
    assert cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == 1.0
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)  # systematic disagreement
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 0])


def test_a_judge_that_agrees_75_percent_of_the_time_may_still_be_only_moderately_useful():
    a = judge_agreement(["good", "good", "bad", "bad"], ["good", "bad", "bad", "bad"])
    assert a["accuracy"] == 0.75 and a["kappa"] == pytest.approx(0.5)


# --- code graders ---------------------------------------------------------------------------------------------
class Out(BaseModel):
    answer: str
    total: float


def test_the_lessons_did_it_cover_both_flights_and_activities_is_a_code_check():
    covered = rec()
    covered.tools_used = ["get_activity_suggestions", "get_flight_info"]
    missing = rec()
    missing.tools_used = ["get_flight_info"]
    grader = required_tools("get_flight_info", "get_activity_suggestions")
    assert grader(covered, None).passed
    r = grader(missing, None)
    assert not r.passed and "get_activity_suggestions" in r.detail


def test_forbidden_tools_step_and_cost_limits():
    r = rec(cost=0.05, steps=9)
    r.tools_used = ["search", "delete_account"]
    assert not forbidden_tools("delete_account")(r, None).passed
    assert forbidden_tools("send_money")(r, None).passed
    assert not max_steps(8)(r, None).passed and max_steps(9)(r, None).passed
    assert not max_cost(0.04)(r, None).passed and max_cost(0.05)(r, None).passed


def test_schema_and_custom_graders():
    assert schema_valid(Out)(rec(), {"answer": "x", "total": 1.5}).passed
    bad = schema_valid(Out)(rec(), {"answer": "x"})
    assert not bad.passed and "total" in bad.detail
    assert custom("positive_total", lambda r, o: o["total"] > 0)(rec(), {"total": 2}).passed


def test_grade_all_runs_every_grader_and_reports_each_result():
    r = rec(steps=2)
    r.tools_used = ["search"]
    results = grade_all(r, {"answer": "x", "total": 1}, [required_tools("search"), max_steps(1), schema_valid(Out)])
    assert [(x.name, x.passed) for x in results] == [("required_tools", True), ("max_steps", False), ("schema_valid", True)]


# --- the LLM judge and its safeguards -----------------------------------------------------------------------------
def quality_judge(question, first, second):
    return "A" if "correct" in first else ("B" if "correct" in second else "tie")


def test_a_judge_that_follows_quality_gives_the_same_winner_in_both_orders():
    r = pairwise_judge(quality_judge, "q", "correct answer", "wrong answer")
    assert r == {"winner": "1", "consistent": True}
    r2 = pairwise_judge(quality_judge, "q", "wrong answer", "correct answer")
    assert r2 == {"winner": "2", "consistent": True}


def test_a_judge_that_always_prefers_the_first_slot_is_exposed_as_position_biased():
    always_first = lambda q, a, b: "A"
    r = pairwise_judge(always_first, "q", "one", "two")
    assert r == {"winner": "inconsistent", "consistent": False}  # it did not pick a winner, so no fake winner is reported
    assert position_bias_rate([r, r, pairwise_judge(quality_judge, "q", "correct", "wrong")]) == pytest.approx(2 / 3)


def test_ties_are_consistent_ties():
    assert pairwise_judge(lambda q, a, b: "tie", "q", "x", "y") == {"winner": "tie", "consistent": True}
    assert position_bias_rate([]) == 0.0
