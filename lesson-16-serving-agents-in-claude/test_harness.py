import pytest

import harness


@pytest.fixture(scope="module")
def by():
    return {(o.scenario, o.strategy): o.metrics for o in harness.evaluate()}


def test_a_burst_without_admission_control_overruns_the_provider_limit_and_most_requests_fail(by):
    m = by[("overload", "unbounded")]
    assert m["peak_in_provider"] > 5 and m["provider_429s"] > 20 and m["failed_503"] > 20 and m["ok"] < 20


def test_with_admission_control_every_request_succeeds_and_the_provider_is_never_over_its_limit(by):
    m = by[("overload", "bounded_with_queue")]
    assert m["ok"] == 40 and m["provider_429s"] == 0 and m["peak_in_provider"] <= 5 and m["failed_503"] == 0


def test_with_a_small_queue_the_excess_is_refused_politely_rather_than_sent_to_the_provider(by):
    m = by[("overload", "bounded_small_queue")]
    assert m["overloaded_429"] > 0 and m["ok"] + m["overloaded_429"] == 40 and m["provider_429s"] == 0 and m["peak_in_provider"] <= 5


def test_retrying_five_times_into_an_outage_hits_the_provider_100_times_and_the_breaker_caps_it_at_5(by):
    assert by[("outage", "retry_5x_no_breaker")]["provider_calls_during_outage"] == 100
    m = by[("outage", "retry_3x_with_breaker")]
    assert m["provider_calls_during_outage"] == 5 and m["all_failed"] is True


def test_both_recover_on_the_first_request_after_the_circuit_pause(by):
    assert by[("outage", "retry_5x_no_breaker")]["first_request_after_reset"] == "ok" and by[("outage", "retry_3x_with_breaker")]["first_request_after_reset"] == "ok"


def test_the_lessons_cache_shows_one_customers_order_to_another_and_the_served_version_does_not(by):
    n, s = by[("cache_leak", "lesson_design")], by[("cache_leak", "served")]
    assert n["other_customer_saw_the_order"] and n["served_from_cache"]
    assert s["owner_saw_the_order"] and not s["other_customer_saw_the_order"] and not s["other_customer_cached"]


def test_a_public_question_asked_five_times_costs_one_model_call_when_served(by):
    s = by[("cache_leak", "served")]
    assert s["public_question_model_calls"] == 1 and s["public_question_cache_hits"] == 4


def test_a_bad_release_hurts_many_users_when_shipped_to_everyone_and_few_under_a_canary(by):
    everyone, canary = by[("release", "ship_to_everyone")], by[("release", "canary_bad_candidate")]
    assert everyone["error_responses"] > 250 and canary["error_responses"] < 60 and canary["final_state"] == "rolled_back"
    assert canary["served_candidate"] < 30


def test_a_good_candidate_is_promoted_through_all_stages(by):
    m = by[("release", "canary_good_candidate")]
    assert m["final_state"] == "promoted"


def test_the_lessons_four_case_gate_blocks_good_agents_a_third_of_the_time_and_ships_bad_ones_13_percent(by):
    m = by[("gate", "lesson_rule_4_cases")]
    assert m["good_agent_(90%)_blocked"] == pytest.approx(0.344, abs=0.001) and m["bad_agent_(60%)_shipped"] == pytest.approx(0.13, abs=0.001)


def test_the_interval_gate_almost_never_ships_a_bad_agent_and_is_honest_when_it_cannot_tell(by):
    bad, good, tiny = by[("gate", "interval_gate_100_cases_bad_agent")], by[("gate", "interval_gate_100_cases_good_agent")], by[("gate", "interval_gate_4_cases_good_agent")]
    assert bad["pass"] == 0.0 and bad["fail"] > 0.95
    assert good["pass"] > 0.6 and good["fail"] == 0.0 and good["inconclusive"] > 0.05  # even 100 cases cannot always confirm a 90% agent against an 80% bar
    assert tiny["pass"] == 0.0 and tiny["inconclusive"] == 1.0


def test_ten_mixed_queries_split_between_the_tiers_and_the_cache(by):
    m = by[("cost", "ten_mixed_queries")]
    assert (m["small"], m["large"], m["from_cache"], m["escalations"]) == (6, 3, 4, 3) and m["total_cost"] != "0"


def test_the_table_prints_every_row(capsys):
    outcomes = harness.evaluate()
    harness.print_table(outcomes)
    assert len(capsys.readouterr().out.strip().splitlines()) == len(outcomes)
