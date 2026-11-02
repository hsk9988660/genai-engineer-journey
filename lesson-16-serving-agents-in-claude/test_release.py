import pytest

from release import GateResult, Rollout, bucket, evaluate, gate, naive_gate_error_rates, wilson_interval


# --- the interval ------------------------------------------------------------------------------------------------------------------------
def test_with_no_data_the_interval_is_everything():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_the_interval_matches_known_values_and_stays_inside_zero_and_one():
    lo, hi = wilson_interval(50, 100)
    assert round(lo, 3) == 0.404 and round(hi, 3) == 0.596
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    for passes in range(0, 21):
        lo, hi = wilson_interval(passes, 20)
        assert 0 <= lo <= passes / 20 <= hi <= 1


def test_more_data_narrows_the_interval():
    a, b = wilson_interval(9, 10), wilson_interval(900, 1000)
    assert (b[1] - b[0]) < (a[1] - a[0])


# --- the gate ----------------------------------------------------------------------------------------------------------------------------
def test_too_few_cases_is_inconclusive_even_if_every_case_passed():
    r = gate(4, 4)
    assert r.decision == "inconclusive" and not r.ship and "only 4 cases" in r.reason


def test_enough_good_results_pass_and_ship():
    r = gate(96, 100)
    assert r.decision == "pass" and r.ship and r.lower >= 0.8


def test_enough_bad_results_fail():
    r = gate(60, 100)
    assert r.decision == "fail" and not r.ship and r.upper < 0.8


def test_a_result_near_the_requirement_is_inconclusive_and_does_not_ship():
    r = gate(85, 100)
    assert r.decision == "inconclusive" and not r.ship and "more cases" in r.reason


def test_the_requirement_and_minimum_are_configurable():
    assert gate(48, 50, required=0.9, min_cases=50).decision == "inconclusive"
    assert gate(95, 100, required=0.5, min_cases=10).decision == "pass"


def test_impossible_counts_are_rejected():
    for passes, n in ((-1, 10), (11, 10), (5, 0)):
        with pytest.raises(ValueError, match="between 0 and n"):
            gate(passes, n)


def test_evaluate_runs_every_case_through_the_function_it_is_given():
    seen = []
    r = evaluate(lambda case: seen.append(case) or case % 10 != 0, list(range(100)), min_cases=30)
    assert len(seen) == 100 and r.passes == 90 and isinstance(r, GateResult)


def test_the_lessons_four_case_rule_blocks_good_agents_and_ships_bad_ones_at_these_exact_rates():
    r = naive_gate_error_rates(0.9, 0.6, n=4, need=0.8)
    assert round(r["good_agent_blocked"], 4) == 0.3439 and round(r["bad_agent_shipped"], 4) == 0.1296


def test_with_more_cases_the_naive_rule_gets_better_and_with_fewer_it_is_a_coin_flip():
    big = naive_gate_error_rates(0.9, 0.6, n=100, need=0.8)
    assert big["good_agent_blocked"] < 0.01 and big["bad_agent_shipped"] < 0.001
    assert naive_gate_error_rates(0.5, 0.5, n=1, need=0.8)["bad_agent_shipped"] == 0.5


# --- buckets ----------------------------------------------------------------------------------------------------------------------------
def test_a_user_always_lands_in_the_same_bucket_for_the_same_salt():
    assert bucket("alice", "s") == bucket("alice", "s") and 0 <= bucket("alice", "s") < 100


def test_a_different_salt_reshuffles_users_so_the_same_people_are_not_always_first():
    assert [bucket(f"u{i}", "a") for i in range(50)] != [bucket(f"u{i}", "b") for i in range(50)]


def test_buckets_are_spread_evenly():
    counts = [0] * 10
    for i in range(10_000):
        counts[bucket(f"user-{i}", "s") // 10] += 1
    assert all(850 < c < 1150 for c in counts)


# --- the rollout --------------------------------------------------------------------------------------------------------------------------
def feed(r, version, n, bad):
    for i in range(n):
        r.observe(version, i >= bad)


def test_at_the_first_stage_only_about_one_percent_of_users_get_the_candidate():
    r = Rollout("s", "c", salt="t")
    n = sum(r.version_for(f"user-{i}") == "c" for i in range(10_000))
    assert r.percent == 1 and 50 < n < 150


def test_a_user_who_is_in_the_candidate_slice_stays_in_it_as_the_slice_grows():
    r = Rollout("s", "c", salt="t", min_samples=1)
    early = {f"u{i}" for i in range(2000) if r.version_for(f"u{i}") == "c"}
    feed(r, "c", 5, 0)
    assert r.evaluate() == "advanced" and r.percent == 10
    assert all(r.version_for(u) == "c" for u in early)


def test_nothing_happens_until_the_candidate_has_enough_samples():
    r = Rollout("s", "c", min_samples=20)
    feed(r, "c", 19, 19)
    assert r.evaluate() == "waiting" and r.state == "running"


def test_a_candidate_that_is_clearly_worse_is_rolled_back_and_everyone_returns_to_stable():
    r = Rollout("s", "c", min_samples=20)
    feed(r, "s", 100, 2)
    feed(r, "c", 20, 10)  # 50% errors against about 2%
    assert r.evaluate() == "rolled_back" and r.state == "rolled_back" and r.percent == 0
    assert all(r.version_for(f"u{i}") == "s" for i in range(500)) and r.history[-1][0] == "rolled_back"


def test_one_unlucky_error_in_a_small_sample_is_not_enough_to_roll_back():
    r = Rollout("s", "c", min_samples=8)
    feed(r, "s", 100, 2)
    feed(r, "c", 8, 1)  # 12.5% observed, but eight samples cannot tell that from 2%
    assert r.evaluate() == "advanced" and r.state == "running"


def test_each_stage_must_prove_itself_with_fresh_traffic():
    r = Rollout("s", "c", min_samples=10)
    feed(r, "c", 10, 0)
    assert r.evaluate() == "advanced" and r.error_rate("c") is None
    assert r.evaluate() == "waiting"


def test_a_healthy_candidate_is_promoted_after_the_last_stage_and_then_everyone_gets_it():
    r = Rollout("s", "c", min_samples=5)
    results = []
    for _ in range(4):
        feed(r, "c", 5, 0)
        results.append(r.evaluate())
    assert results == ["advanced", "advanced", "advanced", "promoted"] and r.state == "promoted" and r.percent == 100
    assert all(r.version_for(f"u{i}") == "c" for i in range(500))


def test_a_finished_rollout_ignores_further_evaluation():
    r = Rollout("s", "c", min_samples=1)
    feed(r, "c", 5, 5)
    r.evaluate()
    assert r.evaluate() == "done"


def test_unknown_versions_are_ignored_and_stages_can_be_customised():
    r = Rollout("s", "c", min_samples=1, stages=(50, 100))
    r.observe("other", False)
    assert r.error_rate("s") is None and r.percent == 50
    feed(r, "c", 2, 0)
    assert r.evaluate() == "advanced" and r.percent == 100
