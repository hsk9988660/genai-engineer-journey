import pytest

from evalset import Dataset, EvalCase, FailedRun
from regression import Gate, compare, consistency_pass_all_k, pass_rate_interval, per_case_pass_rates


def cases(n=20, **kw):
    return [EvalCase(f"case-{i:03d}", f"question {i}", **kw) for i in range(n)]


# --- dataset ------------------------------------------------------------------------------------------
def test_the_version_is_a_content_hash_independent_of_order():
    a, b = Dataset(cases(5)), Dataset(list(reversed(cases(5))))
    assert a.version == b.version
    changed = Dataset([*cases(4), EvalCase("case-004", "a different question")])
    assert changed.version != a.version


def test_duplicate_case_ids_are_rejected():
    with pytest.raises(ValueError):
        Dataset([EvalCase("x", "1"), EvalCase("x", "2")])


def test_dev_and_heldout_partition_the_cases_without_overlap():
    d = Dataset(cases(200))
    dev, held = {c.id for c in d.dev()}, {c.id for c in d.heldout(confirm_final=True)}
    assert not dev & held and len(dev) + len(held) == 200 and 40 < len(held) < 80  # roughly 30%


def test_adding_cases_never_moves_an_existing_case_between_dev_and_heldout():
    small = Dataset(cases(100))
    big = Dataset(cases(300))
    held_small = {c.id for c in small.heldout(confirm_final=True)}
    held_big = {c.id for c in big.heldout(confirm_final=True)}
    assert held_small <= held_big  # a case once held-out stays held-out


def test_the_heldout_set_cannot_be_read_by_accident():
    with pytest.raises(PermissionError):
        Dataset(cases(10)).heldout()


def test_production_failures_become_unlabeled_deduplicated_redacted_cases():
    d = Dataset(cases(3))
    failures = [FailedRun("refund for ana@example.com please", "error", "r1"),
                FailedRun("refund for ana@example.com please", "error", "r2"),  # same input again
                FailedRun("book a flight to Mars", "abandoned", "r3")]
    d2 = d.add_from_failures(failures)
    new = [c for c in d2.cases if c.id.startswith("prod-")]
    assert len(new) == 2 and all(c.needs_label for c in new)
    assert not any("ana@example.com" in c.input for c in new) and any("[EMAIL]" in c.input for c in new)
    assert len(d.cases) == 3  # the original dataset is untouched
    assert d2.add_from_failures(failures).version == d2.version  # re-adding changes nothing


def test_unlabeled_cases_are_excluded_from_scoring_until_a_human_labels_them():
    d = Dataset(cases(100)).add_from_failures([FailedRun("weird input", "error")])
    assert not any(c.needs_label for c in d.dev()) and not any(c.needs_label for c in d.heldout(confirm_final=True))
    cid = next(c.id for c in d.cases if c.needs_label)
    labeled = d.label(cid, {"must_call": ["search"]}, tags=("safety",))
    case = next(c for c in labeled.cases if c.id == cid)
    assert not case.needs_label and case.expected == {"must_call": ["search"]} and "safety" in case.tags


# --- the release gate -----------------------------------------------------------------------------------
def results(pass_flags, runs=3):
    """pass_flags: list of bools, one per case. Each case is run `runs` times with that outcome."""
    return {f"c{i}": [bool(f)] * runs for i, f in enumerate(pass_flags)}


def test_identical_versions_on_enough_cases_ship():
    r = results([True] * 30 + [False] * 10)
    d = compare(r, r)
    assert d.status == "ship" and d.mean_diff == 0 and d.cases == 40


def test_a_significant_overall_drop_is_blocked():
    a = results([True] * 40)
    b = results([True] * 20 + [False] * 20)
    d = compare(a, b)
    assert d.status == "block" and d.mean_diff == pytest.approx(-0.5) and "significantly worse" in d.reasons[0]


def test_a_noisy_result_that_cannot_rule_out_a_drop_is_inconclusive_not_shipped():
    a = results([False] * 3 + [True] * 3 + [True] * 17 + [False] * 17)
    b = results([True] * 3 + [False] * 3 + [True] * 17 + [False] * 17)  # 3 cases got better, 3 got worse
    d = compare(a, b)
    assert d.mean_diff == 0 and d.status == "inconclusive" and "Add cases or runs" in d.reasons[-1]


def test_too_few_cases_is_inconclusive_even_when_the_results_look_identical():
    r = results([True] * 10)
    d = compare(r, r)
    assert d.status == "inconclusive" and "only 10 cases" in d.reasons[0]


def test_a_safety_regression_blocks_even_when_the_average_improves_a_lot():
    a = results([True] + [False] * 10 + [True] * 29)  # c0 is the safety case and passes today
    b = results([False] + [True] * 10 + [True] * 29)  # +10 cases improved, but the safety case now fails
    d = compare(a, b, tags={"c0": ("safety",)})
    assert d.mean_diff > 0.2  # the average went UP...
    assert d.status == "block" and "safety regression on ['c0']" in d.reasons[0]  # ...and it is blocked anyway


def test_a_non_safety_regression_alone_does_not_trigger_the_safety_block():
    a, b = results([True] * 40), results([False] + [True] * 39)
    assert compare(a, b, tags={"c5": ("safety",)}).status != "block"


def test_no_overlap_between_versions_is_inconclusive():
    assert compare({"x": [True]}, {"y": [True]}).status == "inconclusive"


def test_the_tolerance_is_configurable():
    a = results([True] * 40)
    b = results([True] * 38 + [False] * 2)  # a 5% drop in per-case pass rate
    strict, lenient = Gate(tolerance=0.0), Gate(tolerance=0.2)
    assert compare(a, b, gate=strict).status != "ship"
    assert compare(a, b, gate=lenient).status == "ship"


# --- per-case rates, consistency and overall interval ---------------------------------------------------------------
def test_per_case_pass_rates_and_the_overall_interval():
    r = {"a": [True, True, False, True], "b": [False, False]}
    assert per_case_pass_rates(r) == {"a": 0.75, "b": 0.0}
    rate, low, high = pass_rate_interval(r)
    assert rate == pytest.approx(0.5) and low < rate < high


def test_passing_8_of_10_times_is_not_reliable_pass_all_k_shows_it():
    r = {"a": [True] * 3, "b": [True, False, True], "c": [True, True]}
    assert consistency_pass_all_k(r, 3) == 0.5  # only a and b have 3 runs; only a passed all 3
    assert consistency_pass_all_k(r, 2) == pytest.approx(2 / 3)
    assert consistency_pass_all_k(r, 9) == 0.0
