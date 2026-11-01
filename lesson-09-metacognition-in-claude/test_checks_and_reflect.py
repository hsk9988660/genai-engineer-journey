import json

import pytest

from checks import Constraints, Itinerary, check_itinerary, fingerprint, minutes
from fakes import C, FakeClient, act, broken, good, submit
from memory import ReflectionMemory
from reflect import ItineraryGenerator, solve_itinerary, solve_with_checks


def codes(payload, c=C):
    return sorted(f.code for f in check_itinerary(Itinerary.model_validate(payload), c))


# --- the deterministic checkers -----------------------------------------------------------------
def test_a_valid_itinerary_has_no_failures():
    assert codes(good()) == []


@pytest.mark.parametrize("problem", ["no_dinner", "overlap", "total_mismatch", "budget", "too_many"])
def test_each_problem_is_detected_by_exactly_its_own_check(problem):
    assert codes(broken(problem)) == [problem]


def test_activities_that_touch_are_not_an_overlap():
    p = good()
    p["days"][0]["activities"][1]["start"] = "11:00"  # lunch starts the minute the museum ends
    assert codes(p) == []


def test_invalid_times_are_reported_without_crashing_the_other_checks():
    p = good()
    p["days"][0]["activities"][0]["start"] = "25:99"
    p["days"][0]["activities"][1]["end"] = "11:00"  # ends before it starts
    assert codes(p).count("bad_time") == 2


def test_wrong_day_count_and_duplicate_day_numbers():
    p = good()
    p["days"] = p["days"][:1]
    assert "days_count" in codes(p)
    q = good()
    q["days"][1]["day"] = 1
    assert "duplicate_day" in codes(q)


def test_dinner_must_start_at_or_after_1800_and_be_a_meal():
    p = good()
    p["days"][1]["activities"][2]["start"] = "17:59"
    assert "no_dinner" in codes(p)
    q = good()
    q["days"][1]["activities"][2]["kind"] = "sight"
    assert "no_dinner" in codes(q)
    assert codes(good(), Constraints(budget_usd=300, num_days=2, max_activities_per_day=3, dinner_required=False)) == []


def test_messages_name_the_specific_problem():
    msgs = [f.message for f in check_itinerary(Itinerary.model_validate(broken("overlap")), C)]
    assert "Lunch" in msgs[0] and "Museum" in msgs[0] and "Day 1" in msgs[0]


def test_minutes_parsing():
    assert minutes("00:00") == 0 and minutes("18:00") == 1080 and minutes("23:59") == 1439
    assert minutes("24:00") is None and minutes("7:30") is None and minutes("noon") is None


def test_fingerprint_ignores_wording_and_order():
    a = check_itinerary(Itinerary.model_validate(broken("no_dinner", "overlap")), C)
    assert fingerprint(a) == fingerprint(list(reversed(a)))


# --- the generic loop -----------------------------------------------------------------------------
def make_loop(candidates, failures_by_candidate):
    seen = []

    def generate(feedback, n):
        seen.append(feedback)
        return candidates[n - 1]

    return generate, (lambda cand: failures_by_candidate[cand]), seen


from checks import Failure  # noqa: E402

F1, F2, F3 = Failure("a", "fix a"), Failure("b", "fix b"), Failure("c", "fix c")


def test_a_first_try_pass_makes_exactly_one_attempt():
    gen, chk, seen = make_loop(["x"], {"x": []})
    r = solve_with_checks(gen, chk)
    assert r.status == "passed" and len(r.attempts) == 1 and seen == [None] and r.passed


def test_failures_are_fed_back_as_specific_messages_and_the_fix_is_accepted():
    gen, chk, seen = make_loop(["bad", "good"], {"bad": [F1, F2], "good": []})
    r = solve_with_checks(gen, chk)
    assert r.status == "passed" and r.best == "good" and seen == [None, ["fix a", "fix b"]]


def test_the_cap_stops_a_loop_that_keeps_changing_its_failures():
    gen, chk, _ = make_loop(["1", "2", "3"], {"1": [F1], "2": [F2], "3": [F3]})
    r = solve_with_checks(gen, chk, max_attempts=3)
    assert r.status == "capped" and len(r.attempts) == 3 and not r.passed


def test_the_same_failures_twice_in_a_row_stop_the_loop_early():
    gen, chk, _ = make_loop(["1", "2", "3", "4"], {"1": [F1], "2": [F1], "3": [], "4": []})
    r = solve_with_checks(gen, chk, max_attempts=4)
    assert r.status == "stalled" and len(r.attempts) == 2  # it did not burn attempts 3 and 4


def test_the_best_candidate_is_returned_not_the_last_one():
    gen, chk, _ = make_loop(["ok-ish", "worse", "worst"], {"ok-ish": [F1], "worse": [F2, F3], "worst": [F1, F2, F3]})
    r = solve_with_checks(gen, chk, max_attempts=3)
    assert r.best == "ok-ish" and len(r.best_failures) == 1


# --- with the model (scripted) ----------------------------------------------------------------------
def test_a_valid_first_itinerary_costs_one_call():
    client = FakeClient([submit(good())])
    r, usage = solve_itinerary(client, "2 days in Lisbon", C)
    assert r.status == "passed" and usage.calls == 1 and usage.total == 150


def test_the_revision_prompt_contains_the_previous_itinerary_and_the_exact_failures():
    client = FakeClient([submit(broken("no_dinner", "total_mismatch")), submit(good())])
    r, usage = solve_itinerary(client, "2 days in Lisbon", C)
    assert r.status == "passed" and len(r.attempts) == 2 and usage.calls == 2
    second = json.loads(client.calls[1]["messages"][0]["content"])
    assert any("Day 2 has no dinner" in m for m in second["automatic_check_failures"])
    assert any("total_cost_usd" in m for m in second["automatic_check_failures"])
    assert second["previous_itinerary"]["total_cost_usd"] == 150.0
    assert "automatic_check_failures" not in json.loads(client.calls[0]["messages"][0]["content"])


def test_a_model_that_returns_the_same_broken_itinerary_stalls_and_stops_spending():
    client = FakeClient([submit(broken("overlap"))] * 4)
    r, usage = solve_itinerary(client, "x", C, max_attempts=4)
    assert r.status == "stalled" and usage.calls == 2 and not r.passed


def test_a_schema_error_is_retried_once_inside_a_single_attempt():
    client = FakeClient([submit({"days": "oops"}), submit(good())])
    r, usage = solve_itinerary(client, "x", C)
    assert r.status == "passed" and len(r.attempts) == 1 and usage.calls == 2


def test_validated_failures_become_lessons_and_the_next_run_sees_them():
    memory = ReflectionMemory()
    client = FakeClient([submit(broken("no_dinner")), submit(good()), submit(good())])
    solve_itinerary(client, "run 1", C, memory=memory)
    assert [l.code for l in memory.active()] == ["no_dinner"]
    solve_itinerary(client, "run 2", C, memory=memory)
    assert "dinner" in client.calls[2]["system"] and "dinner" not in client.calls[0]["system"].lower().split("lessons")[0]


def test_the_generator_alone_remembers_its_last_candidate():
    client = FakeClient([submit(good())])
    gen = ItineraryGenerator(client, "t", C)
    assert gen(None, 1) is gen.last
