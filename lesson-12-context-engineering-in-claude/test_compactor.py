from compactor import Compactor, Transcript, Turn, numbers_in, verify_summary
from facts import FactStore
from tokens import estimate_tokens


def T(i, role, text):
    return Turn(i, role, text)


SOURCES = [T(1, "user", "My budget is $3000 for the whole trip."), T(2, "assistant", "Route 5 takes about 7 hours by train."),
           T(3, "user", "Great, tell me more about the trains.")]


# --- number handling ----------------------------------------------------------------------------------------------
def test_numbers_are_compared_without_currency_commas_or_trailing_dots():
    assert numbers_in("$3,000 and 3000.5, then 12.") == {"3000", "3000.5", "12"}
    assert numbers_in("budget $3,000") == numbers_in("budget 3000")


# --- what verification catches ------------------------------------------------------------------------------------
def test_a_faithful_summary_passes():
    assert verify_summary("The budget is $3000. Route 5 takes about 7 hours by train.", SOURCES, FactStore()) == []


def test_an_invented_number_is_caught():
    problems = verify_summary("The budget is $3000 and a ticket costs $4500.", SOURCES, FactStore())
    assert any("appear nowhere in the source" in p and "4500" in p for p in problems)


def test_a_dropped_user_number_is_caught_unless_the_fact_store_preserves_it():
    problems = verify_summary("They discussed trains.", SOURCES, FactStore())
    assert any("dropped numbers the user stated" in p and "3000" in p for p in problems)
    store = FactStore()
    store.set("budget", "3000", 1, "user")
    assert verify_summary("They discussed trains.", SOURCES, store) == []  # the number is safe in the fact store, so dropping it is fine


def test_a_dropped_assistant_number_is_not_treated_as_a_problem():
    store = FactStore()
    store.set("budget", "3000", 1, "user")
    assert verify_summary("Trains were discussed and a budget was set.", SOURCES, store) == []  # route 5 / 7 hours were assistant filler


def stale_store():
    s = FactStore()
    s.set("dates", "April", 2, "user")
    s.set("dates", "October", 9, "user")
    return s


def test_a_superseded_value_still_in_the_summary_is_caught_case_insensitively():
    for text in ("They planned to travel in April.", "they planned to travel in april."):
        problems = verify_summary(text, [T(1, "assistant", "x")], stale_store())
        assert any("no longer the user's dates" in p for p in problems), problems


def test_the_current_value_and_unrelated_words_are_not_flagged():
    store = stale_store()
    assert verify_summary("They will travel in October.", [T(1, "assistant", "x")], store) == []
    assert verify_summary("They like Aprilfest food markets.", [T(1, "assistant", "x")], store) == []  # whole-word match only


# --- the compactor: replace only when verified ------------------------------------------------------------------------
# no digits in the user turns here, so the dropped-number check has nothing to complain about in these tests
LONG = [T(i, "user" if i % 2 else "assistant", f"Here is a long sentence about topic {chr(96 + i)} that goes on and on for a while.") for i in range(1, 11)]


def test_a_verified_smaller_summary_replaces_the_turns_and_records_what_it_covers():
    c = Compactor(lambda turns, fb: "They talked about several topics.", FactStore())
    r = c.compact(LONG)
    assert r.status == "compacted" and r.summary.source_ids == tuple(range(1, 11))
    assert r.summary.tokens_before == sum(estimate_tokens(t.text) for t in LONG) and r.summary.tokens_after < r.summary.tokens_before


def test_a_summary_that_is_not_smaller_is_refused():
    r = Compactor(lambda turns, fb: "x" * 5000, FactStore()).compact(LONG)
    assert r.status == "kept_raw" and "not smaller" in r.problems[0]


def test_a_rejected_summary_is_retried_with_the_problems_as_feedback():
    seen = []

    def summarizer(turns, feedback):
        seen.append(feedback)
        return "Topics discussed. The price is $4500." if feedback is None else "Topics discussed."

    r = Compactor(summarizer, FactStore()).compact(LONG)
    assert r.status == "compacted" and seen[0] is None and "4500" in seen[1][0]


def test_a_summarizer_that_keeps_failing_ends_in_kept_raw_after_exactly_one_retry():
    calls = []

    def always_bad(turns, feedback):
        calls.append(1)
        return "Topics. Budget $4500."

    r = Compactor(always_bad, FactStore()).compact(LONG)
    assert r.status == "kept_raw" and r.summary is None and r.problems and len(calls) == 2


def test_nothing_to_compact_is_a_no_op():
    assert Compactor(lambda t, f: "x", FactStore()).compact([]).status == "nothing_to_do"


# --- the archive ------------------------------------------------------------------------------------------------------
def test_the_transcript_is_append_only_and_compaction_never_edits_it():
    t = Transcript()
    for i in range(10):
        t.append("user" if i % 2 == 0 else "assistant", f"message {i}")
    before = [(x.id, x.role, x.text) for x in t.all()]
    Compactor(lambda turns, fb: "gist", FactStore()).compact(t.all())
    assert [(x.id, x.role, x.text) for x in t.all()] == before
    assert [x.text for x in t.get([2, 5])] == ["message 1", "message 4"]


def test_the_dropped_number_check_is_conservative_a_good_summary_that_loses_any_user_digit_is_refused():
    """Documented limitation: real users mention numbers constantly (12 km, 3 nights). Every one the summary omits counts as a
    problem, so compaction will often fall back to keeping the raw turns. Safe, but it costs tokens. Tune this for real traffic."""
    turns = [T(1, "user", "We walked 12 km and it was lovely."), T(2, "assistant", "That sounds like a long, lovely walk to remember.")]
    r = Compactor(lambda t, fb: "They went on a long, lovely walk.", FactStore()).compact(turns * 5)
    assert r.status == "kept_raw" and "dropped numbers" in r.problems[0] and "12" in r.problems[0]
