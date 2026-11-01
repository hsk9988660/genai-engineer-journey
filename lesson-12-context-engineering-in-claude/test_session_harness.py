import re

import pytest

import harness
from assembler import UnstablePrefix
from compactor import Turn
from fakes import FakeClient, text_reply
from session import ContextManager

SYSTEM = "You are a travel assistant. Use only the facts you are given."


def faithful(turns, feedback):
    return "They discussed several trip topics."


def make(summarizer=faithful, **kw):
    return ContextManager(SYSTEM, summarizer, keep_recent=4, compact_over_tokens=100, **kw)


def chat(cm, n):
    """n question/answer pairs of realistic multi-word text. Letters, not digits, so the dropped-number check has nothing to flag."""
    for i in range(n):
        cm.add_user(f"question {chr(97 + i % 26)} " + "tell me more about the places we could visit " * 2)
        cm.add_assistant("answer " + "there are many lovely places and quiet corners worth seeing " * 2)


# --- compaction inside a session ----------------------------------------------------------------------------------------
def test_a_short_conversation_is_not_compacted():
    cm = make()
    chat(cm, 2)
    assert cm.maybe_compact() == "skipped" and cm.summary is None


def test_a_long_conversation_is_compacted_keeping_only_the_recent_turns_live():
    cm = make()
    chat(cm, 10)
    assert cm.maybe_compact() == "compacted"
    live = cm.live_history()
    assert len(live) == 4 and cm.compacted_through == live[0].id - 1
    assert cm.summary.source_ids == tuple(range(1, cm.compacted_through + 1))
    assert len(cm.transcript.all()) == 20  # the archive still has every turn


def test_compaction_makes_the_next_call_smaller():
    big, small = make(), make()
    big.compact_over = 10**9  # never compacts
    for cm in (big, small):
        chat(cm, 15)
    assert small.build("next?").record.total_tokens < big.build("next?").record.total_tokens


def test_pinned_facts_survive_a_summary_that_forgets_the_numbers():
    cm = make(harness.lossy_summarizer)
    cm.add_user("My budget is $3000 for the whole trip.")
    cm.facts.set("budget", "3000", 1, "user")
    cm.add_assistant("Noted, that is a comfortable budget for the trip.")
    chat(cm, 10)
    assert cm.maybe_compact() == "compacted"
    assert "3000" not in cm.summary.text  # the lossy summary really did drop it
    assert "- budget: 3000 (turn 1)" in cm.build("what is my budget?").messages[0]["content"]  # ...but the fact did not


def test_a_hallucinating_summary_is_rejected_and_the_raw_turns_stay():
    cm = make(harness.hallucinating_summarizer)
    chat(cm, 10)
    before = list(cm.live_history())
    assert cm.maybe_compact() == "kept_raw" and cm.summary is None and cm.live_history() == before
    assert "appear nowhere in the source" in cm.compaction_log[-1]["problems"][0]


def test_after_a_failed_compaction_the_manager_waits_instead_of_retrying_every_turn():
    cm = make(harness.hallucinating_summarizer)  # cooldown defaults to 8 turns
    chat(cm, 10)
    assert cm.maybe_compact() == "kept_raw" and len(cm.compaction_log) == 1
    for _ in range(3):
        chat(cm, 1)  # 2 more turns each time
        assert cm.maybe_compact() == "cooling_down"
    assert len(cm.compaction_log) == 1  # no further attempts, so no further wasted model calls
    chat(cm, 3)  # now 8+ turns past the failure
    assert cm.maybe_compact() == "kept_raw" and len(cm.compaction_log) == 2


def test_a_changed_preference_wins_after_compaction_and_the_old_one_is_gone_from_the_whole_context():
    cm = make()
    cm.add_user("I want to fly economy class.")
    cm.facts.set("class", "economy", 1, "user")
    chat(cm, 10)
    assert cm.maybe_compact() == "compacted"
    cm.facts.set("class", "business", 30, "user")
    text = "\n".join(m["content"] for m in cm.build("book it").messages)
    assert "- class: business (turn 30)" in text and "economy" not in text


def test_compacting_twice_treats_the_previous_summary_as_source_so_its_numbers_are_not_flagged_invented():
    def with_budget(turns, feedback):
        return "Budget $3000 noted. Topics were discussed."

    cm = make(with_budget)
    cm.add_user("My budget is $3000 for the whole trip.")
    cm.add_assistant("Noted, thank you for letting me know about that.")
    chat(cm, 8)
    assert cm.maybe_compact() == "compacted"
    chat(cm, 10)  # none of these mention $3000
    assert cm.maybe_compact() == "compacted"  # the second summary repeats $3000, which came from the FIRST summary
    assert [l["status"] for l in cm.compaction_log] == ["compacted", "compacted"]


# --- tool results and building -------------------------------------------------------------------------------------------
def test_a_huge_tool_result_is_offloaded_and_the_context_carries_only_a_preview_and_a_handle():
    cm = make()
    item = cm.add_tool_result("sql_query", "ROW-DATA " * 3000)
    text = "\n".join(m["content"] for m in cm.build("summarise the rows").messages)
    assert item.handle in text and "Preview: ROW-DATA" in text and len(text) < 3000
    assert cm.results.fetch(item.handle).startswith("ROW-DATA")


def test_building_does_not_touch_the_transcript_and_records_accumulate_with_call_ids():
    cm = make()
    chat(cm, 2)
    for _ in range(3):
        cm.build("next?")
    assert len(cm.transcript.all()) == 4 and [r.call_id for r in cm.records] == ["call1", "call2", "call3"]


def test_a_changing_value_in_the_system_prompt_is_refused_when_building():
    cm = ContextManager("Be helpful. Today is 2026-11-02.", faithful)
    with pytest.raises(UnstablePrefix):
        cm.build("hello")


# --- the synthetic conversation -------------------------------------------------------------------------------------------
def test_the_conversation_has_the_planted_facts_a_change_of_mind_and_no_digits_in_user_filler():
    convo = harness.make_conversation(40)
    assert len(convo) == 80 and [r for r, _ in convo] == ["user", "assistant"] * 40
    users = [t for r, t in convo if r == "user"]
    assert users[1] == "My budget is $3000 for the whole trip." and "peanuts" in users[2] and "April" in users[3]
    assert users[23] == "Actually, let's travel in October instead."
    planted = {1, 2, 3, 23}
    assert not any(re.search(r"\d", t) for i, t in enumerate(users) if i not in planted and i != 1)
    assert convo == harness.make_conversation(40) and len(harness.make_conversation(10)) == 20


def test_the_stand_in_extractor_finds_the_planted_facts():
    assert harness.extract_planted("My budget is $3,000 total.") == [("budget", "3000")]
    assert harness.extract_planted("I am allergic to peanuts.") == [("allergy", "peanuts")]
    assert harness.extract_planted("Actually, let's travel in October instead.") == [("dates", "October")]
    assert harness.extract_planted("Tell me about trains.") == []


def test_the_summarizer_test_doubles_behave_as_documented():
    turns = [Turn(1, "user", "My budget is $3000 for the whole trip please."), Turn(2, "user", "I would like to travel in April.")]
    lossy = harness.lossy_summarizer(turns, None)
    assert "3000" not in lossy and not re.search(r"\d", lossy)
    assert "$4500" in harness.hallucinating_summarizer(turns, None)
    told_stale = harness.feedback_aware_summarizer(turns, ["The summary mentions 'April', which is no longer the user's dates."])
    assert "April" not in told_stale and "budget" in told_stale


# --- the strategy comparison: only what the mechanics logically imply ------------------------------------------------------
@pytest.fixture(scope="module")
def outcomes():
    return {o.name: o for o in harness.evaluate(40)}


def test_full_history_keeps_every_fact_but_also_leaks_the_superseded_value(outcomes):
    o = outcomes["full_history"]
    assert all(o.retained.values()) and o.stale_leaked  # the context clash: both April and October are in the prompt


def test_truncation_loses_the_facts_stated_early(outcomes):
    o = outcomes["truncate_last_10"]
    assert not o.retained["budget"] and not o.retained["allergy"] and not o.retained["dates"]


def test_naive_summarization_loses_the_budget_because_the_summary_dropped_the_digits(outcomes):
    o = outcomes["naive_summary"]
    assert not o.retained["budget"] and not all(o.retained.values())


def test_the_managed_strategy_keeps_every_current_fact_with_no_stale_value_and_a_smaller_context(outcomes):
    o, full = outcomes["managed (lossy summarizer)"], outcomes["full_history"]
    assert all(o.retained.values()) and not o.stale_leaked
    assert o.final_tokens < full.final_tokens and o.cumulative_tokens < 0.75 * full.cumulative_tokens
    assert "2 ok, 0 kept raw" in o.note


def test_a_hallucinating_summarizer_never_gets_to_replace_anything_yet_the_facts_still_survive(outcomes):
    o = outcomes["managed (hallucinating summarizer)"]
    assert "0 ok" in o.note and all(o.retained.values()) and not o.stale_leaked
    attempts = int(re.search(r"(\d+) kept raw", o.note).group(1))
    assert 1 <= attempts <= 8  # the cooldown stopped it retrying on all ~23 candidate turns


def test_total_tokens_sent_grow_quadratically_with_conversation_length_under_full_history():
    c10, c20, c40 = (harness.cumulative_tokens(harness.FullHistory, n) for n in (10, 20, 40))
    assert 3.3 < c20 / c10 < 4.8 and 3.5 < c40 / c20 < 4.5  # doubling the turns roughly quadruples the cost


def test_the_table_lists_every_strategy(outcomes, capsys):
    harness.print_table(list(outcomes.values()))
    out = capsys.readouterr().out
    for name in outcomes:
        assert name in out
    assert "LOST" in out and "LEAKED" in out


def test_the_live_evaluation_runs_against_a_client_and_prints_one_answer_per_strategy(capsys):
    client = FakeClient([text_reply("Budget 3000, peanuts, October.")] * 4)
    harness.live_needle_eval(client, n_turns=12)
    out = capsys.readouterr().out
    assert out.count("Budget 3000, peanuts, October.") == 4 and len(client.calls) == 4
