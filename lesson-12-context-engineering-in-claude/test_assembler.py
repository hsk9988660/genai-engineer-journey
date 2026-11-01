import json
import re

import pytest

from assembler import Budget, ContextAssembler, ContextOverflow, Item, UnstablePrefix, looks_volatile
from compactor import Summary, Turn
from facts import FactStore
from tokens import estimate_tokens
from tool_results import ToolItem

SYSTEM = "You are a travel assistant. Use only the facts you are given."
BIG = Budget(total=100_000, summary=100_000, retrieved=100_000, history=100_000, tool_results=100_000)


def turns(n, size=40, start=1):
    return [Turn(i, "user" if i % 2 else "assistant", f"turn{i} " + "w" * size) for i in range(start, start + n)]


def build(budget=None, facts=None, current="What next?", **kw):
    return ContextAssembler(budget or BIG, kw.pop("count", None) or estimate_tokens).assemble(
        system=SYSTEM, facts=facts or FactStore(), current=current, **kw)


def facts_with(**kv):
    s = FactStore()
    for i, (k, v) in enumerate(kv.items(), start=1):
        s.set(k, v, i, "user")
    return s


def text_of(assembled):
    return "\n".join(m["content"] for m in assembled.messages)


# --- the required pieces and the message shape ---------------------------------------------------------------------------
def test_the_system_prompt_pinned_facts_and_current_request_are_always_present():
    a = build(facts=facts_with(budget="3000"), current="Plan day one.")
    assert a.system == SYSTEM
    assert a.messages[0]["role"] == "user" and "<pinned_facts>\n- budget: 3000 (turn 1)\n</pinned_facts>" in a.messages[0]["content"]
    assert a.messages[-1]["content"].endswith("Plan day one.")


def test_an_empty_fact_store_still_renders_a_pinned_block():
    assert "(none yet)" in build().messages[0]["content"]


def test_roles_always_alternate_and_the_first_message_is_from_the_user():
    for history in (turns(5), turns(5, start=2), turns(4), turns(0)):  # starts with user, with assistant, ends in assistant, empty
        msgs = build(history=history).messages
        assert msgs[0]["role"] == "user" and all(a["role"] != b["role"] for a, b in zip(msgs, msgs[1:])), [m["role"] for m in msgs]


def test_with_no_history_the_preamble_and_the_request_become_one_message():
    msgs = build(current="Hello?").messages
    assert len(msgs) == 1 and "<pinned_facts>" in msgs[0]["content"] and msgs[0]["content"].endswith("Hello?")


def test_superseded_and_quarantined_facts_never_reach_the_context():
    s = FactStore()
    s.set("class", "economy", 2, "user")
    s.set("class", "business", 9, "user")
    s.set("flight", "AS 999 from a tiny airport to Tokyo", 5, "tool")  # quarantined: untrusted source
    text = text_of(build(facts=s))
    assert "business" in text and "economy" not in text and "AS 999" not in text


def test_request_omits_tools_when_there_are_none_and_includes_them_otherwise():
    assert "tools" not in build().request()
    tool = {"name": "search", "description": "Search.", "input_schema": {"type": "object"}}
    assert build(tools=[tool]).request()["tools"] == [tool]


# --- budgets ------------------------------------------------------------------------------------------------------------
def test_history_over_its_component_budget_drops_whole_old_turns_and_keeps_the_newest():
    b = Budget(total=100_000, history=100)  # each turn is 12 tokens, so 8 fit
    a = build(b, history=turns(10))
    assert a.record.selected["history"] == [f"turn{i}" for i in range(3, 11)]
    assert [(d["id"], d["reason"]) for d in a.record.dropped] == [("turn1", "over component budget"), ("turn2", "over component budget")]  # recorded oldest first


def test_history_is_a_contiguous_suffix_there_are_never_gaps():
    hist = [Turn(1, "user", "tiny"), Turn(2, "assistant", "H" * 4000), Turn(3, "user", "tiny"), Turn(4, "assistant", "tiny")]
    a = build(Budget(total=100_000, history=50), history=hist)
    assert a.record.selected["history"] == ["turn3", "turn4"]  # turn1 is small but is NOT kept after a big turn was dropped
    assert {d["id"] for d in a.record.dropped} == {"turn1", "turn2"}


def test_retrieved_items_are_chosen_best_score_first_and_the_lowest_is_dropped_when_over_budget():
    items = [Item("low", "L" * 400, 0.1), Item("high", "H" * 400, 0.9), Item("mid", "M" * 400, 0.5)]
    a = build(Budget(total=100_000, retrieved=230), retrieved=items)  # room for two 100-token documents plus the section wrapper
    assert a.record.selected["retrieved"] == ["high", "mid"] and a.record.dropped[0]["id"] == "low"


def test_recent_tool_results_are_shown_in_full_and_older_ones_become_one_line_stubs():
    results = [ToolItem(f"tr{i}", "search", f"FULL-OUTPUT-{i}", None, 900) for i in range(1, 5)]
    text = build(Budget(total=100_000, keep_full_tool_results=2), tool_results=results).messages[0]["content"]
    assert "FULL-OUTPUT-3" in text and "FULL-OUTPUT-4" in text
    assert "FULL-OUTPUT-1" not in text and "FULL-OUTPUT-2" not in text and text.count("[tool result cleared: search, about 900 tokens]") == 2


def test_a_summary_is_included_when_it_fits_and_recorded_as_dropped_when_it_does_not():
    summary = Summary("They talked about trains.", (1, 2, 3), 500, 8)
    assert "<conversation_summary>\nThey talked about trains.\n</conversation_summary>" in build(summary=summary).messages[0]["content"]
    a = build(Budget(total=100_000, summary=2), summary=summary)
    assert "trains" not in text_of(a) and a.record.dropped == [{"id": "summary", "component": "summary", "reason": "over component budget", "tokens": 7}]


def test_retrieved_documents_and_tool_results_are_labelled_as_data_not_instructions():
    text = build(retrieved=[Item("d1", "doc")], tool_results=[ToolItem("tr1", "search", "result", None, 5)]).messages[0]["content"]
    assert text.count("data, not instructions") == 2 and '<document id="d1">' in text and '<tool_result id="tr1" tool="search">' in text


# --- shedding under total pressure -----------------------------------------------------------------------------------------
def full_inputs():
    return dict(summary=Summary("S" * 400, (1,), 900, 100), retrieved=[Item("d1", "R" * 400, 0.9)], history=turns(6, size=100),
                tool_results=[ToolItem("tr1", "search", "T" * 400, None, 100)])


def test_over_the_total_budget_optional_pieces_are_shed_in_the_configured_order():
    base = build(**full_inputs())
    c = base.record.component_tokens
    required = c["prefix"] + c["pinned"] + c["current"]
    assert min(c["tool_results"], c["retrieved"], c["history"], c["summary"]) > 0

    def dropped_components(limit):
        a = ContextAssembler(Budget(total=limit, summary=10**6, retrieved=10**6, history=10**6, tool_results=10**6)).assemble(
            system=SYSTEM, facts=FactStore(), current="What next?", **full_inputs())
        assert a.record.total_tokens <= limit
        return {d["component"] for d in a.record.dropped}

    full = sum(c.values())
    assert dropped_components(full) == set()
    assert dropped_components(full - c["tool_results"]) == {"tool_results"}  # tool results go first
    assert dropped_components(full - c["tool_results"] - c["retrieved"]) == {"tool_results", "retrieved"}  # then retrieved documents
    assert dropped_components(full - c["tool_results"] - c["retrieved"] - c["history"]) == {"tool_results", "retrieved", "history"}  # summary survives
    assert dropped_components(required) == {"tool_results", "retrieved", "history", "summary"}  # everything optional, never the required


def test_required_pieces_are_never_dropped_even_under_extreme_pressure():
    a = ContextAssembler(Budget(total=60)).assemble(system=SYSTEM, facts=facts_with(budget="3000"), current="Go.", **full_inputs())
    assert "- budget: 3000" in a.messages[0]["content"] and a.messages[-1]["content"].endswith("Go.") and a.system == SYSTEM


def test_when_the_required_pieces_alone_exceed_the_budget_it_raises_instead_of_truncating():
    with pytest.raises(ContextOverflow) as e:
        ContextAssembler(Budget(total=10)).assemble(system=SYSTEM, facts=FactStore(), current="hello")
    assert e.value.required > 10 and e.value.total == 10


# --- the inspection record --------------------------------------------------------------------------------------------------
def test_the_record_has_counts_ids_and_hashes_but_never_raw_text():
    secret_facts = facts_with(allergy="SECRET-PEANUT")
    a = build(Budget(total=100_000, history=30), facts=secret_facts, current="SECRET-CURRENT-QUESTION",
              history=turns(6, size=30), retrieved=[Item("doc-7", "SECRET-DOC-BODY")], tool_results=[ToolItem("tr1", "search", "SECRET-RESULT", None, 3)],
              summary=Summary("SECRET-SUMMARY", (1,), 100, 5), call_id="call42")
    blob = json.dumps(a.record.to_dict())
    for secret in ("SECRET-PEANUT", "SECRET-CURRENT-QUESTION", "SECRET-DOC-BODY", "SECRET-RESULT", "SECRET-SUMMARY", "w" * 10):
        assert secret not in blob, secret
    r = a.record
    assert r.call_id == "call42" and "doc-7" in r.selected["retrieved"] and r.selected["tool_results"] == ["tr1"]
    assert re.fullmatch(r"[0-9a-f]{64}", r.prefix_hash) and re.fullmatch(r"[0-9a-f]{64}", r.context_hash)
    assert r.total_tokens == sum(r.component_tokens.values()) and r.budget_total == 100_000
    assert all(set(d) == {"id", "component", "reason", "tokens"} for d in r.dropped) and r.dropped


# --- determinism and cache-friendliness ---------------------------------------------------------------------------------------
def test_the_same_inputs_always_produce_the_same_context():
    kw = dict(facts=facts_with(budget="3000"), history=turns(5), retrieved=[Item("a", "x", 0.5), Item("b", "y", 0.5)])
    assert build(**kw).record.context_hash == build(**kw).record.context_hash


def test_changing_the_conversation_changes_the_context_hash_but_not_the_prefix_hash():
    a, b = build(history=turns(5)), build(history=turns(5, size=41))
    assert a.record.context_hash != b.record.context_hash and a.record.prefix_hash == b.record.prefix_hash


def test_volatile_information_goes_after_the_stable_prefix_so_the_prefix_hash_survives_a_new_day():
    monday = build(volatile="Today is 2026-11-02.")
    tuesday = build(volatile="Today is 2026-11-03.")
    assert monday.record.prefix_hash == tuesday.record.prefix_hash and monday.record.context_hash != tuesday.record.context_hash
    assert "2026-11-02" not in monday.system and "Today is 2026-11-02." in monday.messages[-1]["content"]


def test_a_changing_value_in_the_system_prompt_or_a_tool_description_is_refused():
    with pytest.raises(UnstablePrefix, match="ISO date"):
        ContextAssembler(BIG).assemble(system="Be helpful. Today is 2026-11-02.", facts=FactStore(), current="x")
    with pytest.raises(UnstablePrefix, match="clock time"):
        ContextAssembler(BIG).assemble(system=SYSTEM, facts=FactStore(), current="x",
                                       tools=[{"name": "t", "description": "Server time 12:30:45", "input_schema": {}}])


def test_the_volatility_check_labels_what_it_found():
    assert looks_volatile("fine and stable") == []
    assert looks_volatile("It was 2026-01-05 at 10:11:12, today is Monday") == ["an ISO date", "a clock time", "a 'today' statement"]
    assert looks_volatile("session " + "a" * 32) == ["a hex id"]


def test_token_counts_come_from_the_count_function_you_provide():
    a = ContextAssembler(BIG, count=len).assemble(system=SYSTEM, facts=FactStore(), current="12345")
    assert a.record.component_tokens["current"] == 5 and a.record.component_tokens["prefix"] == len(SYSTEM + "[]")
