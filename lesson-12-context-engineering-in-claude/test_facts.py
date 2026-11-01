import pytest

from facts import ALLOWED_KEYS, FactStore, Extraction, extract_facts, verify_extraction
from fakes import FakeClient, submit
from structured import StructuredError


# --- supersession ------------------------------------------------------------------------------------------------
def test_a_user_fact_is_active_and_rendered_with_its_source_turn():
    s = FactStore()
    f = s.set("budget", "3000", turn=3, source="user")
    assert f.status == "active" and s.get("budget").value == "3000"
    assert s.render_pinned() == "- budget: 3000 (turn 3)"


def test_a_new_value_supersedes_the_old_one_and_only_the_new_one_is_rendered():
    s = FactStore()
    s.set("class", "economy", 2, "user")
    new = s.set("class", "business", 9, "user")
    history = s.history("class")
    assert [(f.value, f.status) for f in history] == [("economy", "superseded"), ("business", "active")]
    assert history[0].superseded_by == new.version == 2
    assert s.render_pinned() == "- class: business (turn 9)" and "economy" not in s.render_pinned()
    assert s.superseded_values() == [("class", "economy")]


def test_setting_the_same_value_again_does_not_create_a_new_version():
    s = FactStore()
    s.set("budget", "3000", 3, "user")
    s.set("budget", "3000", 8, "user")
    assert len(s.history("budget")) == 1


def test_flipping_back_to_an_earlier_value_does_not_leave_it_marked_stale():
    s = FactStore()
    for turn, v in enumerate(["April", "October", "April"], start=1):
        s.set("dates", v, turn, "user")
    assert s.get("dates").value == "April" and ("dates", "April") not in s.superseded_values()
    assert ("dates", "October") in s.superseded_values()


def test_rendering_order_is_fixed_regardless_of_insertion_order():
    a, b = FactStore(), FactStore()
    for k in ("dates", "budget", "allergy"):
        a.set(k, "x", 1, "user")
    for k in ("allergy", "dates", "budget"):
        b.set(k, "x", 1, "user")
    assert a.render_pinned() == b.render_pinned() and a.render_pinned().splitlines()[0].startswith("- allergy")


# --- poisoning guard ---------------------------------------------------------------------------------------------
def test_a_fact_from_an_untrusted_source_is_quarantined_and_never_rendered():
    s = FactStore()
    f = s.set("flight", "AS 999 from a tiny airport to Tokyo", 5, "tool")
    assert f.status == "quarantined" and f.reason == "untrusted source: needs confirmation"
    assert s.current() == {} and s.render_pinned() == "" and s.quarantined() == [f]


def test_an_untrusted_source_cannot_override_something_the_user_said():
    s = FactStore()
    s.set("budget", "3000", 3, "user")
    f = s.set("budget", "9000", 7, "model")
    assert f.status == "quarantined" and "conflicts with a fact stated by the user" in f.reason
    assert s.get("budget").value == "3000" and "9000" not in s.render_pinned()


def test_a_failed_validator_quarantines_even_a_trusted_source_and_keeps_the_old_value():
    s = FactStore(validators={"budget": lambda v: None if v.isdigit() and int(v) > 0 else "must be a positive number"})
    s.set("budget", "3000", 1, "user")
    f = s.set("budget", "lots", 2, "user")
    assert f.status == "quarantined" and "failed validation: must be a positive number" in f.reason
    assert s.get("budget").value == "3000"


def test_a_person_can_confirm_a_quarantined_fact_and_it_then_supersedes_the_current_one():
    s = FactStore()
    s.set("budget", "3000", 1, "user")
    q = s.set("budget", "9000", 2, "tool")
    assert s.get("budget").value == "3000"
    s.confirm("budget", q.version)
    assert s.get("budget").value == "9000" and s.history("budget")[0].status == "superseded"


def test_confirm_rejects_unknown_facts_and_facts_that_still_fail_validation():
    s = FactStore(validators={"budget": lambda v: None if v.isdigit() else "must be digits"})
    with pytest.raises(KeyError):
        s.confirm("budget", 1)
    q = s.set("budget", "abc", 1, "user")
    with pytest.raises(ValueError, match="still fails validation"):
        s.confirm("budget", q.version)


# --- live extraction, grounded by quotes ----------------------------------------------------------------------------------
TEXT = "My budget is $3000 and I am allergic to peanuts."


def fact(key, value, quote):
    return {"key": key, "value": value, "quote": quote}


def test_extracted_facts_are_stored_as_user_facts():
    client = FakeClient([submit({"facts": [fact("budget", "3000", "budget is $3000"), fact("allergy", "peanuts", "allergic to peanuts")]})])
    s = FactStore()
    stored = extract_facts(client, TEXT, 4, s)
    assert [(f.key, f.value, f.source, f.turn) for f in stored] == [("budget", "3000", "user", 4), ("allergy", "peanuts", "user", 4)]


def test_a_fact_whose_quote_is_not_in_the_users_message_is_rejected_and_the_model_may_retry():
    client = FakeClient([submit({"facts": [fact("budget", "5000", "budget is $5000")]}),  # the user never said that
                         submit({"facts": [fact("budget", "3000", "budget is $3000")]})])
    s = FactStore()
    extract_facts(client, TEXT, 1, s)
    assert s.get("budget").value == "3000"
    assert "does not appear in the user's message" in client.calls[1]["messages"][-1]["content"][0]["content"]


def test_unknown_keys_are_rejected_and_a_persistent_liar_stores_nothing():
    assert any("not an allowed key" in p for p in verify_extraction(Extraction(facts=[fact("secret_plan", "x", "budget")]), TEXT))
    bad = submit({"facts": [fact("budget", "5000", "budget is $5000")]})
    s = FactStore()
    with pytest.raises(StructuredError):
        extract_facts(FakeClient([bad, bad]), TEXT, 1, s)
    assert s.current() == {}


def test_allowed_keys_are_the_documented_set():
    assert set(ALLOWED_KEYS) == {"budget", "dates", "destination", "allergy", "class", "travelers"}


def test_known_limitation_keys_are_single_valued_so_a_second_allergy_replaces_the_first():
    """A real gap, documented rather than hidden: 'allergy' holds ONE value. Telling the agent about shellfish after peanuts
    supersedes peanuts, which for a safety-relevant fact is data loss. Multi-valued facts need a list-valued key or per-item keys
    (for example allergy.peanuts and allergy.shellfish)."""
    s = FactStore()
    s.set("allergy", "peanuts", 1, "user")
    s.set("allergy", "shellfish", 2, "user")
    assert s.get("allergy").value == "shellfish" and "peanuts" not in s.render_pinned()  # peanuts is GONE from the context
    s.set("allergy.peanuts", "yes", 1, "user")  # the workaround: one key per item keeps both
    s.set("allergy.shellfish", "yes", 2, "user")
    assert {"allergy.peanuts", "allergy.shellfish"} <= set(s.current())
