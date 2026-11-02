import dataclasses
from decimal import Decimal

import pytest

from agentdef import AgentVersion, Tier, ToolSpec
from harness import make_version
from threads import InvalidMessage, NotFound, Scope, ThreadFull, ThreadStore

A, B = Scope("alice"), Scope("bob")

PUBLIC, USER, EFFECT = ToolSpec("lookup", "public"), ToolSpec("orders", "user"), ToolSpec("refund", "effect")


# --- the agent definition and its version id ------------------------------------------------------------------------------------------
def test_equal_definitions_have_the_same_version_id_and_it_is_short_and_stable():
    assert make_version().version_id == make_version().version_id and len(make_version().version_id) == 12


def test_any_change_to_prompt_model_tools_prices_or_routing_changes_the_version_id():
    base = make_version()
    small, large = base.tiers
    variants = [make_version("a different prompt"), make_version(top_keywords=("cancel",)),
                dataclasses.replace(base, tiers=(dataclasses.replace(small, model="other-model"), large)),
                dataclasses.replace(base, tiers=(dataclasses.replace(small, in_price=Decimal("9")), large)),
                dataclasses.replace(base, tiers=(dataclasses.replace(small, tools=("search_policies",)), large)),
                dataclasses.replace(base, tools=base.tools[:-1], tiers=(small, dataclasses.replace(large, tools=("search_policies", "get_order_status"))))]
    ids = {v.version_id for v in variants} | {base.version_id}
    assert len(ids) == len(variants) + 1


def test_a_side_effect_tool_may_only_be_offered_to_the_top_tier():
    cheap, top = Tier("cheap", "m1", ("refund",)), Tier("top", "m2", ("refund",))
    with pytest.raises(ValueError, match="top tier"):
        AgentVersion("a", "p", (cheap, top), (EFFECT,))
    AgentVersion("a", "p", (Tier("cheap", "m1", ()), top), (EFFECT,))  # fine: only the last tier has it


def test_a_tier_cannot_name_a_tool_that_does_not_exist_and_an_agent_needs_a_tier():
    with pytest.raises(ValueError, match="unknown tool"):
        AgentVersion("a", "p", (Tier("t", "m", ("ghost",)),), (PUBLIC,))
    with pytest.raises(ValueError):
        AgentVersion("a", "p", (), ())


def test_tool_scope_must_be_known_and_tools_can_be_looked_up():
    with pytest.raises(ValueError):
        ToolSpec("x", "everyone")
    assert make_version().tool("issue_refund").scope == "effect"


def test_in_the_demo_definition_the_cheap_tier_has_no_refund_tool_and_the_top_tier_has_it():
    small, large = make_version().tiers
    assert "issue_refund" not in small.tools and "issue_refund" in large.tools


# --- threads ------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def store():
    s = ThreadStore(max_messages=6, max_chars=50)
    yield s
    s.close()


def test_thread_ids_are_random_and_url_safe():
    s = ThreadStore()
    ids = {s.create(A) for _ in range(50)}
    assert len(ids) == 50 and all(len(i) >= 16 and i.replace("-", "").replace("_", "").isalnum() for i in ids)


def test_messages_come_back_in_order_and_last_n_returns_the_most_recent(store):
    t = store.create(A)
    for i in range(4):
        store.append(A, t, "user" if i % 2 == 0 else "assistant", f"m{i}")
    assert [m["content"] for m in store.messages(A, t)] == ["m0", "m1", "m2", "m3"]
    assert [m["content"] for m in store.messages(A, t, last=2)] == ["m2", "m3"] and store.count(A, t) == 4


def test_only_user_and_assistant_roles_and_bounded_non_empty_text_are_accepted(store):
    t = store.create(A)
    for role, content in (("system", "x"), ("user", ""), ("user", "x" * 51), ("user", 5)):
        with pytest.raises(InvalidMessage):
            store.append(A, t, role, content)


def test_a_thread_holds_a_bounded_number_of_messages(store):
    t = store.create(A)
    for i in range(6):
        store.append(A, t, "user", f"m{i}")
    with pytest.raises(ThreadFull):
        store.append(A, t, "user", "one more")


def test_another_users_thread_id_behaves_exactly_like_a_missing_one(store):
    t = store.create(A)
    store.append(A, t, "user", "private")
    for call in (lambda: store.messages(B, t), lambda: store.append(B, t, "user", "x"), lambda: store.count(B, t), lambda: store.messages(B, "nope")):
        with pytest.raises(NotFound):
            call()
    assert store.delete(B, t) is False and store.count(A, t) == 1  # the owner's thread is untouched


def test_deleting_a_thread_removes_it_and_its_messages(store):
    t = store.create(A)
    store.append(A, t, "user", "x")
    assert store.delete(A, t) is True
    with pytest.raises(NotFound):
        store.messages(A, t)
    assert store._db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_deleting_a_user_removes_all_their_threads_and_only_theirs(store):
    ta1, ta2, tb = store.create(A), store.create(A), store.create(B)
    store.append(A, ta1, "user", "x")
    store.append(B, tb, "user", "y")
    assert store.delete_user(A) == 2
    assert store.count(B, tb) == 1
    with pytest.raises(NotFound):
        store.messages(A, ta2)


def test_threads_survive_closing_and_reopening_a_file(tmp_path):
    s = ThreadStore(tmp_path / "t.db")
    t = s.create(A)
    s.append(A, t, "user", "remember me")
    s.close()
    s2 = ThreadStore(tmp_path / "t.db")
    assert s2.messages(A, t)[0]["content"] == "remember me"
    s2.close()


def test_a_deleted_thread_is_gone_from_the_database_file_not_just_hidden(tmp_path):
    s = ThreadStore(tmp_path / "t.db", max_chars=100)
    t = s.create(A)
    s.append(A, t, "user", "ZEBRA-MARKER-8842")
    s.close()
    assert b"ZEBRA-MARKER-8842" in (tmp_path / "t.db").read_bytes()  # the control: it is in the file before deletion
    s = ThreadStore(tmp_path / "t.db")
    s.delete(A, t)
    s.close()
    assert b"ZEBRA-MARKER-8842" not in (tmp_path / "t.db").read_bytes()


def test_scope_needs_a_real_user_id_and_a_hostile_one_is_just_a_string(store):
    for bad in ("", None, "x" * 201):
        with pytest.raises(ValueError):
            Scope(bad)
    evil = Scope("x' OR '1'='1")
    t = store.create(A)
    with pytest.raises(NotFound):
        store.messages(evil, t)


def test_ping_reports_a_working_store(store):
    assert store.ping() is True
