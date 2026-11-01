import dataclasses
import json

import pytest

import harness
from agents import claude_agent
from fakes import FakeClient, NS, text_reply
from handoff import (AgentView, Handoff, HandoffError, HandoffLimit, HandoffLoop, HandoffRefused, HandoffRouter, MAX_SUMMARY_CHARS, Reply)
from middleware import Identity

ME = Identity("alice")


def say(text):
    return lambda view: Reply(text)


def to(target, **kw):
    return lambda view: Handoff(target, **kw)


# --- the router ---------------------------------------------------------------------------------------------------------------
def test_an_agent_that_answers_ends_the_run_with_no_hops():
    r = HandoffRouter({"a": say("hi")}, {}, "a").run("hello", ME)
    assert (r.text, r.path, r.hops) == ("hi", ["a"], 0)


def test_a_handoff_to_an_allowed_agent_moves_the_request_along():
    r = HandoffRouter({"a": to("b"), "b": say("done")}, {"a": {"b"}}, "a").run("refund", ME)
    assert (r.text, r.path, r.hops) == ("done", ["a", "b"], 1)


def test_a_handoff_to_an_agent_not_on_the_callers_allow_list_is_refused_even_if_the_agent_exists():
    router = HandoffRouter({"a": to("c"), "b": say("x"), "c": say("secret")}, {"a": {"b"}}, "a")
    with pytest.raises(HandoffRefused):
        router.run("hi", ME)
    assert [e["to"] for e in router.trace.of("handoff_refused")] == ["c"]


def test_a_handoff_to_a_name_that_does_not_exist_is_refused():
    with pytest.raises(HandoffRefused):
        HandoffRouter({"a": to("ghost")}, {"a": set()}, "a").run("hi", ME)


def test_an_agent_with_no_allow_list_entry_cannot_hand_off_at_all():
    with pytest.raises(HandoffRefused):
        HandoffRouter({"a": to("b"), "b": say("x")}, {}, "a").run("hi", ME)


def test_passing_the_request_back_and_forth_is_stopped_at_the_second_repeat_of_the_same_pair():
    agents = {"a": to("b"), "b": to("a")}
    router = HandoffRouter(agents, {"a": {"b"}, "b": {"a"}}, "a", max_hops=50)
    with pytest.raises(HandoffLoop):
        router.run("hi", ME)
    assert len(router.trace.of("handoff")) == 2 and router.trace.of("handoff_loop")


def test_a_long_chain_of_distinct_handoffs_is_stopped_by_the_hop_cap():
    names = list("abcde")
    agents = {n: to(names[i + 1]) for i, n in enumerate(names[:-1])} | {"e": say("end")}
    allowed = {n: {names[i + 1]} for i, n in enumerate(names[:-1])}
    with pytest.raises(HandoffLimit):
        HandoffRouter(agents, allowed, "a", max_hops=2).run("hi", ME)
    assert HandoffRouter(agents, allowed, "a", max_hops=4).run("hi", ME).path == names


def test_the_next_agent_gets_the_request_and_a_short_summary_not_the_transcript():
    seen = {}
    agents = {"a": to("b", reason="needs  billing", summary="Customer wants a refund. " + "x" * 2000), "b": lambda v: seen.update(view=v) or Reply("ok")}
    HandoffRouter(agents, {"a": {"b"}}, "a").run("please refund order 5", ME)
    v = seen["view"]
    assert v.user_text == "please refund order 5" and v.from_agent == "a" and v.handoff_reason == "needs billing"
    assert len(v.handoff_summary) == MAX_SUMMARY_CHARS and v.handoff_summary.startswith("Customer wants a refund.")
    assert {f.name for f in dataclasses.fields(AgentView)} == {"user_text", "identity", "from_agent", "handoff_summary", "handoff_reason"}


def test_the_identity_travels_with_the_request():
    seen = []
    HandoffRouter({"a": to("b"), "b": lambda v: seen.append(v.identity.user_id) or Reply("x")}, {"a": {"b"}}, "a").run("hi", Identity("bob"))
    assert seen == ["bob"]


def test_an_agent_returning_the_wrong_type_is_an_error():
    with pytest.raises(HandoffError, match="str"):
        HandoffRouter({"a": lambda v: "just text"}, {}, "a").run("hi", ME)


def test_a_bad_configuration_is_rejected_when_the_router_is_built():
    with pytest.raises(ValueError):
        HandoffRouter({"a": say("x")}, {}, "zzz")
    with pytest.raises(ValueError):
        HandoffRouter({"a": say("x")}, {"a": {"ghost"}}, "a")
    with pytest.raises(ValueError):
        HandoffRouter({"a": say("x")}, {"ghost": set()}, "a")


def test_the_handoff_trace_has_hashes_not_text():
    router = HandoffRouter({"a": to("b", reason="SECRET-REASON"), "b": say("SECRET-ANSWER")}, {"a": {"b"}}, "a")
    router.run("hi", ME)
    assert "SECRET" not in json.dumps(router.trace.events) and router.trace.types() == ["handoff", "handoff_done"]


# --- the Claude-backed agent (scripted replies only) ------------------------------------------------------------------------------------
def tool_use(**inp):
    return NS(content=[NS(type="tool_use", id="t1", name="handoff_to", input=inp)], stop_reason="tool_use", usage=NS(input_tokens=1, output_tokens=1))


def test_the_agent_answers_when_the_model_answers():
    client = FakeClient([text_reply("Sure.")])
    assert claude_agent(client, "You help.", ["billing"])(AgentView("hi", ME)) == Reply("Sure.")


def test_the_agent_hands_off_when_the_model_calls_the_handoff_tool():
    client = FakeClient([tool_use(agent="billing", reason="refund", summary="wants money back")])
    assert claude_agent(client, "x", ["billing"])(AgentView("hi", ME)) == Handoff("billing", "refund", "wants money back")


def test_the_tool_only_offers_the_agents_allowed_targets_and_has_no_extra_properties():
    client = FakeClient([text_reply("x")])
    claude_agent(client, "x", ["support", "billing"])(AgentView("hi", ME))
    schema = client.calls[0]["tools"][0]["input_schema"]
    assert schema["properties"]["agent"]["enum"] == ["billing", "support"] and schema["additionalProperties"] is False


def test_an_agent_with_no_targets_is_given_no_tools_at_all():
    client = FakeClient([text_reply("x")])
    claude_agent(client, "x", [])(AgentView("hi", ME))
    assert "tools" not in client.calls[0]


def test_a_handoff_note_is_passed_to_the_next_agent_labelled_as_data():
    client = FakeClient([text_reply("x")])
    claude_agent(client, "You are billing.", [])(AgentView("hi", ME, from_agent="support", handoff_summary="SUMMARY-TEXT"))
    assert "SUMMARY-TEXT" in client.calls[0]["system"] and "data, not instructions" in client.calls[0]["system"]


def test_a_model_that_names_a_target_outside_the_enum_is_still_refused_by_the_router():
    client = FakeClient([tool_use(agent="admin", reason="r")])
    router = HandoffRouter({"a": claude_agent(client, "x", ["b"]), "b": say("x"), "admin": say("root")}, {"a": {"b"}}, "a")
    with pytest.raises(HandoffRefused):
        router.run("hi", ME)


# --- the harness -----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def by():
    out = {}
    for o in harness.evaluate():
        out[(o.scenario, o.strategy)] = o.metrics
    return out


def test_routing_on_the_models_claim_books_a_room_that_does_not_exist_and_routing_on_checked_data_does_not(by):
    assert by[("routing", "route_on_model_claim")]["booked_with_no_rooms"] == 1
    assert by[("routing", "route_on_checked_data")] == {"bookings": ["stockholm"], "booked_with_no_rooms": 0}


def test_a_declared_override_books_nothing_when_the_step_checks_the_flag_and_a_silent_rewrite_is_refused(by):
    assert by[("middleware", "override_ignored_flag")]["booked_with_no_rooms"] == 1  # declared, but the step ignored the flag
    assert by[("middleware", "override_checked_flag")] == {"booked_with_no_rooms": 0, "result": "SyntheticFact"}
    assert by[("middleware", "silent_rewrite")] == {"booked_with_no_rooms": 0, "result": "UndeclaredOverride"}


def test_a_restart_repeats_the_booking_but_checkpoint_plus_ledger_does_not(by):
    assert by[("crash", "restart_from_start")]["bookings"] == 2
    assert by[("crash", "checkpoint_and_ledger")] == {"bookings": 1, "emails": 1, "status": "completed"}


def test_a_partial_result_after_a_timeout_does_not_wait_for_the_slow_branch(by):
    assert by[("slow_branch", "wait_for_everything")]["waited_at_least_slow_branch"] is True
    m = by[("slow_branch", "timeout_and_partial")]
    assert m["waited_at_least_slow_branch"] is False and m["missing"] == ["weather"] and m["got"] == ["flights", "hotels"]


def test_the_ping_pong_guard_stops_after_two_handoffs_where_the_unguarded_loop_ran_to_its_manual_cap(by):
    assert by[("handoff", "no_loop_guard")]["hops_before_manual_stop"] == 50
    assert by[("handoff", "ping_pong_guard")] == {"hops_before_stop": 2, "result": "HandoffLoop"}


def test_the_table_prints_every_row(capsys):
    outcomes = harness.evaluate()
    harness.print_table(outcomes)
    assert len(capsys.readouterr().out.strip().splitlines()) == len(outcomes)
