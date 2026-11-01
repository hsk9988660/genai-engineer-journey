import json

from fakes import ScriptedClient, last_tool_results, submit
from router import SUPPORT_AGENTS, run_support
from runtime import Budget, Runtime


def go(queues, agents=None, **kw):
    client = ScriptedClient(queues)
    rt = Runtime(client, budget=kw.pop("budget", None))
    return run_support(rt, "I was charged twice this month", agents or SUPPORT_AGENTS, **kw), client, rt


def handoff(to, note="customer was charged twice"):
    return submit({"action": "handoff", "content": note, "to": to})


def answer(text="Refund issued."):
    return submit({"action": "answer", "content": text})


def test_triage_hands_off_to_a_specialist_who_answers():
    out, client, rt = go({"triage": [handoff("billing")], "billing": [answer()]})
    assert out.status == "answered" and out.answer == "Refund issued." and out.path == ["triage", "billing"] and out.hops == 1


def test_a_simple_question_is_answered_by_triage_with_no_handoff():
    out, client, rt = go({"triage": [answer("Our hours are 9 to 5.")]})
    assert out.status == "answered" and out.path == ["triage"] and out.hops == 0


def test_the_specialist_gets_the_customers_message_and_a_note_not_the_previous_agents_prompt():
    out, client, rt = go({"triage": [handoff("billing", "double charge on the March invoice")], "billing": [answer()]})
    seen = json.loads(client.calls_for("billing")[0]["messages"][0]["content"])
    assert seen == {"customer_message": "I was charged twice this month", "note_from_previous_agent": "double charge on the March invoice"}
    assert "greet the customer" not in client.calls_for("billing")[0]["messages"][0]["content"]  # triage's instructions
    assert "[ROLE: billing]" in client.calls_for("billing")[0]["system"]


def test_the_agent_is_told_who_it_may_still_hand_off_to():
    out, client, rt = go({"triage": [answer()]})
    assert "['triage'" not in client.calls[0]["system"] and "billing" in client.calls[0]["system"]
    out, client, rt = go({"triage": [handoff("billing")], "billing": [answer()]})
    assert "'triage'" not in client.calls_for("billing")[0]["system"].split("hand off to one of")[1]


def test_a_handoff_to_an_unknown_agent_is_rejected_and_corrected():
    out, client, rt = go({"triage": [handoff("sales"), handoff("billing")], "billing": [answer()]})
    assert out.status == "answered" and out.path == ["triage", "billing"]
    assert "Cannot hand off" in last_tool_results(client.calls_for("triage")[1])[0]["content"]


def test_ping_pong_is_blocked_the_conversation_never_returns_to_an_agent_that_had_it():
    out, client, rt = go({"triage": [handoff("billing")], "billing": [handoff("triage")] * 3})
    assert out.status == "escalated" and out.path == ["triage", "billing"]
    assert "could not complete a valid turn" in out.reason and "human" in out.answer
    assert len(client.calls_for("billing")) == 3  # the model was told no, twice, then the system escalated


def test_a_long_chain_is_cut_off_by_the_hop_cap_and_escalates_to_a_human():
    agents = {n: f"You are {n}." for n in ["a", "b", "c", "d", "e"]}
    out, client, rt = go({"a": [handoff("b")], "b": [handoff("c")], "c": [handoff("d")], "d": [answer()]},
                         agents=agents, start="a", max_hops=2)
    assert out.status == "escalated" and out.hops == 3 and "hop limit (2)" in out.reason and out.path == ["a", "b", "c"]
    assert client.calls_for("d") == []  # the fourth agent never ran


def test_the_budget_caps_a_support_conversation():
    out, client, rt = go({"triage": [handoff("billing")], "billing": [answer()]}, budget=Budget(max_calls=1))
    assert out.status == "over_budget" and "call limit" in out.reason


def test_the_trace_records_each_agent_in_the_conversation():
    out, client, rt = go({"triage": [handoff("billing")], "billing": [answer()]})
    assert [s.agent for s in rt.tracer.spans] == ["pipeline", "triage", "billing"]
    assert all(s.parent == 0 for s in rt.tracer.spans[1:]) and rt.tracer.spans[0].status == "ok"
