import threading

import pytest
from pydantic import BaseModel

from fakes import ScriptedClient, last_tool_results, submit, tool_names, turn, use
from runtime import AgentError, Budget, BudgetExceeded, Runtime, Tracer


class Answer(BaseModel):
    text: str


def client(*replies, role="x"):
    return ScriptedClient({role: list(replies)})


def rt(c, **kw):
    return Runtime(c, **kw)


SYS = "[ROLE: x] be brief"
ECHO = {"name": "echo", "description": "echo", "input_schema": {"type": "object", "properties": {"v": {"type": "string"}}}}


def test_returns_a_validated_object_and_records_usage():
    r = rt(client(submit({"text": "hi"})))
    assert r.run_agent("a", SYS, "q", Answer).text == "hi"
    s = r.tracer.spans[0]
    assert (s.agent, s.status, s.input_tokens, s.output_tokens) == ("a", "ok", 100, 50)
    assert r.budget.calls == 1 and r.tracer.total_tokens() == 150


def test_validation_problems_are_sent_back_and_retried():
    c = client(submit({"text": "bad"}), submit({"text": "good"}))
    out = rt(c).run_agent("a", SYS, "q", Answer, validate=lambda o: [] if o.text == "good" else ["say good"])
    assert out.text == "good"
    assert "say good" in last_tool_results(c.calls[1])[0]["content"]


def test_schema_violations_are_reported_too():
    c = client(submit({"wrong": 1}), submit({"text": "ok"}))
    assert rt(c).run_agent("a", SYS, "q", Answer).text == "ok"
    assert "Schema errors" in last_tool_results(c.calls[1])[0]["content"]


def test_gives_up_after_max_retries_and_the_span_records_the_error():
    c = client(*[submit({"text": "bad"})] * 5)
    r = rt(c)
    with pytest.raises(AgentError):
        r.run_agent("a", SYS, "q", Answer, validate=lambda o: ["nope"], max_retries=2)
    assert len(c.calls) == 3 and r.tracer.spans[0].status == "error" and "nope" in r.tracer.spans[0].note


def test_tools_run_and_their_results_come_back():
    c = client(turn(use("t1", "echo", v="ping")), submit({"text": "done"}))
    r = rt(c)
    r.run_agent("a", SYS, "q", Answer, tools=[ECHO], tool_impls={"echo": lambda v: {"said": v}})
    assert "ping" in last_tool_results(c.calls[1])[0]["content"]


def test_unknown_tool_and_tool_exceptions_become_error_results():
    def boom(**kw):
        raise RuntimeError("kaput")

    c = client(turn(use("t1", "launch"), use("t2", "echo", v="x")), submit({"text": "done"}))
    rt(c).run_agent("a", SYS, "q", Answer, tools=[ECHO], tool_impls={"echo": boom})
    res = last_tool_results(c.calls[1])
    assert [r["is_error"] for r in res] == [True, True] and "No such tool" in res[0]["content"] and "kaput" in res[1]["content"]


def test_an_agent_sees_only_the_tools_it_was_given_plus_submit():
    c = client(submit({"text": "x"}))
    rt(c).run_agent("a", SYS, "q", Answer)
    assert tool_names(c.calls[0]) == ["submit_result"] and c.calls[0]["tool_choice"] == {"type": "any"}


def test_every_run_starts_with_a_fresh_context():
    c = client(submit({"text": "first"}), submit({"text": "second"}))
    r = rt(c)
    r.run_agent("a", SYS, "SECRET-ONE", Answer)
    r.run_agent("b", SYS, "question two", Answer)
    assert len(c.calls[1]["messages"]) == 1 and "SECRET-ONE" not in str(c.calls[1]["messages"])


def test_step_limit():
    c = client(*[turn(use(f"t{i}", "echo", v="x")) for i in range(9)])
    with pytest.raises(AgentError, match="step limit"):
        rt(c).run_agent("a", SYS, "q", Answer, tools=[ECHO], tool_impls={"echo": lambda v: v}, max_steps=3)


def test_an_agent_that_makes_no_tool_call_is_an_error():
    from types import SimpleNamespace as NS
    c = client(NS(content=[NS(type="text", text="hi")], stop_reason="end_turn", usage=NS(input_tokens=1, output_tokens=1)))
    with pytest.raises(AgentError, match="no tool call"):
        rt(c).run_agent("a", SYS, "q", Answer)


# --- budget -------------------------------------------------------------------------------------
def test_budget_stops_on_call_count_before_making_the_call():
    c = client(submit({"text": "a"}), submit({"text": "b"}))
    r = rt(c, budget=Budget(max_calls=1))
    r.run_agent("a", SYS, "q", Answer)
    with pytest.raises(BudgetExceeded, match="call limit"):
        r.run_agent("a", SYS, "q", Answer)
    assert len(c.calls) == 1


def test_budget_stops_on_spend():
    b = Budget(max_cost_usd=0.001, usd_per_mtok_in=1000.0, usd_per_mtok_out=1000.0)
    r = rt(client(submit({"text": "a"}), submit({"text": "b"})), budget=b)
    r.run_agent("a", SYS, "q", Answer)
    with pytest.raises(BudgetExceeded, match="spend limit"):
        r.run_agent("a", SYS, "q", Answer)


def test_the_call_cap_is_exact_when_many_agents_race_for_the_last_slots():
    b = Budget(max_calls=100)
    granted, lock = [], threading.Lock()

    def worker():
        for _ in range(50):
            try:
                b.begin_call()
            except BudgetExceeded:
                continue
            with lock:
                granted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(20)]  # 1000 attempts for 100 slots
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(granted) == 100 and b.calls == 100


def test_spend_is_summed_exactly_under_concurrent_charges():
    b = Budget()
    threads = [threading.Thread(target=lambda: [b.charge(10, 5) for _ in range(100)]) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert b.cost_usd == pytest.approx(2000 * (10 * 3 + 5 * 15) / 1e6)


# --- tracer -------------------------------------------------------------------------------------
def test_tracer_builds_a_tree_and_rolls_up_by_agent():
    now = [0.0]
    t = Tracer(clock=lambda: now[0])
    root = t.start("pipeline", "demo")
    a = t.start("reader", "task 1", root)
    t.add_usage(a, 100, 50, 0.01)
    now[0] = 0.25
    t.finish(a)
    b = t.start("reader", "task 2", root)
    t.add_usage(b, 10, 5, 0.002)
    t.finish(b, "error", "boom")
    t.finish(root)
    assert t.by_agent()["reader"] == {"runs": 2, "tokens": 165, "cost_usd": pytest.approx(0.012)}
    lines = t.render().splitlines()
    assert lines[0].startswith("pipeline demo") and lines[1].startswith("  reader task 1") and "250ms" in lines[1]
    assert "[error]" in lines[2] and "boom" in lines[2]


def test_tracer_is_safe_across_threads():
    t = Tracer()
    threads = [threading.Thread(target=lambda: [t.finish(t.start("w")) for _ in range(100)]) for _ in range(10)]
    [x.start() for x in threads]
    [x.join() for x in threads]
    assert len(t.spans) == 1000 and len({s.id for s in t.spans}) == 1000
