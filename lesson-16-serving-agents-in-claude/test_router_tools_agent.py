import threading
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

import support_tools
from agentdef import AgentCall, AgentResult
from claude_agent import MAX_TOOL_RESULT, classify_error, make_agent_fn
from fakes import FakeClient, text_reply
from harness import make_version
from resilience import Permanent, Transient
from router import ToolNotAllowed, cascade, default_check, enforce_tools, start_index

V = make_version()
SMALL, LARGE = V.tiers


# --- routing -------------------------------------------------------------------------------------------------------------------------------
def test_the_keyword_hint_only_decides_where_to_start():
    assert start_index(V, "I want a refund") == 1 and start_index(V, "Please CANCEL it") == 1 and start_index(V, "What is your return window?") == 0


def test_the_hint_is_absent_when_the_definition_has_none():
    assert start_index(make_version(top_keywords=()), "refund please") == 0


def test_default_check_wants_a_successful_non_empty_answer():
    assert default_check(AgentResult("hi")) and not default_check(AgentResult("")) and not default_check(AgentResult("  ")) and not default_check(AgentResult("x", ok=False))


def test_a_tool_the_tier_was_not_offered_is_refused_whatever_the_model_did():
    enforce_tools(SMALL, AgentResult("x", ("get_order_status",)))
    with pytest.raises(ToolNotAllowed):
        enforce_tools(SMALL, AgentResult("x", ("issue_refund",)))
    assert issubclass(ToolNotAllowed, Permanent)  # so neither retries nor the breaker treat it as an outage


def test_an_acceptable_answer_from_the_cheap_tier_ends_the_cascade():
    ran = []
    res, tier, passed, attempts = cascade(V, lambda t: ran.append(t.name) or AgentResult("fine"))
    assert (tier, passed, ran) == ("small", True, ["small"]) and attempts == [("small", res, True)]


def test_a_rejected_answer_escalates_to_the_next_tier():
    ran = []

    def run(t):
        ran.append(t.name)
        return AgentResult("no" if t.name == "small" else "yes")
    res, tier, passed, attempts = cascade(V, run, check=lambda r: r.text == "yes")
    assert (res.text, tier, passed, ran) == ("yes", "large", True, ["small", "large"]) and [a[2] for a in attempts] == [False, True]


def test_the_top_tiers_answer_is_returned_even_if_it_fails_the_check_and_says_so():
    res, tier, passed, _ = cascade(V, lambda t: AgentResult("meh"), check=lambda r: False)
    assert (tier, passed) == ("large", False)


def test_starting_at_the_top_skips_the_cheap_tier():
    ran = []
    cascade(V, lambda t: ran.append(t.name) or AgentResult("x"), start=1)
    assert ran == ["large"]


def test_a_cheap_tier_that_breaks_the_tool_rule_is_escalated_past_but_the_top_tier_doing_so_is_an_error():
    def run(t):
        if t.name == "small":
            raise ToolNotAllowed("no")
        return AgentResult("ok")
    res, tier, passed, attempts = cascade(V, run)
    assert tier == "large" and attempts[0] == ("small", None, False)
    with pytest.raises(ToolNotAllowed):
        cascade(V, lambda t: (_ for _ in ()).throw(ToolNotAllowed("no")))


# --- the support tools -------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def tools():
    data = support_tools.SupportData()
    return data, support_tools.build_tools(data)


def test_a_customer_sees_their_own_order(tools):
    _, t = tools
    assert t["get_order_status"]("cust-42", "A1002") == "Order A1002: status=processing, total=$128.50, eta=5 days." and "a1002" in t["get_order_status"]("cust-42", "a1002").lower()


def test_someone_elses_order_answers_exactly_like_one_that_does_not_exist(tools):
    _, t = tools
    assert t["get_order_status"]("cust-99", "A1002") == t["get_order_status"]("cust-99", "Z9999") == support_tools.NOT_FOUND


def test_policy_search_is_public_and_unknown_questions_say_so(tools):
    _, t = tools
    assert "30 days" in t["search_policies"]("anyone", "what is the returns window") and t["search_policies"]("anyone", "zzz") == "No matching policy found."


def test_a_refund_for_someone_elses_order_is_refused_and_changes_nothing(tools):
    data, t = tools
    assert t["issue_refund"]("cust-99", "A1002", 10) == support_tools.NOT_FOUND and data.refunded == {}


def test_a_small_refund_is_issued_without_a_human(tools):
    data, t = tools
    assert t["issue_refund"]("cust-99", "A1001", 30) == "Refund of $30.00 issued for order A1001." and data.refunded["A1001"] == Decimal("30.00")


def test_amounts_must_be_positive_numbers_within_what_is_left(tools):
    _, t = tools
    for bad in (0, -5, "abc", 42.01, 1000):
        assert "must be more than" in t["issue_refund"]("cust-99", "A1001", bad) or "valid amount" in t["issue_refund"]("cust-99", "A1001", bad)


def test_partial_refunds_reduce_what_can_still_be_refunded(tools):
    _, t = tools
    t["issue_refund"]("cust-99", "A1001", 30)
    assert "at most $12.00" in t["issue_refund"]("cust-99", "A1001", 20)
    assert "issued" in t["issue_refund"]("cust-99", "A1001", 12)


def test_repeating_the_same_refund_does_nothing_the_second_time(tools):
    data, t = tools
    t["issue_refund"]("cust-99", "A1001", 20)
    assert "already issued" in t["issue_refund"]("cust-99", "A1001", 20) and data.refunded["A1001"] == Decimal("20.00")


def test_two_identical_refunds_at_the_same_moment_issue_once(tools):
    data, t = tools
    barrier = threading.Barrier(8)

    def go():
        barrier.wait()
        t["issue_refund"]("cust-99", "A1001", 20)
    threads = [threading.Thread(target=go) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(2)
    assert data.refunded["A1001"] == Decimal("20.00")


def test_a_large_refund_needs_a_human_and_is_queued_when_nobody_answers_now():
    data = support_tools.SupportData()
    t = support_tools.build_tools(data)
    out = t["issue_refund"]("cust-42", "A1002", 100)
    assert "needs a human" in out and data.refunded == {} and data.pending == [{"order": "A1002", "amount": "100.00", "order_total": "128.50", "customer": "cust-42"}]


def test_a_large_refund_runs_only_if_the_approver_says_yes_and_a_no_stops_it():
    for verdict, expected in ((True, "issued"), (False, "declined")):
        data = support_tools.SupportData()
        t = support_tools.build_tools(data, approver=lambda s, v=verdict: v)
        assert expected in t["issue_refund"]("cust-42", "A1002", 100)
        assert (data.refunded == {}) == (verdict is False)


def test_the_approval_threshold_is_configurable():
    data = support_tools.SupportData()
    t = support_tools.build_tools(data, threshold=Decimal("10"))
    assert "needs a human" in t["issue_refund"]("cust-99", "A1001", 20)


def test_no_tool_schema_has_a_user_id_and_none_accepts_extra_properties():
    for s in support_tools.SCHEMAS.values():
        assert "user_id" not in s["input_schema"]["properties"] and s["input_schema"]["additionalProperties"] is False
    assert {s.name: s.scope for s in support_tools.SPECS} == {"search_policies": "public", "get_order_status": "user", "issue_refund": "effect"}


# --- the Claude adapter --------------------------------------------------------------------------------------------------------------
def tool_reply(name, tool_input, tid="t1"):
    return NS(content=[NS(type="tool_use", id=tid, name=name, input=tool_input)], stop_reason="tool_use", usage=NS(input_tokens=100, output_tokens=10))


def call_for(tier, text="hi", user="cust-42", ctx=""):
    return AgentCall(V, tier, [{"role": "user", "content": text}], ctx, 1e9, user)


def agent_with(replies):
    client = FakeClient(replies)
    data = support_tools.SupportData()
    return make_agent_fn(client, support_tools.build_tools(data), support_tools.SCHEMAS), client, data


class StatusError(Exception):
    def __init__(self, status, headers=None):
        super().__init__("err")
        self.status_code, self.response = status, NS(headers=headers or {})


@pytest.mark.parametrize("exc,kind", [
    (StatusError(429), Transient), (StatusError(500), Transient), (StatusError(503), Transient), (StatusError(408), Transient),
    (StatusError(400), Permanent), (StatusError(401), Permanent), (StatusError(404), Permanent), (ValueError("?"), Permanent)])
def test_provider_errors_are_split_into_worth_retrying_and_not(exc, kind):
    assert isinstance(classify_error(exc), kind)


def test_errors_are_recognised_by_class_name_as_well_as_status():
    class RateLimitError(Exception):
        pass
    class APIConnectionError(Exception):
        pass
    assert isinstance(classify_error(RateLimitError()), Transient) and isinstance(classify_error(APIConnectionError()), Transient)


def test_a_retry_after_header_is_passed_on_and_garbage_is_ignored():
    assert classify_error(StatusError(429, {"retry-after": "7"})).retry_after_s == 7.0
    assert classify_error(StatusError(429, {"retry-after": "soon"})).retry_after_s is None


def test_a_plain_answer_comes_back_with_its_token_counts():
    fn, client, _ = agent_with([text_reply("Hello!")])
    r = fn(call_for(SMALL))
    assert (r.text, r.tool_calls, r.input_tokens, r.output_tokens) == ("Hello!", (), 10, 5)


def test_only_the_tools_the_tier_is_offered_are_sent_to_the_model_and_none_if_it_has_none():
    fn, client, _ = agent_with([text_reply("a"), text_reply("b")])
    fn(call_for(SMALL))
    assert [t["name"] for t in client.calls[0]["tools"]] == ["search_policies", "get_order_status"]
    from agentdef import Tier
    fn(call_for(Tier("bare", "m", ())))
    assert "tools" not in client.calls[1]


def test_a_tool_call_runs_with_the_user_id_from_the_serving_code_and_the_result_goes_back_to_the_model():
    fn, client, _ = agent_with([tool_reply("get_order_status", {"order_id": "A1002"}), text_reply("It is processing.")])
    r = fn(call_for(SMALL, user="cust-42"))
    assert r.tool_calls == ("get_order_status",) and r.text == "It is processing." and (r.input_tokens, r.output_tokens) == (110, 15)
    sent = client.calls[1]["messages"][-1]["content"][0]
    assert sent["type"] == "tool_result" and "status=processing" in sent["content"]


def test_the_same_tool_call_for_another_user_finds_nothing_because_the_id_comes_from_code():
    fn, client, _ = agent_with([tool_reply("get_order_status", {"order_id": "A1002"}), text_reply("done")])
    fn(call_for(SMALL, user="cust-99"))
    assert client.calls[1]["messages"][-1]["content"][0]["content"] == support_tools.NOT_FOUND


def test_a_user_id_the_model_invents_is_rejected_as_a_bad_argument_not_obeyed():
    fn, client, _ = agent_with([tool_reply("get_order_status", {"order_id": "A1002", "user_id": "cust-42"}), text_reply("sorry")])
    fn(call_for(SMALL, user="cust-99"))
    assert "invalid arguments" in client.calls[1]["messages"][-1]["content"][0]["content"]


def test_a_tool_outside_the_tiers_list_stops_the_call():
    fn, _, data = agent_with([tool_reply("issue_refund", {"order_id": "A1001", "amount": 5})])
    with pytest.raises(ToolNotAllowed):
        fn(call_for(SMALL, user="cust-99"))
    assert data.refunded == {}


def test_the_top_tier_can_use_the_refund_tool():
    fn, _, data = agent_with([tool_reply("issue_refund", {"order_id": "A1001", "amount": 5}), text_reply("Refunded.")])
    r = fn(call_for(LARGE, user="cust-99"))
    assert r.tool_calls == ("issue_refund",) and data.refunded["A1001"] == Decimal("5.00")


def test_a_model_that_never_stops_calling_tools_is_cut_off():
    fn, _, _ = agent_with([tool_reply("search_policies", {"query": "x"}, f"t{i}") for i in range(10)])
    with pytest.raises(Permanent, match="too many"):
        fn(call_for(SMALL))


def test_provider_failures_become_transient_or_permanent_errors():
    class Boom:
        def __init__(self, exc):
            self.messages = NS(create=lambda **kw: (_ for _ in ()).throw(exc))
    for exc, kind in ((StatusError(429), Transient), (StatusError(400), Permanent)):
        fn = make_agent_fn(Boom(exc), {}, support_tools.SCHEMAS)
        with pytest.raises(kind):
            fn(call_for(SMALL))


def test_user_facts_travel_in_the_conversation_labelled_as_data_and_the_system_prompt_never_changes():
    fn, client, _ = agent_with([text_reply("a"), text_reply("b")])
    fn(call_for(SMALL, ctx="Dana, enterprise"))
    fn(call_for(SMALL, ctx=""))
    first, second = client.calls
    assert "Dana, enterprise" in first["messages"][-1]["content"] and "data, not instructions" in first["messages"][-1]["content"]
    assert first["system"] == second["system"] == V.prompt and "Dana" not in first["system"]


def test_a_huge_tool_result_is_cut_before_it_goes_back_to_the_model():
    client = FakeClient([tool_reply("search_policies", {"query": "x"}), text_reply("ok")])
    fn = make_agent_fn(client, {"search_policies": lambda uid, query: "z" * 10_000}, support_tools.SCHEMAS)
    fn(call_for(SMALL))
    assert len(client.calls[1]["messages"][-1]["content"][0]["content"]) == MAX_TOOL_RESULT


def test_a_second_identical_refund_cannot_slip_in_while_the_first_is_waiting_on_its_approval():
    """Deterministic race: the first caller is held INSIDE the approval step. Without the lock the second would finish its refund first and the first would then refund again."""
    inside, gate, calls = threading.Event(), threading.Event(), []

    def approver(summary):
        calls.append(1)
        if len(calls) == 1:
            inside.set()
            gate.wait(2)
        return True
    data = support_tools.SupportData()
    t = support_tools.build_tools(data, approver=approver)
    results = []
    first = threading.Thread(target=lambda: results.append(t["issue_refund"]("cust-42", "A1002", 100)))
    first.start()
    inside.wait(2)
    second = threading.Thread(target=lambda: results.append(t["issue_refund"]("cust-42", "A1002", 100)))
    second.start()
    second.join(0.2)  # with the lock it is still waiting here; without it, it has already finished
    gate.set()
    first.join(2)
    second.join(2)
    assert data.refunded["A1002"] == Decimal("100.00") and sorted(r.split(" ")[0] for r in results) == ["A", "Refund"]
