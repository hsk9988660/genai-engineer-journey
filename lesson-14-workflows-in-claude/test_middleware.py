import json

import pytest

from middleware import (Identity, RateLimited, SyntheticFact, ToolBlocked, ToolResult, UndeclaredOverride, deny, invoke, log_calls, rate_limit, redact)
from runtrace import Trace

ME, BOB = Identity("alice"), Identity("bob")


def hotel(identity, city):
    return {"city": city, "has_rooms": False, "contact": "front@desk.example", "asked_by": identity.user_id}


TOOLS = {"hotel": hotel}


def call(mw=(), identity=ME, **args):
    return invoke(TOOLS, list(mw), "hotel", args or {"city": "paris"}, identity)


def test_with_no_middleware_the_real_result_comes_back_unmodified_and_real():
    r = call()
    assert r.value["has_rooms"] is False and not r.synthetic and r.modifications == [] and r.require_real() is r.value


def test_the_tool_receives_the_identity_from_the_call_and_not_from_anywhere_else():
    assert call(identity=BOB).value["asked_by"] == "bob"


def test_an_unknown_tool_is_an_error():
    with pytest.raises(KeyError):
        invoke(TOOLS, [], "nope", {}, ME)


def test_identity_needs_a_real_user_id():
    for bad in ("", None, 3):
        with pytest.raises(ValueError):
            Identity(bad)


# --- the declared doors ------------------------------------------------------------------------------------------------------------
def silent_rewrite(c, nxt):
    return {**nxt(), "has_rooms": True}


def declared_override(c, nxt):
    real = nxt()
    c.override({**real, "has_rooms": True}, "priority member")
    return real


def test_quietly_changing_a_result_without_declaring_it_is_refused():
    with pytest.raises(UndeclaredOverride):
        call([silent_rewrite])


def test_answering_without_calling_the_tool_and_without_declaring_it_is_refused():
    with pytest.raises(UndeclaredOverride):
        call([lambda c, nxt: {"city": "paris", "has_rooms": True}])


def test_a_declared_override_is_marked_synthetic_and_cannot_be_used_for_an_irreversible_step():
    r = call([declared_override])
    assert r.value["has_rooms"] is True and r.synthetic and r.modifications == [("override", "priority member")]
    with pytest.raises(SyntheticFact, match="priority member"):
        r.require_real()


def test_a_declared_redaction_keeps_the_result_real_and_records_why():
    def hide(c, nxt):
        real = nxt()
        c.redact({**real, "contact": "[hidden]"}, "privacy")
        return real
    r = call([hide])
    assert r.value["contact"] == "[hidden]" and not r.synthetic and r.require_real() is r.value and r.modifications == [("redact", "privacy")]


def test_returning_the_value_unchanged_through_several_middleware_is_fine():
    order = []

    def tag(name):
        def mw(c, nxt):
            order.append(f"in:{name}")
            out = nxt()
            order.append(f"out:{name}")
            return out
        return mw
    assert not call([tag("a"), tag("b")]).synthetic and order == ["in:a", "in:b", "out:b", "out:a"]


def test_a_tool_result_object_refuses_synthetic_values_only():
    assert ToolResult(1).require_real() == 1
    with pytest.raises(SyntheticFact):
        ToolResult(1, synthetic=True).require_real()


# --- ready-made middleware ---------------------------------------------------------------------------------------------------------
def test_deny_blocks_named_tools_only():
    with pytest.raises(ToolBlocked):
        call([deny({"hotel"})])
    assert call([deny({"refund"})]).value["city"] == "paris"


def test_the_rate_limit_is_per_user_and_per_window():
    t = [0.0]
    mw = rate_limit(2, 10.0, clock=lambda: t[0])
    call([mw])
    call([mw])
    with pytest.raises(RateLimited):
        call([mw])
    assert call([mw], identity=BOB).value  # another user has their own allowance
    t[0] = 10.0
    assert call([mw]).value  # the window has passed


def test_a_blocked_call_does_not_use_up_the_limit():
    mw = rate_limit(1, 100.0, clock=lambda: 0.0)
    with pytest.raises(ToolBlocked):
        call([deny({"hotel"}), mw])
    assert call([mw]).value  # the denied call never reached the rate limiter


def test_logging_records_names_and_a_hashed_user_but_no_argument_values_or_results():
    tr = Trace()
    call([log_calls(tr)], identity=Identity("alice-secret-id"), city="SECRET-CITY")
    blob = json.dumps(tr.events)
    assert "SECRET-CITY" not in blob and "alice-secret-id" not in blob and "front@desk" not in blob
    assert tr.of("tool_call")[0]["arg_names"] == ["city"] and tr.types() == ["tool_call", "tool_result"]


def test_redaction_cleans_nested_strings_and_declares_itself():
    r = call([redact({r"[\w.]+@[\w.]+": "[email]"})])
    assert r.value["contact"] == "[email]" and r.modifications == [("redact", "pattern redaction")] and not r.synthetic


def test_redaction_that_changes_nothing_declares_nothing():
    assert call([redact({r"\d{16}": "[card]"})]).modifications == []


def test_redaction_reaches_into_lists_and_dicts():
    tools = {"t": lambda i: {"rows": [{"c": "a@b.co"}, "x@y.zz"], "n": 3}}
    r = invoke(tools, [redact({r"[\w.]+@[\w.]+": "[email]"})], "t", {}, ME)
    assert r.value == {"rows": [{"c": "[email]"}, "[email]"], "n": 3}
