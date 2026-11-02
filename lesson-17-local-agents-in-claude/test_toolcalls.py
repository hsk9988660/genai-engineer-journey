import json

import pytest

from fakes import chat_reply
from toolcalls import BadCall, MAX_ARGS_CHARS, MAX_CALLS_PER_MESSAGE, ParsedCall, ToolCallError, extract_calls, parse_arguments, validate_args

SCHEMA = {"type": "object", "properties": {"path": {"type": "string", "maxLength": 20}, "n": {"type": "integer", "minimum": 1}, "x": {"type": "number", "exclusiveMinimum": 0},
                                           "flag": {"type": "boolean"}, "mode": {"type": "string", "enum": ["a", "b"]}}, "required": ["path"]}


# --- validating arguments ------------------------------------------------------------------------------------------------------------
def test_valid_arguments_have_no_problems_and_optional_ones_may_be_left_out():
    assert validate_args(SCHEMA, {"path": "auth.py"}) == [] and validate_args(SCHEMA, {"path": "a", "n": 3, "x": 0.5, "flag": True, "mode": "a"}) == []


def test_a_missing_required_argument_is_named():
    assert validate_args(SCHEMA, {}) == ["missing required argument 'path'"]


def test_an_argument_the_tool_does_not_have_is_always_a_problem_and_the_allowed_ones_are_listed():
    p = validate_args(SCHEMA, {"path": "a", "encoding": "utf8"})
    assert len(p) == 1 and "unexpected argument 'encoding'" in p[0] and "'path'" in p[0]


@pytest.mark.parametrize("args,fragment", [
    ({"path": 5}, "must be a string"), ({"path": "x" * 21}, "longer than 20"), ({"path": "a", "n": "3"}, "must be an integer"), ({"path": "a", "n": True}, "must be an integer"),
    ({"path": "a", "n": 1.5}, "must be an integer"), ({"path": "a", "n": 0}, "at least 1"), ({"path": "a", "x": "1"}, "must be a number"), ({"path": "a", "x": False}, "must be a number"),
    ({"path": "a", "x": 0}, "more than 0"), ({"path": "a", "flag": "yes"}, "true or false"), ({"path": "a", "mode": "c"}, "one of")])
def test_each_kind_of_wrong_value_is_reported(args, fragment):
    assert any(fragment in p for p in validate_args(SCHEMA, args))


def test_a_string_without_its_own_limit_is_capped_at_a_thousand_characters():
    s = {"type": "object", "properties": {"q": {"type": "string"}}}
    assert validate_args(s, {"q": "x" * 1000}) == [] and validate_args(s, {"q": "x" * 1001})


def test_arguments_that_are_not_an_object_are_rejected():
    for bad in ([], "text", 5, None):
        assert validate_args(SCHEMA, bad) == ["arguments must be a JSON object"]


def test_a_tool_with_no_parameters_accepts_only_an_empty_object():
    s = {"type": "object", "properties": {}}
    assert validate_args(s, {}) == [] and validate_args(s, {"a": 1})


# --- parsing the argument string ---------------------------------------------------------------------------------------------------
def test_none_and_empty_mean_no_arguments_and_an_object_passes_through():
    assert parse_arguments(None) == {} and parse_arguments("") == {} and parse_arguments({"a": 1}) == {"a": 1}


def test_json_text_becomes_a_dict():
    assert parse_arguments('{"path": "auth.py"}') == {"path": "auth.py"}


@pytest.mark.parametrize("raw", ['{"path": "auth.py"', "not json", "[1, 2]", '"just a string"', "5", 5, ["a"]])
def test_broken_or_non_object_arguments_raise_a_tool_call_error_not_a_json_error(raw):
    with pytest.raises(ToolCallError):
        parse_arguments(raw)


def test_oversized_arguments_are_refused_before_parsing():
    with pytest.raises(ToolCallError, match="longer than"):
        parse_arguments('{"a": "' + "x" * MAX_ARGS_CHARS + '"}')


# --- reading a whole message ---------------------------------------------------------------------------------------------------------
def msg(*calls, content=None):
    return chat_reply(content, calls)["choices"][0]["message"]


def test_a_structured_call_is_parsed_with_its_id_name_and_arguments():
    [c] = extract_calls(msg(("read_file", '{"path": "a.py"}')))
    assert c == ParsedCall("call_0", "read_file", {"path": "a.py"}, False)


def test_several_calls_in_one_message_come_back_in_order_and_are_capped():
    calls = [("t", "{}")] * (MAX_CALLS_PER_MESSAGE + 3)
    got = extract_calls(msg(*calls))
    assert len(got) == MAX_CALLS_PER_MESSAGE and [c.id for c in got] == [f"call_{i}" for i in range(MAX_CALLS_PER_MESSAGE)]


def test_broken_json_arguments_become_a_bad_call_that_keeps_the_name_and_id_so_it_can_be_answered():
    [c] = extract_calls(msg(("read_file", '{"path": ')))
    assert isinstance(c, BadCall) and c.name == "read_file" and c.id == "call_0" and "not valid JSON" in c.reason


def test_a_call_with_no_function_or_no_name_is_a_bad_call():
    m = {"tool_calls": [{"id": "x"}, {"id": "y", "function": {"arguments": "{}"}}, {"id": "z", "function": {"name": "", "arguments": "{}"}}, "junk"]}
    got = extract_calls(m)
    assert [type(c) for c in got] == [BadCall] * 4 and got[0].id == "x"


def test_a_missing_call_id_gets_a_generated_one():
    [c] = extract_calls({"tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]})
    assert c.id == "call_0"


def test_arguments_given_as_an_object_instead_of_a_string_are_accepted():
    [c] = extract_calls({"tool_calls": [{"id": "1", "function": {"name": "t", "arguments": {"a": 1}}}]})
    assert c.args == {"a": 1}


TEXT = '<tool_call>\n{"name": "read_file", "arguments": {"path": "auth.py"}}\n</tool_call>'


def test_a_call_written_in_the_message_text_is_parsed_and_marked_as_from_text():
    [c] = extract_calls({"content": "Let me look. " + TEXT})
    assert c == ParsedCall("text_0", "read_file", {"path": "auth.py"}, True)


def test_text_calls_are_ignored_when_not_asked_for():
    assert extract_calls({"content": TEXT}, accept_text=False) == []


def test_a_garbled_text_call_is_a_bad_call_not_an_exception():
    got = extract_calls({"content": '<tool_call>\n{"name": "x", "arguments": {broken}}\n</tool_call>'})
    assert got == [] or isinstance(got[0], BadCall)
    got = extract_calls({"content": '<tool_call>{"arguments": {}}</tool_call>'})
    assert isinstance(got[0], BadCall) and "no name" in got[0].reason


def test_a_structured_call_wins_over_text_in_the_same_message():
    got = extract_calls({"content": TEXT, "tool_calls": [{"id": "1", "function": {"name": "other", "arguments": "{}"}}]})
    assert [c.name for c in got] == ["other"]


def test_plain_text_and_odd_messages_give_no_calls_and_never_raise():
    for m in ({"content": "just an answer"}, {"content": None}, {}, {"tool_calls": []}, {"tool_calls": "nope"}, None, "string"):
        assert extract_calls(m) == []


def test_only_the_first_four_text_calls_in_a_message_are_read():
    text = "\n".join(f'<tool_call>{{"name": "t{i}", "arguments": {{}}}}</tool_call>' for i in range(MAX_CALLS_PER_MESSAGE + 3))
    got = extract_calls({"content": text})
    assert [c.name for c in got] == [f"t{i}" for i in range(MAX_CALLS_PER_MESSAGE)]
