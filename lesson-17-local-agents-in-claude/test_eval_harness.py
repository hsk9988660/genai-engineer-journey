import json

import pytest

import evaluate_local
import harness
from agentloop import ModelUnavailable, ToolDef
from evalharness import EvalReport, ToolCase, classify, run_tool_eval, wilson_interval
from fakes import ScriptedChat, chat_reply
from fakeserver import FakeLocalServer
from retrieval import BM25Index
from sandbox import Sandbox
from tools import build_tools

TOOLS = {"read_file": ToolDef("read_file", "", {"type": "object", "properties": {"path": {"type": "string", "maxLength": 50}}, "required": ["path"]}, lambda path: path),
         "list_files": ToolDef("list_files", "", {"type": "object", "properties": {}}, lambda: "")}
CASE = ToolCase("show auth.py", "read_file", {"path": "auth.py"})


def msg(*calls, content=None):
    return chat_reply(content, calls)["choices"][0]["message"]


# --- classifying one reply ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("message,expected", [
    (msg(("read_file", '{"path": "auth.py"}')), "correct"),
    (msg(("read_file", '{"path": "AUTH.PY "}')), "correct"),
    (msg(("read_file", '{"path": "utils.py"}')), "wrong_args"),
    (msg(("list_files", "{}")), "wrong_tool"),
    (msg(("read_file", '{"path": "auth.py", "encoding": "utf8"}')), "invalid_args"),
    (msg(("read_file", "{}")), "invalid_args"),
    (msg(("read_file", '{"path": "auth.py"')), "bad_json"),
    (msg(("readfile", '{"path": "auth.py"}')), "unknown_tool"),
    (msg(content="Sure, here is the file."), "no_call"),
])
def test_every_reply_falls_into_exactly_one_outcome(message, expected):
    assert classify(message, CASE, TOOLS) == expected


def test_a_wildcard_argument_accepts_any_non_empty_string_but_not_a_missing_or_blank_one():
    tools = {"search_docs": ToolDef("search_docs", "", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, lambda query: "")}
    case = ToolCase("q", "search_docs", {"query": "*"})
    assert classify(msg(("search_docs", '{"query": "passwords"}')), case, tools) == "correct"
    assert classify(msg(("search_docs", '{"query": "  "}')), case, tools) == "wrong_args"
    assert classify(msg(("search_docs", "{}")), case, tools) == "invalid_args"


def test_extra_expected_or_missing_keys_make_the_arguments_wrong():
    case = ToolCase("q", "read_file", {"path": "a"})
    assert classify(msg(("read_file", '{"path": "a"}')), case, TOOLS) == "correct"
    assert classify(msg(("list_files", "{}")), ToolCase("q", "list_files", {}), TOOLS) == "correct"


def test_a_call_written_as_text_counts_only_if_the_evaluation_accepts_text_calls():
    text = msg(content='<tool_call>\n{"name": "read_file", "arguments": {"path": "auth.py"}}\n</tool_call>')
    assert classify(text, CASE, TOOLS) == "no_call" and classify(text, CASE, TOOLS, accept_text=True) == "correct"


# --- running an evaluation ---------------------------------------------------------------------------------------------------------
def test_attempts_are_counted_by_outcome_and_by_prompt():
    replies = [chat_reply(calls=[("read_file", '{"path": "auth.py"}')])] * 3 + [chat_reply(calls=[("list_files", "{}")]), chat_reply("words")]
    rep = run_tool_eval(ScriptedChat(replies), "m", TOOLS, [CASE], repeats=5, warmup=0)
    assert rep.attempts == 5 and rep.counts["correct"] == 3 and rep.counts["wrong_tool"] == 1 and rep.counts["no_call"] == 1 and sum(rep.counts.values()) == 5
    assert rep.per_case["show auth.py"]["correct"] == 3


def test_server_failures_are_counted_as_errors_not_raised():
    rep = run_tool_eval(ScriptedChat([ModelUnavailable("down"), {"choices": []}, chat_reply(calls=[("read_file", '{"path": "auth.py"}')])]), "m", TOOLS, [CASE], repeats=3, warmup=0)
    assert rep.counts["error"] == 2 and rep.counts["correct"] == 1


def test_rates_come_with_wilson_intervals_and_valid_calls_include_the_wrong_ones():
    replies = [chat_reply(calls=[("read_file", '{"path": "auth.py"}')])] * 8 + [chat_reply(calls=[("read_file", '{"path": "x"}')]), chat_reply(calls=[("read_file", "{")])]
    rep = run_tool_eval(ScriptedChat(replies), "m", TOOLS, [CASE], repeats=10, warmup=0)
    rate, lo, hi = rep.correct
    assert rate == 0.8 and (round(lo, 3), round(hi, 3)) == tuple(round(x, 3) for x in wilson_interval(8, 10)) and lo < 0.8 < hi
    assert rep.valid_call[0] == 0.9


def test_more_attempts_give_a_narrower_interval_for_the_same_observed_rate():
    a, b = wilson_interval(8, 10), wilson_interval(80, 100)
    assert (b[1] - b[0]) < (a[1] - a[0]) and wilson_interval(0, 0) == (0.0, 1.0)


def test_latency_is_summarised_with_the_first_runs_discarded():
    ticks = iter(range(0, 1000, 2))
    rep = run_tool_eval(ScriptedChat([chat_reply(calls=[("list_files", "{}")], completion_tokens=20)] * 4), "m", TOOLS, [ToolCase("q", "list_files", {})], repeats=4, warmup=1, clock=lambda: next(ticks))
    assert rep.latency["n"] == 3 and rep.latency["discarded"] == 1 and rep.latency["median_s"] == 2 and rep.latency["tokens_per_s_median"] == 10.0


def test_an_empty_report_has_zero_rates():
    assert EvalReport(0, {}, {}).correct[0] == 0.0


# --- the command line against a scripted server -------------------------------------------------------------------------------------
def expected_call_for(prompt: str):
    table = {"What files": ("list_files", "{}"), "Show me the contents": ("read_file", '{"path": "auth.py"}'), "How many functions": ("analyze_code", '{"path": "auth.py"}'),
             "Find all the TODO": ("find_todos", "{}"), "documentation": ("search_docs", '{"query": "passwords"}')}
    return next(v for k, v in table.items() if k in prompt)


def test_the_script_measures_a_model_that_always_calls_the_right_tool(capsys):
    def respond(req, i):
        prompt = req["messages"][-1]["content"]
        return 200, chat_reply(calls=[expected_call_for(prompt)]), 0
    with FakeLocalServer(respond) as srv:
        code = evaluate_local.main(["--base-url", srv.url, "--model", "qwen", "--repeats", "2"])
    out = capsys.readouterr().out
    assert code == 0 and "attempts: 10" in out and "correct call: 100%" in out and "95% interval" in out and "latency:" in out


def test_the_script_reports_a_model_that_never_calls_a_tool(capsys):
    with FakeLocalServer(lambda req, i: (200, chat_reply("I would rather talk."), 0)) as srv:
        evaluate_local.main(["--base-url", srv.url, "--model", "m", "--repeats", "2"])
    out = capsys.readouterr().out
    assert "no_call" in out and "correct call: 0%" in out


def test_the_script_refuses_a_non_local_address_unless_told_otherwise(capsys):
    assert evaluate_local.main(["--base-url", "https://api.example.com/v1", "--model", "m"]) == 2
    assert "this machine" in capsys.readouterr().err


def test_a_server_that_fails_every_request_is_reported_as_errors_not_a_crash(capsys):
    with FakeLocalServer(lambda req, i: (500, {"error": "x"}, 0)) as srv:
        assert evaluate_local.main(["--base-url", srv.url, "--model", "m", "--repeats", "1"]) == 0
    assert "error" in capsys.readouterr().out


# --- the scenario harness -----------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def by():
    return {(o.scenario, o.strategy): o.metrics for o in harness.evaluate()}


def test_the_lessons_loop_crashes_on_the_first_broken_tool_call_and_the_hardened_loop_recovers(by):
    assert by[("malformed_calls", "lesson_loop")]["result"] == "crashed: JSONDecodeError"
    h = by[("malformed_calls", "hardened_loop")]
    assert h == {"result": "answered", "invalid_calls": 3, "repairs": 3, "valid_calls": 1}


def test_a_call_written_as_text_is_taken_for_the_final_answer_by_the_lessons_loop(by):
    assert by[("text_format_call", "lesson_loop")]["final_answer_is_raw_tag"] is True
    assert by[("text_format_call", "hardened_loop")] == {"final_answer_is_raw_tag": False, "tool_calls_run": 1}


def test_the_lessons_file_tool_returns_secrets_crashes_on_binary_and_reads_a_whole_big_file(by):
    n = by[("file_probes", "lesson_read_file")]
    assert n["secrets_returned_unflagged"] == 2 and n["crashes"] == 1 and n["chars_from_big_file"] == 240_000
    assert n["per_probe"][".env"].endswith("(SECRET, unflagged)") and n["per_probe"]["blob.bin"] == "crashed: UnicodeDecodeError"


def test_the_sandbox_denies_env_flags_the_config_password_caps_the_big_file_and_survives_binary(by):
    s = by[("file_probes", "sandbox")]
    assert s["secrets_returned_unflagged"] == 0 and s["secrets_returned_but_flagged"] == 1 and s["crashes"] == 0 and s["chars_from_big_file"] < 21_000
    assert s["per_probe"][".env"] == "denied" and s["per_probe"]["../outside.txt"] == "denied" and "(secret, flagged)" in s["per_probe"]["config.txt"]


def test_without_a_threshold_every_unrelated_question_gets_documents_and_with_one_none_do(by):
    a, b = by[("retrieval", "always_top_k (the lesson's behavior)")], by[("retrieval", "coverage_threshold")]
    assert a["unrelated_given_documents"] == 3 and b["unrelated_given_documents"] == 0 and a["related_answered"] == b["related_answered"] == 2


def test_routing_by_request_alone_sends_the_password_to_the_cloud_and_taint_tracking_does_not(by):
    a, b = by[("hybrid", "route_by_request_only")], by[("hybrid", "taint_tracking")]
    assert a["secret_sent_to_cloud"] and a["second_question_served_by"] == "cloud"
    assert not b["secret_sent_to_cloud"] and b["second_question_served_by"] == "local" and b["cloud_calls"] == 0


def test_a_gullible_model_reaches_the_secret_through_the_lessons_tools_but_not_through_the_sandbox(by):
    assert by[("injection", "lesson_tools")]["secret_reached_the_model"] is True
    s = by[("injection", "sandboxed_tools")]
    assert s["secret_reached_the_model"] is False and s["the_model_obeyed_and_was_refused"] is True and s["tainted"] is False


def test_a_7b_model_does_not_fit_at_16_bit_on_these_machines_and_four_bit_fits_with_room_for_a_long_conversation(by):
    eight, sixteen = by[("resources", "8GB_machine_longest_conversation_tokens")], by[("resources", "16GB_machine_longest_conversation_tokens")]
    assert eight["fp16"] == 0 and sixteen["fp16"] == 0 and eight["int8"] == 0 and sixteen["int8"] == 32768 and 20_000 < eight["int4"] < 32768
    big = by[("resources", "int4_at_32k_tokens_on_8GB")]
    assert big["fits"] is False and big["weights_gb"] == 3.8 and big["kv_cache_gb"] == pytest.approx(1.88, abs=0.01)


def test_ten_attempts_cannot_pin_down_a_rate_but_two_hundred_get_close(by):
    small, large = by[("eval", "10_attempts_true_rate_80%")], by[("eval", "200_attempts_true_rate_80%")]
    assert small["width_points"] > 30 and large["width_points"] < 15


def test_the_table_prints_every_row(capsys):
    outcomes = harness.evaluate()
    harness.print_table(outcomes)
    assert len(capsys.readouterr().out.strip().splitlines()) == len(outcomes)


def test_an_optional_argument_the_case_did_not_expect_makes_the_call_wrong_even_though_it_is_valid():
    tools = {"read_file": ToolDef("read_file", "", {"type": "object", "properties": {"path": {"type": "string"}, "tail": {"type": "integer"}}, "required": ["path"]}, lambda **k: "")}
    case = ToolCase("q", "read_file", {"path": "a"})
    assert classify(msg(("read_file", '{"path": "a"}')), case, tools) == "correct"
    assert classify(msg(("read_file", '{"path": "a", "tail": 5}')), case, tools) == "wrong_args"
