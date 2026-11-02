import time
from types import SimpleNamespace as NS

import pytest

from agentloop import LocalAgent
from fakes import FakeClient, ScriptedChat, chat_reply, text_reply
from hybrid import Answer, CloudError, ConversationState, HybridAssistant, claude_cloud_fn
from latency import LatencyStats
from resources import EXAMPLE_SHAPES, ModelShape, estimate, kv_cache_gb, max_context_tokens, weights_gb
from sandbox import Sandbox
from tools import build_tools

SECRET = "sk-testtesttesttest1234"
HARD = lambda q: "in depth" in q


@pytest.fixture
def proj(tmp_path):
    (tmp_path / "auth.py").write_text("def login(u, p):\n    return True\n", encoding="utf8")
    (tmp_path / "config.txt").write_text("password = hunter2hunter2\n", encoding="utf8")
    return tmp_path


def local(proj, replies):
    return LocalAgent(ScriptedChat(replies), "m", build_tools(Sandbox(proj)))


class Cloud:
    def __init__(self, fail=False):
        self.prompts, self.fail = [], fail

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise CloudError("down")
        return "cloud answer"


# --- who answers -------------------------------------------------------------------------------------------------------------------------
def test_an_easy_question_is_answered_locally_and_the_answer_says_so(proj):
    cloud = Cloud()
    a = HybridAssistant(local(proj, [chat_reply("local answer")]), cloud, HARD).ask("what is login?", ConversationState())
    assert a == Answer("local answer", "local", "simple or no cloud configured: local model", False, False, "answered") and cloud.prompts == []


def test_a_hard_question_in_a_clean_conversation_goes_to_the_cloud(proj):
    cloud = Cloud()
    a = HybridAssistant(local(proj, []), cloud, HARD).ask("explain in depth how login works", ConversationState())
    assert a.served_by == "cloud" and a.text == "cloud answer" and len(cloud.prompts) == 1 and "explain in depth" in cloud.prompts[0]


def test_with_no_cloud_configured_a_hard_question_is_answered_locally(proj):
    a = HybridAssistant(local(proj, [chat_reply("local")]), None, HARD).ask("explain in depth", ConversationState())
    assert a.served_by == "local"


def test_offline_means_local_even_for_a_hard_question(proj):
    cloud = Cloud()
    a = HybridAssistant(local(proj, [chat_reply("local")]), cloud, HARD, is_offline=lambda: True).ask("explain in depth", ConversationState())
    assert a.served_by == "local" and "offline" in a.reason and cloud.prompts == []


def test_when_the_cloud_fails_the_local_model_answers_and_the_answer_admits_it(proj):
    a = HybridAssistant(local(proj, [chat_reply("local fallback")]), Cloud(fail=True), HARD).ask("explain in depth", ConversationState())
    assert (a.served_by, a.fell_back, a.text) == ("local", True, "local fallback") and "failed" in a.reason


def test_if_the_local_model_cannot_finish_the_answer_says_so_and_names_nobody(proj):
    a = HybridAssistant(LocalAgent(ScriptedChat([RuntimeError]) if False else ScriptedChat([__import__("agentloop").ModelUnavailable("down")]), "m", {}), None).ask("hello", ConversationState())
    assert a.served_by == "none" and a.status == "error" and "could not finish" in a.text


# --- sensitivity ----------------------------------------------------------------------------------------------------------------------
def test_a_sensitive_question_taints_the_conversation_and_keeps_it_local(proj):
    cloud, state = Cloud(), ConversationState()
    a = HybridAssistant(local(proj, [chat_reply("local")]), cloud, HARD).ask(f"explain in depth why {SECRET} was rejected", state)
    assert a.served_by == "local" and a.tainted and state.tainted and "stays local" in a.reason and cloud.prompts == []


def test_personal_data_in_a_question_also_taints_it(proj):
    state = ConversationState()
    HybridAssistant(local(proj, [chat_reply("ok")]), Cloud(), HARD).ask("draft an email to jane.doe@example.com", state)
    assert state.tainted and "question itself" in state.reasons[0]


def test_a_tool_that_reads_a_secret_taints_the_conversation_so_the_next_hard_question_cannot_reach_the_cloud(proj):
    cloud, state = Cloud(), ConversationState()
    h = HybridAssistant(local(proj, [chat_reply(calls=[("read_file", '{"path": "config.txt"}')]), chat_reply("it sets a password"), chat_reply("local deep answer")]), cloud, HARD)
    first = h.ask("what is in config.txt?", state)
    second = h.ask("explain in depth how to secure it", state)
    assert first.tainted and second.served_by == "local" and cloud.prompts == [] and "tool returned" in state.reasons[0]


def test_taint_is_never_cleared_within_a_conversation(proj):
    state = ConversationState()
    state.taint("earlier secret")
    h = HybridAssistant(local(proj, [chat_reply("a"), chat_reply("b")]), Cloud(), HARD)
    assert h.ask("simple one", state).tainted and h.ask("another simple one", state).tainted and state.tainted


def test_reasons_are_not_repeated():
    s = ConversationState()
    s.taint("same")
    s.taint("same")
    assert s.reasons == ["same"]


def test_a_clean_tool_result_does_not_taint_and_the_cloud_stays_available(proj):
    cloud, state = Cloud(), ConversationState()
    h = HybridAssistant(local(proj, [chat_reply(calls=[("read_file", '{"path": "auth.py"}')]), chat_reply("it defines login")]), cloud, HARD)
    h.ask("what is in auth.py?", state)
    assert not state.tainted and h.ask("explain in depth", state).served_by == "cloud"


def test_the_cloud_sees_earlier_clean_questions_and_answers_but_never_tool_output(proj):
    cloud, state = Cloud(), ConversationState()
    h = HybridAssistant(local(proj, [chat_reply(calls=[("read_file", '{"path": "auth.py"}')]), chat_reply("it defines login")]), cloud, HARD)
    h.ask("what is in auth.py?", state)
    h.ask("explain in depth", state)
    assert "what is in auth.py?" in cloud.prompts[0] and "it defines login" in cloud.prompts[0] and "def login(u, p)" not in cloud.prompts[0]


def test_a_tainted_turn_is_not_remembered_for_later_cloud_prompts(proj):
    state = ConversationState()
    h = HybridAssistant(local(proj, [chat_reply(calls=[("read_file", '{"path": "config.txt"}')]), chat_reply("it has a password hunter2hunter2")]), Cloud(), HARD)
    h.ask("what is in config.txt?", state)
    assert state.history == []


def test_once_tainted_the_local_model_is_not_given_earlier_turns_as_history(proj):
    state = ConversationState()
    state.history.append(("earlier q", "earlier a"))
    state.taint("x")
    client = ScriptedChat([chat_reply("ok")])
    HybridAssistant(LocalAgent(client, "m", {}), None).ask("q", state)
    assert [m["content"] for m in client.calls[0]["messages"][1:]] == ["q"]


# --- the Claude cloud function ---------------------------------------------------------------------------------------------------------
def test_the_claude_function_makes_one_plain_call_and_returns_the_text():
    client = FakeClient([text_reply("Hello from Claude")])
    out = claude_cloud_fn(client, "claude-sonnet-5-5")("explain in depth")
    call = client.calls[0]
    assert out == "Hello from Claude" and call["model"] == "claude-sonnet-5-5" and call["messages"] == [{"role": "user", "content": "explain in depth"}] and "tools" not in call


def test_any_failure_of_the_claude_call_becomes_a_cloud_error():
    client = NS(messages=NS(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))))
    with pytest.raises(CloudError, match="RuntimeError"):
        claude_cloud_fn(client, "m")("q")


# --- resources -----------------------------------------------------------------------------------------------------------------------
SHAPE = ModelShape("test", 7.0, 28, 4, 128, 32768)


def test_weights_are_parameters_times_bytes():
    assert weights_gb(7, "fp16") == 14.0 and weights_gb(7, "int8") == 7.0 and weights_gb(7, "int4") == 3.5 and weights_gb(7, "fp32") == 28.0


def test_unknown_precision_or_non_positive_parameters_are_rejected():
    for args in ((7, "int3"), (0, "fp16"), (-1, "fp16")):
        with pytest.raises(ValueError):
            weights_gb(*args)


def test_the_kv_cache_follows_the_formula_and_grows_linearly_with_tokens():
    assert kv_cache_gb(SHAPE, 1) == pytest.approx(2 * 28 * 4 * 128 * 2 / 1e9) and kv_cache_gb(SHAPE, 0) == 0
    assert kv_cache_gb(SHAPE, 2000) == pytest.approx(2 * kv_cache_gb(SHAPE, 1000)) and kv_cache_gb(SHAPE, 1000, bytes_per_value=1) == pytest.approx(kv_cache_gb(SHAPE, 1000) / 2)
    with pytest.raises(ValueError):
        kv_cache_gb(SHAPE, -1)


def test_an_estimate_adds_weights_cache_and_overhead_and_compares_with_free_memory():
    e = estimate(SHAPE, "int4", 8000, 8)
    assert e.weights == 3.5 and e.kv_cache == pytest.approx(0.46, abs=0.01) and e.overhead == 0.35 and e.available == 6.0 and e.fits and e.headroom == pytest.approx(6.0 - e.total, abs=0.01)


def test_a_model_that_does_not_fit_says_so():
    assert not estimate(SHAPE, "fp16", 1000, 8).fits and not estimate(SHAPE, "int4", 32768, 6).fits


def test_asking_for_more_context_than_the_model_supports_is_an_error():
    with pytest.raises(ValueError, match="context"):
        estimate(SHAPE, "int4", 40000, 64)


def test_the_longest_fitting_conversation_is_zero_when_the_weights_alone_do_not_fit_and_capped_at_the_models_limit():
    assert max_context_tokens(SHAPE, "fp16", 8) == 0 and max_context_tokens(SHAPE, "int4", 64) == 32768


def test_the_longest_conversation_agrees_with_the_estimate():
    n = max_context_tokens(SHAPE, "int4", 7)  # 7 GB, so memory (not the model's own 32k limit) is what binds
    assert 0 < n < 32768 and estimate(SHAPE, "int4", n, 7).fits and not estimate(SHAPE, "int4", n + 2000, 7).fits


def test_more_memory_never_shortens_the_longest_conversation():
    sizes = [max_context_tokens(SHAPE, "int4", ram) for ram in (6, 8, 12, 16)]
    assert sizes == sorted(sizes)


def test_the_example_shape_is_labelled_as_unverified():
    [name] = EXAMPLE_SHAPES
    assert "verify" in name


# --- latency ----------------------------------------------------------------------------------------------------------------------------
def test_warmup_runs_are_discarded_and_the_summary_has_the_median_and_p95():
    s = LatencyStats()
    for v in (10.0, 1.0, 2.0, 3.0, 4.0, 100.0):
        s.record(v)
    out = s.summary(discard_warmup=1)
    assert out["n"] == 5 and out["median_s"] == 3.0 and out["p95_s"] == 100.0 and out["min_s"] == 1.0 and out["max_s"] == 100.0 and "iqr_s" in out


def test_tokens_per_second_is_the_median_of_the_per_run_rates():
    s = LatencyStats()
    s.record(9.9, 1)
    s.record(2.0, 40)
    s.record(4.0, 40)
    s.record(1.0, 30)
    assert s.summary(discard_warmup=1)["tokens_per_s_median"] == 20.0


def test_no_data_after_warmup_gives_an_empty_summary_and_few_samples_have_no_iqr():
    s = LatencyStats()
    s.record(1.0)
    assert s.summary(discard_warmup=1) == {"n": 0, "discarded": 1}
    s.record(2.0)
    s.record(3.0)
    assert "iqr_s" not in s.summary(discard_warmup=1)


def test_a_negative_duration_is_rejected_and_timed_uses_the_given_clock():
    s = LatencyStats()
    with pytest.raises(ValueError):
        s.record(-1)
    ticks = iter([5.0, 7.5])
    assert s.timed(lambda: "result", clock=lambda: next(ticks)) == "result" and s.samples == [(2.5, None)]
