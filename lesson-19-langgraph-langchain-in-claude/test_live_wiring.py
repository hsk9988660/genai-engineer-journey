"""The live runner talks to the real API, so these tests only check the wiring, with a dummy key. No network call is made."""

import pytest

pytest.importorskip("langchain_anthropic")

import run_live  # noqa: E402
from stage5_approval import ApprovalRequest  # noqa: E402

REQUEST = ApprovalRequest("issue_refund", {"order_id": "A1001", "amount": 89.0, "reason": "x"}, "Refund $89.00", ["approve", "edit", "reject"])


@pytest.fixture(autouse=True)
def dummy_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")


def test_the_live_desk_builds_with_the_same_graph_as_the_offline_one():
    app = run_live.build_live_desk()
    assert set(app.get_graph().nodes) == {"__start__", "triage", "refund_flow", "answer_faq", "escalate", "finish", "__end__"}


def test_the_model_name_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    assert run_live.live_model().model == "claude-haiku-5-5"


def test_the_default_model_is_the_one_the_rest_of_the_course_uses(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert run_live.live_model().model == "claude-sonnet-5-5"


@pytest.mark.parametrize("answers, expected", [
    (["a"], {"decisions": [{"type": "approve"}]}),
    (["r", "Too risky."], {"decisions": [{"type": "reject", "message": "Too risky."}]}),
    (["r", ""], {"decisions": [{"type": "reject", "message": "A reviewer rejected this refund."}]}),
    (["e", "40"], {"decisions": [{"type": "edit", "edited_action": {"name": "issue_refund", "args": {"order_id": "A1001", "amount": 40.0, "reason": "x"}}}]}),
])
def test_the_terminal_reviewer_builds_the_right_decision(monkeypatch, answers, expected):
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    assert run_live.review(REQUEST) == expected
