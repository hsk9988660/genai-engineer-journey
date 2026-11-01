"""Scripted stand-in for the Anthropic client, a controllable clock, and helpers for building run records."""

from types import SimpleNamespace as NS

from telemetry import RunRecord


class Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def reply(input_tokens=100, output_tokens=50, cache_read=None, stop="end_turn", text="ok"):
    usage = NS(input_tokens=input_tokens, output_tokens=output_tokens)
    if cache_read is not None:
        usage.cache_read_input_tokens = cache_read
    return NS(content=[NS(type="text", text=text)], stop_reason=stop, usage=usage)


def submit(payload: dict, input_tokens=100, output_tokens=50):
    block = NS(type="tool_use", id="sub", name="submit_result", input=payload)
    return NS(content=[block], stop_reason="tool_use", usage=NS(input_tokens=input_tokens, output_tokens=output_tokens))


class FakeClient:
    """Replies pop in order. Each call can advance a Clock, to give calls a duration."""

    def __init__(self, replies, clock: Clock | None = None, seconds_per_call: float = 0.0):
        self.replies, self.calls = list(replies), []
        self.clock, self.seconds = clock, seconds_per_call
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw | {"messages": kw["messages"][:]})
        if self.clock:
            self.clock.advance(self.seconds)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def rec(outcome="success", cost=0.01, latency=1000.0, steps=3, run_id="r", tool_calls=0, tool_failures=0, fallbacks=0,
        feedback=None, retries=0) -> RunRecord:
    return RunRecord(run_id=run_id, outcome=outcome, cost_usd=cost, latency_ms=latency, steps=steps, tool_calls=tool_calls,
                     tool_failures=tool_failures, fallbacks=fallbacks, feedback=feedback, retries=retries)
