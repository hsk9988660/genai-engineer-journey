"""Hard limits: budgets, per-user rate limits, and a kill switch.

Cost is an attack surface ("denial of wallet"): an attacker, or just a confused agent, can burn money through
long loops and expensive tools. Limits are enforced in code, not requested in the prompt.
"""

import threading
import time
from collections import defaultdict, deque


class KillSwitch:
    """Thread-safe stop button. Check `tripped` before every model call and tool call."""

    def __init__(self):
        self._event, self.reason = threading.Event(), ""

    def trip(self, reason: str = "operator stop") -> None:
        self.reason = reason
        self._event.set()

    @property
    def tripped(self) -> bool:
        return self._event.is_set()


class Budget:
    """Caps steps, tool calls and total spend (model tokens + tool costs) for one run.

    The default token prices are EXAMPLES only. Set the real prices for your model; they change.
    """

    def __init__(self, max_steps: int = 8, max_tool_calls: int = 10, max_usd: float = 1.0,
                 usd_per_mtok_in: float = 3.0, usd_per_mtok_out: float = 15.0):
        self.max_steps, self.max_tool_calls, self.max_usd = max_steps, max_tool_calls, max_usd
        self.price_in, self.price_out = usd_per_mtok_in, usd_per_mtok_out
        self.steps = self.tool_calls = 0
        self.spent_usd = 0.0

    def fresh(self) -> "Budget":
        """A new budget with the same limits and zeroed counters. Use one per run so spend never carries over."""
        return Budget(self.max_steps, self.max_tool_calls, self.max_usd, self.price_in, self.price_out)

    def charge_step(self, input_tokens: int, output_tokens: int) -> None:
        self.steps += 1
        self.spent_usd += input_tokens / 1e6 * self.price_in + output_tokens / 1e6 * self.price_out

    def charge_tool(self, cost_usd: float) -> None:
        self.tool_calls += 1
        self.spent_usd += cost_usd

    def would_exceed(self, cost_usd: float) -> bool:
        return self.spent_usd + cost_usd > self.max_usd or self.tool_calls + 1 > self.max_tool_calls

    def exhausted(self) -> str | None:
        if self.steps >= self.max_steps:
            return "step limit reached"
        if self.spent_usd >= self.max_usd:
            return "spend limit reached"
        return None


class RateLimiter:
    """Sliding-window limit per user: at most `max_calls` in any `window_s` seconds."""

    def __init__(self, max_calls: int = 30, window_s: float = 60.0, clock=time.monotonic):
        self.max_calls, self.window_s, self.clock = max_calls, window_s, clock
        self._calls: dict[str, deque] = defaultdict(deque)

    def allow(self, user_id: str) -> bool:
        now, calls = self.clock(), self._calls[user_id]
        while calls and now - calls[0] >= self.window_s:
            calls.popleft()
        if len(calls) >= self.max_calls:
            return False
        calls.append(now)
        return True
