"""Measure a local model's tool calling instead of asserting it.

The lesson says Qwen "emit[s] well-formed tool-call structures consistently" and shows three prompts. This asks the model each question several times and sorts every attempt into
exactly one outcome:
  correct          a valid call to the expected tool with the expected arguments
  wrong_args       the expected tool, valid arguments, but not the expected values
  wrong_tool       a valid call to a different tool
  invalid_args     the expected or another tool, but its arguments break the schema
  bad_json         a tool call whose arguments are not JSON
  unknown_tool     a tool name that does not exist
  no_call          the model answered in words instead of calling a tool
  error            the server failed or timed out
Each rate comes with a 95% Wilson interval, because "9 of 10" and "90 of 100" are very different amounts of evidence. Run it against your own server with `evaluate_local.py`.
I have NOT run it against a real model; the tests use a scripted server.
"""

import math
import time
from dataclasses import dataclass, field

from agentloop import ModelError
from latency import LatencyStats
from toolcalls import BadCall, ParsedCall, extract_calls, validate_args

OUTCOMES = ("correct", "wrong_args", "wrong_tool", "invalid_args", "bad_json", "unknown_tool", "no_call", "error")


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class ToolCase:
    prompt: str
    tool: str
    args: dict  # the expected arguments, compared exactly (strings case-insensitively; "*" means any non-empty string)


@dataclass
class EvalReport:
    attempts: int
    counts: dict
    latency: dict
    per_case: dict = field(default_factory=dict)

    def rate(self, *outcomes: str) -> tuple[float, float, float]:
        k = sum(self.counts.get(o, 0) for o in outcomes)
        lo, hi = wilson_interval(k, self.attempts)
        return (k / self.attempts if self.attempts else 0.0, lo, hi)

    @property
    def correct(self) -> tuple[float, float, float]:
        return self.rate("correct")

    @property
    def valid_call(self) -> tuple[float, float, float]:
        """A call that parsed and passed its schema, whether or not it was the right one."""
        return self.rate("correct", "wrong_args", "wrong_tool")


def _same(a, b) -> bool:
    if b == "*":  # "any non-empty string": for free-text arguments such as a search query
        return isinstance(a, str) and bool(a.strip())
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def classify(message: dict, case: ToolCase, tools: dict, accept_text: bool = False) -> str:
    calls = extract_calls(message, accept_text)
    if not calls:
        return "no_call"
    call = calls[0]
    if isinstance(call, BadCall):
        return "bad_json"
    tool = tools.get(call.name)
    if tool is None:
        return "unknown_tool"
    if validate_args(tool.parameters, call.args):
        return "invalid_args"
    if call.name != case.tool:
        return "wrong_tool"
    return "correct" if set(call.args) == set(case.args) and all(_same(call.args[k], v) for k, v in case.args.items()) else "wrong_args"


def run_tool_eval(client, model: str, tools: dict, cases: list, repeats: int = 5, system: str = "Use the tools to answer. Call exactly one tool.", warmup: int = 1,
                  clock=time.monotonic, accept_text: bool = False) -> EvalReport:
    schemas = [t.schema() for t in tools.values()]
    counts, per_case, stats = {o: 0 for o in OUTCOMES}, {}, LatencyStats()
    for case in cases:
        for _ in range(repeats):
            t0 = clock()
            try:
                resp = client.chat(model, [{"role": "system", "content": system}, {"role": "user", "content": case.prompt}], schemas)
                msg = resp["choices"][0]["message"]
                outcome = classify(msg, case, tools, accept_text)
                usage = resp.get("usage") or {}
                stats.record(clock() - t0, usage.get("completion_tokens"))
            except (ModelError, KeyError, IndexError, TypeError):
                outcome = "error"
                stats.record(clock() - t0)
            counts[outcome] += 1
            per_case.setdefault(case.prompt, {o: 0 for o in OUTCOMES})[outcome] += 1
    return EvalReport(sum(counts.values()), counts, stats.summary(discard_warmup=warmup), per_case)
