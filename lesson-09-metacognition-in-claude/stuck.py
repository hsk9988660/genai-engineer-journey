"""Notice when an agent is going in circles, and do something about it.

Three signals, all computed from what actually happened (not from the model's opinion of itself):
  repeat       the exact same tool call with the exact same arguments, more than `max_repeat` times
  errors       `max_consecutive_errors` failures in a row
  no_new_info  the last `window` tool results were byte-for-byte identical

Policy: the first time it happens the agent gets a corrective note and a chance to change approach. The second time the
run STOPS and asks the user for help. An agent that has twice been told it is stuck and still is should not keep spending.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Verdict:
    stuck: bool
    kind: str = ""
    message: str = ""


class StuckDetector:
    def __init__(self, max_repeat: int = 2, max_consecutive_errors: int = 3, window: int = 3):
        self.max_repeat, self.max_consecutive_errors, self.window = max_repeat, max_consecutive_errors, window
        self.reset()

    def reset(self) -> None:
        self._calls: Counter = Counter()
        self._errors = 0
        self._digests: list[str] = []

    def observe(self, tool: str, args: dict, ok: bool, result) -> Verdict:
        key = (tool, json.dumps(args, sort_keys=True, default=str))
        self._calls[key] += 1
        self._errors = 0 if ok else self._errors + 1
        self._digests.append(hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest())
        self._digests = self._digests[-self.window:]
        if self._calls[key] > self.max_repeat:
            return Verdict(True, "repeat", f"You have called {tool} with the same arguments {self._calls[key]} times.")
        if self._errors >= self.max_consecutive_errors:
            return Verdict(True, "errors", f"The last {self._errors} tool calls all failed.")
        if len(self._digests) == self.window and len(set(self._digests)) == 1:
            return Verdict(True, "no_new_info", f"The last {self.window} tool results were identical, so you are learning nothing new.")
        return Verdict(False)


@dataclass
class AgentRun:
    status: str  # answered | stuck | capped
    answer: str = ""
    steps: int = 0
    stuck_events: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


SUBMIT = {"name": "submit_answer", "description": "Give the final answer to the user. Call once when done.",
          "input_schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}


def run_monitored_agent(client, system: str, user: str, tools: dict, detector: StuckDetector | None = None,
                        max_steps: int = 10, max_guidance: int = 1, model: str = "claude-sonnet-5-5") -> AgentRun:
    """`tools` maps name -> (schema, function). The agent finishes by calling submit_answer."""
    detector = detector or StuckDetector()
    run, guidance = AgentRun("capped"), 0
    schemas = [t[0] for t in tools.values()] + [SUBMIT]
    messages = [{"role": "user", "content": user}]
    for step in range(1, max_steps + 1):
        run.steps = step
        resp = client.messages.create(model=model, max_tokens=1500, system=system, tools=schemas,
                                      tool_choice={"type": "any"}, messages=messages)
        run.input_tokens += resp.usage.input_tokens
        run.output_tokens += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})
        calls = [b for b in resp.content if b.type == "tool_use"]
        final = next((b for b in calls if b.name == "submit_answer"), None)
        if final is not None:
            run.status, run.answer = "answered", str(final.input.get("answer", ""))
            return run
        results, verdicts = [], []
        for block in calls:
            fn = tools.get(block.name, (None, None))[1]
            if fn is None:
                out, ok = {"error": f"No such tool: {block.name}"}, False
            else:
                try:
                    out, ok = fn(**block.input), True
                except Exception as exc:
                    out, ok = {"error": str(exc)}, False
            results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": not ok,
                            "content": json.dumps(out, default=str)})
            verdict = detector.observe(block.name, block.input, ok, out)
            if verdict.stuck:
                verdicts.append(verdict)
        if verdicts:
            run.stuck_events.extend(verdicts)
            guidance += 1
            if guidance > max_guidance:
                run.status = "stuck"
                run.answer = (f"I am not making progress: {verdicts[0].message} "
                              "Could you clarify what you need, or suggest what I should try?")
                return run
            detector.reset()
            results.append({"type": "text", "text": f"[MONITOR] {verdicts[0].message} Try a different approach or a different tool, "
                                                    "or call submit_answer with what you know and say what is missing."})
        messages.append({"role": "user", "content": results})
    return run
