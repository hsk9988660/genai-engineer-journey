"""A hardened tool-calling agent on the Claude API.

On top of the basic loop from lesson 01 this adds what production tool use needs:

  * argument validation with Pydantic (errors go back to the model so it can fix the call)
  * parallel execution of independent tool calls, with a per-call timeout
  * human approval for tools with side effects (deny by default)
  * a taint guard against indirect prompt injection (see `taint_policy`)
  * repeated-identical-call detection, result truncation, and an audit log
"""

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable

import anthropic
from pydantic import BaseModel, ValidationError

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")


@dataclass
class Tool:
    name: str
    description: str  # the model reads this to decide WHEN to call the tool: write it carefully
    input_model: type[BaseModel]  # validates arguments AND produces the JSON schema
    fn: Callable
    side_effects: bool = False  # changes the outside world (money, email, deletes): needs human approval
    untrusted_output: bool = False  # result may contain attacker-controlled text (web pages, emails)

    @property
    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_model.model_json_schema()}


def deny_all(tool_name: str, args: dict, tainted: bool) -> bool:
    return False  # safe default: nothing with side effects runs unless you pass a real approver


class ToolAgent:
    """taint_policy: what to do when a side-effect tool is requested AFTER untrusted content was read.

    "block": refuse outright. The conservative default; the model cannot be talked into acting by a web page.
    "ask":   require approval as usual, and tell the approver that untrusted content was read.
    """

    def __init__(self, instructions, tools, client=None, model=MODEL, max_steps=8, approve=deny_all,
                 taint_policy="block", tool_timeout_s=10.0, max_result_chars=4000,
                 max_identical_calls=2, max_workers=4):
        self.instructions = instructions
        self.tools = {t.name: t for t in tools}
        self._client, self.model, self.max_steps, self.approve = client, model, max_steps, approve
        self.taint_policy, self.tool_timeout_s = taint_policy, tool_timeout_s
        self.max_result_chars, self.max_identical_calls, self.max_workers = max_result_chars, max_identical_calls, max_workers
        self.audit: list[dict] = []
        self.tainted = False
        self.input_tokens = self.output_tokens = 0

    @property
    def client(self):
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def run(self, user_msg: str) -> str:
        messages = [{"role": "user", "content": user_msg}]
        self.audit, self.tainted, self._seen = [], False, Counter()
        for _ in range(self.max_steps):
            resp = self.client.messages.create(
                model=self.model, max_tokens=1500, system=self.instructions,
                tools=[t.schema for t in self.tools.values()], messages=messages)
            self.input_tokens += resp.usage.input_tokens
            self.output_tokens += resp.usage.output_tokens
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text")
            calls = [b for b in resp.content if b.type == "tool_use"]
            # Every tool_use needs a tool_result, all in ONE user message, in any order.
            messages.append({"role": "user", "content": self._execute(calls)})
        return "Stopped: step limit reached."

    # -- one model turn: validate -> gate -> run in parallel -> collect ----------------------
    def _execute(self, calls) -> list:
        outcomes: dict[str, tuple] = {}  # tool_use id -> (status, payload)
        runnable = []  # (block, tool, validated args)
        for block in calls:
            tool = self.tools.get(block.name)
            if tool is None:
                outcomes[block.id] = ("unknown", {"error": f"No such tool: {block.name}. Available: {sorted(self.tools)}"})
                continue
            try:
                args = tool.input_model.model_validate(block.input)
            except ValidationError as exc:
                outcomes[block.id] = ("invalid", {"error": "Invalid arguments. Fix them and retry.", "details": str(exc)[:600]})
                continue
            key = (block.name, json.dumps(block.input, sort_keys=True))
            self._seen[key] += 1
            if self._seen[key] > self.max_identical_calls:
                outcomes[block.id] = ("repeat", {"error": "You already made this exact call several times. "
                                                 "Use a different approach or answer with what you have."})
                continue
            if tool.side_effects:
                if self.tainted and self.taint_policy == "block":
                    outcomes[block.id] = ("blocked", {"error": "Blocked: this action is not allowed after reading untrusted "
                                                      "content (web page, email, file) in this conversation."})
                    continue
                if not self.approve(block.name, block.input, self.tainted):
                    outcomes[block.id] = ("denied", {"error": "The user did not approve this action."})
                    continue
            runnable.append((block, tool, args))

        started = {}
        pool = ThreadPoolExecutor(max_workers=self.max_workers)
        futures = {}
        for block, tool, args in runnable:
            started[block.id] = time.monotonic()
            futures[pool.submit(tool.fn, **args.model_dump())] = (block, tool)
        done, pending = wait(futures, timeout=self.tool_timeout_s)
        pool.shutdown(wait=False, cancel_futures=True)  # don't hang on a stuck tool
        for fut, (block, tool) in futures.items():
            if fut in pending:
                outcomes[block.id] = ("timeout", {"error": f"{tool.name} timed out after {self.tool_timeout_s}s."})
            elif fut.exception():
                outcomes[block.id] = ("error", {"error": str(fut.exception())})
            else:
                outcomes[block.id] = ("ok", fut.result())

        results, tainted_now = [], False
        for block in calls:
            status, payload = outcomes[block.id]
            tool = self.tools.get(block.name)
            self.audit.append({"tool": block.name, "args": block.input, "status": status,
                               "tainted_at_call": self.tainted,
                               "ms": round((time.monotonic() - started[block.id]) * 1000) if block.id in started else 0})
            if status == "ok" and tool.untrusted_output:
                tainted_now = True
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": self._to_text(payload), "is_error": status != "ok"})
        # Taint applies to LATER turns: a call issued in the same turn was decided before the model saw the content.
        self.tainted = self.tainted or tainted_now
        return results

    def _to_text(self, payload) -> str:
        text = payload.model_dump_json() if isinstance(payload, BaseModel) else json.dumps(payload, default=str)
        if len(text) > self.max_result_chars:
            text = text[: self.max_result_chars] + f"... [truncated {len(text) - self.max_result_chars} chars; narrow your request]"
        return text
