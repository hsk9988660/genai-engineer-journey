"""Shared plumbing for every agent: isolated runs, a trace tree, and one budget for the whole system.

* Each `run_agent` call starts with a FRESH message list. That is context isolation: a worker's long reading never
  reaches the orchestrator, only the structured result it submits.
* Every run becomes a span in a tree (who called whom, tokens, cost, time, outcome). Without this, a multi-agent
  system is a black box.
* One thread-safe `Budget` caps API calls and dollars across ALL agents, including those running in parallel.
  Example token prices below are placeholders. Set the real prices for your model.
"""

import json
import os
import threading
import time
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")


class AgentError(Exception):
    """An agent could not produce a valid result within its retry or step limit."""


class BudgetExceeded(Exception):
    pass


@dataclass
class Span:
    id: int
    parent: int | None
    agent: str
    label: str
    started: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    status: str = "running"
    note: str = ""


class Tracer:
    def __init__(self, clock=time.monotonic):
        self.clock, self.spans, self._lock = clock, [], threading.Lock()

    def start(self, agent: str, label: str = "", parent: int | None = None) -> int:
        with self._lock:
            span = Span(len(self.spans), parent, agent, label, self.clock())
            self.spans.append(span)
            return span.id

    def add_usage(self, span_id: int, input_tokens: int, output_tokens: int, cost: float) -> None:
        with self._lock:
            s = self.spans[span_id]
            s.input_tokens += input_tokens
            s.output_tokens += output_tokens
            s.cost_usd += cost

    def finish(self, span_id: int, status: str = "ok", note: str = "") -> None:
        with self._lock:
            s = self.spans[span_id]
            s.status, s.note = status, note
            s.duration_ms = round((self.clock() - s.started) * 1000)

    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.spans)

    def by_agent(self) -> dict:
        out: dict = {}
        for s in self.spans:
            row = out.setdefault(s.agent, {"runs": 0, "tokens": 0, "cost_usd": 0.0})
            row["runs"] += 1
            row["tokens"] += s.input_tokens + s.output_tokens
            row["cost_usd"] += s.cost_usd
        return out

    def render(self) -> str:
        """The span tree, indented by depth."""
        children: dict = {}
        for s in self.spans:
            children.setdefault(s.parent, []).append(s)
        lines = []

        def walk(parent, depth):
            for s in children.get(parent, []):
                tok = s.input_tokens + s.output_tokens
                lines.append(f"{'  ' * depth}{s.agent}{(' ' + s.label) if s.label else ''}  "
                             f"[{s.status}] {tok} tok ${s.cost_usd:.4f} {s.duration_ms}ms{(' - ' + s.note) if s.note else ''}")
                walk(s.id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)


class Budget:
    def __init__(self, max_calls: int = 40, max_cost_usd: float = 1.0, usd_per_mtok_in: float = 3.0,
                 usd_per_mtok_out: float = 15.0):
        self.max_calls, self.max_cost_usd = max_calls, max_cost_usd
        self.price_in, self.price_out = usd_per_mtok_in, usd_per_mtok_out
        self.calls, self.cost_usd, self._lock = 0, 0.0, threading.Lock()

    def begin_call(self) -> None:
        """Atomically check the limits AND reserve one call. Two parallel agents can never both take the last slot.

        The call cap is therefore exact. The dollar cap is not: a call's cost is only known after it returns, so
        calls already in flight can push spend a little past `max_cost_usd`.
        """
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"call limit reached ({self.max_calls})")
            if self.cost_usd >= self.max_cost_usd:
                raise BudgetExceeded(f"spend limit reached (${self.max_cost_usd:.2f})")
            self.calls += 1

    def charge(self, input_tokens: int, output_tokens: int) -> float:
        cost = input_tokens / 1e6 * self.price_in + output_tokens / 1e6 * self.price_out
        with self._lock:
            self.cost_usd += cost
        return cost


SUBMIT_DESCRIPTION = "Submit your final result. Call exactly once when you are done."


class Runtime:
    def __init__(self, client, tracer: Tracer | None = None, budget: Budget | None = None, model: str = MODEL):
        self.client, self.tracer, self.budget, self.model = client, tracer or Tracer(), budget or Budget(), model

    def run_agent(self, agent: str, system: str, user: str, output_model: type[BaseModel], parent: int | None = None,
                  label: str = "", tools=(), tool_impls=None, validate=None, max_retries: int = 2, max_steps: int = 5):
        """Run one agent to completion in an isolated context and return a validated `output_model`.

        `tools` are the ONLY tools this agent can see; `tool_impls` maps their names to Python callables.
        The agent finishes by calling `submit_result`. `validate(obj)` returns a list of problems, which are
        sent back to the agent for a retry.
        """
        tool_impls = tool_impls or {}
        span = self.tracer.start(agent, label, parent)
        status, note = "error", ""
        try:
            submit = {"name": "submit_result", "description": SUBMIT_DESCRIPTION,
                      "input_schema": output_model.model_json_schema()}
            all_tools = [*tools, submit]
            messages = [{"role": "user", "content": user}]  # fresh context: nothing is shared with other agents
            retries = 0
            for _ in range(max_steps):
                self.budget.begin_call()
                resp = self.client.messages.create(model=self.model, max_tokens=2000, system=system, tools=all_tools,
                                                   tool_choice={"type": "any"}, messages=messages)
                cost = self.budget.charge(resp.usage.input_tokens, resp.usage.output_tokens)
                self.tracer.add_usage(span, resp.usage.input_tokens, resp.usage.output_tokens, cost)
                messages.append({"role": "assistant", "content": resp.content})
                calls = [b for b in resp.content if b.type == "tool_use"]
                if not calls:
                    raise AgentError("the agent made no tool call")
                results, submit_block = [], None
                for block in calls:
                    if block.name == "submit_result":
                        submit_block = block
                    elif block.name in tool_impls:
                        try:
                            out, is_error = tool_impls[block.name](**block.input), False
                        except Exception as exc:
                            out, is_error = {"error": str(exc)}, True
                        results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": is_error,
                                        "content": json.dumps(out, default=str)})
                    else:
                        results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                        "content": json.dumps({"error": f"No such tool: {block.name}"})})
                if submit_block is not None:
                    try:
                        obj = output_model.model_validate(submit_block.input)
                        problems = validate(obj) if validate else []
                    except ValidationError as exc:
                        obj, problems = None, [f"Schema errors: {str(exc)[:500]}"]
                    if not problems:
                        status = "ok"
                        return obj
                    retries += 1
                    if retries > max_retries:
                        raise AgentError("; ".join(problems))
                    results.append({"type": "tool_result", "tool_use_id": submit_block.id, "is_error": True,
                                    "content": json.dumps({"error": "Rejected. Fix every problem and resubmit.",
                                                           "problems": problems})})
                messages.append({"role": "user", "content": results})
            raise AgentError("step limit reached")
        except Exception as exc:
            note = f"{type(exc).__name__}: {str(exc)[:120]}"
            raise
        finally:
            self.tracer.finish(span, status, note)
