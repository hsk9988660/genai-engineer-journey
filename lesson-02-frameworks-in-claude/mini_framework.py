"""A tiny agent framework on the Claude API (~120 lines, no dependencies beyond `anthropic`).

Lesson 02's Microsoft examples use `@tool`, `provider.as_agent(...)`, `agent.run(...)`,
streaming and threads. This file builds each of those by hand so you can see that a
"framework" is mostly convenience wrapped around the loop from lesson 01.

    Microsoft Agent Framework / Azure         ->  This file
    -------------------------------------------------------------------------
    FoundryChatClient + AzureCliCredential    ->  anthropic.Anthropic() (reads ANTHROPIC_API_KEY)
    @tool(approval_mode="never_require")      ->  @tool
    @tool (approval needed)                   ->  @tool(requires_approval=True)
    provider.as_agent(name, instructions, ..) ->  Agent(name, instructions, tools=[...])
    await agent.run("...")                    ->  agent.run("...")
    agent.run(..., stream=True)               ->  agent.run(..., on_text=print)
    thread + create_message + run             ->  the Agent object itself keeps the history
"""

import inspect
import json
import os
import typing

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _json_type(annotation) -> str:
    origin = typing.get_origin(annotation) or annotation  # list[str] -> list
    return _JSON_TYPES.get(origin, "string")


def tool(fn=None, *, requires_approval: bool = False):
    """Turn a typed Python function into a tool the model can request.

    The JSON schema is built from the function name, type hints and docstring. The model
    sees only that schema, so the docstring is effectively a prompt: write it carefully.
    """

    def wrap(f):
        doc = inspect.getdoc(f)
        if not doc:
            raise ValueError(f"tool '{f.__name__}' needs a docstring; the model reads it")
        hints = typing.get_type_hints(f)
        props, required = {}, []
        for name, param in inspect.signature(f).parameters.items():
            props[name] = {"type": _json_type(hints.get(name, str))}
            if param.default is inspect.Parameter.empty:
                required.append(name)
        f.schema = {
            "name": f.__name__,
            "description": doc,
            "input_schema": {"type": "object", "properties": props, "required": required},
        }
        f.requires_approval = requires_approval
        return f

    return wrap(fn) if fn else wrap


def ask_in_terminal(tool_name: str, args: dict) -> bool:
    return input(f"  Allow {tool_name}({json.dumps(args)})? [y/N] ").strip().lower() == "y"


class Agent:
    """Instructions + tools + conversation history + the agent loop."""

    def __init__(self, name, instructions, tools=(), client=None, model=MODEL,
                 max_steps=6, approve=ask_in_terminal, verbose=False):
        self.name, self.instructions = name, instructions
        self.tools = {t.__name__: t for t in tools}
        self._client = client
        self.model, self.max_steps, self.approve, self.verbose = model, max_steps, approve, verbose
        self.messages: list = []  # the "thread": resent to the model on every call
        self.input_tokens = self.output_tokens = 0

    @property
    def client(self):
        if self._client is None:  # created lazily so importing never needs an API key
            self._client = anthropic.Anthropic()
        return self._client

    def run(self, user_msg: str, on_text=None) -> str:
        """Send a message and run the loop until the model answers.

        on_text: optional callback that receives text chunks as they stream in.
        """
        self.messages.append({"role": "user", "content": user_msg})
        for step in range(1, self.max_steps + 1):  # always cap the loop
            resp = self._call(on_text)
            self.input_tokens += resp.usage.input_tokens
            self.output_tokens += resp.usage.output_tokens
            self.messages.append({"role": "assistant", "content": resp.content})
            if self.verbose:
                print(f"[{self.name} step {step}] stop_reason={resp.stop_reason}")
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text")
            self.messages.append({"role": "user", "content": self._run_tools(resp.content)})
        return "Stopped: step limit reached."

    def _call(self, on_text):
        params = dict(model=self.model, max_tokens=1024, system=self.instructions, messages=self.messages)
        if self.tools:  # the API rejects an empty tools list
            params["tools"] = [t.schema for t in self.tools.values()]
        if on_text is None:
            return self.client.messages.create(**params)
        with self.client.messages.stream(**params) as stream:
            for chunk in stream.text_stream:
                on_text(chunk)
            return stream.get_final_message()

    def _run_tools(self, content) -> list:
        results = []
        for block in content:
            if block.type != "tool_use":
                continue
            if self.verbose:
                print(f"  -> {block.name}({json.dumps(block.input)})")
            fn, is_error = self.tools.get(block.name), False
            if fn is None:
                out, is_error = {"error": f"No such tool: {block.name}"}, True
            elif fn.requires_approval and not self.approve(block.name, block.input):
                out, is_error = {"error": "The user denied this action."}, True
            else:
                try:
                    out = fn(**block.input)  # YOUR code runs the tool, never the model
                except Exception as exc:  # report to the model so it can recover
                    out, is_error = {"error": str(exc)}, True
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(out, default=str), "is_error": is_error})
        return results
