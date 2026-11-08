"""A scripted chat model, so every test in this lesson runs offline with no API key.

LangChain agents talk to any object that follows the `BaseChatModel` interface. This one ignores the prompt and
replays a list of messages that you wrote in advance. It also records what it was sent (`seen`) and which tools were
bound to it (`bound`), so a test can check what the agent actually showed the model.

It can stream too: `stream()` yields the next scripted message word by word, so the streaming stage can show tokens.
"""

import itertools
import json
import re
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

_ids = itertools.count(1)  # tool-call ids are unique across the whole test run, like a real provider's


class ScriptExhausted(RuntimeError):
    """The agent asked the model for more turns than the script has."""


class ScriptedChatModel(BaseChatModel):
    script: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[dict] = Field(default_factory=list)
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        # A real model sends the tool schemas to the provider here. The script does not need them, but we record them.
        names = [convert_to_openai_tool(tool)["function"]["name"] for tool in tools]
        self.bound.append({"tools": names, "kwargs": kwargs})
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen.append(list(messages))
        if self.cursor >= len(self.script):
            raise ScriptExhausted(f"the script has {len(self.script)} turns and the agent asked for turn {self.cursor + 1}")
        message = self.script[self.cursor]
        self.cursor += 1
        return message

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        if message.tool_calls:
            chunks = [AIMessageChunk(content=message.content, tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i} for i, c in enumerate(message.tool_calls)])]
        else:
            text = message.content if isinstance(message.content, str) else ""
            chunks = [AIMessageChunk(content=piece) for piece in re.findall(r"\S+\s*", text)] or [AIMessageChunk(content="")]
        if message.usage_metadata:
            chunks.append(AIMessageChunk(content="", usage_metadata=message.usage_metadata))
        for chunk in chunks:
            if run_manager and isinstance(chunk.content, str) and chunk.content:
                run_manager.on_llm_new_token(chunk.content, chunk=ChatGenerationChunk(message=chunk))
            yield ChatGenerationChunk(message=chunk)


class FlakyChatModel(ScriptedChatModel):
    """Fails with a connection error for the first `failures` calls, then follows its script. For testing retries and fallbacks."""

    failures: int = 0
    attempts: int = 0

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ConnectionError("the provider is unreachable")
        return super()._next(messages)


def say(text: str, *, tokens: tuple[int, int] | None = None) -> AIMessage:
    """A final answer with no tool call. `tokens=(input, output)` attaches usage numbers, like a real provider does."""
    usage = {"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)} if tokens else None
    return AIMessage(content=text, usage_metadata=usage)


def call(name: str, args: dict[str, Any] | None = None, *, call_id: str | None = None, text: str = "", tokens: tuple[int, int] | None = None) -> AIMessage:
    """A turn in which the model asks for one tool call."""
    return calls((name, args or {}, call_id), text=text, tokens=tokens)


def calls(*requests: tuple, text: str = "", tokens: tuple[int, int] | None = None) -> AIMessage:
    """A turn in which the model asks for several tool calls at once."""
    tool_calls = []
    for request in requests:
        name, args, call_id = (tuple(request) + (None, None))[:3]
        tool_calls.append({"name": name, "args": args or {}, "id": call_id or f"call_{next(_ids)}", "type": "tool_call"})
    usage = {"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)} if tokens else None
    return AIMessage(content=text, tool_calls=tool_calls, usage_metadata=usage)
