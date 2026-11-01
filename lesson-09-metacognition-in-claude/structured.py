"""One validated model call returning a Pydantic object, with token accounting."""

import json
import os
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")


class StructuredError(Exception):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


def call_structured(client, system: str, user: str, model_cls: type[BaseModel], usage: Usage | None = None,
                    max_retries: int = 1, model: str = MODEL) -> BaseModel:
    """Force the model to answer in `model_cls`. Schema errors are sent back for up to `max_retries` retries."""
    usage = usage if usage is not None else Usage()
    submit = {"name": "submit_result", "description": "Submit the result. Call once.", "input_schema": model_cls.model_json_schema()}
    messages, retries = [{"role": "user", "content": user}], 0
    while True:
        resp = client.messages.create(model=model, max_tokens=3000, system=system, tools=[submit],
                                      tool_choice={"type": "tool", "name": "submit_result"}, messages=messages)
        usage.calls += 1
        usage.input_tokens += resp.usage.input_tokens
        usage.output_tokens += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})
        block = next((b for b in resp.content if b.type == "tool_use" and b.name == "submit_result"), None)
        if block is None:
            raise StructuredError("the model did not submit a result")
        try:
            return model_cls.model_validate(block.input)
        except ValidationError as exc:
            retries += 1
            if retries > max_retries:
                raise StructuredError(f"invalid output after {max_retries} retries: {str(exc)[:300]}") from exc
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                                          "content": json.dumps({"error": "Schema errors", "details": str(exc)[:500]})}]})
