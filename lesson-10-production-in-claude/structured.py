"""One validated model call returning a Pydantic object. `user` may be text or a list of content blocks (e.g. an image)."""

import json
import os

from pydantic import BaseModel, ValidationError

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")


class StructuredError(Exception):
    pass


def call_structured(client, system: str, user, model_cls: type[BaseModel], max_retries: int = 1, model: str = MODEL) -> BaseModel:
    submit = {"name": "submit_result", "description": "Submit the result. Call once.", "input_schema": model_cls.model_json_schema()}
    messages, retries = [{"role": "user", "content": user}], 0
    while True:
        resp = client.messages.create(model=model, max_tokens=3000, system=system, tools=[submit],
                                      tool_choice={"type": "tool", "name": "submit_result"}, messages=messages)
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
