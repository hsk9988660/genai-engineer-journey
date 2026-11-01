"""One validated model call returning a Pydantic object (used by the live fact extractor)."""

import json
import os

from pydantic import BaseModel, ValidationError

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")


class StructuredError(Exception):
    pass


def call_structured(client, system: str, user: str, model_cls: type[BaseModel], validate=None, max_retries: int = 1,
                    model: str = MODEL) -> BaseModel:
    """`validate(obj)` returns a list of problems. Problems go back to the model for a retry, up to `max_retries`."""
    submit = {"name": "submit_result", "description": "Submit the result. Call once.", "input_schema": model_cls.model_json_schema()}
    messages, retries = [{"role": "user", "content": user}], 0
    while True:
        resp = client.messages.create(model=model, max_tokens=2000, system=system, tools=[submit],
                                      tool_choice={"type": "tool", "name": "submit_result"}, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        block = next((b for b in resp.content if b.type == "tool_use" and b.name == "submit_result"), None)
        if block is None:
            raise StructuredError("the model did not submit a result")
        try:
            obj = model_cls.model_validate(block.input)
            problems = validate(obj) if validate else []
        except ValidationError as exc:
            problems = [f"Schema errors: {str(exc)[:400]}"]
        if not problems:
            return obj
        retries += 1
        if retries > max_retries:
            raise StructuredError("; ".join(problems))
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                                      "content": json.dumps({"error": "Rejected", "problems": problems})}]})
