"""Ties memory to a conversation: the harness retrieves before a turn and writes after it; the model gets only three narrow tools.

Why harness-driven: if the model alone decides when to save and recall, it forgets to, or it is talked into doing it. Retrieval and
extraction therefore run in YOUR code on every turn. The model's tools are a small extra: remember, forget, list.

The tools take NO user id. The user comes from the `MemoryScope` your code built, so a prompt-injected "read alice's memories" has nothing to
aim at. A model-made `remember` becomes a trusted memory only if its quote appears verbatim in the user's latest message; otherwise it is quarantined.
"""

import json

from extract import apply_candidates
from retrieval import render_block, retrieve
from store import KEYS, MemoryScope, MemoryStore, MemoryStoreError, norm

TOOL_SCHEMAS = [
    {"name": "remember", "description": "Save a durable fact the user just told you about themselves. `quote` must be the user's exact words.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["key", "value", "quote"],
                      "properties": {"key": {"type": "string", "enum": sorted(KEYS)}, "value": {"type": "string", "maxLength": 120},
                                     "quote": {"type": "string", "maxLength": 300}}}},
    {"name": "forget", "description": "Hide one remembered fact because the user asked you to forget it.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["memory_id"], "properties": {"memory_id": {"type": "integer"}}}},
    {"name": "list_memories", "description": "List what is currently remembered about this user.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {}}},
]


class MemoryManager:
    def __init__(self, store: MemoryStore, extractor, consent_sensitive: bool = False, budget_tokens: int = 300):
        self.store, self.extractor, self.consent_sensitive, self.budget = store, extractor, consent_sensitive, budget_tokens

    def before_turn(self, scope: MemoryScope, user_text: str) -> str:
        """The `<user_memory>` block to put in this turn's context (after the stable prefix, since it changes)."""
        return render_block([s.memory for s in retrieve(self.store, scope, user_text, self.budget)])

    def after_turn(self, scope: MemoryScope, user_text: str, turn: int | None = None) -> list[dict]:
        """Write memories from the USER's message only. The signature has no parameter for assistant or tool text, on purpose."""
        return apply_candidates(self.store, scope, self.extractor(user_text), user_text, turn, self.consent_sensitive)


class MemoryTools:
    """Executes the model's memory tool calls. Construct one per turn, bound to that turn's scope and the user's latest message."""

    def __init__(self, store: MemoryStore, scope: MemoryScope, last_user_text: str, consent_sensitive: bool = False):
        self.store, self.scope, self.last_user_text, self.consent = store, scope, last_user_text, consent_sensitive

    def handle(self, name: str, args: dict) -> str:
        schema = next((t["input_schema"] for t in TOOL_SCHEMAS if t["name"] == name), None)
        if schema is None:
            return json.dumps({"error": f"unknown tool {name!r}"})
        extra = set(args) - set(schema["properties"])
        if extra:  # includes any attempt to pass user_id: there is no such parameter, and we do not silently ignore it
            return json.dumps({"error": f"unexpected arguments: {sorted(extra)}"})
        try:
            if name == "remember":
                return self._remember(args)
            if name == "forget":
                return json.dumps({"forgotten": self.store.forget(self.scope, int(args["memory_id"]))})  # soft only: a model never hard-deletes
            return json.dumps({"memories": [{"id": m.id, "key": m.key, "value": m.value} for m in self.store.active(self.scope)]})
        except (MemoryStoreError, KeyError, TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc)})

    def _remember(self, a: dict) -> str:
        quote = str(a.get("quote", ""))
        grounded = len(quote.strip()) >= 3 and norm(quote) in norm(self.last_user_text)
        m = self.store.remember(self.scope, a["key"], a["value"], "user" if grounded else "model", evidence=quote if grounded else None,
                                consent=self.consent and grounded)
        return json.dumps({"id": m.id, "status": m.status, "reason": m.reason})
