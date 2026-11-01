"""A versioned fact store: the runtime state object that replaces re-reading a transcript.

Why it exists. If "economy" and later "business class" both sit in the prompt, the model has to guess which wins. Here a new
value explicitly SUPERSEDES the old one: only the current value is rendered into the context, and the history is kept for audit.

Why it is also a poisoning guard. A fact's SOURCE decides whether it may enter the context directly:
  trusted sources (the user, the system)   -> active immediately, superseding any older value
  anything else (a tool result, the model) -> QUARANTINED until a person or validator confirms it
  a per-key validator that fails           -> QUARANTINED with the reason
  an untrusted source that CONTRADICTS a user-stated fact -> QUARANTINED. It never silently overrides the user.
Quarantined facts are never rendered, so a hallucinated flight or an injected claim cannot be reused turn after turn.
"""

import re
from dataclasses import dataclass

from pydantic import BaseModel

from structured import call_structured

TRUSTED_SOURCES = frozenset({"user", "system"})
ALLOWED_KEYS = ("budget", "dates", "destination", "allergy", "class", "travelers")


@dataclass
class Fact:
    key: str
    value: str
    turn: int
    source: str
    version: int
    status: str = "active"  # active | superseded | quarantined
    reason: str = ""  # why it is quarantined
    superseded_by: int | None = None  # the version that replaced it


class FactStore:
    def __init__(self, validators: dict | None = None):
        self._facts: dict[str, list[Fact]] = {}
        self.validators = validators or {}  # key -> fn(value) -> problem string or None

    def get(self, key: str) -> Fact | None:
        return next((f for f in reversed(self._facts.get(key, [])) if f.status == "active"), None)

    def current(self) -> dict[str, Fact]:
        return {k: f for k in sorted(self._facts) if (f := self.get(k))}

    def history(self, key: str) -> list[Fact]:
        return list(self._facts.get(key, []))

    def quarantined(self) -> list[Fact]:
        return [f for facts in self._facts.values() for f in facts if f.status == "quarantined"]

    def superseded_values(self) -> list[tuple[str, str]]:
        """(key, value) pairs that are no longer true. A summary that still mentions one has gone stale."""
        active = {(f.key, f.value) for f in self.current().values()}
        return [(f.key, f.value) for facts in self._facts.values() for f in facts
                if f.status == "superseded" and (f.key, f.value) not in active]

    def set(self, key: str, value, turn: int, source: str) -> Fact:
        history = self._facts.setdefault(key, [])
        value = str(value).strip()
        fact = Fact(key, value, turn, source, version=len(history) + 1)
        current = self.get(key)
        problem = self.validators.get(key, lambda v: None)(value)
        if problem:
            fact.status, fact.reason = "quarantined", f"failed validation: {problem}"
        elif source not in TRUSTED_SOURCES:
            fact.status = "quarantined"
            fact.reason = ("conflicts with a fact stated by the user" if current and current.source in TRUSTED_SOURCES and current.value != value
                           else "untrusted source: needs confirmation")
        elif current and current.value == value:
            return current  # nothing changed: do not churn versions
        history.append(fact)
        if fact.status == "active" and current:
            current.status, current.superseded_by = "superseded", fact.version
        return fact

    def confirm(self, key: str, version: int) -> Fact:
        """A person (or a validator you trust) approves a quarantined fact. It then supersedes the current value."""
        fact = next((f for f in self._facts.get(key, []) if f.version == version and f.status == "quarantined"), None)
        if fact is None:
            raise KeyError(f"no quarantined fact {key} v{version}")
        problem = self.validators.get(key, lambda v: None)(fact.value)
        if problem:
            raise ValueError(f"still fails validation: {problem}")
        current = self.get(key)
        if current:
            current.status, current.superseded_by = "superseded", fact.version
        fact.status, fact.reason = "active", ""
        return fact

    def render_pinned(self) -> str:
        """Only current, trusted facts, in a fixed order. Superseded and quarantined facts never appear."""
        return "\n".join(f"- {f.key}: {f.value} (turn {f.turn})" for f in self.current().values())


# --- live extraction: the model proposes facts, code verifies them against what the user actually said ---------------------------
class ExtractedFact(BaseModel):
    key: str
    value: str
    quote: str  # words copied exactly from the user's message


class Extraction(BaseModel):
    facts: list[ExtractedFact]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip("\"'").lower()


def verify_extraction(ex: Extraction, user_text: str) -> list[str]:
    problems = []
    for f in ex.facts:
        if f.key not in ALLOWED_KEYS:
            problems.append(f"{f.key!r} is not an allowed key. Allowed: {list(ALLOWED_KEYS)}.")
        elif len(_norm(f.quote)) < 3 or _norm(f.quote) not in _norm(user_text):
            problems.append(f"The quote {f.quote!r} for {f.key} does not appear in the user's message. Only extract what they said.")
    return problems


EXTRACT_SYSTEM = f"""Extract preferences and constraints the user STATED in their message. Allowed keys: {list(ALLOWED_KEYS)}.
For each fact give the key, a short value, and a quote copied EXACTLY from the message. Extract nothing the user did not say."""


def extract_facts(client, user_text: str, turn: int, store: FactStore) -> list[Fact]:
    """Facts extracted from the USER'S OWN message are stored as user facts. Ungrounded or unknown-key proposals are rejected."""
    ex = call_structured(client, EXTRACT_SYSTEM, user_text, Extraction, validate=lambda e: verify_extraction(e, user_text))
    return [store.set(f.key, f.value, turn, "user") for f in ex.facts]
