"""Compaction you can trust: a summary replaces raw turns ONLY if it passes deterministic checks. Otherwise the raw turns stay.

Summarization is lossy and irreversible. A summary can drop a constraint or invent one, and then it becomes the single source of
truth. So before a summary is allowed to replace anything, code checks three things:

  invented numbers   a number, price or date in the summary that appears NOWHERE in the source (a hallucination)
  dropped numbers    a number the USER stated that the summary lost and that is not preserved in the fact store
  stale values       a value that has been SUPERSEDED (the user changed their mind) but still appears in the summary

These checks are conservative on purpose: a false alarm just keeps the raw turns (safe, costs tokens), while a missed problem
silently corrupts the agent's memory. They are not a full guarantee: a summary can still mis-state something that contains no
number. And the raw transcript is always archived, so a bad summary can be re-done.
"""

import re
from dataclasses import dataclass, field

from facts import FactStore
from tokens import estimate_tokens

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class Turn:
    id: int
    role: str  # user | assistant
    text: str


class Transcript:
    """The append-only archive. Compaction never edits it, so nothing is ever truly lost."""

    def __init__(self):
        self._turns: list[Turn] = []

    def append(self, role: str, text: str) -> Turn:
        turn = Turn(len(self._turns) + 1, role, text)
        self._turns.append(turn)
        return turn

    def get(self, ids) -> list[Turn]:
        wanted = set(ids)
        return [t for t in self._turns if t.id in wanted]

    def all(self) -> list[Turn]:
        return list(self._turns)


def numbers_in(text: str) -> set[str]:
    """'$3,000' and '3000' count as the same number."""
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}


def verify_summary(summary: str, sources: list[Turn], store: FactStore) -> list[str]:
    problems = []
    source_numbers = set().union(*(numbers_in(t.text) for t in sources)) if sources else set()
    fact_numbers = set().union(*(numbers_in(f.value) for f in store.current().values())) if store.current() else set()
    summary_numbers = numbers_in(summary)

    invented = sorted(summary_numbers - source_numbers - fact_numbers)
    if invented:
        problems.append(f"The summary contains numbers that appear nowhere in the source: {invented}. Remove them.")
    user_numbers = set().union(*(numbers_in(t.text) for t in sources if t.role == "user")) if sources else set()
    dropped = sorted(user_numbers - summary_numbers - fact_numbers)
    if dropped:
        problems.append(f"The summary dropped numbers the user stated: {dropped}. Keep them.")
    for key, value in store.superseded_values():
        if re.search(rf"\b{re.escape(value)}\b", summary, re.IGNORECASE):
            problems.append(f"The summary mentions {value!r}, which is no longer the user's {key}. Leave out superseded values.")
    return problems


@dataclass
class Summary:
    text: str
    source_ids: tuple
    tokens_before: int
    tokens_after: int


@dataclass
class CompactionResult:
    status: str  # compacted | kept_raw | nothing_to_do
    summary: Summary | None = None
    problems: list = field(default_factory=list)


class Compactor:
    def __init__(self, summarize_fn, store: FactStore, count=estimate_tokens, max_retries: int = 1):
        """summarize_fn(turns, feedback) -> text. `feedback` is a list of problems from the previous attempt, or None."""
        self.summarize_fn, self.store, self.count, self.max_retries = summarize_fn, store, count, max_retries

    def compact(self, turns: list[Turn]) -> CompactionResult:
        if not turns:
            return CompactionResult("nothing_to_do")
        before = sum(self.count(t.text) for t in turns)
        feedback, problems = None, []
        for _ in range(self.max_retries + 1):
            text = self.summarize_fn(turns, feedback)
            problems = verify_summary(text, turns, self.store)
            if not problems and self.count(text) >= before:
                problems = [f"The summary ({self.count(text)} tokens) is not smaller than the source ({before} tokens)."]
            if not problems:
                return CompactionResult("compacted", Summary(text, tuple(t.id for t in turns), before, self.count(text)))
            feedback = problems
        return CompactionResult("kept_raw", None, problems)  # fail SAFE: keep the raw turns
