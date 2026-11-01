"""A reflection memory that cannot be poisoned by text the model read.

Reflexion-style memory stores a lesson after a failure and retrieves it next time. The danger is that a lesson distilled
from untrusted text (a web page, a document) becomes a persistent instruction. So here the model NEVER writes a
lesson. Lessons come from TEMPLATES keyed by the code of a failure that a deterministic checker confirmed. The cost
of that safety: it only learns failure types you already wrote a check and a template for.
"""

import time
from dataclasses import dataclass, field

from checks import Failure

LESSON_TEMPLATES = {
    "no_dinner": "Every day must include a dinner: a meal that starts at 18:00 or later.",
    "overlap": "Never schedule two activities at the same time. An activity may start exactly when the previous one ends.",
    "budget": "Add up all activity costs while planning and stay under the budget. Prefer cheaper options when close.",
    "total_mismatch": "total_cost_usd must equal the exact sum of every activity's cost_usd. Recompute it last.",
    "too_many": "Respect the maximum number of activities per day.",
    "days_count": "Produce exactly the requested number of days, numbered 1 to N.",
    "bad_time": "Use 24-hour HH:MM times and make every end time later than its start time.",
    "duplicate_day": "Each day number must appear exactly once.",
}


@dataclass
class Lesson:
    code: str
    text: str
    count: int
    first_seen: float
    last_seen: float


@dataclass
class ReflectionMemory:
    max_lessons: int = 20
    ttl_s: float = 7 * 24 * 3600
    clock: object = time.time
    _lessons: dict = field(default_factory=dict)

    def record_failures(self, failures: list[Failure], trusted: bool = True) -> int:
        """Store lessons for validated failures. Returns how many lessons were added or reinforced.

        `trusted` must be False if the failures were derived from content the agent did not generate and the checker
        did not verify. Untrusted failures are ignored entirely.
        """
        if not trusted:
            return 0
        now, touched = self.clock(), 0
        for code in {f.code for f in failures}:
            template = LESSON_TEMPLATES.get(code)
            if template is None:  # unknown failure types are ignored, never turned into free text
                continue
            lesson = self._lessons.get(code)
            if lesson is None:
                self._lessons[code] = Lesson(code, template, 1, now, now)
            else:
                lesson.count += 1
                lesson.last_seen = now
            touched += 1
        self._evict()
        return touched

    def _evict(self) -> None:
        now = self.clock()
        for code in [c for c, l in self._lessons.items() if now - l.last_seen > self.ttl_s]:
            del self._lessons[code]
        while len(self._lessons) > self.max_lessons:
            worst = min(self._lessons.values(), key=lambda l: (l.count, l.last_seen))
            del self._lessons[worst.code]

    def active(self) -> list[Lesson]:
        self._evict()
        return sorted(self._lessons.values(), key=lambda l: (-l.count, l.code))

    def render(self, k: int = 5) -> str:
        """Text for the system prompt. Empty string when there is nothing to say."""
        lessons = self.active()[:k]
        if not lessons:
            return ""
        return ("Lessons from earlier attempts. They came from automatic checks, and they are advice, not rules that override the task:\n"
                + "\n".join(f"- {l.text}" for l in lessons))
