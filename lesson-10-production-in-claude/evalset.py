"""A versioned evaluation dataset with a held-out split and a privacy-aware path from production failures.

* The dataset VERSION is a hash of its content, so every score can say exactly which dataset produced it.
* The dev/held-out split is decided by a hash of each case's id: it never changes when you add cases, so a case can't
  drift from held-out into dev and quietly leak into your tuning.
* Tune prompts on `dev()` only. `heldout()` is for the final check, and asking for it takes an explicit confirmation.
  That is a soft guard against peeking, not a lock: discipline still matters.
* Production failures become cases WITHOUT auto-labeling (a human must say what the right behavior is) and with PII
  redacted from the input first.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field

from redact import redact_text


@dataclass(frozen=True)
class EvalCase:
    id: str
    input: str
    tags: tuple = ()
    expected: dict = field(default_factory=dict)
    needs_label: bool = False  # True until a human has written the expected behavior


@dataclass(frozen=True)
class FailedRun:
    input: str
    outcome: str
    run_id: str = ""


def _bucket(case_id: str) -> float:
    return int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) / 0x100000000


class Dataset:
    def __init__(self, cases):
        self.cases = sorted(cases, key=lambda c: c.id)
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("case ids must be unique")

    @property
    def version(self) -> str:
        canonical = json.dumps([asdict(c) for c in self.cases], sort_keys=True, default=list)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def _in_heldout(self, case: EvalCase, holdout_fraction: float) -> bool:
        return _bucket(case.id) < holdout_fraction

    def dev(self, holdout_fraction: float = 0.3) -> list[EvalCase]:
        return [c for c in self.cases if not self._in_heldout(c, holdout_fraction) and not c.needs_label]

    def heldout(self, holdout_fraction: float = 0.3, confirm_final: bool = False) -> list[EvalCase]:
        if not confirm_final:
            raise PermissionError("the held-out set is for the final check only: pass confirm_final=True when you really mean it")
        return [c for c in self.cases if self._in_heldout(c, holdout_fraction) and not c.needs_label]

    def add_from_failures(self, failures, tag: str = "from_production") -> "Dataset":
        """Return a NEW dataset with a case per distinct failed input. Unlabeled, so excluded from scoring until a human labels it."""
        existing = {c.id for c in self.cases}
        new = []
        for f in failures:
            clean = redact_text(f.input)[0]
            case_id = "prod-" + hashlib.sha256(clean.encode()).hexdigest()[:10]
            if case_id in existing:
                continue
            existing.add(case_id)
            new.append(EvalCase(case_id, clean, (tag, f"outcome:{f.outcome}"), {}, needs_label=True))
        return Dataset([*self.cases, *new])

    def label(self, case_id: str, expected: dict, tags: tuple = ()) -> "Dataset":
        """Return a new dataset where a human has written the expected behavior for one case."""
        out = []
        for c in self.cases:
            out.append(EvalCase(c.id, c.input, (*c.tags, *tags), expected, needs_label=False) if c.id == case_id else c)
        return Dataset(out)
