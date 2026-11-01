"""Knowing what you don't know, as numbers you can check.

"I'm 90% sure" is only useful if answers given at 90% confidence are right about 90% of the time. Verbalized confidence
from a language model is not reliably calibrated, so measure it on labelled data before trusting any threshold.

  coverage          the share of questions the system chooses to answer
  selective accuracy  accuracy on only the questions it answered
  ECE               expected calibration error: the average gap between stated confidence and actual accuracy

A good abstention policy trades a little coverage for a lot of accuracy, and the AbstentionPolicy rule never answers
without grounding, however confident the model says it is.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    confidence: float  # 0..1, as stated by the model
    correct: bool  # ground truth


def coverage_and_accuracy(preds: list[Prediction], threshold: float) -> tuple[float, float | None]:
    answered = [p for p in preds if p.confidence >= threshold]
    coverage = len(answered) / len(preds) if preds else 0.0
    accuracy = sum(p.correct for p in answered) / len(answered) if answered else None
    return coverage, accuracy


def risk_coverage(preds: list[Prediction], thresholds=(0.0, 0.5, 0.7, 0.9)) -> list[dict]:
    rows = []
    for t in thresholds:
        cov, acc = coverage_and_accuracy(preds, t)
        rows.append({"threshold": t, "coverage": cov, "selective_accuracy": acc})
    return rows


def expected_calibration_error(preds: list[Prediction], bins: int = 5) -> float:
    if not preds:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [p for p in preds if lo <= p.confidence < hi or (b == bins - 1 and p.confidence == 1.0)]
        if in_bin:
            acc = sum(p.correct for p in in_bin) / len(in_bin)
            conf = sum(p.confidence for p in in_bin) / len(in_bin)
            total += abs(acc - conf) * len(in_bin) / len(preds)
    return total


@dataclass
class AbstentionPolicy:
    threshold: float = 0.7
    require_grounding: bool = True

    def decide(self, confidence: float, grounded: bool) -> tuple[str, str]:
        """Returns ("answer"|"abstain", reason). Grounding is checked FIRST and cannot be outvoted by confidence."""
        if self.require_grounding and not grounded:
            return "abstain", "no verified supporting evidence"
        if confidence < self.threshold:
            return "abstain", f"confidence {confidence:.2f} is below {self.threshold:.2f}"
        return "answer", "grounded and confident enough"
