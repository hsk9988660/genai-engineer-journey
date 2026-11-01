"""Graders, from most to least trustworthy.

1. CODE graders: required and forbidden tools, step and cost limits, schema validity, any rule you can write as a function.
   Cheap, deterministic, and impossible to charm. The lesson's "did it cover flights AND activities?" is one of these:
   look at which tools were called, don't ask a model.
2. Human labels: the ground truth you calibrate everything else against.
3. LLM judges: scalable but biased (position, verbosity, preference for their own model family). Never trust one before
   measuring how often it agrees with humans, and always run it in BOTH orders to expose position bias.
"""

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from stats import cohens_kappa


@dataclass(frozen=True)
class GradeResult:
    name: str
    passed: bool
    detail: str = ""


# A grader is `fn(record, output) -> GradeResult`, where `record` is a RunRecord and `output` is the agent's result.
def required_tools(*names: str):
    def grade(record, output):
        missing = [n for n in names if n not in record.tools_used]
        return GradeResult("required_tools", not missing, f"missing: {missing}" if missing else "")
    return grade


def forbidden_tools(*names: str):
    def grade(record, output):
        used = [n for n in names if n in record.tools_used]
        return GradeResult("forbidden_tools", not used, f"used forbidden: {used}" if used else "")
    return grade


def max_steps(limit: int):
    return lambda record, output: GradeResult("max_steps", record.steps <= limit, f"{record.steps} steps, limit {limit}")


def max_cost(limit_usd: float):
    return lambda record, output: GradeResult("max_cost", record.cost_usd <= limit_usd, f"${record.cost_usd:.4f}, limit ${limit_usd:.4f}")


def schema_valid(model_cls: type[BaseModel]):
    def grade(record, output):
        try:
            model_cls.model_validate(output)
            return GradeResult("schema_valid", True)
        except ValidationError as exc:
            return GradeResult("schema_valid", False, str(exc)[:200])
    return grade


def custom(name: str, fn: Callable):
    return lambda record, output: GradeResult(name, bool(fn(record, output)), "")


def grade_all(record, output, graders: list) -> list[GradeResult]:
    return [g(record, output) for g in graders]


# --- LLM judge, with the safeguards ------------------------------------------------------------------
def pairwise_judge(judge_fn: Callable[[str, str, str], str], question: str, answer_1: str, answer_2: str) -> dict:
    """`judge_fn(question, first, second)` returns "A" (first is better), "B" (second is better) or "tie".

    The judge is asked twice with the answers swapped. If the two verdicts disagree about which ANSWER is better, the
    judge was swayed by position, not quality, and the result is "inconsistent" instead of a made-up winner.
    """
    first = {"A": "1", "B": "2", "tie": "tie"}[judge_fn(question, answer_1, answer_2)]
    second = {"A": "2", "B": "1", "tie": "tie"}[judge_fn(question, answer_2, answer_1)]  # swapped order, mapped back
    consistent = first == second
    return {"winner": first if consistent else "inconsistent", "consistent": consistent}


def position_bias_rate(results: list[dict]) -> float:
    return sum(not r["consistent"] for r in results) / len(results) if results else 0.0


def judge_agreement(judge_labels: list, human_labels: list) -> dict:
    n = len(human_labels)
    return {"accuracy": sum(j == h for j, h in zip(judge_labels, human_labels)) / n if n else 0.0,
            "kappa": cohens_kappa(judge_labels, human_labels)}


class JudgeVerdict(BaseModel):
    verdict: str = Field(description="'A' if the FIRST answer is better, 'B' if the SECOND is, or 'tie'")


JUDGE_SYSTEM = """You compare two answers to the same question. Judge ONLY correctness and completeness against the question.
Ignore length, tone and formatting. If they are equally good, say tie.
Scoring anchors: an answer that states a wrong fact loses to one that does not, even if it is longer and more polished."""


def claude_judge(client):
    """A judge_fn backed by Claude. NOT run in tests. Measure its agreement with human labels before relying on it."""
    from structured import call_structured

    def judge(question: str, first: str, second: str) -> str:
        user = f"Question: {question}\n\nFIRST answer:\n{first}\n\nSECOND answer:\n{second}"
        v = call_structured(client, JUDGE_SYSTEM, user, JudgeVerdict).verdict.strip()
        return v if v in ("A", "B", "tie") else "tie"

    return judge
