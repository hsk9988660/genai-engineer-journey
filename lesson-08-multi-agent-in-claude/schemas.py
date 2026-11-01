"""The messages agents exchange. Structured, never free prose, so every handoff can be validated in code."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    doc_id: str
    quote: str = Field(description="Words copied EXACTLY from a document you fetched")


class Handoff(BaseModel):
    """Orchestrator to worker: everything the worker is allowed to know."""
    task_id: int = Field(ge=1)
    question: str
    doc_ids: list[str] = Field(description="The only documents this worker may read")
    acceptance: str = Field(description="What a good answer looks like")


class SubTasks(BaseModel):
    tasks: list[Handoff]


class WorkerResult(BaseModel):
    """Worker to orchestrator: a small summary plus evidence, never the raw documents."""
    task_id: int
    status: Literal["ok", "failed"]
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)


class Claim(BaseModel):
    text: str
    sources: list[int] = Field(description="task_ids of the worker results that support this claim")


class Report(BaseModel):
    title: str
    claims: list[Claim]


class Verdict(BaseModel):
    ok: bool
    problems: list[str] = Field(default_factory=list, description="Each unsupported claim and why")


class AgentTurn(BaseModel):
    """One turn of a support agent: answer the customer, or hand the conversation to another agent."""
    action: Literal["answer", "handoff"]
    content: str = Field(description="The answer to the customer, or a short note for the next agent")
    to: Optional[str] = Field(default=None, description="Agent to hand off to (only when action is handoff)")


class SingleAnswer(BaseModel):
    answer: str
