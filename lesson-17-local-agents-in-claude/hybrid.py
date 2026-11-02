"""Local-first with a cloud model, without leaking what stayed local.

The lesson routes by whether the REQUEST is sensitive or hard. But sensitivity belongs to the CONTEXT: once a tool has read `.env` into the conversation, sending the next, "hard"
question to a cloud model sends that secret with it. Here a conversation carries a `taint` flag. It is set when the user's text or any tool result looks sensitive, it is NEVER cleared
within the conversation, and a tainted conversation is not allowed to reach the cloud at all.

Other rules:
  * the cloud side receives only the question and previous clean question/answer pairs, never raw tool output
  * if the cloud call fails the assistant falls back to the local model, and the answer says so (`fell_back`)
  * every answer states who produced it (`served_by`) and why, so a quieter model is never passed off as a louder one
Detecting sensitivity is a heuristic (see sandbox.looks_sensitive). The cloud function here makes one plain Claude call with no tools.
"""

from dataclasses import dataclass, field

from sandbox import looks_personal, looks_sensitive


class CloudError(Exception):
    pass


@dataclass
class ConversationState:
    tainted: bool = False
    reasons: list = field(default_factory=list)
    history: list = field(default_factory=list)  # [(question, answer)] of turns that were clean

    def taint(self, reason: str) -> None:
        self.tainted = True
        if reason not in self.reasons:
            self.reasons.append(reason)


@dataclass(frozen=True)
class Answer:
    text: str
    served_by: str  # local | cloud | none
    reason: str
    fell_back: bool = False
    tainted: bool = False
    status: str = "answered"


def claude_cloud_fn(client, model: str, system: str = "You are a concise engineering assistant."):
    """Returns `cloud_fn(prompt) -> text` that makes one Claude call. Any failure becomes `CloudError`. Scripted-reply tests only: not run against the real API."""
    def cloud_fn(prompt: str) -> str:
        try:
            resp = client.messages.create(model=model, max_tokens=800, system=system, messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in resp.content if b.type == "text")
        except Exception as exc:
            raise CloudError(type(exc).__name__) from exc
    return cloud_fn


class HybridAssistant:
    def __init__(self, local_agent, cloud_fn=None, is_hard=lambda q: False, is_offline=lambda: False):
        self.local, self.cloud_fn, self.is_hard, self.is_offline = local_agent, cloud_fn, is_hard, is_offline

    def _cloud_prompt(self, question: str, state: ConversationState) -> str:
        context = "\n".join(f"Q: {q}\nA: {a}" for q, a in state.history[-3:])
        return (f"Earlier in this conversation:\n{context}\n\n" if context else "") + question

    def ask(self, question: str, state: ConversationState) -> Answer:
        if looks_sensitive(question) or looks_personal(question):
            state.taint("the question itself looks sensitive")
        want_cloud, why = False, "simple or no cloud configured: local model"
        if self.cloud_fn and self.is_hard(question):
            if state.tainted:
                why = "this conversation touched sensitive data, so it stays local"
            elif self.is_offline():
                why = "offline: local model"
            else:
                want_cloud = True
        fell_back = False
        if want_cloud:
            try:
                text = self.cloud_fn(self._cloud_prompt(question, state))
                state.history.append((question, text))
                return Answer(text, "cloud", "hard question, nothing sensitive in the conversation", tainted=False)
            except CloudError:
                fell_back, why = True, "the cloud model failed: answered locally instead"
        rec = self.local.run(question, [m for q, a in state.history[-3:] for m in ({"role": "user", "content": q}, {"role": "assistant", "content": a})] if not state.tainted else None)
        if rec.tainted:
            state.taint("a tool returned content that looks sensitive")
        if rec.status != "answered":
            return Answer(f"I could not finish: {rec.reason}", "none", why, fell_back, state.tainted, rec.status)
        if not state.tainted:
            state.history.append((question, rec.answer))
        return Answer(rec.answer, "local", why, fell_back, state.tainted)
