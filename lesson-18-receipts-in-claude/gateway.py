"""Putting receipts around a Claude agent's tools.

`wrap_impls` takes the same `impl(user_id, **args)` functions lesson 16's tools use and returns versions that go through a `GuardedExecutor`, so every call leaves an intent and a result
receipt, and a tool that changes the world needs an approval.
  * the USER ID is part of the recorded action (`{"user_id": ..., **args}`), so a receipt and an approval are about one user's action, and it comes from code, never from the model
  * the model does not choose, hold or see approvals. They are put in an `ApprovalBroker` by the system (for example by a person clicking "approve" in a UI)
  * a refusal comes back to the model as text ("Not performed: approval_required"), so it can tell the user, in the same way lesson 17 returns sandbox refusals
`run_turn` is a minimal Claude tool-use loop to show the wiring. It is tested with scripted replies, not the real API.
"""

from approval import ApprovalRefused, GuardedExecutor, action_digest


class ApprovalBroker:
    """Where the system places approvals a human has granted, keyed by the exact action they approve."""

    def __init__(self):
        self._by_digest: dict[str, dict] = {}

    def add(self, approval: dict) -> None:
        self._by_digest[approval["action_digest"]] = approval

    def find(self, tool: str, args: dict):
        try:
            return self._by_digest.get(action_digest(tool, args))
        except ValueError:  # arguments that cannot be canonicalised match no approval
            return None


def wrap_impls(executor: GuardedExecutor, impls: dict, broker: ApprovalBroker) -> dict:
    wrapped = {}
    for name, impl in impls.items():
        def call(user_id, _name=name, _impl=impl, **args):
            recorded = {"user_id": user_id, **args}

            def run(**a):
                uid = a.pop("user_id")
                return _impl(uid, **a)
            try:
                return executor.execute(_name, run, recorded, broker.find(_name, recorded))
            except ApprovalRefused as exc:
                return f"Not performed: {exc.reason}"
        wrapped[name] = call
    return wrapped


def run_turn(client, model: str, system: str, user_text: str, schemas: list, impls: dict, user_id: str, max_rounds: int = 4) -> str:
    messages = [{"role": "user", "content": user_text}]
    for _ in range(max_rounds):
        resp = client.messages.create(model=model, max_tokens=500, system=system, messages=messages, tools=schemas)
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            return "".join(b.text for b in resp.content if b.type == "text")
        results = []
        for b in calls:
            fn = impls.get(b.name)
            try:
                out = fn(user_id, **b.input) if fn else f"Unknown tool {b.name}"
            except TypeError:
                out = "Error: invalid arguments"
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)[:2000]})
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": results})
    return "Stopped: too many tool rounds."
