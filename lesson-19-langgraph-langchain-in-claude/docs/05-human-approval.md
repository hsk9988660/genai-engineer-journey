# Stage 5: a person approves before money moves

**Code:** [stage5_approval.py](../stage5_approval.py) | **Tests:** [test_stage5_approval.py](../test_stage5_approval.py) | **Run:** `python stage5_approval.py`

## The idea

Some actions are too risky to leave to a model: moving money, deleting data, sending an email. The usual answer is **human in the loop**: the agent pauses, a person looks, and the agent continues only if they agree.

In LangGraph a pause is an **interrupt**:

1. A node calls `interrupt(...)`. LangGraph saves the full state with the checkpointer and stops.
2. `invoke` returns early. The result contains `__interrupt__` with whatever you passed.
3. Minutes or days later you call `invoke(Command(resume=...), same_thread_config)`.
4. The graph loads the saved state and continues from the pause.

Because the state is saved, the program does not have to stay running while a person decides. This is why an interrupt **needs a checkpointer**.

You rarely call `interrupt` yourself for tools. `HumanInTheLoopMiddleware` does it for you.

## The code

```python
HumanInTheLoopMiddleware(interrupt_on={
    "issue_refund": {
        "allowed_decisions": ["approve", "edit", "reject"],
        "description": describe,
        "when": needs_approval,
    },
})
```

Only `issue_refund` is listed, so the read-only tools never pause (`test_read_only_tools_never_pause`). The pause happens **after the model asks for the tool and before the tool runs**. In a paused thread, `get_state(config).next` shows the middleware node that is waiting.

### Three decisions

| Decision | Resume payload | What the agent does |
|---|---|---|
| approve | `{"decisions": [{"type": "approve"}]}` | Runs the tool as the model asked |
| edit | `{"decisions": [{"type": "edit", "edited_action": {"name": ..., "args": {...}}}]}` | Runs the tool with the reviewer's arguments, and tells the model a human changed the call |
| reject | `{"decisions": [{"type": "reject", "message": "..."}]}` | Does **not** run the tool. The model receives the reason as the tool's result and can respond to the customer |

The helpers `approve()`, `edit(...)`, `reject(...)` and `resume(...)` build these. `pending_approval(result)` reads the pause out of an `invoke` result and returns `None` when the run finished. If one turn asks for several paused tools, the decisions list needs one entry per tool, in order.

Run the demo to see all three:

```
--- reject
paused before running. refunds so far: 0
reviewer sees: Refund $89.00 on order A1001 (Trail Shoes, order total $89.00, still refundable $89.00). Reason given: 'too small'
tool     : User rejected the tool call for `issue_refund` with reason: Please offer store credit instead.
refunds issued: []
```

## Three lessons from the design

### 1. Approval is not authorization

A person can type a wrong number. A tired reviewer can click approve. So the rules in `desk.py` check **every** refund again when it runs, whoever approved it. Four tests show it:

- An edit to $500 on an $89 order is refused (`more than the $89.00`).
- An approved refund of $240 is refused, because an agent may never refund over $200 (`agent limit`).
- An approved refund on an order outside the 30-day window is refused.
- An approved refund on **another customer's** order is refused.

Do not treat the approval click as the safety control. It is one layer. The rules in code are another, and they do not get tired.

### 2. Show the reviewer facts that code looked up

The reviewer sees this text, built by `describe`:

> Refund $89.00 on order A1001 (Trail Shoes, order total $89.00, still refundable $89.00). Reason given: 'too small'

The item name, the order total and the refundable amount come from the **desk**, not from the model. If the model claims it is refunding "a $5 sock" and the order is a $400 tent, the reviewer sees the tent. If the order cannot be verified, the text says so (`test_the_reviewer_is_told_when_the_order_cannot_be_verified`). Approvals work only when the person sees the truth.

### 3. The `when` predicate must be stable

Small refunds (up to $25.00) should not need a person; asking for every $3 refund would train reviewers to click without reading. `when=needs_approval` decides per call:

```python
def needs_approval(request) -> bool:
    try:
        return to_cents(request.tool_call["args"]["amount"]) > AUTO_APPROVE_LIMIT_CENTS
    except Exception:
        return True
```

Two rules, both from LangChain's own documentation of this option:

- **The predicate runs again when the run resumes.** If it gave a different answer the second time, the reviewer's decision would be skipped, and the tool would run even if they had rejected it. So it must be a pure function of the call's arguments. Never base it on something that changes during the pause, such as how much is left on the order.
- **When it cannot read the arguments, it answers "yes, ask a person".** An unreadable amount is the case a human should see (`test_an_unreadable_amount_pauses_and_is_refused_even_if_approved`).

The limit is inclusive: exactly $25.00 runs alone, $25.01 pauses (`TestRiskTiers`). A threshold like this also invites **splitting**: ten $25 refunds instead of one $250. The desk's per-order total and the call limit in stage 6 are what stop that, not this predicate.

## What can go wrong

- **No checkpointer.** An interrupt cannot save its state. Always pass one.
- **Resuming on the wrong thread**, or with a different customer's context. The resume needs the same `thread_id`, and your code should check that the person resuming is allowed to decide for that ticket.
- **Double resume.** A reviewer clicks twice. Make your resume endpoint idempotent in your own code. (The refund itself is already safe: its idempotency key means a replay pays once.)
- **A forgotten pause.** A paused thread waits forever. Production systems need a queue of pending approvals and a timeout policy.

## Try it

1. Make `describe` also show how many refunds this customer received in the last 30 days. Where does that number come from? Does it break the stability rule for `when`? (It should not, if you only use it in `describe`. Why?)
2. Add a second risky tool, `send_apology_email`, that always needs approval but only allows `approve` and `reject`.
3. Resume with a reject that has an empty message. What does the model see?

## Check yourself

- What exactly is saved when the agent pauses, and what makes resuming possible?
- A reviewer approves a refund that is above the agent limit. What happens, and which layer stopped it?
- Why must `needs_approval` give the same answer before and after the pause?
