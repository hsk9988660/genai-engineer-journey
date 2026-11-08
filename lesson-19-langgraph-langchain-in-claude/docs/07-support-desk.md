# Stage 7: the whole support desk as one workflow

**Code:** [stage7_support_desk.py](../stage7_support_desk.py) | **Tests:** [test_stage7_support_desk.py](../test_stage7_support_desk.py) | **Run:** `python stage7_support_desk.py`

## The idea

You now have every part. This stage joins them into one graph that handles a customer message from start to finish:

```
START ─► triage ─┬─► refund_flow ─┐
                 ├─► answer_faq  ─┼─► finish ─► END
                 └─► escalate    ─┘
```

An agent that loops on its own is flexible and hard to predict. A **workflow** has boxes and arrows that you decide, and models work *inside* the boxes. For a support desk, that is usually what you want: the same message always takes the same path, and you can draw the path, test it and explain it to an auditor. (This is the point lesson 14 makes without a framework.)

## Who does what

| Node | Uses a model? | What it does |
|---|---|---|
| `triage` | Yes (structured output, stage 3) | Classifies, then **code verifies** the claims and **decides the route** |
| `refund_flow` | Yes (an agent, stages 1, 5, 6) | Looks things up, may issue a refund. Pauses for a person above $25 |
| `answer_faq` | Yes (no tools) | Answers using only the help articles that code found |
| `escalate` | **No** | Opens a ticket and writes the message to the customer |
| `finish` | **No** | Builds the `resolution` from the ledger |

Three of five nodes make no model call, and the two that decide anything important (`triage`'s routing and `finish`'s resolution) are code.

## Routing is a plain function

```python
def decide_route(triage: dict) -> tuple[str, str]:
    if not triage["verified"]:
        return "escalate", "the request could not be verified: " + ...
    if triage["urgency"] >= URGENT_AT:
        return "escalate", f"urgency {triage['urgency']} of 5"
    if triage["category"] == "refund":
        return "refund_flow", "a verified refund request"
    if triage["category"] == "faq":
        if triage["faq_hits"]:
            return "answer_faq", "help articles match"
        return "escalate", "no help article matches"
    return "escalate", "not a refund or a help question"
```

The order of the checks is the policy:

1. **Unverified goes to a person**, whatever the category. A confident but wrong classification must not reach a money-moving agent.
2. **Urgent goes to a person**, even a clean refund request. A furious customer should not be handled by a bot.
3. Only then does the category matter.
4. A help question with no matching article goes to a person, **not to a model that would make something up**.

`TestTheRoutingDecision` is a table with a row for every branch and for the order between them. The conditional edge is then just `lambda state: state["route"]`. The model never chooses an arrow.

## An agent as a node

```python
refund_agent = build_hardened_agent(models.agent, desk, trail, approval=True, checkpointer=None)
graph.add_node("refund_flow", refund_agent)
```

A compiled graph is itself runnable, so it can be a node of another graph. Two details matter:

- **`checkpointer=None`**: a nested graph should *inherit* the parent's checkpointer. The parent owns the saved state, including the pause inside `refund_flow`.
- **Shared keys**: parent and subgraph both have `messages`. That is safe, because `add_messages` ignores ids it has seen (the trap from stage 2 is about additive lists). `test_a_resume_does_not_duplicate_messages` checks it after a pause and resume.

The verified customer context also flows into the nested agent: the `refund_flow` tools still read `runtime.context.customer_id` (`test_the_verified_customer_reaches_the_nested_agent`).

## The pause, end to end

```python
paused = handle(app, "Please refund order A1001, the shoes are too small.", thread_id="t1")
request = pending_approval(paused)         # what the reviewer sees
done = resume(app, approve(), "t1")        # later, from a different request, even a different process
print(done["resolution"])                  # {'status': 'refunded', 'refunded_cents': 8900, 'refund_ids': ['R-0001']}
```

While paused, the parent state is saved: `app.get_state(config).next == ("refund_flow",)`. Several tickets can wait at once, on different threads, and be resumed in any order (`test_two_paused_refunds_resume_independently_in_any_order`).

## `finish` reads facts, not the model's words

The `resolution` says what actually happened, taken from the ledger:

```python
paid = [r for r in desk.refunds if r.key.startswith(config["configurable"]["thread_id"] + ":")]
status = "refunded" if paid else "declined"
```

If the model says "I have refunded you" but the ledger shows nothing, the status is `declined`. Rejected, edited-and-refused, or refused by a business rule: all show as `declined` with `refunded_cents: 0`.

**A bug this stage caught in itself.** My first version attributed refunds by position: "everything added to the ledger since this ticket began". It passed the simple tests. Then a test with two tickets paused at once failed: ticket 1 was credited with ticket 2's $50. The fix is in the refund's idempotency key: it now starts with the thread id (`t1:toolu_...`, built by `request_key` in `tools.py`), so each refund names the conversation that caused it. The same change also fixed a second problem: a scripted model (and some real providers) can repeat tool-call ids, and two different conversations then collided. Tests: `test_each_resolution_counts_only_its_own_refund`, `test_the_same_tool_call_id_in_two_threads_does_not_collide`.

## Run it

```
$ python stage7_support_desk.py
TICKET 1: a refund
  paused for a person: Refund $89.00 on order A1001 (Trail Shoes, order total $89.00, still refundable $89.00). Reason given: 'shoes are too small'
  resolution: {'status': 'refunded', 'refunded_cents': 8900, 'refund_ids': ['R-0001']}
  customer sees: I have refunded $89.00 for order A1001.

TICKET 2: a help question
  events: ['triage -> answer_faq', 'answer_faq', 'finish']
  customer sees: Standard shipping takes 3 to 5 business days.

TICKET 3: urgent
  events: ['triage -> escalate', 'escalate', 'finish']
  reason: urgency 5 of 5
  customer sees: I have passed this to a person. Your reference is T-0001.
```

`path_of(app, text, thread_id=...)` lists the nodes a ticket visited, using `stream_mode="updates"`. Use it when you debug a route.

## Streaming

`app.stream(input, config, context=..., stream_mode="updates")` yields one chunk per node as it finishes, which is what lets a UI show progress ("checking your order...") before the answer is ready. Other modes: `"values"` (full state after each step) and `"messages"` (model tokens as they arrive). A pause arrives as a chunk with the key `__interrupt__`.

## What can go wrong

- **A model-chosen route.** If you let the triage model name the next node, a hostile message can pick it. Here the model only fills the schema; code reads it.
- **Letting `triage` see too little or too much.** It gets only the customer's last message. In a longer thread you would pass a summary. Think about what a hostile *earlier* message could do.
- **An approval nobody answers.** A paused ticket waits forever. Add a queue, an owner and a timeout policy outside this code.
- **State that grows.** `events` is an additive list that every node appends to. Fine for a ticket, a problem for a thread that lives for months.
- **One shared `Desk`.** The demo uses an in-memory desk. In production the ledger is a database with a unique constraint on the idempotency key, so the "pay once" rule survives a crash and several servers.

## Try it

1. Add a fourth route, `warranty_claim`, for category `warranty`. Where do you add it, and which tests change?
2. Make `escalate` skip creating a ticket if the same customer already has an open one with the same summary. Write the test first.
3. Replace the `answer_faq` node by an agent with the `search_faq` tool. What do you gain? What do you lose compared with code doing the search?
4. Run `app.get_graph().draw_mermaid()` and compare the picture with the diagram at the top of this page.

## Check yourself

- Name two nodes that never call a model. Why is that a good thing?
- Why must `decide_route` check `verified` before it looks at `category`?
- The reviewer rejects the refund. What are the `status` and `refunded_cents` in the resolution, and where do they come from?
- Why does the nested agent get `checkpointer=None`?
