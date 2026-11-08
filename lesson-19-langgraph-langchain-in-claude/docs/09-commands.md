# Stage 9: `Command` and `interrupt()` in your own nodes

**Code:** [stage9_commands.py](../stage9_commands.py) | **Tests:** [test_stage9_commands.py](../test_stage9_commands.py) | **Run:** `python stage9_commands.py`

## The idea

In stage 5 a person approved a refund through middleware. In stage 7 the route was chosen by a conditional edge. This stage shows the two lower-level tools behind them:

- **`Command(update=..., goto=...)`** lets a node change the state **and** choose the next node in one step.
- **`interrupt(value)`** lets a node pause the run and ask a question. The caller resumes with `Command(resume=answer)`, and `interrupt()` then returns `answer`.

The example is a small front door for refunds. If the customer's message has no order id, the graph **asks** which order they mean. It checks the answer, asks again if it is wrong, and after two bad answers hands the request to a person.

```
START -> find_order ─┬─ id found ───────────────► quote ─► END
                     └─ no id ──► clarify ─┬─ good answer ─► quote
                                    ▲      ├─ bad answer ──┐ (asks again)
                                    └──────┘                └─ 2nd bad answer ─► handoff ─► END
```

## `Command` in a node

```python
def find_order(state) -> Command[Literal["quote", "clarify"]]:
    found = ORDER_ID.search(state["text"])
    if found:
        return Command(update={"order_id": found.group(), "log": [...]}, goto="quote")
    return Command(update={"log": ["no order id in the message"]}, goto="clarify")
```

There are no edges from `find_order` in the code. The `Command` is the route. The return annotation `Command[Literal["quote", "clarify"]]` is not decoration: it tells LangGraph where the node may go, so the **drawn graph** still shows the arrows (`test_the_drawn_graph_shows_every_route_a_command_can_take`).

**When to use which?**

| | Conditional edge (stages 2, 7) | `Command` (this stage) |
|---|---|---|
| The choice is... | a pure function of the state | made inside the node, from what the node just found out |
| You can test the route... | without running the node | only by running the node |
| The graph shows the route... | always | only if you annotate the return type |

Prefer the conditional edge when you can. Use `Command` when update-and-route belong together, as in `clarify`.

## `interrupt()` in a node

```python
reply = interrupt({"question": "Which order do you mean? For example A1001.", "attempt": asked + 1})
```

The value you pass goes to the caller, under `__interrupt__` in the result. `pending_question(result)` reads it. The caller answers with `reply(app, "A1001", thread_id=...)`, which calls `invoke(Command(resume="A1001"), same_config)`. This needs a checkpointer, because the state must be saved while the graph waits.

### The sharp edge: code before `interrupt()` runs twice

When you resume, LangGraph does **not** continue from the middle of the node. It runs the node again **from its first line**, and `interrupt()` now returns the answer instead of pausing. So anything above the `interrupt()` call happens once before the pause and once after:

```
side effect before the interrupt runs 2 times; after it, 1 time
```

(`side_effect_runs(before_interrupt)` builds both versions; `TestTheSharpEdge`.) The rule:

> **Do nothing irreversible before an `interrupt()`.** Reading is fine. Sending an email, charging a card or writing a row is not. Put those after the interrupt, or make them idempotent.

`clarify` follows the rule: above its interrupt it only reads state (`test_the_clarify_node_has_nothing_irreversible_before_its_interrupt`). Each answer is also a separate run of the node: a bad answer returns `Command(goto="clarify")` and the node runs again with `questions_asked` increased. That keeps each interrupt as the first thing that happens in a fresh run.

## Input and output schemas

```python
StateGraph(ClarifyState, input_schema=Request, output_schema=Answer, context_schema=DeskContext)
```

- `Request` (`text`) is all the caller may provide. An extra key such as `order_id` is **ignored**, so a caller cannot pre-seed a working value (`test_a_caller_cannot_pre_seed_a_working_key`).
- `Answer` (`outcome`, `log`) is all the caller gets back. The working keys (`order_id`, `questions_asked`) stay inside (`test_the_caller_gets_only_the_output_schema`).

This is the graph's public API. Everything else is private.

## A footgun: resuming a thread that is not waiting

On a thread that never started, `Command(resume=...)` did not fail clearly in the version tested. The resume value was treated as the graph's input, and the run died with `KeyError: 'text'`. `reply()` therefore checks `app.get_state(config).interrupts` first and raises `NoPendingQuestion` with a plain message. Your own resume endpoints should do the same, and an unexpected resume should never be able to start a graph by accident.

## Run it

```
2. no id: the graph asks, the customer answers
   asked: {'question': 'Which order do you mean? For example A1001.', 'attempt': 1}
   {'outcome': {'status': 'quote', 'order_id': 'A1001', 'eligible': True, 'refundable': '$89.00'}, 'log': ['no order id in the message', 'customer chose A1001']}

3. two bad answers: a person takes over
   asked again: {'question': ..., 'attempt': 2}
   {'outcome': {'status': 'handoff', 'detail': 'no valid order id after 2 questions'}, 'log': [..., 'bad answer 1', 'bad answer 2, giving up']}
```

## What can go wrong

- **A side effect before the interrupt** (above).
- **An endless question loop.** `MAX_QUESTIONS = 2` is a cap in code. Without one, a confused customer could be asked forever.
- **Trusting the answer.** The reply is checked: the right shape *and* an order that exists *for this customer*. Another customer's id counts as a bad answer (`test_another_customers_order_id_counts_as_a_bad_answer`).
- **Several interrupts in one node.** They are matched to resume values in order. It works, but each resume re-runs the node from the top, which makes the first rule even more important.

## Try it

1. Add a third kind of question: if the order is outside the return window, ask "Do you want store credit instead?" with an interrupt. Where do you put it so nothing irreversible comes first?
2. Change `MAX_QUESTIONS` to 3. Which test breaks, and what does that tell you about how tightly the tests pin the behaviour?
3. Make `quote` write a ticket. Is it safe to do that in `clarify`? Why or why not?

## Check yourself

- What does `Command(update=..., goto=...)` do that a plain return value cannot?
- A node has `print("hello")`, then `interrupt(...)`. How many times is "hello" printed over one pause and one resume?
- How does an `input_schema` protect the graph?
- Why does `reply()` check for a pending question first?
