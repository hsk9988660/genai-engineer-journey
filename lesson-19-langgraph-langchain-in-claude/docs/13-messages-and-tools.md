# Stage 13: messages and tools in depth

**Code:** [stage13_messages_tools.py](../stage13_messages_tools.py) | **Tests:** [test_stage13_messages_tools.py](../test_stage13_messages_tools.py) | **Run:** `python stage13_messages_tools.py`

This stage is about the exact objects that pass between your code and the model.

## Part 1: messages

### What a message holds

```python
message = AIMessage("hello")
message.text            # 'hello'
message.content_blocks  # [{'type': 'text', 'text': 'hello'}]
```

- `.text` is the plain text, joined from any text blocks.
- `.content_blocks` shows **any** message as a list of typed blocks (text, tool call, image, ...), whatever the provider's own format is. Code that reads blocks works across providers.
- An `AIMessage` also carries `.tool_calls` (what the model asked for) and `.usage_metadata` (token counts). The fake model can attach numbers with `say("hi", tokens=(12, 3))`.

### History must stay valid

A `ToolMessage` must come **after** the `AIMessage` that asked for it. If you cut a history in the wrong place, a provider rejects the request. The tempting way to shorten history is a slice:

```python
history[-4:]       # the last 4 messages
```

On the example history this starts with a `ToolMessage`, an **orphan**: its request is gone. `orphans()` in the test file finds such messages. `trim_messages` cuts in a valid place:

```python
trim_messages(messages, max_tokens=keep, token_counter=len, strategy="last", start_on="human", include_system=True)
```

It keeps the system message, keeps the newest messages, and starts on a person's message, so no tool result is left without its request (`test_the_safe_window_never_leaves_an_orphan` checks every window size from 2 to 7). `token_counter=len` counts **messages**. In production you pass a real token counter.

### Shrinking the saved history

`RemoveMessage(id=...)` deletes a message from saved state, through the `add_messages` reducer from stage 2:

```python
add_messages(history, [RemoveMessage(id="h1"), RemoveMessage(id="a1")])
```

A `before_model` middleware can use it so the history **stored** in the thread shrinks too, not only what one call sees:

```python
return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}   # replace the whole list with the kept part
```

(`trim_history`; `test_the_trimming_middleware_shrinks_what_the_model_sees_and_what_is_saved`.) Compare stage 15's `SummarizationMiddleware`, which keeps the gist instead of dropping the old messages.

## Part 2: tools that say more

### Limits in the schema

```python
class RefundArgs(BaseModel):
    order_id: str = Field(pattern=r"^[A-Z]\d{4}$", description="An order id such as A1001")
    amount: float = Field(gt=0, le=200, description="Dollars, at most 200")
    reason: str = Field(min_length=3)
```

The limits do two jobs. They are **sent to the model** as part of the tool's schema (`exclusiveMinimum`, `maximum`, `pattern`), so a good model respects them. And they are **enforced before your function runs**: an amount of 500 or an order id in lowercase becomes an error `ToolMessage` and the function is never called, so no refund is made (`test_a_call_outside_the_limits_never_reaches_the_function`, six cases). The schema is one layer. The desk's rules are another, still behind it (`test_the_schema_limit_and_the_desk_limit_are_separate_layers`).

The tool also takes `runtime: ToolRuntime[DeskContext]` next to a custom `args_schema`. It is injected and stays hidden from the model (`test_the_runtime_argument_is_not_in_the_schema`).

### Two ways to say no

| | Return `{"error": ...}` (stages 1 to 7) | Raise `ToolException` (this stage) |
|---|---|---|
| `ToolMessage.status` | `"success"` | `"error"` |
| The model sees | the text | the text |
| Your code can tell it failed by... | reading the content | reading `status` |

Both reach the model. The raised form is **machine-readable**: logs, metrics and the audit trail can count failures without parsing text. To make a `ToolException` come back as a message instead of ending the run, set `handle_tool_error = True` on the tool:

```python
refund_with_exception.handle_tool_error = True
```

(`@tool` does not take it as an argument, so it is set on the tool object.) Without it the exception propagates, as in stage 6. My first version of the receipt tool forgot this and let `OrderNotFound` escape, which would have ended the run (the stage 6 trap again). I caught it on review before running it. The tool now raises `ToolException`, and a test checks that another customer's receipt is an error with no details.

### Text for the model, data for your app

```python
@tool(response_format="content_and_artifact")
def order_receipt(order_id, runtime) -> tuple[str, dict]:
    return f"{order.item}, {money(order.total_cents)}", receipt
```

The tool returns **two** things. The first, a short line, is the message content the model reads. The second, the full receipt, becomes `ToolMessage.artifact`, which your application can use (draw a receipt in the UI) and the model never needs to read (`test_the_model_gets_one_line_and_the_app_gets_the_payload`). It saves tokens and keeps structured data structured.

### A tool that writes to the agent's state

```python
@tool
def flag_for_followup(reason: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    return Command(update={"followup": reason, "messages": [ToolMessage(..., tool_call_id=tool_call_id)]})
```

- `InjectedToolCallId` gives the tool its own call id, hidden from the model.
- Returning a `Command(update=...)` writes keys into the **agent's state**, not only a message. The agent needs a state schema that has the key: `class FollowupState(AgentState): followup: str`, passed as `state_schema=` (`test_a_tool_can_write_a_key_in_the_agent_state`).
- You must include the `ToolMessage` yourself, or the tool call is left without an answer.

### A tool that reads the state

`InjectedState` gives a tool a view of the state, hidden from the model like `ToolRuntime`. `messages_so_far` returns `"2 messages"` on a first turn.

## What the model is actually told

The fake model records each `bind_tools` call, so you can see the real contract:

```python
model.bound[0]   # {'tools': ['refund_with_exception', 'order_receipt', ...], 'kwargs': {'tool_choice': None}}
```

With `response_format=Answer`, the schema becomes **one more tool** and the model is forced to use a tool:

```python
{'tools': [..., 'Answer'], 'kwargs': {'tool_choice': 'any'}}
```

(`TestWhatTheModelIsToldAbout`.) That is how structured output works under the hood: "call this tool to give your final answer".

## What can go wrong

- **Orphaned tool messages** after cutting history. Always use a function that respects the pairing.
- **Trusting the schema alone.** It checks shape and range, not facts.
- **A `ToolException` without `handle_tool_error`.** The run ends.
- **A state-writing tool with no matching key in the state schema.**
- **Big artifacts in state.** Artifacts are stored with the thread; large ones bloat checkpoints.

## Try it

1. Add `Field(max_length=200)` to `reason` and a test that a 201-character reason never reaches the function.
2. Make `messages_so_far` count only the human messages.
3. Make `flag_for_followup` refuse a second flag in the same conversation, using `InjectedState`.

## Check yourself

- Why is `history[-4:]` dangerous?
- Where do the limits in `RefundArgs` go, and who enforces them?
- What do you gain by raising `ToolException` instead of returning an error dictionary?
- What does `content_and_artifact` let you do?
