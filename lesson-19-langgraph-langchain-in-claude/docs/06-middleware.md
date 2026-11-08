# Stage 6: middleware, the hooks that make an agent safe to run

**Code:** [stage6_middleware.py](../stage6_middleware.py) | **Tests:** [test_stage6_middleware.py](../test_stage6_middleware.py) | **Run:** `python stage6_middleware.py`

## The idea

Production agents need things that have nothing to do with the task: logs, retries, budgets, redaction, approvals. If you write those inside every tool, you repeat yourself and miss cases. **Middleware** wraps the two things an agent does, calling the model and calling a tool, so you add a concern once, in one place.

The hooks in LangChain v1:

| Hook | Runs | Typical use |
|---|---|---|
| `before_agent`, `after_agent` | Once per run | Load or save things; final checks |
| `before_model`, `after_model` | Around each model step | Trim history; check what the model asked for (approval lives here) |
| `wrap_model_call` | Around the model request | Change the request, retry, fall back to another model |
| `wrap_tool_call` | Around each tool run | Audit, retry, block, rewrite results |
| `dynamic_prompt` | Before the model call | Build the system prompt from the context |

You write one with a decorator (`@wrap_tool_call`, `@dynamic_prompt`) or by subclassing `AgentMiddleware`. LangChain also ships ready-made ones; this stage uses three of them.

## The five layers in this stage

### Dynamic prompt

```python
@dynamic_prompt
def desk_prompt(request: ModelRequest) -> str:
    context = request.runtime.context
    return f"{SYSTEM_PROMPT}\nYou are talking to {context.customer_name} (customer id {context.customer_id}). Today is ..."
```

The prompt is built **per request** from verified context. Two customers get two prompts. It replaces the system message, so pass `system_prompt=None` to `create_agent` to avoid saying it twice (`test_there_is_exactly_one_system_message`).

### Audit

```python
@wrap_tool_call
def audit(request, handler):
    ...
    response = handler(request)      # runs the next layer, and finally the tool
    ...
    self.records.append({...})
    return response
```

`handler(request)` means "carry on". Code before it runs before the tool, code after it runs after. `AuditTrail` records, for each call: a sequence number, the tool, its arguments (text clipped to 80 characters), and an outcome:

| Outcome | Meaning |
|---|---|
| `ok` | The tool ran and returned data |
| `refused` | The tool ran and said no (`{"error": ...}`), for example a business rule |
| `error` | Something broke: a schema error or an exhausted retry |
| `exception` | An exception passed through this layer |

The audit changes nothing the model sees (`test_the_record_does_not_change_what_the_model_receives`). Keep audit **content-light**: record the arguments you need to investigate, not whole customer messages (lesson 13 goes deeper).

### Retry

```python
ToolRetryMiddleware(max_retries=2, retry_on=(TemporaryOutage,), initial_delay=0.0, jitter=False)
```

When the order system is down for a moment, running the tool again is right. The important word is `retry_on`: **only temporary failures are retried.** A business refusal like "order not found" is not an exception (it is returned as data), so it is never retried (`test_business_refusals_are_not_retried`). Retrying a refused refund would be at best pointless and at worst a duplicate payment.

Here `on_failure="error"` is set, which means: when the retries run out, **raise** the last exception instead of hiding it. The next layer, error handling, deals with it. The demo sets `desk.outages = 2` and the model never notices.

### Error handling: a bug must not end the run

Not every failure is a business refusal or an outage. A tool can simply have a **bug**. Here is what `create_agent` does with one, by default (`test_by_default_a_bug_in_a_tool_ends_the_run`):

```
RuntimeError: secret internal detail     <- the whole run crashes
```

That is loud, which is good while you develop. In front of a customer it is not what you want. `ToolErrorMiddleware` is the opt-in fix:

```python
ToolErrorMiddleware(on_error=report_tool_failure)
```

`on_error` receives the exception and decides what the model is told. Return text and the exception becomes an error message; return `None` and it keeps propagating. Two choices in `report_tool_failure` are deliberate:

- It **names the exception type** (`RuntimeError`) but not its message. The raw message can hold internal detail, such as a file path, a SQL fragment or a key. `test_the_model_is_not_shown_the_raw_exception_text` proves the text never reaches the model.
- It tells the model **not to call the tool again** and to say a person will follow up. An honest failure message is better than a made-up answer.

Notice what is *not* retried: a `RuntimeError` is not a `TemporaryOutage`, so retry leaves it alone (`test_an_unexpected_exception_is_not_retried`). Retrying a bug only repeats it, and repeating a tool that moves money is dangerous. When I first wrote the retry layer, my only test used a business refusal, which is never an exception, so a version that retried every exception still passed. A mutation check found that gap; this test closes it.

### Limit

```python
ToolCallLimitMiddleware(run_limit=8)
```

Stops an agent from calling tools forever. Calls over the limit **do not run**; the model gets "Tool call limit exceeded" as the result and can wrap up. Every call still gets an answer (`test_every_call_gets_an_answer_even_after_the_limit`). The budget is **per run**, so the next customer message starts fresh (`test_the_budget_is_per_run_so_a_new_run_starts_fresh`). There is also a `thread_limit` for a whole conversation.

Remember from stage 1 that the framework's own `recursion_limit` is huge by default in this version. This middleware is the limit you can reason about.

## Order matters

For `wrap_tool_call`, **the first middleware in the list is the outermost layer.** In `build_hardened_agent`:

```python
middleware = [prompt_middleware(desk), trail.middleware, ToolErrorMiddleware(...), ToolRetryMiddleware(...), ToolCallLimitMiddleware(...)]
```

Audit is **outside** retry and error handling, so a call that needed three attempts is **one** record, and a failure is recorded as `error` with the message the model saw. Retry sits *inside* error handling, so a retry that gave up reaches it. Swap them and the audit sees every attempt:

```
audit outside retry:  ["ok"]
audit inside  retry:  ["exception", "exception", "ok"]
```

Both tests are in the file (`test_the_audit_outside_retry_sees_one_logical_call`, `test_the_audit_inside_retry_sees_every_attempt`). Neither order is wrong. They answer different questions: "what did the agent do?" versus "what did our systems go through?". Choose on purpose and write a test for the one you chose.

## Putting it together

`build_hardened_agent(model, desk, trail, approval=True)` is the agent that stage 7 uses: dynamic prompt, audit, error handling, retry, limit, and the approval from stage 5. `TestAllLayersTogether` checks that the layers work in one agent. Note the audit sees the refund only **when it runs**, after approval; a paused call is not in the trail yet.

A detail you will meet in stage 7: `checkpointer=AUTO` by default means "use `InMemorySaver` if there is an approval step". Pass `checkpointer=None` explicitly when this agent will be a **subgraph**, so it inherits the parent graph's checkpointer.

## What can go wrong

- **Middleware that hides failures.** A hook that catches an exception and returns a made-up success hurts you at 3 a.m. (lesson 14 builds middleware that cannot lie quietly). Every hook here either passes the real result through or returns an honest error. The error layer also logs nothing by itself: the audit trail is where the failure is recorded.
- **Retrying something that is not safe to repeat.** Retry only idempotent calls. `issue_refund` is safe to repeat only because the desk uses an idempotency key.
- **Logging secrets.** The audit clips text and records arguments. Do not log customer messages or tokens just because they are available.

## Try it

1. Write a `wrap_tool_call` middleware that blocks `issue_refund` after 3 refunds in one thread, returning an honest error message. Put it **outside** the audit. What does the trail show? Now put it **inside**.
2. Add `ModelCallLimitMiddleware(run_limit=6)`. Make the scripted model never stop asking for a tool and see what happens.
3. Add a `before_model` hook that prints how many messages are about to be sent. Run the stage 4 demo with it.

## Check yourself

- What does `handler(request)` do, and what happens if a middleware never calls it?
- Why is `retry_on` a tuple of exception types and not "any error"?
- Audit outside retry, or inside? Give one reason for each.
