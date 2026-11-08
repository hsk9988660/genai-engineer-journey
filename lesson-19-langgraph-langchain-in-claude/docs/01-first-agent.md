# Stage 1: your first LangChain agent

**Code:** [stage1_first_agent.py](../stage1_first_agent.py), [tools.py](../tools.py) | **Tests:** [test_stage1_agent.py](../test_stage1_agent.py) | **Run:** `python stage1_first_agent.py`

## The idea

An agent is a language model in a loop with tools. You wrote that loop by hand in lesson 01. `create_agent` is that same loop, packaged:

```
ask the model ─► did it request tools? ─ no ─► return its answer
      ▲                  │ yes
      └── send the results back ◄── run each requested tool (your code, not the model)
```

The model never runs anything. It only **asks**. Your code runs the tool and sends the result back as a message.

## The code

The whole agent is assembled in one function:

```python
def build_agent(model, desk, *, extra_tools=(), middleware=(), checkpointer=None, store=None, system_prompt=SYSTEM_PROMPT):
    return create_agent(
        model,
        [*make_tools(desk), *extra_tools],
        system_prompt=system_prompt,
        middleware=list(middleware),
        checkpointer=checkpointer,
        store=store,
        context_schema=DeskContext,
    )
```

`create_agent` takes a model and a list of tools. Everything else is optional, and each option is a later stage: `checkpointer` and `store` (stage 4), `middleware` (stages 5 and 6). `context_schema` is explained below.

### A tool is a function with a good docstring

```python
@tool
def lookup_order(order_id: str, runtime: ToolRuntime[DeskContext]) -> dict:
    """Look up ONE of this customer's orders by its id (for example A1001). Returns the item, the total, and the delivery status."""
```

`@tool` reads the function name, the type hints and the docstring and turns them into a schema. **The model sees only that schema.** So the docstring is a prompt. Vague docstrings cause wrong or missed tool calls. Notice the docstring of `issue_refund`: it says the tool moves real money and when to call it.

### The argument the model never sees

Look again at `lookup_order`. It has a second parameter, `runtime`. It is **not** in the schema the model receives. LangChain injects it when the tool runs. Through `runtime.context` the tool reads `customer_id`, a fact that your code verified when the conversation began:

```python
agent.invoke({"messages": [...]}, context=DeskContext(customer_id="C-100"))
```

This is the most important safety habit in this stage: **identity comes from verified context, never from the model.** A model can be tricked into asking for another customer's order. It cannot be tricked into supplying a different `customer_id`, because there is no such argument to supply. Two tests prove this:

- `test_the_model_never_sees_the_runtime_argument` checks the schema of every tool.
- `test_the_model_cannot_pass_a_customer_id_of_its_own` makes the scripted model pass `customer_id="C-200"` anyway. The extra field is ignored and the order is still not found.

### Errors are data

When the business rules say no, the tool **returns** `{"error": "..."}` instead of raising. The model reads it and can explain it to the customer. This is the same idea as lesson 01. One error is the exception: `TemporaryOutage` is raised on purpose, so the retry middleware in stage 6 can catch it.

## Run it

```
$ python stage1_first_agent.py
customer : What is in order A1001?
model    : asks for lookup_order(order_id='A1001')
tool     : {"order_id": "A1001", "item": "Trail Shoes", "total": "$89.00", "status": "delivered", ...
model    : Order A1001 is a pair of Trail Shoes for $89.00. It was delivered on 2026-10-20.

model calls: 2
```

Two model calls, one tool call. The first call ended with a tool request, the second with an answer. `format_messages` in [show.py](../show.py) prints any conversation this way. Use it whenever you want to see what an agent did.

## What can go wrong

- **A runaway loop.** The loop stops by itself only when the model stops asking for tools. What stops it otherwise is `recursion_limit` in the run config. In `langgraph` 1.2.20 (the version tested here) its default is **10,007 steps**. Older tutorials say 25. Do not rely on the default: set a small `recursion_limit`, or use `ToolCallLimitMiddleware` (stage 6). `test_a_runaway_loop_is_stopped_by_the_recursion_limit` shows the error you get.
- **A tool the model invented.** If the model asks for `delete_all_orders`, LangChain does not crash. It sends back an error message that says it is not a valid tool, and the loop continues. `test_a_tool_the_model_invented_becomes_an_error_message_and_the_run_continues`.
- **A bad argument.** If the model sends `amount="lots"`, LangChain's schema check refuses it before your function runs, and the model gets an error message. The desk's own rules are a second layer behind it.
- **A tool call needs an answer.** Every tool call in the history needs a result. If you ever build the history yourself, a dangling call makes the next model request invalid (stage 2 shows how to close one).

## Compare with lesson 01

| Lesson 01 (by hand) | Stage 1 (`create_agent`) |
|---|---|
| You write the loop and the step cap | The loop is built in. The cap is `recursion_limit` |
| You write the JSON schema for each tool | `@tool` writes it from the function |
| You catch tool exceptions yourself | Domain errors are returned as data. Invalid tools and bad arguments become error messages |
| Calls the Anthropic API directly | Works with any chat model that follows the LangChain interface |

You gain less code. You give up some visibility, which stage 2 gets back.

## Try it

1. Add a fifth tool, `list_my_orders`, that returns the ids of the current customer's orders. Write a test that another customer cannot see them.
2. Change the docstring of `issue_refund` to something vague like "Refund." What would a real model do differently? (You cannot see it with the scripted model. Write down your prediction, then try `run_live.py` if you have a key.)
3. In the demo, make the scripted model ask for `lookup_order` with `order_id="A1004"`. Whose order is it? What does the customer `C-100` get back?

## Check yourself

- Where does the loop live, and who runs the tools?
- Why is `runtime` missing from the schema the model sees?
- A tool hits a business rule. Should it raise or return an error dictionary? Why? What is the one exception, and why?
