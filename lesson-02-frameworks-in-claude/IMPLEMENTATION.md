# Lesson 02: implementation guide

This explains **how the code works**. The [README](README.md) says why the lesson exists. The point of the lesson is that a "framework" is mostly **convenience wrapped around the loop from lesson 01**, so this guide shows which convenience lives where.

---

## 1. The system on one page

```
 @tool                                  Agent(name, instructions, tools=[...])
   │ reads the function's name,            │
   │ type hints and docstring              │  agent.run("question", on_text=print)
   ▼                                       ▼
 f.schema   (what the model sees)       the loop (lesson 01) + conversation history
 f.requires_approval                       │
                                           ├─ _call(on_text)    one model call, streamed or not
                                           └─ _run_tools(content)  run each requested tool, with approval
```

Two files: `mini_framework.py` (about 130 lines, the framework) and `examples.py` (the lesson's README examples rewritten with it).

---

## 2. A run, from start to end

### Where it starts
`Agent.run(user_msg, on_text=None)`.

### What happens, in order

1. **The message joins the history.** `self.messages` is the "thread": it is kept on the object and resent to the model on every call, which is how a second `run` remembers the first.
2. **The loop**, up to `max_steps`.
3. **`_call(on_text)`** builds the parameters: model, `max_tokens`, `system=instructions`, `messages`. It adds `tools` **only if there are any**, because the API rejects an empty list. With no callback it calls `messages.create`; with a callback it calls `messages.stream` and passes each text chunk to `on_text` as it arrives, then returns the final message.
4. **Count tokens and record the reply** (the whole content, tool requests included).
5. **Done?** If `stop_reason != "tool_use"` the text blocks are joined and returned.
6. **`_run_tools(content)`** handles each `tool_use` block:
   - unknown name: an error result, `is_error=True`
   - the tool is marked `requires_approval` **and** the `approve` callback says no: `{"error": "The user denied this action."}`
   - otherwise call `fn(**block.input)` in `try/except`; an exception becomes an error result
   - results are serialised with `json.dumps(out, default=str)` so odd types do not crash serialisation
7. **Results go back as one user message**, and the loop continues.

### Where it ends
It returns the final text, or `"Stopped: step limit reached."` after `max_steps`. The history stays on `self.messages`, so the next `run` continues the same conversation.

---

## 3. Reference

### `@tool`
Turns a typed function into a tool. It builds the JSON schema from:
- the **name** (`f.__name__`)
- the **docstring**, which becomes the description. A tool with no docstring raises `ValueError`, because the model reads it
- the **type hints**, mapped to JSON types (`str`, `int`, `float`, `bool`, `list`, `dict`; anything else becomes `"string"`; `list[str]` is read as `list`)
- which parameters have **no default**, which become `required`

It also sets `f.requires_approval`. Use it bare (`@tool`) or with an argument (`@tool(requires_approval=True)`).

### `Agent`
`Agent(name, instructions, tools=(), client=None, model=MODEL, max_steps=6, approve=ask_in_terminal, verbose=False)`.
- the client is created **lazily**, so importing the module never needs an API key
- token counts accumulate in `input_tokens` / `output_tokens`
- `ask_in_terminal` is the default approver and asks `y/N` on the keyboard; pass your own function for anything else

### The mapping to the Microsoft examples
The docstring of `mini_framework.py` has a table: `FoundryChatClient` becomes `anthropic.Anthropic()`, `@tool(approval_mode=...)` becomes `@tool(requires_approval=...)`, `provider.as_agent(...)` becomes `Agent(...)`, streaming becomes `on_text`, a thread becomes the agent's own history.

### `examples.py`
One function per example (`hello`, `travel`, `weather`, `chain`, `plan_and_execute`, `thread`, `streaming`). Run one with `python examples.py travel`; running it with no argument lists them.

---

## 4. Design decisions

- **The framework is the loop plus conveniences, nothing more.** Schema generation, history, streaming, an approval hook. If you can read lesson 01, you can read this.
- **The docstring is the description.** Forcing one prevents the most common tool bug (a vague or missing description).
- **Approval is a callback**, so the same agent works in a terminal, a web app or a test.
- **Lazy client.** Tests and imports never need credentials.

## 5. What it deliberately does not do

No argument validation (a wrong type reaches your function), no parallel tool calls, no timeouts, no protection against untrusted tool output. Lesson 04 adds those.

## 6. Running and testing

`python examples.py` lists the examples (most need an API key). `python -m pytest -q` runs 11 offline tests with a scripted client.
