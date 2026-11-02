# Lesson 01: implementation guide

This explains **how the code works**: where a run starts, what happens in each step, and where it ends. The [README](README.md) says why the lesson exists. Everything lives in one file, `agent.py`, so this guide follows it from top to bottom.

---

## 1. The system on one page

```
 user question
      │
      ▼
 run_agent()  ──────────────────────────────────────────────┐
      │   repeat, at most max_steps times:                  │
      │   1  client.messages.create(system, tools, messages)│  the model call
      │   2  append the model's whole reply to `messages`   │
      │   3  stop_reason != "tool_use"?  → return the text  │
      │   4  otherwise run each requested tool IN YOUR CODE │
      │   5  send the results back as ONE user message      │
      └─────────────────────────────────────────────────────┘
      ▼
 AgentResult(answer, steps, tokens, stopped_early, trace)
```

The model never runs anything. It only **asks** for a tool call. Your code runs it, and the result goes back as text. That one idea is the whole lesson.

---

## 2. A run, from start to end

### Where it starts
`run_agent(user_msg, client=None, max_steps=6, verbose=True)`. Running `python agent.py "..."` calls it from the `__main__` block, using the words you typed (or a default question).

### What happens, in order

1. **Setup.** If no client is passed, `anthropic.Anthropic()` is created (it reads `ANTHROPIC_API_KEY`). `messages` starts as one user message. An empty `AgentResult` collects the answer, step count, token counts and a trace of printed lines.
2. **The loop**, `for step in range(1, max_steps + 1)`. The cap is deliberate: an agent loop without one can run forever.
3. **Ask the model.** `client.messages.create(model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)`. `TOOLS` is the list of schemas (name, description, parameters). The model sees only these, never your Python code, so a tool's description is effectively a prompt.
4. **Count tokens** from `resp.usage`, and log `stop_reason`.
5. **Record the model's turn.** The whole `resp.content` (text blocks *and* `tool_use` blocks) is appended as an assistant message. Skipping the `tool_use` blocks would make the next call invalid.
6. **Did the model answer?** If `stop_reason != "tool_use"`, join the text blocks into `result.answer` and return. This is the normal ending.
7. **Otherwise run the tools.** For each `tool_use` block:
   - look the name up in `REGISTRY`; an unknown name becomes `{"error": "No such tool: ..."}` with `is_error=True`
   - call `fn(**block.input)` inside `try/except`; an exception becomes `{"error": str(exc)}` with `is_error=True`, so one bad call **cannot crash the loop**
   - build a `tool_result` block with the same `tool_use_id`, the JSON result, and `is_error`
8. **Send all results back in a single user message**, then loop.

### Where it ends
- **Normal:** the model replied without asking for a tool. `stopped_early` is `False`.
- **Capped:** after `max_steps` model calls with no final answer, `stopped_early=True` and `answer="Stopped: step limit reached before the agent finished."`

---

## 3. Reference

### The tools
- `DESTINATIONS`: the demo data.
- `get_destinations()` returns the sorted names. `get_destination_details(name)` returns that destination's climate, beach and budget, **or an error dictionary** for an unknown name ("Call get_destinations first."). Returning the error as data lets the model read it and recover.
- `TOOLS`: the schemas the model sees. `REGISTRY`: name to function. They are two parallel lists you must keep in step; lesson 02 removes that duplication.

### `AgentResult`
`answer`, `steps`, `input_tokens`, `output_tokens`, `stopped_early`, `trace` (a list of the lines printed).

### Configuration
`MODEL` comes from the `ANTHROPIC_MODEL` environment variable, defaulting to `claude-sonnet-5-5`. `SYSTEM_PROMPT` tells the model to check what is available before recommending anything.

---

## 4. Design decisions

- **Errors are data, not exceptions.** A tool that fails returns an error the model can read. The alternative (letting the exception propagate) ends the whole run for one bad argument.
- **Always cap the loop.** `max_steps` bounds both time and cost.
- **Keep the whole assistant turn.** The API expects the `tool_use` blocks to be in history before the matching `tool_result`.
- **One user message for all results.** Every `tool_use` needs a `tool_result`, and they go back together.
- **Print every step.** `verbose` shows the loop so you can watch it; the trace is also kept on the result.

## 5. What this deliberately does not do

No argument validation, no parallel calls, no timeouts, no approval for risky tools, no defence against instructions hidden in tool results. Those arrive in lessons 03, 04 and 06. This file is the loop and nothing else.

## 6. Running and testing

`python agent.py "I want a warm beach holiday."` needs an API key. `python -m pytest -q` runs 5 offline tests with a scripted fake client (no key, no network).
