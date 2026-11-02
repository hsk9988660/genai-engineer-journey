# Lesson 03: implementation guide

This explains **how the code works**. The [README](README.md) says why the lesson exists. The lesson has three patterns, all in `patterns.py`: clear instructions, structured output, and single-responsibility agents.

---

## 1. The system on one page

```
 plan_trip(client, request, budget_usd)
      │
      ▼
 destination_expert  ──► run_structured(... Shortlist ...)   model proposes picks (claims)
      │
      ▼
 CODE re-checks each pick against the ground truth (_DB)      available? within budget?
      │            none left → NoAvailableDestination
      ▼
 logistics_planner   ──► run_structured(... Itinerary ...)    sees ONLY the verified top pick
      ▼
 Itinerary  (a validated Pydantic object)
```

The idea: **the model proposes, code decides.** Structured output makes the model's answer something code can check.

---

## 2. `run_structured`: the heart of the file

### Where it starts
`run_structured(client, system, user_msg, output_model, tools=(), max_steps=6, max_retries=2, model=MODEL)`.

### What happens, in order

1. **The final answer is a tool call.** A tool named `submit_result` is added whose `input_schema` is `output_model.model_json_schema()`. The model "answers" by calling it.
2. **A tool call is forced every turn.** `tool_choice={"type": "any"}` means the model can never reply in free text. It must call one of your tools or `submit_result`.
3. **Each turn:** the model's content is recorded, then each `tool_use` block is handled:
   - `submit_result`: the input is validated with `output_model.model_validate(...)`. **Valid: return the object.** Invalid: count a retry; after `max_retries` raise `StructuredOutputError`; otherwise send the validation error back as an error result so the model can fix it and call `submit_result` again
   - one of your tools: run it, errors become error results
   - anything else: `{"error": "No such tool: ..."}`
4. **No tool call at all** raises `StructuredOutputError("model returned no tool call")`.
5. **Results go back in one user message** and the loop continues.

### Where it ends
It returns a **validated** object, or raises `StructuredOutputError` (invalid after the retries, no tool call, or the step limit). It never returns unvalidated data.

---

## 3. The three patterns

### Pattern 1: clear instructions
`VAGUE_INSTRUCTIONS` ("You help with travel.") versus `CONCIERGE_INSTRUCTIONS` (a role, ordered duties, a reason for the key rule, and what to do when it cannot find an answer). `python patterns.py` prints both answers to the same question so you can compare.

### Pattern 2: structured output
`run_structured`, above. Models: `Destination`, `Shortlist`, `Itinerary`.

### Pattern 3: single-responsibility agents
- `destination_expert`: has the `get_destination_details` tool and must submit a `Shortlist`. Its instructions say it must **not** plan flights or itineraries.
- `logistics_planner`: has no tools, gets one chosen destination as JSON, must submit an `Itinerary`. Its instructions say it must **not** change the destination.
- `plan_trip` connects them with **code in between**: each of the expert's picks is looked up in `_DB` (the ground truth); only picks that are actually available and within the budget survive; the planner is given only the first verified one. The model's `available` and budget fields are claims, not facts.

### `@tool`
Builds a tool schema from the function's signature and docstring. Unlike lesson 02's, every parameter is required (`"required": list(props)`).

### Errors
`StructuredOutputError` (the model never produced valid output) and `NoAvailableDestination` (nothing survived the checks).

---

## 4. Design decisions

- **A forced tool call is more reliable than asking for JSON in text.** The schema is enforced by the API on the way in and by Pydantic on the way out.
- **Validation errors go back to the model.** It usually fixes the problem on the next try, and the retry budget stops it looping.
- **Code owns the rules that must be right.** Availability and budget are checked against data, not against what the model said.
- **Narrow agents.** An agent with one job is easier to prompt, test and replace.

## 5. What it does not do

A valid shape is not a correct answer: the checks here cover availability and budget only. It does not defend against bad tool output, and the "database" is a dictionary.

## 6. Running and testing

`python patterns.py` needs an API key. `python -m pytest -q` runs 10 offline tests with a scripted client, including the retry path and the ground-truth re-check.
