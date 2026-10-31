# Lesson 03: three agent design patterns on the Claude API

The notebook in lesson 03 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) teaches three patterns. `patterns.py` implements them with the Claude API.

| Pattern | Where in `patterns.py` | Idea |
|---|---|---|
| 1. Clear instructions | `CONCIERGE_INSTRUCTIONS` vs `VAGUE_INSTRUCTIONS` | Role, ordered steps, and what NOT to do |
| 2. Structured output | `run_structured()` | The final answer is a tool call with a Pydantic schema. It is validated, and errors go back to the model for a retry. |
| 3. Single responsibility | `destination_expert`, `logistics_planner`, `plan_trip` | One job per agent, with structured data passed between them |

## Run

```bash
pip install anthropic pytest
python -m pytest -q                   # 10 offline tests, no API key
set ANTHROPIC_API_KEY=sk-ant-...      # macOS/Linux: export ANTHROPIC_API_KEY=...
python patterns.py                    # live demo: vague vs clear, then the pipeline
```

## Design decisions worth studying
1. **`tool_choice={"type": "any"}`** forces a tool call every turn, so the model can't answer in free text. It may call your tools first, then must call `submit_result`.
2. **Validation errors are returned as a tool result** with `is_error=True`. The model sees what was wrong and fixes it. Retries are capped (`max_retries`) and raise `StructuredOutputError`.
3. **Code re-checks the model's claims.** In `plan_trip`, a pick's `available` and budget fields come from the model. They are verified against the database before the planner runs. A schema guarantees shape, not truth.
4. **Hard rules live in code, not the prompt.** Budget and availability filtering is an `if` statement. `NoAvailableDestination` stops the pipeline before the planner agent is called, which saves a model call.
5. **Only verified data goes to the next agent**, as JSON rather than pasted prose.

## Exercises
1. Remove the verification loop in `plan_trip` and craft a test where the model lies. Watch it fail.
2. Add a field with a constraint (`estimated_budget_usd: int = Field(ge=0)`) and confirm a negative value triggers a retry.
3. Merge the two agents into one. Compare the number of model calls and the quality of the result.
4. Write 5 test requests with expected destinations. This is the start of an eval set.
