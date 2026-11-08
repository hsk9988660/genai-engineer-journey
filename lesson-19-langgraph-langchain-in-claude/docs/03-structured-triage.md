# Stage 3: structured output, and checking it against the truth

**Code:** [stage3_triage.py](../stage3_triage.py) | **Tests:** [test_stage3_triage.py](../test_stage3_triage.py) | **Run:** `python stage3_triage.py`

## The idea

Code cannot do much with a paragraph of text. It can do a lot with a typed object. **Structured output** makes the model fill in a schema instead of writing prose:

```python
class Triage(BaseModel):
    category: Literal["refund", "faq", "other"]
    order_id: str | None = None
    urgency: int = Field(ge=1, le=5)
    summary: str = Field(max_length=300)

structured = model.with_structured_output(Triage)
result = structured.invoke(messages)        # a Triage object, not a string
```

`with_structured_output` uses the model's tool-calling feature under the hood: the schema becomes a tool, and the model "calls" it with its answer. That is why the scripted model in the tests answers with `call("Triage", {...})`.

## A schema checks the shape, not the truth

This is the central lesson of the stage. Pydantic can tell you that `urgency` is an integer from 1 to 5. It cannot tell you that `order_id` is real, or that it came from the customer's message, or that it belongs to this customer. A model can return a perfectly shaped object that is perfectly wrong.

So the stage has two layers:

| Layer | Catches | Handled by |
|---|---|---|
| 1. The schema | Wrong shape: `urgency=9`, `category="complaint"`, text instead of an object | pydantic. The model gets **one more try**, with the error as feedback |
| 2. `verify` | Wrong facts: an order id that is not in the message, does not exist, or belongs to someone else; a refund with no order id | code that owns the ground truth |

```python
def verify(triage, text, desk, customer_id) -> list[str]:
    problems = []
    ids_in_text = set(ORDER_ID.findall(text))
    if triage.order_id is not None:
        if triage.order_id not in ids_in_text:
            problems.append(f"order id {triage.order_id} does not appear in the message")
        else:
            try:
                desk.get_order(customer_id, triage.order_id)
            except DeskError as exc:
                problems.append(str(exc))
    elif triage.category == "refund":
        problems.append("a refund request needs an order id")
    return problems
```

The first check is the clever one. The customer's message is ground truth. If the model "found" an order id that is not written in the message, it invented it. Tests: `TestVerification`.

## Retry once, then give up safely

```python
for attempt in range(1, max_attempts + 1):
    try:
        parsed = structured.invoke(messages)
        ...
    except (ValidationError, OutputParserException) as exc:
        ...  # tell the model what was wrong, ask again
```

- The retry **replaces** the feedback each time instead of piling it up (`test_the_retry_does_not_pile_up_old_feedback`). Message lists that grow on every retry waste tokens.
- It is capped. Every retry costs money, and a model that fails twice usually fails a third time.
- After the last attempt it returns a **safe fallback**: category `other`, `verified=False`, with the customer's own words as the summary. It never raises, and it never pretends to have understood. Stage 7 sends unverified tickets to a person.

The result type carries what the next stage needs: `TriageResult(triage, verified, problems, attempts)`.

## Run it

```
$ python stage3_triage.py
a good answer: category=refund verified=True attempts=1 problems=[]
an invented order id: category=refund verified=False attempts=1 problems=['order id A9999 does not appear in the message']
an invalid shape, fixed on retry: category=faq verified=True attempts=2 problems=[]
```

## Two ways to get structured output

This stage calls `with_structured_output` on the model. `create_agent` has a different option, `response_format=Triage`, which makes an *agent* finish by returning an object (in the result under `structured_response`). Use `with_structured_output` for a single classification step like this one. Use `response_format` when the agent works with tools first and the final answer must be typed.

## What can go wrong

- **The model returns plain text.** The parser gives back `None`. The code treats that as a failure and retries (`test_plain_text_instead_of_an_object_counts_as_a_failure`).
- **A valid object that is wrong.** Layer 2 exists for this. No schema can replace it.
- **Over-trusting a small enum.** `category` is one of three values, so it always "looks valid". Test your classifier on messages that do not fit (a lesson 10 evaluation set) before you trust the routing built on it.

## Try it

1. Add a field `language: Literal["en", "other"]` to `Triage`. Which tests need an update? Which do not?
2. Add a check to `verify`: if the message contains **two** different order ids and the model returned one, mark it unverified. Write the test first.
3. Make the fallback urgency `5` instead of `3`. What changes downstream in stage 7, and is that what you want?

## Check yourself

- What does `with_structured_output` do under the hood?
- Name one thing a schema can check and one thing it cannot.
- Why is the number of retries capped, and why does the fallback not raise?
