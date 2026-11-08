# Stage 12: LangChain runnables and the `|` pipe

**Code:** [stage12_runnables.py](../stage12_runnables.py) | **Tests:** [test_stage12_runnables.py](../test_stage12_runnables.py) | **Run:** `python stage12_runnables.py`

Stages 1 to 11 used LangChain mostly through `create_agent`. This stage opens the library underneath: the **runnable**, the interface that almost everything in LangChain shares.

## The idea

A prompt, a chat model, an output parser, a Python function, a whole agent and a LangGraph graph are all **runnables**. They all have the same methods:

| Method | Meaning |
|---|---|
| `.invoke(x)` | one input, one output |
| `.batch([x, y, z])` | many inputs, outputs **in the same order** |
| `.stream(x)` | the output in pieces |
| `.ainvoke`, `.abatch`, `.astream` | the async twins |

The `|` operator joins runnables: `a | b` runs `a` and gives its output to `b`. The result is itself a runnable, so chains nest and can be joined again. This is **LCEL**, the LangChain Expression Language.

## The reply chain

The example drafts a reply to a customer:

```python
RunnableLambda(clean_input) | build_prompt() | model | StrOutputParser() | RunnableLambda(add_signature)
```

```
the chain is made of: ['RunnableLambda', 'ChatPromptTemplate', 'ScriptedChatModel', 'StrOutputParser', 'RunnableLambda']
```

A chain written with `|` is a flat `RunnableSequence`. Its `.steps` list the pieces (`test_the_chain_is_a_flat_sequence_of_five_steps`). Read it left to right:

1. **`clean_input`** (plain code) removes emails and phone numbers from the message. The model **never sees them**. This is a rule in code, not a request in a prompt (`test_personal_data_never_reaches_the_model`).
2. **`ChatPromptTemplate`** turns the dict `{"name", "message"}` into messages: a system message with the rules, then a human message.
3. **The model** returns an `AIMessage`.
4. **`StrOutputParser`** takes out the text.
5. **`add_signature`** (plain code) adds the shop's signature. The model is never trusted to sign.

Your own code belongs in `RunnableLambda` steps like these, not in a prompt.

### Prompts

```python
ChatPromptTemplate.from_messages([
    ("system", REPLY_RULES),
    MessagesPlaceholder("history", optional=True),
    ("human", "Customer {name} writes:\n{message}"),
])
```

- `{name}` and `{message}` are variables. A missing one is an error, not a blank (`test_a_missing_variable_is_an_error_not_a_blank`).
- `MessagesPlaceholder("history")` inserts a whole list of earlier messages at that spot. With `optional=True` it can be left out (`test_earlier_messages_go_where_the_placeholder_is`).
- Curly braces in the **customer's** text are safe. The text is substituted into the variable and not read as a template again (`test_curly_braces_in_the_customers_text_are_not_treated_as_variables`). Curly braces in the **template itself** must be doubled (`{{`).
- `.partial(format_instructions=...)` fills one variable now and leaves the rest for later.

## Parallel branches and typed output

```python
RunnableLambda(clean_input) | RunnableParallel(
    sentiment=sentiment_prompt | sentiment_model | PydanticOutputParser(pydantic_object=Sentiment),
    summary=summary_prompt | summary_model | StrOutputParser(),
)
```

`RunnableParallel` runs its branches on the **same input** and returns a dict. Here two questions about one message are asked at once:

```
analysis: {'sentiment': Sentiment(label='negative', reason='the boots leak'), 'summary': "The customer's boots leak."}
```

- `PydanticOutputParser` turns the model's JSON text into a `Sentiment` object. Its `get_format_instructions()` text goes in the prompt so the model knows the format (`test_the_format_instructions_were_filled_into_the_prompt_once`).
- A label outside the schema, or text that is not JSON, raises `OutputParserException` (`test_a_label_outside_the_schema_is_a_parse_error`). Stage 3 showed how to retry on that.
- Both branches got the same redacted message (`test_both_branches_see_the_same_redacted_message`).

`RunnablePassthrough.assign(reply=chain)` **keeps the input** and adds a key: you get `{"name", "message", "reply"}` back (`TestPassthroughAssign`). Use it when a later step needs both the original input and a new result.

## Resilience on any runnable

```python
reply = build_reply_chain(primary).with_retry(retry_if_exception_type=(ConnectionError,), stop_after_attempt=2)
safe = reply.with_fallbacks([RunnableLambda(fallback_reply)], exceptions_to_handle=(ConnectionError,))
```

- `.with_retry` retries the **listed** errors. One blip is absorbed (`test_a_connection_error_is_retried`).
- `.with_fallbacks` runs the next runnable when the first raises. Here, after every attempt fails, a **safe canned reply** goes out (`test_when_every_attempt_fails_a_safe_canned_reply_is_sent`). The canned text promises nothing (`test_the_canned_reply_never_promises_anything`).
- `exceptions_to_handle` limits which errors trigger the fallback. A different error is **not hidden** (`test_an_error_that_is_not_listed_is_not_hidden_by_the_fallback`). Without that limit a fallback can mask real bugs.

These wrap one runnable. Middleware (stages 6 and 15) does the same inside an agent loop.

## Batch and stream

```python
chain.batch(items, config={"max_concurrency": 2}, return_exceptions=True)
```

- Results come back **in the order of the inputs**, however the work was scheduled (`test_results_come_back_in_the_order_of_the_inputs`).
- With `return_exceptions=True` a failure is returned **in its place** as an exception object, and the other items still finish (`test_draft_many_returns_a_failure_in_place_instead_of_raising`).
- `max_concurrency` limits parallelism, which matters for rate limits.

A plain chain `prompt | model | StrOutputParser()` streams word by word, and ends with an empty marker chunk (`test_a_plain_chain_streams_word_by_word_and_ends_with_an_empty_marker`).

**A trap:** a `RunnableLambda` at the **end** of a chain holds the stream. The signature step needs the whole text, so the chain emits one chunk when the model is done (`test_a_function_at_the_end_holds_the_stream_until_the_text_is_complete`). If you want to stream to a user, keep post-processing out of the chain, or accept that it will not stream.

## Two bugs this stage found in its own code

The redaction function looked simple. Running the demo caught the first bug and writing the tests caught the second:

1. The email pattern swallowed the full stop at the end of a sentence: "or sam@example.com." became "or [email]" with the full stop gone.
2. The phone pattern left the opening bracket of `(415) 555-0100` behind: "call ([phone]".

Both are fixed, with tests. The tests also pin two **known limits** honestly: a date such as `2026-10-20` is treated as a phone number, and names are not found at all (`TestKnownLimits`). A regex is a weak privacy tool; lesson 06 explains what a real one needs.

## What can go wrong

- **Putting rules in the prompt** that belong in a `RunnableLambda`.
- **Over-long chains.** A twelve-step `|` chain is hard to debug. Break it into named pieces.
- **Retrying a non-idempotent step.** `with_retry` repeats the runnable. Wrap only calls that are safe to repeat.
- **A fallback that hides bugs** (above).

## Try it

1. Add a `RunnableLambda` after the parser that rejects a reply containing the word "refund". What should happen to the rejected reply?
2. Make `draft_many` use `ainvoke`/`abatch` and an `asyncio.run`.
3. Use `.with_config(run_name="reply")` and a callback handler (stage 15) to print the name of each step as it runs.

## Check yourself

- What does `a | b | c` build, and what methods does the result have?
- Where does your own code belong in a chain?
- What is the difference between `.with_retry` and `.with_fallbacks`?
- Why does the last `RunnableLambda` stop the stream from being word by word?
