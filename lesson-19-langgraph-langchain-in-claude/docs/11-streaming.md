# Stage 11: streaming

**Code:** [stage11_streaming.py](../stage11_streaming.py) | **Tests:** [test_stage11_streaming.py](../test_stage11_streaming.py) | **Run:** `python stage11_streaming.py`

## The idea

An agent can take many seconds. A user who sees nothing for ten seconds thinks the app is broken. Streaming sends partial results as they happen: a progress line, the first words of the answer. LangGraph streams in several **modes**. You pick one, or pass a list.

| Mode | Each chunk is... | Use it for |
|---|---|---|
| `"values"` | the **whole state** after each step | a UI that redraws from full state |
| `"updates"` | `{node_name: what_that_node_returned}` | "step 3 of 5 finished" |
| `"messages"` | `(token, metadata)` for each model token | typing the answer word by word |
| `"custom"` | whatever a node passes to its stream writer | your own progress events |
| `"debug"` | detailed task records | debugging |

With a **list** of modes, every chunk is a pair `(mode, data)`:

```python
for mode, data in app.stream(input, config, context=..., stream_mode=["updates", "messages", "custom"]):
    ...
```

`_event(mode, data)` turns raw chunks into small typed events a UI could use: `{"kind": "node"}`, `{"kind": "token"}`, `{"kind": "progress"}`, `{"kind": "pause"}`.

## A help question, streamed

```
{'kind': 'node', 'node': 'triage'}
{'kind': 'token', 'text': 'Standard ', 'node': 'answer_faq'}
{'kind': 'token', 'text': 'shipping ', 'node': 'answer_faq'}
 ... (one event per word)
{'kind': 'node', 'node': 'answer_faq'}
{'kind': 'node', 'node': 'finish'}
```

What the tests establish:

- The tokens **spell the final answer** (`test_the_tokens_spell_the_final_answer`).
- Every token arrives **before** its node is reported finished (`test_tokens_arrive_before_their_node_is_reported_finished`). That is the whole point of streaming.
- Each token carries the **name of the node** that made it (`metadata["langgraph_node"]`), so a UI can show tokens only from the answering node.
- The triage model's structured-output call produces **no text tokens**: it is a tool call, and tool-call chunks have no text (`test_a_structured_output_call_shows_no_text_tokens`).
- `ScriptedChatModel` can stream, which is how all of this runs offline. It yields the next scripted message one word at a time.

### A surprise in `"messages"` mode

`"messages"` streams model tokens **and also any message a node returns**. The escalation note is written by plain code, not by a model, and it still shows up, whole, in one event from the `escalate` node:

```python
tokens = [e for e in events if e["kind"] == "token"]
assert len(tokens) == 1 and tokens[0]["node"] == "escalate"
```

(`test_a_message_written_by_code_arrives_whole_not_word_by_word`.) So "a token event" does not always mean "a model is typing". If your UI animates tokens, check the node name, or expect whole-message events.

## Custom progress events

A node can emit its own events. In stage 8 each worker takes a `StreamWriter` and calls it:

```python
def check_order(state, writer: StreamWriter) -> dict:
    ...
    writer({"checked": state["order_id"]})
```

```
{'kind': 'progress', 'checked': 'A1001'}
{'kind': 'node', 'node': 'check_order'}
{'kind': 'progress', 'checked': 'A1002'}
```

Use `"custom"` for anything that is not a model token or a state change: "searching the knowledge base...", "3 of 10 orders checked".

## Pauses and nested graphs

- A pause arrives as a chunk with the key `__interrupt__`, which `_event` turns into `{"kind": "pause"}`. It ends the stream (`test_a_refund_that_needs_a_person_ends_the_stream_with_a_pause`).
- By default you do **not** see inside a nested graph. The refund agent is a node of the desk, so you only see `refund_flow` finish. Pass `subgraphs=True` and each chunk becomes `(namespace, data)`, with the namespace telling you which subgraph it came from:

```
[('', 'triage'), ('refund_flow', 'model'), ('refund_flow', '__interrupt__'), ('', '__interrupt__')]
```

(`nested_nodes`; `TestSubgraphs`.)

## `values` mode

Each chunk is the full state. The first has no `report`, the last equals what `invoke` returns (`TestValuesMode`). It is simple and heavy: use it when your UI wants to redraw, not when you want a cheap progress signal.

## Async

Every method has an async twin: `astream`, `ainvoke`. A web server uses these so one slow agent does not block the others:

```python
async for mode, data in app.astream(input, config, context=..., stream_mode=MODES):
    ...
```

`ticket_events_async` gives exactly the same events as the sync version (`test_the_async_stream_gives_the_same_events`).

## What can go wrong

- **Streaming tokens to a customer before checking them.** Whatever the model says is already on the screen. Any output check (redaction, a claim check) must run before you stream, or you need a design that can retract text. This is a real tension with the "verify before showing" rules of the earlier stages.
- **Treating every `messages` event as model output** (above).
- **Backpressure.** A slow client with a fast stream needs a buffer or a drop policy. Not covered here.
- **Dropped connections.** If a client disconnects mid-stream, the run may still finish and act. Decide what should happen to a half-delivered ticket.

## Try it

1. Add a `writer({"stage": "triage started"})` call at the top of the `triage` node of stage 7. Stream a ticket and watch where it appears.
2. Stream the stage 10 quote graph with `stream_mode="values"` and print how the state grows.
3. Stream with `subgraphs=True` through a full approved refund and list every `(namespace, node)` pair.

## Check yourself

- Which mode would you use for a progress bar, and which for typing an answer?
- Why does a code-written message appear in `"messages"` mode?
- What does `subgraphs=True` change in each chunk?
