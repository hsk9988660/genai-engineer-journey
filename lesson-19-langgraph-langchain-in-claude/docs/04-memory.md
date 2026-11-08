# Stage 4: memory, in two kinds

**Code:** [stage4_memory.py](../stage4_memory.py) | **Tests:** [test_stage4_memory.py](../test_stage4_memory.py) | **Run:** `python stage4_memory.py`

## The idea

A language model remembers nothing. Every request is new. "Memory" means your code sends the right earlier messages again. LangGraph gives you two tools for this, and they do different jobs.

| | Short-term memory | Long-term memory |
|---|---|---|
| What it holds | The conversation so far | Facts that outlive a conversation ("contact me by email") |
| LangGraph name | **Checkpointer**, keyed by `thread_id` | **Store**, keyed by a namespace |
| Lifetime | One thread | Across threads |
| Scope key | `thread_id` | `("preferences", customer_id)` |

## Short-term memory: a checkpointer and a thread

```python
agent = create_agent(model, tools, checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t1"}}
agent.invoke({"messages": [...]}, config)       # turn 1
agent.invoke({"messages": [...]}, config)       # turn 2 continues turn 1
```

After **every step** the graph saves its whole state under the thread id. When you call it again with the same id, it loads that state, adds your new message and carries on. A different thread id starts clean.

Run the demo and read the middle of the output:

```
the model saw, on the second question:
  system   : You are a support agent ...
  customer : What is in order A1001?
  model    : asks for lookup_order(order_id='A1001')
  tool     : {"order_id": "A1001", "item": "Trail Shoes", ...
  model    : A1001 is a pair of Trail Shoes for $89.00.
  customer : How much was that?
```

The model "remembers" the shoes because the whole earlier conversation was sent again. Nothing more. This also means **memory costs tokens**: a long thread is a long prompt. Stage 6 and the context-engineering lesson (12) deal with that.

### Facts worth knowing (each has a test)

- **A checkpointer makes `thread_id` mandatory.** Calling the agent without one raises `ValueError`. (`test_an_agent_with_a_checkpointer_refuses_to_run_without_a_thread_id`)
- **Without a checkpointer there is no memory at all.** Two calls are two unrelated conversations.
- **The saver is separate from the agent.** Two different agent objects that share one saver continue the same thread. This is how a web server can rebuild the agent on every request.
- **Every step is saved.** `checkpoint_history` lists them, oldest first. This is what makes replay and "time travel" debugging possible. `get_state(config).next` tells you what would run next, and is empty when the run is finished (it is not empty when the run is paused, which is stage 5).

`InMemorySaver` lives in RAM and disappears when the program stops. It is right for learning and tests. In production you swap in a database-backed saver (SQLite, Postgres). The agent code does not change.

## Long-term memory: a store

Some facts must survive a new conversation: "this customer wants email, not phone". These go in a **store**.

```python
@tool
def remember_preference(preference: str, runtime: ToolRuntime[DeskContext]) -> str:
    """Save ONE lasting preference of this customer ..."""
    namespace = ("preferences", runtime.context.customer_id)
    runtime.store.put(namespace, uuid.uuid4().hex[:8], {"text": preference.strip()[:200]})
    return "saved"
```

Read the namespace line again. It contains the customer id, and that id comes from `runtime.context`, which your code verified. **The tool has no `customer_id` parameter, so the model cannot choose whose notes to read or write.** This is the same rule as stage 1, and the same defect that lesson 13 found in the original course notebook (a model-supplied `user_id` leaked another user's data).

Tests that pin it down:

- `test_the_tools_do_not_take_a_customer_id`: the schema has no such field.
- `test_a_customer_id_invented_by_the_model_is_ignored`: the model passes `customer_id="C-200"`, and the note is saved under `C-100` anyway.
- `test_another_customer_sees_nothing`.

Two more habits in the tool: the note is **clipped** to 200 characters, and the docstring tells the model not to save order details or anything sensitive. A docstring is a request, not a rule. For real data you would also validate the content in code (lesson 13 shows how: typed keys, consent, verified deletion).

## Which one do I use?

- "What did the customer just say?" is short-term. The checkpointer already handles it.
- "What should I remember next month?" is long-term. Save it deliberately, in a namespace you control.
- Never use long-term memory as a dumping ground for whole conversations. Save small, typed facts.

## What can go wrong

- **Forgetting the thread id** in a web handler, so every request starts fresh. Or **reusing one thread id for everyone**, so customers see each other's history. Derive the thread id from a verified user and a conversation, never from user input alone.
- **Unbounded history.** A thread grows forever. Trim or summarize it (`trim_messages`, `SummarizationMiddleware`).
- **Memory as a prompt-injection channel.** Whatever you store gets shown to the model later. Do not store text from tool results or web pages without checking it (lessons 12 and 13).

## Try it

1. Make `recall_preferences` return at most 5 notes, newest first. You will need to store a timestamp. How do you keep the tests deterministic?
2. Write a test that a note saved by `C-100` is invisible to `C-200` **through the whole agent**, not only through the tool.
3. Replace `InMemorySaver` with `SqliteSaver` (install `langgraph-checkpoint-sqlite`). Stop the program between turns. Does the thread survive?

## Check yourself

- Which of the two memories survives a new `thread_id`?
- Why does the namespace use `runtime.context` and not a tool argument?
- A thread has 200 messages. What is the cost, and what can you do about it?
