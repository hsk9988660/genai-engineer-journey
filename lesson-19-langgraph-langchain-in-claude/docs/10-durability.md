# Stage 10: durability, time travel and node retries

**Code:** [stage10_durability.py](../stage10_durability.py) | **Tests:** [test_stage10_durability.py](../test_stage10_durability.py) | **Run:** `python stage10_durability.py`

Three things decide whether a LangGraph system survives the real world.

## 1. Durable persistence

`InMemorySaver` (stage 4) lives in RAM. When the process ends, every paused ticket is gone. `SqliteSaver` writes each checkpoint to a file:

```python
def open_saver(path):
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))

app = build_desk(models, desk, checkpointer=open_saver("tickets.db"))
```

Nothing else in the graph changes. The test that matters is `pause_then_restart`:

1. Build the desk on a SQLite file. Start a refund. It **pauses** for a person. Nothing has been paid.
2. Close the connection. The "process" ends. Everything in memory is thrown away: the graph, the models.
3. Open the file again and build a **brand-new** graph with **brand-new** model objects.
4. Resume the same `thread_id` with an approval. The ticket finishes: `{'status': 'refunded', 'refunded_cents': 8900, ...}`.

Look at step 3. The new models are given **only the turns still to come** (`say("Refund done.")`). They have no triage turn and no tool-call turn, because those already happened and the saved state remembers them. If the saved state were ignored, the desk would ask for those turns again and the scripted fake would raise `ScriptExhausted`. The run finishing proves the saved state was used (`test_the_second_process_never_repeats_a_model_turn`). That is what **durable execution** means: a crash or a restart costs you nothing already done.

For contrast, `test_an_in_memory_saver_forgets_everything_after_a_restart` builds a second desk on a new `InMemorySaver` and finds an empty thread.

Facts worth knowing:

- **Close your connections.** On Windows a SQLite file stays locked until every connection to it is closed. The demo first crashed when its temp folder could not be deleted. `pause_then_restart` closes both savers in `try/finally`.
- **SQLite is for one machine.** For several servers use the Postgres saver (`langgraph-checkpoint-postgres`).
- **The desk's ledger is still a Python list** in this lesson. Real durability also needs the refund ledger in a database with a unique key (stage 7 explains why).
- The checkpoints really are rows in the file: `test_the_checkpoints_are_really_in_the_file`.

## 2. Time travel

Every step of a thread is a checkpoint. The quote graph (`load -> price -> finish`) is small on purpose so you can see them:

```
history: [(-1, ('__start__',)), (0, ('load',)), (1, ('price',)), (2, ('finish',)), (3, ())]
```

Each entry says what would run **next** from that point. The functions:

| Function | Does |
|---|---|
| `history(app, thread)` | all checkpoints, oldest first |
| `checkpoint_before(app, thread, "price")` | the config of the moment just before `price` |
| `replay(app, config)` | run forward again from there. Nodes before it do **not** run again |
| `what_if(app, config, {"fee_pct": 0})` | change the state at that moment, then run forward |

```
replay from before 'price' ran: {'price': 1, 'finish': 1}          # 'load' did not run again
what if the fee were 0%: quote $89.00 | nodes run: {'price': 1, 'finish': 1}
```

A "what if" is an experiment you can do on a finished run, for example: "what would this customer have been quoted with no restocking fee?". Three things to know:

- **Replay and fork write new checkpoints on the same thread.** The old ones stay (`test_a_fork_adds_checkpoints_and_keeps_the_old_ones`: the demo's history grew from 5 to 11 checkpoints). But the thread's **latest state becomes the fork** (`test_the_threads_latest_state_becomes_the_fork`). If you want the original run untouched, copy it to a new thread first.
- **Replay re-executes real nodes.** If a later node sends an email or pays a refund, replaying it does it again. Time travel is safe for pure steps and dangerous for side effects, unless they are idempotent (stage 7).
- Other threads are never affected (`test_another_thread_is_untouched_by_a_fork`).

## 3. Node retries

Stage 6 retried *tool calls* with middleware. A `RetryPolicy` retries any *node*:

```python
graph.add_node("lookup", lookup,
               retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.0, jitter=False, retry_on=TemporaryOutage))
```

```
node retry, 2 outages, 3 attempts: {'order_id': 'A1001', 'item': 'Trail Shoes'}
```

The same discipline as stage 6 applies, and the tests pin it:

- `max_attempts` counts the first try, so 3 absorbs two outages. A third outage raises `TemporaryOutage` to the caller.
- **Only the listed exception is retried.** `OrderNotFound` is raised once and not retried (`test_an_error_that_is_not_listed_is_not_retried`).
- `max_attempts=1` means no retry.

Use a retry policy for a node that calls something flaky and is safe to repeat. Do not use one around a node that moves money, unless the node is idempotent.

## Run it

```
$ python stage10_durability.py
after a restart: {'status': 'refunded', 'refunded_cents': 8900, 'refund_ids': ['R-0001']}

base run (fee 10%): ['loaded A1001', 'fee 10%', 'quote $80.10']
history: [(-1, ('__start__',)), (0, ('load',)), (1, ('price',)), (2, ('finish',)), (3, ())]
replay from before 'price' ran: {'price': 1, 'finish': 1}
what if the fee were 0%: quote $89.00 | nodes run: {'price': 1, 'finish': 1}
history is now 11 checkpoints long; the old ones are still there
```

## What can go wrong

- **Resuming in a different code version.** If you change the graph while tickets are paused, the saved state may not fit the new nodes. Version your graphs, and drain or migrate paused threads before a deploy.
- **Unbounded history.** Every step is a row. Prune old threads.
- **Secrets in checkpoints.** The whole state is stored in plain form. Keep secrets out of state, and think about retention (lesson 13).
- **Forgetting to close a connection**, as above.

## Try it

1. Kill the program for real (`os._exit(1)`) between `handle` and `resume`, in a child process. Does the ticket survive?
2. Run `what_if` from before `load` with a different `order_id`. What does the history look like afterwards?
3. Wrap `finish` with a side effect (append to a list). Replay twice. How many entries are there? What does that tell you about replay?

## Check yourself

- Why did the second "process" get models with fewer scripted turns?
- What is the difference between replaying and forking?
- Why is a retry policy safe on `lookup` but risky on a refund node?
