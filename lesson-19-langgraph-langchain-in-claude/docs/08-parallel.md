# Stage 8: parallel work in LangGraph

**Code:** [stage8_parallel.py](../stage8_parallel.py) | **Tests:** [test_stage8_parallel.py](../test_stage8_parallel.py) | **Run:** `python stage8_parallel.py`

This is the first stage of **Part 2**, a deeper look at LangGraph and LangChain. Stages 1 to 7 gave you the working desk. Part 2 takes each library feature you have not seen yet and shows it with a small, tested example.

## The idea

A customer writes: "Can you check orders A1001, A1002 and A1003?" Checking them one after another wastes time. LangGraph can run the checks at the same time. To use that safely you need to understand the **superstep**.

LangGraph runs a graph in rounds. In each round every node that is ready runs "at once". Their updates are merged into the state when the round ends, and the next round starts from the merged state. Three rules follow.

### Rule 1: fan-out with `Send`

```python
def fan_out(state):
    return [Send("check_order", {"customer_id": state["customer_id"], "order_id": oid}) for oid in state["order_ids"]]

graph.add_conditional_edges("plan", fan_out, ["check_order", "collect"])
```

A conditional edge may return a list of `Send(node, input)`. Each one starts one copy of `check_order` with **its own private input** (`CheckInput`, which is not part of the main state). The number of copies is decided at run time: one per order id. This pattern is called map-reduce: map a worker over many items, then reduce the results.

### Rule 2: fan-in needs a reducer

Every worker returns `{"results": [entry]}`. The key `results` has the reducer `operator.add`, so the entries are concatenated. Without a reducer, LangGraph refuses:

```
concurrent writes, no reducer : InvalidUpdateError   ("Can receive only one value per step. Use an Annotated key ...")
concurrent writes, reducer    : ['from a', 'from b']
```

This is a good design. The alternative, silently keeping one of two writes, would lose data without telling you. (`concurrent_writes` in the file shows both.)

### Rule 3: a join waits only if you ask

Branch `a` takes one step. Branch `b` takes two (`b1`, then `b2`). How many times does `join` run?

```python
graph.add_edge("a", "join"); graph.add_edge("b2", "join")   # separate edges: join runs 2 times
graph.add_edge(["a", "b2"], "join")                          # a list: join runs once, after both
```

Separate edges mean "run `join` after `a`" and "run `join` after `b2`", which happen in different rounds. This is an easy mistake to make, and nothing warns you. `join_runs(wait_for_all)` counts the runs: 2 versus 1.

In the report graph, `check_order -> collect` is a single edge, but all the workers of one `Send` round finish in the **same** round, so `collect` runs once.

## The report graph

```
START -> plan -> (fan_out) -> check_order x N -> collect -> END
                     └──────── no orders ──────────────────► collect
```

- `plan` removes duplicate ids and **caps the width** (`MAX_ORDERS = 10`). A fan-out whose width a customer controls must always be capped by code. Over the cap, `plan` raises `TooManyOrders` *before any worker starts* (`test_one_more_is_refused_before_any_worker_starts`).
- With **no** orders, `fan_out` would return an empty list and nothing would ever reach `collect`, so the graph would end without a report. It returns the string `"collect"` instead (`test_no_orders_still_produces_a_report`).
- A worker turns a missing order into an **error entry** (`{"order_id": ..., "error": ...}`) instead of failing. One bad id does not stop the others.
- `collect` **sorts** the results by order id. Workers finish in no guaranteed order, so never rely on arrival order. The test runs the same ids forwards and backwards and gets the same report.
- Another customer's order is reported as missing, with no details (`test_another_customers_order_is_reported_as_missing_not_leaked`).

## Run it

```
$ python stage8_parallel.py
{'order_id': 'A1001', 'item': 'Trail Shoes', 'eligible': True, 'refundable_cents': 8900, ...}
{'order_id': 'A1002', 'item': 'Camp Stove', 'eligible': True, 'refundable_cents': 24000, ...}
{'order_id': 'A1003', 'item': 'Rain Jacket', 'eligible': False, 'refundable_cents': 0, 'reason': 'outside the 30-day return window ...'}
{'order_id': 'A1004', 'error': 'no order A1004 found for this customer'}

found=3 missing=['A1004'] refundable_total=$329.00
```

## What can go wrong

- **An unbounded fan-out.** `Send` per item is cheap to write and expensive to run. Cap it.
- **Relying on order.** Results arrive in no fixed order. Sort, or key them by id.
- **A worker that raises.** In this stage an outage in one worker is **not hidden**: the whole run raises `TemporaryOutage` (`test_an_outage_in_a_worker_is_not_hidden`). Stage 10 shows `RetryPolicy`, the right tool.
- **Parallel is not always faster.** For tiny pure-Python work, running workers concurrently buys nothing. It pays when workers wait on a network, a database or a model.
- **Parallel model calls.** Each worker could call a model. Mind your rate limit (stage 15) and your cost.

## Try it

1. Make `check_order` a model call that writes a one-line summary of each order. What do you do about the rate limit, and about a model that fails on one order?
2. Add a second kind of worker, `check_shipping`, started by the same `fan_out` with a different `Send`. Both write `results`. What do you need to change so `collect` can tell them apart?
3. Raise `MAX_ORDERS` to 11 and watch which test fails. Is the failure telling you something useful?

## Check yourself

- What happens when two nodes in the same round write the same key and it has no reducer?
- Why does `collect` sort?
- `add_edge("a", "join")` plus `add_edge("b2", "join")`, or `add_edge(["a", "b2"], "join")`: which one waits for both?
- Why must the width of a `Send` fan-out be capped in code?
