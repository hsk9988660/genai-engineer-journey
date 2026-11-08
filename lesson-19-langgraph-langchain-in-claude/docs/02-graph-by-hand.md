# Stage 2: the same agent, built by hand as a graph

**Code:** [stage2_graph_by_hand.py](../stage2_graph_by_hand.py) | **Tests:** [test_stage2_graph.py](../test_stage2_graph.py) | **Run:** `python stage2_graph_by_hand.py`

## The idea

`create_agent` returned something called a `CompiledStateGraph`. That is a **LangGraph** object. In this stage you build the same agent with LangGraph directly. Once you have done it, `create_agent` stops being a black box, and you can build workflows that are not a plain tool loop (stage 7).

LangGraph has four ideas. Learn these and you know the library.

1. **State.** A typed dictionary that every step reads and writes.
2. **Nodes.** Plain functions: state in, a *partial* state update out.
3. **Edges.** What runs next. A *conditional* edge calls a function of the state to choose.
4. **Reducers.** The rule for merging an update into a key of the state.

## State and reducers

```python
class DeskState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tools_used: Annotated[list[str], operator.add]
    model_calls: int
```

Each key has its own merge rule:

| Key | Reducer | What a node's update does |
|---|---|---|
| `messages` | `add_messages` | **Appends** the new messages. If a new message has the same `id` as an old one, it **replaces** it |
| `tools_used` | `operator.add` | **Concatenates** lists. It never removes duplicates |
| `model_calls` | none | **Overwrites**. The latest value wins |

A node returns only the keys it changes. That is why `call_model` returns `{"messages": [response], "tools_used": [...], "model_calls": n + 1}`. LangGraph applies each reducer. Tests: `TestState`.

## Nodes and edges

```python
graph = StateGraph(DeskState, context_schema=DeskContext)
graph.add_node("model", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_node("give_up", give_up)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", route_after_model, {"tools": "tools", "give_up": "give_up", END: END})
graph.add_edge("tools", "model")
graph.add_edge("give_up", END)
app = graph.compile()
```

- `ToolNode(tools)` is a ready-made node. It runs every tool call in the last AI message and returns one `ToolMessage` for each. It also injects `runtime`, so the context from stage 1 still works.
- `route_after_model` is the conditional edge. It is plain Python that reads the state:

```python
def route_after_model(state):
    last = state["messages"][-1]
    if not last.tool_calls:
        return END                       # the model answered
    if state["model_calls"] >= max_model_calls:
        return "give_up"                 # too many turns
    return "tools"
```

This is the whole agent loop in six lines, and **you can read where the loop stops**. Run the demo to see the graph as a Mermaid diagram:

```
__start__ ─► model ─► tools ─► model ...
                │
                ├─► give_up ─► __end__
                └─► __end__
```

Paste the printed diagram into <https://mermaid.live> to see it drawn.

## The step cap, done properly

`give_up` does something small and important. When the cap is reached and the model still wants tools, it does not just stop. It first answers every pending tool call with an error message (`"not run: the step limit was reached"`), then adds a final message. Why? Because a history that contains a tool call with no result is **invalid**: the next model request on that thread would be rejected. `test_every_tool_call_in_the_history_has_a_result` checks this, and `test_the_unanswered_tool_call_is_closed_and_not_executed` checks that the call really did not run (the refund that hit the cap was not paid).

## It really is the same agent

`test_it_gives_the_same_conversation_as_create_agent` runs the same script through your graph and through `create_agent`, and compares the messages. They match.

What `create_agent` adds on top is everything that comes in later stages (middleware hooks, structured output, interrupts) wired into the same kind of graph.

## The trap: shared reducers and subgraphs

A graph can be used as a node inside another graph (stage 7 does this). There is a trap, and the demo shows it:

```
subgraph with a shared additive key : ['before', 'before', 'sub did work']
subgraph started empty              : ['before', 'sub did work']
```

A subgraph returns its **whole** final state. If parent and subgraph share a key whose reducer adds (`operator.add`), the parent adds the old items a second time. `'before'` appears twice.

- Messages are safe, because `add_messages` skips an id it has already seen.
- A plain `operator.add` list is not safe.
- The fix used here: call the subgraph from a small node that starts it with an empty list, so it returns only what it added. Another fix is to give the subgraph a different schema. The simplest rule: **do not share an additive key with a subgraph unless it deduplicates.**

Tests: `TestTheSharedReducerTrap`.

## Run it

```
$ python stage2_graph_by_hand.py
customer : Can I get a refund for A1003?
model    : asks for check_refund_policy(order_id='A1003')
tool     : {"order_id": "A1003", "eligible": false, "refundable": "$0.00", "reason": "outside the 30-day return window...
model    : Order A1003 is outside the 30-day return window, so I cannot refund it.

tools_used=['check_refund_policy']  model_calls=2
```

## Compare with `create_agent`

| | Hand-built graph | `create_agent` |
|---|---|---|
| Lines of code | about 40 | 1 call |
| You can change the routing | Yes, anywhere | Only through middleware and options |
| Step cap | `max_model_calls`, with clean shutdown | `recursion_limit`, which raises an error |
| Extra state keys | Any you like (`tools_used`) | Through `state_schema` |

Use `create_agent` when the standard loop fits. Drop to a hand-built graph when you need a different shape. Stage 7 does the second.

## Try it

1. Add a node `check_budget` between `model` and `tools` that blocks any tool call to `issue_refund` when `tools_used` already contains it. Add a conditional edge for it.
2. Change `tools_used` to use no reducer. Run the demo. What do you get in the final state, and why?
3. Draw the graph for your idea on paper first, then build it. This is how real LangGraph work is done.

## Check yourself

- What does a node return? Is it the whole state?
- A key has no reducer and two nodes write it in the same step. What happens? (Look at the LangGraph documentation for concurrent writes, then try it.)
- Why does `give_up` answer the pending tool calls before it ends?
- Why can `add_messages` be used in a subgraph, but a plain `operator.add` list cannot?
