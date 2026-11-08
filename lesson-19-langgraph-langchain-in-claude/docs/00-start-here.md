# Start here

This is a hands-on project in two parts. **Part 1** (stages 1 to 7) builds a customer-support agent for an outdoor-gear shop with **LangChain** and **LangGraph**, one stage at a time. **Part 2** (stages 8 to 15) goes deeper into each library with small, tested examples on the same shop: LangGraph's parallel work, `Command` and `interrupt()`, durability and time travel, and streaming; then LangChain's runnables, messages and tools, retrieval, and the operations around a model (cost, cache, rate limits, fallbacks). Every stage has three files: a short explanation (this folder), the code (next to the README), and tests that prove the code does what the explanation says.

You do not need an API key. A scripted fake model plays the part of Claude, so everything runs offline and costs nothing.

## What you will build

A customer writes a message. The desk decides what to do with it:

```
                        ┌──────────────► refund_flow ──┐   an agent with tools, that pauses
                        │                              │   for a human before money moves
 customer ─► triage ────┼──────────────► answer_faq ───┼─► finish ─► resolution
 message     (a schema, │                              │
              checked   └──────────────► escalate ─────┘   opens a ticket for a person
              by code)
```

- `triage` turns free text into a typed object, then code checks the model's claims.
- `refund_flow` is an agent. It looks things up, and when it wants to issue a refund over $25, a person must approve.
- `answer_faq` answers from help articles that code found. If none match, nobody guesses.
- `escalate` makes a ticket. It never calls a model.
- Code, not the model, decides which of the three branches runs.

## LangChain and LangGraph in plain words

| | LangChain | LangGraph |
|---|---|---|
| What it is | The parts: a common interface to chat models, tools, messages, and a ready-made agent (`create_agent`) | The engine: a graph of steps that share state, can be saved, paused and resumed |
| You use it for | "Give a model some tools and let it work" | "I need to control exactly what runs, in what order, and be able to pause" |
| Relationship | `create_agent` builds a LangGraph graph and returns it | Runs it |

In stage 1 you use the ready-made agent. In stage 2 you build the same agent yourself as a graph, so you can see there is no magic. From then on you mix both.

## Twelve words you need

| Word | Meaning here |
|---|---|
| **Message** | One item in the conversation: `HumanMessage`, `AIMessage` (may carry tool calls), `ToolMessage` (a tool's result), `SystemMessage` |
| **Tool** | A Python function the model may ask to run. The model sees only its name, description and argument schema |
| **Agent loop** | Ask the model, run the tools it asked for, send results back, repeat until it answers |
| **State** | A typed dictionary that every step reads and updates. The conversation is one key of it |
| **Reducer** | The rule for merging an update into one key of the state: append, concatenate, or overwrite |
| **Node** | A step. A function from state to a partial state update |
| **Edge** | What runs next. A *conditional* edge picks using a function of the state |
| **Graph** | Nodes and edges together. You *compile* it, then call `invoke` or `stream` |
| **Checkpointer / thread** | Saves the state after every step under a `thread_id`. This gives you conversation memory, pauses and replay |
| **Store** | Long-term memory that outlives a conversation, organised by namespace |
| **Interrupt** | A pause that waits for a person. The state is saved, `invoke` returns, and you resume later |
| **Middleware** | A layer around the model call or the tool call: logging, retries, limits, approvals |

## The stages

| Stage | You learn | Code | Explanation |
|---|---|---|---|
| 0 | The business rules, with no model at all | `desk.py`, `tools.py` | this file |
| 1 | `create_agent`, `@tool`, the verified context | `stage1_first_agent.py` | [01](01-first-agent.md) |
| 2 | The same agent as a hand-built graph: state, reducers, edges | `stage2_graph_by_hand.py` | [02](02-graph-by-hand.md) |
| 3 | Structured output, and checking it against the truth | `stage3_triage.py` | [03](03-structured-triage.md) |
| 4 | Memory: threads, checkpoints, a store | `stage4_memory.py` | [04](04-memory.md) |
| 5 | Human approval with interrupts | `stage5_approval.py` | [05](05-human-approval.md) |
| 6 | Middleware: audit, retry, limits, a dynamic prompt | `stage6_middleware.py` | [06](06-middleware.md) |
| 7 | Everything as one workflow graph | `stage7_support_desk.py` | [07](07-support-desk.md) |
| 8 | **LangGraph:** parallel work with `Send`, reducers, joins | `stage8_parallel.py` | [08](08-parallel.md) |
| 9 | **LangGraph:** `Command`, `interrupt()` in a node, input and output schemas | `stage9_commands.py` | [09](09-commands.md) |
| 10 | **LangGraph:** SQLite persistence, time travel, node retries | `stage10_durability.py` | [10](10-durability.md) |
| 11 | **LangGraph:** every streaming mode, subgraphs, async | `stage11_streaming.py` | [11](11-streaming.md) |
| 12 | **LangChain:** runnables, the `\|` pipe, prompts, parsers, retry and fallbacks | `stage12_runnables.py` | [12](12-runnables.md) |
| 13 | **LangChain:** messages, trimming, tools with schemas, artifacts, state-writing tools | `stage13_messages_tools.py` | [13](13-messages-and-tools.md) |
| 14 | **LangChain:** retrieval (RAG): splitting, embeddings, a vector store, checked citations | `stage14_rag.py` | [14](14-rag.md) |
| 15 | **LangChain:** callbacks, caching, rate limits, model fallback, summarising | `stage15_model_ops.py` | [15](15-model-ops.md) |

When you finish, read [16-cheat-sheet.md](16-cheat-sheet.md). It lists the main LangChain and LangGraph functions in one place.

## Stage 0: the business core

Open `desk.py`. It has **no LangChain in it**, on purpose. It holds the orders, the refund rules (30-day window, $200 limit for an agent, no refund above what is left on the order), a refund ledger, tickets and help articles. The tools in `tools.py` are thin wrappers around it.

Why start here? Because the rules about money must not depend on a model behaving well. A model can be fooled, confused, or edited by a human reviewer who types a wrong number. If the rules live in plain Python, they still hold. Every stage after this one leans on that.

Skim `desk.py` and `test_desk.py` before stage 1. Two details are worth seeing now:

- Money is stored as whole **cents** (integers). `to_cents(1.005)` is refused, not rounded.
- `get_order` gives the **same message** for "no such order" and "someone else's order", so the answer cannot be used to probe other customers' orders.

## How to run it

```bash
cd lesson-19-langgraph-langchain-in-claude
pip install -r requirements.txt         # langchain, langgraph (+ sqlite saver, text splitters), numpy, pytest, langchain-anthropic

python stage1_first_agent.py            # every stage has a demo that prints what happened
python stage7_support_desk.py
python stage14_rag.py
python -m pytest -q                     # the offline tests for this lesson (about 30 seconds)
```

Developed and tested with Python 3.14.2, `langchain` 1.4.9, `langgraph` 1.2.20 and `langchain-core` 1.7.6. These libraries change quickly. If something differs in your version, the tests will tell you what.

## Why a scripted model

`fakes.py` defines `ScriptedChatModel`. You give it the list of replies it will make, in order. It ignores the prompt. It also records every list of messages it was sent, in `model.seen`, so a test can ask "what did the agent actually show the model?".

```python
model = ScriptedChatModel(script=[
    call("lookup_order", {"order_id": "A1001"}),   # turn 1: the model asks for a tool
    say("It is a pair of Trail Shoes."),            # turn 2: the model answers
])
```

This tests everything **around** the model: the loop, the tools, the rules, the pauses. It cannot test whether Claude writes good answers. That needs the live API (see `run_live.py`) and an evaluation set (lesson 10).

## Reading order and time

Plan about 30 to 45 minutes per stage if you run each demo and read the test names. Part 1 (stages 1 to 7) must be done in order. After it, the stages of Part 2 are independent of each other, so read the ones you need. Stages 8 to 11 are LangGraph, stages 12 to 15 are LangChain. Test names are written as sentences on purpose, so `python -m pytest -v` reads like a specification.
