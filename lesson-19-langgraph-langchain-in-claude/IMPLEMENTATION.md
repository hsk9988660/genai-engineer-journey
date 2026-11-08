# Lesson 19: implementation guide

This explains **how the final system works**: where a ticket starts, what happens at each node, where it pauses, and where it ends. The [README](README.md) says why the lesson exists and what building it showed. The stage-by-stage teaching is in [docs/](docs/00-start-here.md). Read the docs first if you are new to LangChain or LangGraph.

This guide covers the **final desk** (stage 7), which is the heart of the lesson. Section 6 lists the Part 2 modules (stages 8 to 15), which are separate, smaller programs on the same shop.

The one idea: **a workflow you can draw is better than an agent that wanders, when the cost of a wrong move is money.** The model works inside the boxes. Code owns the arrows, the rules and the record of what happened.

---

## 1. The system on one page

```
 handle(app, text, customer_id=, thread_id=)            stage7_support_desk.py
      │   context = DeskContext(customer_id, customer_name)      verified by YOUR code
      ▼
 ┌─ triage ───────────────────────────────────────────────────────────────────────┐
 │  triage_request()  (stage3)   model fills Triage; schema check; one retry       │
 │  verify()                     id in the message? order exists? customer's own?  │
 │  decide_route()               plain Python -> route + reason                    │
 └──────────────┬───────────────────────────┬───────────────────────────┬──────────┘
                ▼ "refund_flow"             ▼ "answer_faq"              ▼ "escalate"
   hardened agent (a subgraph)      faq_search() finds articles    desk.open_ticket()
    prompt · audit · errors ·       model answers from them only   AIMessage written by code
    retry · limit · approval        (no match never gets here)
    tools: lookup_order,
    check_refund_policy,
    issue_refund, search_faq
                └───────────────────────────┬───────────────────────────┘
                                            ▼
                          finish: resolution from the ledger  ──►  END
```

Underneath all of it, `desk.py` holds the rules and the ledger. No LangChain code is imported there.

---

## 2. A ticket, from start to end

### Where it starts
`handle(app, text, customer_id=..., customer_name=..., thread_id=...)` calls `app.invoke({"messages": [HumanMessage(text)]}, config, context=DeskContext(...))`. The `thread_id` is the key under which everything is saved. The context carries identity beside the conversation, not inside it.

### What happens, in order

1. **`triage` node.** It reads the last message and calls `triage_request(models.triage, text, desk, customer_id)`:
   - The model is asked for a `Triage` object (`category`, `order_id`, `urgency`, `summary`) through `with_structured_output`.
   - If the shape is wrong, or the model returns text, it is asked **once more** with the error. After the last attempt, a fallback `other`, unverified, is returned.
   - `verify` checks the claims: an order id must appear in the message, exist, and belong to this customer; a refund needs an id.
   - `decide_route` turns the verified record into a route and a reason. The node returns `triage`, `route`, `reason` and an `events` entry.
2. **The conditional edge** is `lambda state: state["route"]`. It maps the route to one of three nodes.
3. **`refund_flow`** is the hardened agent used as a node. Its loop:
   - the model asks for tools; `ToolErrorMiddleware`, `ToolRetryMiddleware` and `ToolCallLimitMiddleware` wrap each call, with the audit trail outermost;
   - `issue_refund` above $25 triggers an **interrupt** from `HumanInTheLoopMiddleware`. The whole graph is saved and `invoke` returns with `__interrupt__`;
   - on resume, an approved or edited call runs through the tool. The desk checks every rule again;
   - the model writes its closing message.
4. **`answer_faq`** (only reached when articles matched) sends the model the articles and the rule "answer using ONLY these".
5. **`escalate`** opens a ticket and writes the customer message in code.
6. **`finish`** builds the `resolution` from facts: refunds whose key starts with this ticket's thread id, or the ticket id.

### Where it pauses and resumes
`pending_approval(result)` reads the pause. A person decides. `resume(app, approve() | edit(...) | reject(msg), thread_id, customer_id)` calls `app.invoke(Command(resume=...), config, context=...)`. The context must be supplied again: it is per call, not saved.

### Where it ends
`finish` always runs last, and the result holds `messages`, `triage`, `route`, `reason`, `events` and `resolution`. The resolution statuses are `refunded`, `declined`, `answered` and `escalated`.

---

## 3. Reference

### `desk.py`: the rules (no LangChain)
- `Desk(orders=None, today=TODAY)`: `get_order`, `policy_for`, `issue_refund`, `refunded_cents`, `open_ticket`, `faq_search`. `outages` is a test hook for simulating a failing order system.
- Errors: `DeskError` base; `OrderNotFound`, `RefundRejected`, `TemporaryOutage`.
- Constants: `RETURN_WINDOW_DAYS = 30`, `AGENT_REFUND_LIMIT_CENTS = 20_000`.
- `to_cents(x)` accepts a clean positive amount with at most two decimals and refuses everything else. `money(cents)` formats for display.

### `tools.py`
- `DeskContext(customer_id, customer_name)`.
- `request_key(runtime)`: `thread_id:tool_call_id`, or the tool-call id alone when there is no thread.
- `make_tools(desk)`: `lookup_order`, `check_refund_policy`, `issue_refund`, `search_faq`.

### Stages
- `stage1_first_agent.build_agent(model, desk, extra_tools=, middleware=, checkpointer=, store=, system_prompt=)` and `ask(agent, text, customer_id=, thread_id=)`.
- `stage2_graph_by_hand.build_graph(model, desk, max_model_calls=, checkpointer=)` and `shared_reducer_gotcha()`.
- `stage3_triage.Triage`, `triage_request(...)`, `verify(...)`, `TriageResult`.
- `stage4_memory.build_memory_agent(...)`, `remember_preference`, `recall_preferences`, `thread_summary`, `checkpoint_history`.
- `stage5_approval.build_approval_agent(...)`, `approval_middleware(desk)`, `needs_approval`, `pending_approval`, `approve/edit/reject/resume`.
- `stage6_middleware.AuditTrail`, `build_hardened_agent(model, desk, trail, approval=, tool_call_limit=, max_retries=, extra_tools=, checkpointer=, store=)`.
- `stage7_support_desk.DeskModels(triage, agent, faq)`, `decide_route`, `build_desk(models, desk, trail=None, checkpointer=None)`, `handle`, `path_of`.

### Support
- `fakes.ScriptedChatModel(script=[...])` with `seen` and `cursor`; helpers `say(text)`, `call(name, args, call_id=)`, `calls(*requests)`. Tool-call ids are unique across a test run.
- `show.format_messages(messages)` and `show.final_text(messages)`.

---

## 4. Design decisions

- **Rules in `desk.py`, not in the tools or the prompts.** The tools translate between the model's world and the desk's. This is why an approved refund that breaks a rule is still refused, and why the rules have 43 tests that need no model.
- **Identity in context, not in arguments.** `ToolRuntime` hides it from the model's schema.
- **Money as integer cents.** `to_cents` refuses a third decimal instead of rounding.
- **Domain errors as data, bugs as errors.** The model can react to "not eligible". A bug is turned into a message that names only the exception type.
- **The model fills a form; code reads it.** Triage output is checked against the message and the records before anything acts on it.
- **Routing in one pure function,** so a table of cases tests every branch and the order of the checks.
- **One checkpointer, on the parent.** The nested agent gets `checkpointer=None` and inherits it. The pause inside the agent is saved by the parent.
- **Idempotency key = `thread:tool_call_id`.** It makes replays safe, avoids cross-thread collisions, and lets `finish` find the refunds of one ticket without sharing state.
- **`finish` reads the ledger.** What the customer was told and what happened are recorded separately and can be compared.
- **A scripted model.** It tests everything around the model, deterministically and for free. It cannot tell you whether Claude behaves.

## 5. What this deliberately does not do

No real model in the tests, no persistence beyond RAM, no authentication of customers or reviewers, no queue or timeout for pending approvals, no concurrency control in `Desk`, no tracing service, no async, no token streaming. The first three of those are where a real deployment starts; lessons 10, 13 and 16 cover the rest.

---

## 6. Part 2: the deeper modules (stages 8 to 15)

Each module is self-contained, has a `demo()`, and is explained in its own file under [docs/](docs/00-start-here.md). They reuse `desk.py`, `tools.py` and `fakes.py`, and nothing in the final desk depends on them.

| Module | Key entry points | What it shows |
|---|---|---|
| `stage8_parallel.py` | `build_report_graph(desk)`, `order_report(...)`, `concurrent_writes(...)`, `join_runs(...)` | `Send` fan-out, reducers for fan-in, a width cap, joins, custom stream events |
| `stage9_commands.py` | `build_clarify_graph(desk)`, `start`, `reply`, `pending_question`, `side_effect_runs(...)` | `Command(update, goto)`, `interrupt()` in a node, the run-twice rule, input and output schemas, a safe resume |
| `stage10_durability.py` | `open_saver`, `pause_then_restart`, `build_quote_graph`, `history`, `checkpoint_before`, `replay`, `what_if`, `build_lookup_graph` | a SQLite saver surviving a restart, time travel, `RetryPolicy` |
| `stage11_streaming.py` | `ticket_events`, `ticket_events_async`, `report_events`, `nested_nodes`, `spoken_text` | every stream mode, token events, `subgraphs=True`, async |
| `stage12_runnables.py` | `build_reply_chain`, `build_analysis_chain`, `build_drafting_chain`, `build_reliable_reply_chain`, `draft_many`, `redact` | LCEL, prompts, parsers, parallel branches, retry, fallbacks, batch, streaming |
| `stage13_messages_tools.py` | `safe_window`, `trim_history`, `RefundArgs`, `make_demo_tools`, `build_demo_agent` | valid history cuts, schema limits, `ToolException`, artifacts, state-writing and state-reading tools |
| `stage14_rag.py` | `KnowledgeBase`, `HashingEmbeddings`, `split`, `ask_knowledge_base`, `make_search_tool` | splitting, embeddings, filter before ranking, a relevance threshold, verified citations |
| `stage15_model_ops.py` | `UsageTracker`, `CountingCache`, `CountingLimiter`, `cached_model`, `limited_model`, `make_model`, `build_ops_agent` | callbacks, cache, rate limit, retry and fallback order, summarising |

Changes that Part 2 made to the shared modules:

- `fakes.py`: `ScriptedChatModel` can stream word by word, records the tools bound to it (`bound`), and `say`/`call` can attach token counts. `FlakyChatModel` fails a set number of times.
- `tools.py`: `DeskContext` gained `audience` (default `"public"`), used by stage 14.

Two design points carry across Part 2: every module has a **pure-Python core** that you can test with no model (`order_report`, `redact`, `KnowledgeBase.search`), and every LangGraph or LangChain feature is shown next to the **mistake** it invites, with a test that fails if the mistake appears.
