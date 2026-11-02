# Lesson 12: implementation guide

This explains **how the code works**: how the text sent to a model on each call is chosen by code, kept within a budget, kept honest as the conversation grows, and recorded so you can see why. The [README](README.md) says why the lesson exists and what the harness measured.

The one idea: **the context window is a budget, not a transcript.** Every model call is built from scratch from a few well-defined pieces, by deterministic code, not by the model's judgement.

---

## 1. The system on one page

```
 your agent loop
   │ add_user / add_assistant / add_tool_result          ContextManager (session.py)
   ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────┐
 │ transcript (append-only archive, never edited)                                            │
 │ FactStore   (facts.py)      versioned; trusted facts supersede, untrusted ones quarantined │
 │ Compactor   (compactor.py)  summary replaces old turns ONLY if code verifies it            │
 │ ToolResultStore (tool_results.py)  big results offloaded behind a handle; old ones stubbed  │
 │ ToolIndex   (tool_loadout.py)      offer only the tools likely needed this step           │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
   │ build(current, retrieved, volatile)
   ▼
 ContextAssembler.assemble  ──►  Assembled(system, tools, messages) + InspectionRecord (why, in numbers and ids)
   ▼
 client.messages.create(**assembled.request())
```

---

## 2. One model call, from start to end

### Where it starts
`ContextManager.build(current, retrieved=(), volatile="", call_id="")` in `session.py`. `current` is the new user message and is **not** added to the transcript by `build`, so call `add_user(current)` afterwards.

### What happens, in order

1. **Maybe compact** (`maybe_compact`). It returns one of `skipped`, `cooling_down`, `compacted` or `kept_raw`:
   - fewer than `keep_recent` live turns, or fewer tokens than `compact_over_tokens`: `skipped`
   - otherwise the older turns (everything except the last `keep_recent`) are sent to the `Compactor`. The previous summary is included as source material, so numbers it carries are not flagged as invented.
   - the summary replaces those turns **only if it passes verification** (below). Then `compacted_through` moves forward and the live history shrinks.
   - if it fails, the raw turns are kept (`kept_raw`) and the manager **waits `compaction_cooldown_turns` before trying again**. Without that cooldown a summariser that keeps failing would be called, with a retry, on every turn: a silent cost leak.
2. **Assemble** (`ContextAssembler.assemble`), described next.
3. The inspection record is appended to `records`, and the `Assembled` object is returned.

### `assemble`: how the context is built
Order matters: **stable things first**, so a provider's prompt cache can reuse the prefix.

1. **Stable prefix check.** The system prompt plus the tool definitions must not contain anything that changes per call (an ISO date, a clock time, "today is", a 32-hex id). `looks_volatile` is a heuristic that catches common mistakes; a hit raises `UnstablePrefix` telling you to put it in the `volatile` argument. The reason: one changing character at the front invalidates the whole cached prefix.
2. **Required pieces** are the prefix, the pinned facts and the current request (with volatile text). They are never dropped. If they alone exceed `Budget.total`, `ContextOverflow` is raised: that is a configuration bug, and quietly truncating something important would hide it.
3. **Optional pieces each get their own budget:**
   - `summary`: kept if it fits its budget, otherwise dropped
   - `retrieved` documents: best score first until the budget is full
   - `history`: newest first; once one turn does not fit, **everything older is dropped too**, so there are no gaps in the middle of a conversation
   - `tool_results`: the last `keep_full_tool_results` (default 2) in full, older ones replaced by one-line stubs, newest first within the budget
4. **Global shedding.** If the total is still over `Budget.total`, components are shed in `drop_order` (default: tool results, then retrieved, then history, then summary), oldest or lowest-scoring first, until it fits.
5. **Layout.** One user message, the "preamble", holds `<pinned_facts>`, `<conversation_summary>`, `<retrieved_documents>` and `<tool_results>` (the last two say outright that they are data, not instructions). Then the recent turns verbatim, then the current request. `_merge_roles` merges consecutive same-role messages so the conversation always alternates.
6. **Inspection record.** `InspectionRecord`: token count per component, the ids that were **selected**, the ids that were **dropped with the reason** (`over component budget` or `over total budget`), a hash of the prefix and a hash of the whole context. It holds ids and hashes, not the text.

### Where it ends
`Assembled` with `system`, `tools`, `messages`, and `.request()` giving the keyword arguments for `client.messages.create`. The same inputs always give the same context.

---

## 3. Module reference

### `facts.py`: runtime state instead of re-reading a transcript
`FactStore.set(key, value, turn, source)`:
- a **trusted** source (the user, the system) makes the fact **active** and **supersedes** the older value. Only the current value is rendered; history is kept for audit.
- any other source (a tool result, the model) makes it **quarantined**: not shown until `confirm(key, version)` by a person or a trusted validator
- an untrusted source that **contradicts a user-stated fact** is quarantined with that reason: it can never silently override the user
- a per-key **validator** that fails also quarantines it, with the reason
- setting the same value again does nothing (no version churn)

`render_pinned()` returns only current trusted facts in a fixed order, so a hallucinated flight or an injected claim is not reused turn after turn. `superseded_values()` lists values that are no longer true, which the compactor uses.

Live extraction: the model proposes facts, **code verifies them**. `verify_extraction` requires the key to be in `ALLOWED_KEYS` and the `quote` to appear in the user's actual message; failures go back for a retry. Grounded facts are stored as `user` facts.

### `compactor.py`: compaction you can trust
Summarising is lossy and irreversible, so a summary is accepted only after `verify_summary` finds none of:
- **invented numbers**: a number or price in the summary that appears nowhere in the source or the fact store (`$3,000` and `3000` count as equal)
- **dropped numbers**: a number the *user* stated that the summary lost and the fact store does not preserve
- **stale values**: a value that has been superseded but still appears in the summary

It also requires the summary to be **smaller** than the source. On failure the problems go back to the summariser for a retry (`max_retries`, default 1); if it still fails the result is `kept_raw`: **fail safe, keep the raw turns.** The checks are conservative: a false alarm costs tokens, a missed problem silently corrupts memory. They do not catch a mis-stated summary that contains no number. `Transcript` is append-only, so a bad summary can always be redone.

### `tool_results.py`: the biggest source of bloat
`ToolResultStore.add(tool, text)`: a result over `offload_over_tokens` (default 500) is stored outside the context behind an **unguessable handle** (`res_` plus random hex), and the model gets a short preview and how to fetch more. `fetch(handle, offset, length)` returns one **bounded page** (never more than `page_chars`, so a caller cannot pull the whole thing in one call). `stub_text` is the one-line replacement for old results. The full text is kept, so nothing is lost; offloading is for the model's benefit only.

### `tool_loadout.py`: fewer tools per step
Too many tools hurts tool choice. `ToolIndex.select(query, k, always)` returns the `always` tools plus the top `k` by lexical relevance (word overlap weighted by rarity, with light plural stemming) and records what was dropped. Retrieval can **miss** the tool you need, so there is also a `search_tools` fallback the model can call, and `recall_at_k` measures how often the needed tools actually make it into the loadout. Lexical search has a vocabulary gap ("get around Paris" does not share words with `public_transport_info`); tags and good descriptions close it, and the tests show both.

### `tokens.py`
`estimate_tokens` is a heuristic (about 4 characters per token for English prose). It is wrong for code, numbers and non-English text. Pass your own counter (`count=`) for exact budgets.

### `harness.py`: do these mechanisms help?
Compares strategies on a 40-turn synthetic conversation with planted facts: a budget (3000), a peanut allergy, and travel dates (April) that the user later changes to October. For each strategy it asks: is each **current** fact present, does the **stale** value still appear (a context clash), and how many tokens were sent (current and cumulative). Strategies: `FullHistory`, `TruncateLastK`, `NaiveSummary` (no verification, no fact store) and `Managed` (fact store + verified compaction + assembler). The summarisers are **test doubles with known behaviour** (one drops digits, one invents a number), so the results show what the mechanisms do, not how well a real model summarises. `python harness.py live` asks a real model; I have not run it.

### `structured.py` and `fakes.py`
A validated, retrying model call (used for fact extraction), and scripted test doubles.

---

## 4. Design decisions

- **Deterministic assembly.** The same inputs give the same context, so it can be tested and cached.
- **Stable prefix first,** volatile text last.
- **Required versus optional.** What must never be lost is separated from what may be shed, and a too-small budget is an error, not a silent truncation.
- **State is not a transcript.** A fact has a current value and a history.
- **Provenance decides trust.** Untrusted sources are quarantined, so a tool result cannot rewrite what the user said.
- **Summaries are verified or discarded.**
- **Inspect everything.** The record says what was in, what was out and why.

## 5. What it does not do

The token estimator is approximate. Verification checks numbers and superseded values, not meaning. The loadout is lexical, so it can miss a tool. A fact being present in the context does not mean the model uses it, which only a live evaluation can show (not run here). There is no real prompt-cache integration; the stable prefix is what makes one possible.

## 6. Running and testing

`python harness.py` runs offline. `python harness.py live` needs an API key. `python -m pytest -q` runs 90 offline tests.
