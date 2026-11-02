# Lesson 13: implementation guide

This explains **how the code works**: how an agent remembers things about a user across sessions, how a memory gets in, how the right ones are chosen for a turn, and how they are forgotten. The [README](README.md) says why the lesson exists and what the harness measured.

The one idea: **memory is a database with rules, not a list of text the model appends to.** Who the memory belongs to, what may be written, what is shown, and what is deleted are all decided by code.

---

## 1. The system on one page

```
 WRITE PATH (after each turn)                                  READ PATH (before each turn)
 user's message ONLY                                           user's message
    │                                                             │
    ▼  extractor PROPOSES candidates                              ▼  retrieve(): score active memories,
 verify_candidates: key registered? value short?                  pinned keys always, others by
    quote appears verbatim in the user's message?                 relevance + recency + importance, under a token budget
    │ (fail → rejected, never partly applied)                     │
    ▼                                                             ▼
 MemoryStore.remember(scope, ...)  ── SQLite ──►  active / quarantined / superseded / retracted
    trusted source + grounded + not an instruction                render_block → <user_memory> … </user_memory>
    + consent if sensitive  → ACTIVE, else QUARANTINED

 Identity: MemoryScope(user_id, session) is built by YOUR auth code. No method and no model tool takes a user id string.
```

---

## 2. A conversation turn, from start to end

### Where it starts
`MemoryManager` in `manager.py` is the harness. You call it twice per turn, from your own code:

1. `before_turn(scope, user_text)` returns the `<user_memory>` block to put in this turn's context. Put it **after** the stable prefix, since it changes.
2. Run the model (optionally with the three memory tools).
3. `after_turn(scope, user_text, turn)` writes memories.

Retrieval and extraction run in **your code on every turn** because if the model alone decides when to save and recall, it forgets to, or it is talked into doing it. The model's tools are a small extra.

### Read path, in order (`retrieval.retrieve`)
1. Load this user's **active, unexpired** memories (`store.active(scope)`). Quarantined, superseded, retracted and expired ones are never candidates.
2. Score each memory: `0.6 × relevance + 0.25 × recency + 0.15 × importance`. Relevance is the fraction of the memory's words (stop words removed, a trailing plural `s` stripped) that appear in the query. Recency is `0.5 ** (age_days / half_life_days)` with a 30-day half-life. The weights and half-life are **untuned defaults**, not measured values.
3. **Pinned keys** (`allergy`, `mobility`, `diet` by default) are included every turn, whatever the query says, because "what should I eat tonight?" shares no words with "peanuts". If the pinned memories alone exceed the budget, `MemoryBudgetExceeded` is raised: **required memories are never silently dropped.**
4. The rest are added best score first, but only if they have **some relevance** (otherwise they just fill space), up to `k` memories and the token budget. Ties break by id, so the result is deterministic.
5. `render_block` writes one line per memory. `render_line` HTML-escapes the value and collapses whitespace so a stored value **cannot close the block or start a new line**; the block opens with a header saying that these are remembered facts, not instructions. A memory taken from a user's turn is labelled "from the user, turn N".

### Write path, in order (`extract.py`, then `store.py`)
1. **Only the user's text goes in.** `after_turn` has no parameter for assistant or tool text, so those can never become `source="user"` memories.
2. The extractor returns `Candidate`s (key, value, a `quote`, and an action: remember or retract). In real use it is a model (`llm_extract`, a forced structured call, scripted in tests only); `rule_extract` is a regex **test double** so the pipeline runs offline.
3. `verify_candidates` rejects a candidate unless its key is registered, its value is short, and its quote appears verbatim (normalised) in the user's message. `apply_candidates` then writes the survivors, at most `MAX_CANDIDATES` per turn, and reports each outcome (`active`, `quarantined`, `retracted N`, `rejected`).
4. `MemoryStore.remember` decides the status:

| Condition | Result |
|---|---|
| key not in the registry | `UnknownKey` error |
| value empty or over 300 characters | `InvalidMemory` |
| `source="user"` without a quote of at least 3 characters | `InvalidMemory` |
| source not trusted (only `user` and `system` are) | **quarantined**: "untrusted source: needs confirmation" |
| value looks like an instruction (heuristic) | **quarantined** |
| key is sensitive (`allergy`, `mobility`) and no consent | **quarantined**: "needs the user's consent" |
| otherwise | **active** |

   For an active write: the same value again just refreshes it; a new value for a **single-valued** key (budget, dates, destination, class, mobility) marks the old one `superseded`; a **set-valued** key (diet, interest, dislike, allergy) accumulates. More than 500 memories per user raises `MemoryLimit`, checked only when a row would be added. Each write is audited.
5. `retract(scope, key, value)` handles "I am not vegetarian anymore"; only a trusted source may do it.

### Where it ends
A turn ends with the memory block used and the store updated. Everything that happened is in the audit log; `export_user` shows everything held about a user in every status.

---

## 3. Module reference

### `store.py`: the database
SQLite, standard library only. The rules that matter:
- **Identity is a `MemoryScope`** built from your auth system. Every query is parameterised and filtered by `scope.user_id`.
- **Typed keys with cardinality** (the `KEYS` registry), not free text.
- **Quarantine is a first-class status.** `confirm(scope, memory_id, consent)` lets a person approve one (sensitive ones need consent; single-valued keys then supersede the old value).
- **Forgetting.** `forget(scope, id)` is soft (status `retracted`, `restore` undoes it); `forget(..., hard=True)` deletes the row and runs `VACUUM`. `delete_user` erases every row for a user, including superseded and quarantined ones, then `VACUUM`. SQLite's default keeps the text of a deleted row in the file (I checked), so the store turns on `secure_delete` and a test scans the raw file for the deleted text. `purge_expired` removes rows past their expiry.
- **Hash-chained audit.** Every change appends an entry containing the previous hash; the user id appears only as a keyed HMAC (`_user_hash`); `verify_audit` recomputes the chain. As in lesson 06, this is tamper-evident, not tamper-proof.
- `looks_like_instruction` is a heuristic and easy to evade; the real defences are source trust, grounding, the key registry and the safe rendering above.

### `retrieval.py`
As described: `retrieve`, `score_memory`, `render_line`, `render_block`, `words`.

### `extract.py`
As described: `verify_candidates`, `apply_candidates`, `llm_extract`, `rule_extract`.

### `manager.py`
- `MemoryManager.before_turn` / `after_turn`, as above.
- `MemoryTools(store, scope, last_user_text)` executes the model's three tools: `remember`, `forget`, `list_memories`. **They take no user id**: the scope was fixed by your code for this turn, so an injected "read alice's memories" has nothing to aim at. `handle` rejects any unknown tool and **any argument not in the schema** (including an attempt to pass `user_id`) with an error rather than ignoring it. A model's `remember` becomes a trusted `user` memory only if its `quote` appears verbatim in the user's latest message; otherwise it is stored as `source="model"`, which is **quarantined**. The model can only **soft** forget: it never hard-deletes.

### `harness.py`
Compares the notebook-style memory (`NaiveListMemory`: the model passes a user id, every message is appended as text, recall returns everything) with the managed store on four scenarios: **drift** (a changed preference), **growth** (many sessions), **cross-user** (one user reading another's), and **poisoning** (an injected memory). Everything is offline and deterministic with labelled test doubles, so it shows what the mechanisms do and says nothing about how a real model extracts facts or whether it would follow a poisoned memory.

### `structured.py`
The same validated, retrying model call used before.

---

## 4. Design decisions

- **The harness, not the model, drives memory.** Reliable, and harder to manipulate.
- **Identity from your auth, never from the model.**
- **Grounding.** A memory needs the user's own words as evidence.
- **Quarantine instead of refusal.** Untrusted or sensitive input is held for confirmation, not silently saved and not silently lost.
- **Supersede, do not append.** The current budget replaces the old one; history is kept.
- **Pin what is dangerous to forget.** Allergies must not depend on a word match.
- **Deletion is real.** Soft and hard forget, per-user erase, and a check of the raw file.
- **Budgets.** Memory competes for the same context as everything else (lesson 12).

## 5. What it does not do

The scoring weights are untuned. Word-overlap relevance misses paraphrases (no embeddings). The instruction detector is a heuristic. A grounded quote proves the user *said* something, not that it is true or that the extractor understood it. The live extractor has not been run against a real model. There is no encryption of the database file and no multi-process locking beyond SQLite's.

## 6. Running and testing

Everything runs offline. `python harness.py` prints the comparison. `python -m pytest -q` runs 92 offline tests.
