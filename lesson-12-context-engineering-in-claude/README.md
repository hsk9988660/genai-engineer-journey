# Lesson 12: context engineering, as deterministic code

Lesson 12 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) names the right ideas (write, select, compress, isolate; poisoning, distraction, confusion, clash). Its notebook, though, never reduces any history. Its "summarization tool" returns the text it was given, its scratchpad is not implemented, and it never measures a token. This folder builds the machinery and measures it.

> **Context engineering is deciding, for every model call, exactly which tokens go in, in what order, and which stay out.** That is a deterministic job for your code, not something to ask the model to do.

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q          # 90 offline tests, no API key, under a second
python harness.py            # the strategy comparison below (offline: uses labelled test doubles, no model)
set ANTHROPIC_API_KEY=sk-ant-...      # macOS/Linux: export ANTHROPIC_API_KEY=...
python harness.py live       # asks a real model about the planted facts. NOT RUN by me; costs money
```

## What is built

| File | What it is |
|---|---|
| `assembler.py` | **The core.** Builds each model call from a stable prefix, pinned facts, a summary, retrieved data, recent turns and tool results, under per-component token budgets, and returns an inspection record. |
| `facts.py` | A **versioned fact store**: a new value *supersedes* the old one; untrusted-source facts are *quarantined*. Plus a live extractor whose proposed facts must be backed by a verbatim quote. |
| `compactor.py` | **Verified compaction.** A summary replaces raw turns only if it passes deterministic checks. Otherwise the raw turns stay. |
| `tool_results.py` | Offloads huge tool results behind an unguessable handle and bounded page fetches; stubs old results. |
| `tool_loadout.py` | Picks the few tools relevant to this step, with a `search_tools` fallback and a recall measurement. |
| `session.py` | Ties them together for one agent session, with a cooldown after failed compactions. |
| `harness.py` | Compares strategies on a long synthetic conversation. |

## The rules the code enforces
1. **The prefix is stable.** Dates, clock times and ids are refused in the system prompt and tool descriptions (`UnstablePrefix`). Volatile text goes *after* the prefix, so a provider's prompt cache can reuse it.
2. **Required pieces are never silently truncated.** The prefix, pinned facts and current request are always included. If they alone exceed the budget, it raises `ContextOverflow` instead.
3. **Optional pieces have their own budgets and a drop order** (tool results, then retrieved documents, then history, then summary). History is always a contiguous recent suffix, with no gaps.
4. **A summary must pass checks before it replaces anything:** no invented numbers, no dropped user numbers (unless preserved in the fact store), and no superseded values still mentioned. On failure the raw turns stay.
5. **A changed preference supersedes the old one explicitly.** The superseded value never appears in the context.
6. **Untrusted facts are quarantined.** A fact from a tool or the model never enters the context directly, and can never override something the user said.
7. **The inspection record contains no raw text:** token counts, ids of what was selected and dropped (with reasons), and hashes. It is safe to log.
8. **The raw transcript is never edited.** Compaction only changes what the next call receives.

## Measured results

`python harness.py` runs a 40-turn conversation. Early on the user states a budget ($3000), a peanut allergy and travel dates (April). At turn 24 they change the dates to October. The table shows what the context for the *next* call contains.

| Strategy | budget | allergy | dates | stale "April" | tokens in final context | total tokens sent across the conversation |
|---|---|---|---|---|---|---|
| full history | kept | kept | kept | **leaked** (clash) | 3,503 | 68,781 |
| truncate to last 10 turns | **lost** | **lost** | **lost** | clean | 462 | 17,276 |
| naive summary (no checks, no fact store) | **lost** | **lost** | kept | clean | 1,273 | 38,990 |
| **managed**, lossy summarizer | kept | kept | kept | clean | 1,313 | **40,294 (41% less than full history)** |
| **managed**, hallucinating summarizer | kept | kept | kept | clean | 3,016 | 68,549 |

And the cost curve under full history, which is the quadratic growth the lesson never shows:

| Turns | Total tokens sent |
|---|---|
| 10 | 4,137 |
| 20 | 16,965 (4.1x) |
| 40 | 68,781 (4.1x) |

**How to read this honestly.** The summarizers are **labelled test doubles** with known behavior (one drops every digit, one invents a number). So this shows what the *mechanisms* do: pinned facts survive a lossy summary, verification refuses a bad one and the raw turns stay, truncation loses early facts, and full history carries both April and October. It says **nothing about how well a real model summarizes.** A fact being present in the context is also necessary, not sufficient: the model still has to use it.

## Findings from building it (in my own code)
1. **A failing summarizer made the manager retry on every turn.** In the hallucinating run the harness showed compaction attempted on **23 consecutive turns**, each with a retry: a silent cost leak. I added a cooldown, and attempts dropped from 23 to 6 with identical outcomes.
2. **Per-case retrieval measurement exposed three tool-selection flaws that one aggregate number hid.** There was no stemming, so "find a hotel" missed a description saying "hotels". The words "get" and "find" matched tool *names* (`get_visa_rules`, `find_tours`) and pulled in unrelated tools. And the lesson's own example, "how do I get around Paris without a car", selected `rent_car` and not the public-transport tool. I fixed the first two, and tags fix the third. A fourth case, "what do I need to enter Japan", still returns **an empty loadout** because it shares no words with "visa requirements". That is why a `search_tools` fallback is essential.
3. **Repeated summarization compounds loss.** In the naive strategy each re-summarization re-truncates the previous summary, so facts erode over time. That is the behavior of my test double, but compounding loss through repeated summarization is a known real risk, and it is why raw turns are archived.
4. **The dropped-number check is very conservative.** It was refusing my own test fixtures because user turns contained digits. A real user mentions numbers constantly, so many good summaries would be rejected and the raw turns kept. That is safe but costs tokens. A test documents it, and it should be tuned for real traffic.
5. **A vacuous test of mine.** I wrote `assert ... or True`, which can never fail. I caught it in review and replaced it with a real assertion.
6. **A test gap found by reasoning, not by a failing test.** "Handles are unguessable" was only tested as "unique", so a simple counter would have passed. I added a test that would catch it.

## How I checked the tests can fail
I mutation-checked 23 behaviors in a scratch copy: quarantine, supersession, validators, quote verification, current-only rendering, the three summary checks, the not-smaller rule, the fail-safe fallback, the overflow error, prefix stability, the history budget, total-budget shedding, tool stubbing, the "data, not instructions" labels, raw-text-free records, the page cap, handle unpredictability, the cooldown, previous-summary-as-source, stemming, and always-include tools. **All 23 were caught.**

## What is NOT known
- **Nothing here ran against a live model.** I have no API key here. The live fact extractor and `harness.py live` are tested with scripted replies only. I do not know how well a real model summarizes under these checks, or how often they would reject a good summary.
- **Token counts are an estimate** (about 4 characters per token). Code, numbers and non-English text use more. Per-message overhead is not modelled. Plug in your provider's counter and keep a safety margin.
- **I did not test any provider's prompt caching.** The prefix design follows the general idea (stable first), but the actual rules (what is cached, for how long, at what price) are provider-specific. Check them.
- **The synthetic conversation is simple:** three planted facts, one change of mind, and filler with no digits in user turns. Real conversations are messier.

## Limits of the design
- **Facts are single-valued per key.** A second allergy *replaces* the first (a test shows this), which for a safety-relevant fact is data loss. Use per-item keys (`allergy.peanuts`, `allergy.shellfish`), or a list-valued fact type.
- **Verification only catches numbers and superseded words.** A summary can still misstate something that contains neither. The stale-value check also flags a legitimate "they changed from April to October", by design.
- **When compaction is refused, raw history still contains the superseded statement** until it ages out of the history budget. The fact store protects the *pinned* block, not the raw turns.
- **Lexical tool search has a vocabulary gap.** Tags and good descriptions help, but real systems usually need embeddings too, plus the fallback.
- **Quarantined facts need someone to confirm them.** There is no review UI here.
- **A pin is only as good as the extractor.** If extraction misses "I'm allergic to nuts", the fact never enters the store.

## Exercises
1. Run `python harness.py live` and compare the four answers. Does the model actually use a fact when it is present in the context?
2. Replace the test-double summarizer with a real model call. How often does verification reject it, and for which reason?
3. Make facts multi-valued. What should "supersede" mean for a list?
4. Make the dropped-number check smarter: only enforce numbers tied to pinned fact keys. What do you give up?
5. Add `volatile` information (today's date) and confirm the prefix hash stays constant across two simulated days.
