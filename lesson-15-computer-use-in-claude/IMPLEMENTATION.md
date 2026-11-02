# Lesson 15: implementation guide

This explains **how the code works**: how a browser-driving agent observes a page, asks a model for one action, lets code decide, and acts, and how it is kept from being steered by the page. The [README](README.md) says why the lesson exists and what the harness measured.

The one idea: **the page is written by strangers.** Everything the model reads from it is untrusted data, and everything the model asks to do is a proposal that code checks, against a policy the page cannot influence.

---

## 1. The system on one page

```
 ComputerUseAgent.run(task, start_url)                       (agent.py)
      │
      ▼ each step:
 limits ok? (actions, time, tokens, refusals)
      ▼
 browser.snapshot()  ──►  observe.render()   hidden elements dropped, text escaped and capped,
      │                                      numbered [1] [2] …, injection-looking text FLAGGED
      ▼
 model, forced tool `act`: ONE action (navigate | click | type | done), by [number]
      ▼
 parse_action          strict shape check (wrong shape = refused, counted)
      ▼
 stuck check           same action on an unchanged page too often → stop
      ▼
 ActionPolicy.decide   →  allow | approve | deny | handover        (policy.py + urlpolicy.py)
      ▼
 approve: summary built by CODE from the real page → a person → answer must be for THAT summary
      ▼
 re-check the real page just before acting (element still there, same label/link, same prices)
      ▼
 browser acts; every URL hop goes through DomainPolicy
```

The model sees only: the task, the last eight things that happened (one short line each), and the current page. **There is no growing screenshot history**, so cost does not climb with every step.

---

## 2. A run, from start to end

### Where it starts
`ComputerUseAgent(client, browser, policy, limits, trace, approver).run(task, start_url)`. The browser is a small interface (`url`, `snapshot`, `navigate`, `click`, `type_text`); the lesson ships `SimBrowser` (a model of a browser, see `simsite.py`), not a real one.

### What happens, in order

1. **Start URL.** The first navigation is checked by the domain policy. If it is not allowed the run stops: "the start URL is not allowed".
2. **Loop guard.** Before each step the run stops if it hit the action cap (25), the time cap (120 s), the token cap, or more than `max_denied` (5) refused actions.
3. **Observe** (`observe.render`). Described in the module reference below. The trace records the page path, element count, how many hidden elements were dropped, how many were flagged and a short page hash, **never page text**.
4. **Ask** (`_ask`). One call with `tool_choice` forced to the `act` tool. If the model returns no `act` call, that is a refusal (counted) and the loop continues.
5. **Parse** (`parse_action`). The arguments are untrusted input: it must be an object, with no unexpected fields, with the right types for the action (a boolean is not an integer). `navigate` needs `url`, `click` needs `ref`, `type` needs `ref` and `text`, `done` carries an answer (capped at 500 characters). Anything else is `ActionError`, a refusal.
6. **Stuck check.** The key is the action plus the page hash. The same action on an unchanged page more than `max_repeats` (2) times ends the run: "stuck".
7. **Done.** A `done` action ends the run with `status="done"` and the model's answer.
8. **Decide and act** (`_act`):

| Policy decision | What the agent does |
|---|---|
| `deny` | refuse (counted), tell the model why, continue |
| `handover` | stop with `needs_user`: a password or card field is for the user, never the model |
| `approve` | hash the summary, ask the approver (none configured = not approved). Not approved: tell the model, continue. An answer whose `summary_hash` differs from this summary's hash: refused ("the approval was for a different action") |
| `allow` | continue |

   Then, **just before acting**, the real page is read again. If the element is gone, or its fingerprint (path, role, text, link, input type, and for approvals **every price visible on the page**) differs from what was shown (and approved), the action is refused: "the page changed since it was shown (and since the approval)". Only then does the browser navigate, click or type, with every URL hop checked by the domain policy (`NavigationBlocked` is a refusal).
9. A successful action increments the action count and adds a line to the history.

### Where it ends
`RunResult` with `status` (`done`, `stopped` or `needs_user`), a reason, the answer, and counts of actions, refusals, tokens and the history.

---

## 3. Module reference

### `observe.py`: what the model may see
`render(page, max_elements, max_text)` returns an `Observation` (the text for the model, a `refs` map from number to element, the hidden count, flagged texts, and a `page_hash`). Before the model sees anything:
- **hidden elements are dropped** (hidden text is the cheapest injection: the human never sees it, the model does). They get no number, so they can never be clicked.
- every text is whitespace-collapsed, length-capped and **HTML-escaped**, so it cannot close the block or fake another element
- link targets are shown (path only on the same site, **full address when off-site**) so an off-site link looks off-site
- password and card fields appear as "the user must enter this", **never with values**
- `looks_injected` **flags** instruction-looking text (a heuristic, easy to evade). Flagged text is kept and marked, not silently removed
- the whole thing sits under a header saying it is data, not instructions
- elements are numbered per snapshot; the model refers to numbers, never to the site's own ids

### `policy.py`: what the agent may do (default deny)
`ActionPolicy.decide(action, obs)`:
- `done`: allow
- **navigate**: resolve the URL against the current page **first**, then check the domain policy (a relative link cannot smuggle the agent off-site); a sensitive path (`/book`, `/checkout`, `/payment`, `/login`, `/account`, `/settings`, `/messages`, ...) needs approval, anything else is allowed
- the target element must be one **that was shown**, otherwise deny
- **type**: only into a text box; a `password` or `card` field is a **handover**; text over 200 characters is denied; typing on a sensitive page needs approval
- **click on a link**: resolve and check the domain; sensitive destination or sensitive current page needs approval
- **click on a button**: allowed without approval only if its visible label is on a short allow-list (`search`, `accept cookies`, `next`, `show more`, `close`) and the page is not sensitive; every other button needs approval
- the approval **summary is built by code from the real page** (action, page, the element's role and text, where it goes, and every amount visible), never from the model's description. `fingerprint` and `amounts_on` are what the pre-action re-check compares.

**Known limit (a test documents it):** the button allow-list trusts the button's *visible label*, which the page controls, so a hostile page can label a purchase button "Next". The sensitive-path rule is the second line of defence. Neither is complete; a real deployment adds network-level limits and a sandboxed browser profile.

### `urlpolicy.py`: which URLs, on every hop
`DomainPolicy.check(url)` returns the URL or raises `BlockedURL` with a reason. It **parses** rather than pattern-matches, and refuses the classic tricks: the allowed name used as a prefix (`https://stays.example.evil.example`) or suffix without a dot (`evilstays.example`), userinfo (`stays.example@evil.example`), a backslash that browsers read as a slash, any scheme but `https` (`http`, `javascript:`, `data:`, `file:`), IP literals (`127.0.0.1`, `[::1]`), and punycode or non-ASCII look-alike hosts. `resolve(base, href)` resolves a link against its page: the **result** is what gets checked. `safe_display(url)` shows host and path only, because query strings can carry tokens.

### `extract.py`: turning a page into data, with code doing the checking
The lesson's schema asked the model for listings **and** the cheapest, average and range. Here the model (or a deterministic parser) only **points at elements**; code then:
- requires the title and the quoted price text to appear in the element pointed at
- **parses the price itself** (`parse_price`): requires a currency and a **per-night** basis, rejects totals, and never defaults a missing currency
- requires the link to be the element's real link, on the allowed site, in the expected `/rooms/<n>` shape
- drops duplicates, and computes cheapest, average and range with `Decimal` (`summarize`), refusing to mix currencies (`MixedCurrencies`) or answer from nothing (`NoListings`)

Two entry points: `actor_extract` (no model, fixed structure, fails loudly with `LayoutChanged` if the layout changed) and `llm_extract` (the model points, code verifies; scripted tests only). "Verified" means "matches what the page says"; a hostile page can still say something false.

### `simsite.py`
A deterministic, in-memory website, **a model of a browser, not a browser**: it does not render or run JavaScript, and I did not write or run a Playwright adapter. It has the features that matter for agent safety: a cookie banner, a search box, listing cards, hidden and visible injection text, an off-domain ad link, an open redirect, a booking page with a button that spends money, a card field, a login page, a price that changes after you looked at it, and a second layout with different element names. `SimBrowser.navigate/click` call `url_ok` for **every** hop, as request interception would in a real browser.

### `runtrace.py` and `structured.py`
A trace of event types, counts and hashes (never page text), and the validated, retrying model call.

### `harness.py`
The naive behaviour (raw page text with hidden elements, no allow-list, no policy, no approval) against the guarded loop on five scenarios: **injection** (a *gullible* test double that obeys any "GO TO <url>" it reads), **cheapest** (listing extraction), **purchase** (the booking flow with approval and a repricing page), **stuck**, and **layout** (a changed page structure). The "models" are labelled test doubles, so it shows what the mechanisms do when a model misbehaves, not how often a real model is fooled by a real page.

---

## 4. Design decisions

- **Observation is a defence, not just a format.** Dropping hidden elements and escaping text happens before any model sees the page.
- **One action per turn, a short history, no screenshot pile.** Bounded cost and a small attack surface.
- **The model refers to elements by number, never by the site's own ids.** It can only click or type into what it was shown. (It may propose a URL to navigate to, but that goes through the domain policy.)
- **Approvals are built from the page and bound to what was shown.** The re-check just before acting catches a page that changed after the human looked.
- **Credentials are not the model's.** Password and card fields end the run with a handover.
- **Checks run on every hop,** not only on the first request.
- **Limits everywhere:** actions, time, tokens, refusals, repeats.

## 5. What it does not do

No real browser was driven. The button-label allow-list can be fooled by a page that lies about its labels. Injection detection is a heuristic. The domain policy is only as good as the allow-list you give it, and a real deployment needs network-level egress limits and a sandboxed profile. The live model paths (`llm_extract`, the Claude loop) are tested with scripted replies only.

## 6. Running and testing

Everything runs offline against the simulated site (`python harness.py`). `python -m pytest -q` runs 163 offline tests.
