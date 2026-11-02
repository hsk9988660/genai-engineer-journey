# Lesson 06: implementation guide

This explains **how the code works**: how a tool call the model requests passes through code-enforced controls before anything happens. The [README](README.md) says why the lesson exists and what the red-team scorecard showed.

---

## 1. The system on one page

```
 TrustedAgent.run(user_msg)
      │  the model never sees raw PII: the request goes through PiiVault.tokenize
      ▼
 each model turn:  kill switch? budget exhausted?  → stop
      ▼
 for every tool_use block  ──  _handle(block):
      1  unknown tool?            2  kill switch?           3  per-user rate limit
      4  arguments valid?         (the model may be told why: it is about the shape of the call)
      5  PolicyEngine.decide(spec, args, user, state)  →  allow | ask | deny   (from DECLARED tool metadata)
      6  budget would be exceeded?
      7  "ask": ApprovalGate.request → human → Grant bound to the exact arguments
              then ApprovalGate.redeem right before running
      8  run the tool
      9  tokenize PII in the result;  every step writes an audit entry (hash-chained)
```

A denial shows the model only **"This action is not permitted."** It never learns which rule fired, so it cannot reword a request to get around the gate.

---

## 2. A run, from start to end

### Where it starts
`TrustedAgent.run(user_msg)`. `TrustedAgent` is constructed with instructions, a list of `ToolSpec`s, and a `UserContext` (who is signed in, set by **your** auth code, never by the model).

### What happens, in order

1. **Fresh per-run state.** A new `RunState` (the taint flag and tool-call count), a new `PiiVault`, and `budget_template.fresh()` so spend never carries over from a previous run. An audit entry `run_start` is written.
2. **Tokenise the request.** `vault.tokenize` replaces emails, phone numbers and similar with placeholders such as `[EMAIL_1]`. The model sees only placeholders.
3. **The loop.**
   - stop if the **kill switch** is tripped or the **budget** is exhausted (step or spend limit)
   - call the model, charge the step's tokens to the budget
   - no `tool_use`: the answer text is **detokenised** (placeholders swapped back for the real values) and the run ends `completed`
4. **For each requested tool call,** `_handle` runs the checks below, in this order. After each turn, if an untrusted tool returned successfully the run becomes **tainted** (applies to later turns).
5. Each result is JSON, passed through `vault.tokenize` again before the model sees it, and everything goes back in one user message.

### `_handle`: the gauntlet

| Step | Check | On failure |
|---|---|---|
| 1 | tool is known | `unknown` |
| 2 | kill switch not tripped | `killed` |
| 3 | `RateLimiter.allow(user_id)` (sliding window per user) | `rate_limited` |
| 4 | arguments validate against the tool's Pydantic model | `invalid`, with the validation detail (safe to explain) |
| 5 | **policy** decides | `deny` is `denied` |
| 6 | `budget.would_exceed(cost)` | `budget` |
| 7 | verdict `ask`: human approval, then redeem the grant | `declined` or `approval_mismatch` |
| 8 | the tool runs; an exception is audited (redacted) and the model sees "The tool failed." | `error` |
| 9 | charge the tool's cost, count the call, audit the result | `ok` |

Every refusal writes a `decision` audit entry with the reason, but the model only ever sees `NOT_PERMITTED` (except for invalid arguments and a plain "not approved").

### Where it ends
`_end(reason, text)` writes a `run_end` entry (reason, steps, money spent) and returns the answer, or `"Stopped: <reason>."` Reasons include `completed`, `step limit reached`, `spend limit reached`, and `kill switch: ...`.

---

## 3. Module reference

### `policy.py`: decide from what the TOOL is
`ToolSpec` carries facts the **developer declares**: `side_effects`, `irreversible`, `cost_usd` or `cost_fn(args)`, `required_scope`, `untrusted_output`, `allow_pii_args`. `PolicyEngine.decide` then returns:
- **deny** if the tool is unknown, the user lacks the required scope, or a risky tool (side effects or irreversible) is requested **after untrusted content was read** (taint)
- **ask** if the tool is irreversible, has side effects, or costs at least the threshold (default $25), with the reasons listed
- **allow** for everything else ("read-only")

The lesson's notebook classified risk by matching keywords in an action description **the model wrote**. That fails three ways: substring bugs, incomplete word lists, and the model (or whoever steers it) controls the words.

### `approvals.py`: approval bound to the exact action
`ApprovalGate(approver, ttl_s=120, timeout_s=60)`.
- `request` computes `args_hash(tool, args)` (SHA-256 of canonical JSON) **before** the approver runs, and gives the approver a **deep copy**. (A bug found while building: hashing after the approver ran let it change the arguments and have the grant match the changed version.) It runs the approver in a thread with a timeout. Only the exact string `"approve"` counts. A different word, an exception, a timeout or no answer is a denial: **fail closed**.
- `redeem(grant, tool, args)` is called with the arguments the agent is *about to run*. It refuses a replay (a grant is consumed by **any** attempt to use it), an expired grant, or any difference in tool or arguments.

### `audit.py`: tamper-evident log
Each entry includes the hash of the previous one; `verify` returns whether the chain is intact and the index of the first bad entry. Entries are PII-redacted **before** hashing. Documented limits: cutting off the **end** of the log is not detectable from the log alone (anchor `head()` elsewhere), and anyone who can rewrite the whole file and recompute every hash can forge it. Lesson 18 builds the signed version of this idea.

### `limits.py`: hard limits
`KillSwitch` (thread-safe), `Budget` (steps, tool calls, and money for model tokens plus tool costs; the default token prices are examples, set real ones), `RateLimiter` (per-user sliding window). Cost is an attack surface ("denial of wallet"), so limits live in code, not in the prompt.

### `redact.py`: two different PII jobs
- `redact_text` / `redact_obj`: **irreversible**, for logs and traces. Credit-card-like numbers are checked with a Luhn test.
- `PiiVault`: **reversible**, per run. The model sees placeholders; real values return only at the boundary of tools declared `allow_pii_args` (and in the final answer). A tool such as `fetch_webpage` is **not** given real values, so the model cannot use it to smuggle an address out.
These are regex heuristics: they miss names, addresses and free-form identifiers.

### `demo_tools.py`
Stub tools for a support agent (`search_flights`, `find_location`, `read_policy`, `lookup_customer`, `send_email`, `issue_refund` with a cost computed from the amount, `delete_account`, `upload_file`, `fetch_webpage`), each declared with the metadata above. They return canned data and never touch the outside world.

### `redteam.py`: a scorecard with no model and no key
Part 1 compares a **keyword baseline** (my own re-transcription of the notebook's logic, labelled as such) with the metadata policy on a set of scenarios, counting **unsafe auto-runs** and **over-blocks**. Part 2 attacks the approval gate: swapping the amount after approval, replaying a used approval, using an expired one, reusing it for a different tool, and an approver that answers with the wrong word, crashes, or times out (plus a control that the exact approved action is allowed). Part 3 attacks the PII vault: real values must not reach the model, a token smuggled into a URL must not be expanded, a tool without `allow_pii_args` never gets real PII, and a token that was never issued leaks nothing.

---

## 4. Design decisions

- **Rules from declared facts, not model words.** The model is the thing being controlled, so it cannot supply the inputs to the control.
- **Say nothing useful on denial.** A model that learns the rule can rephrase around it.
- **Bind the approval to the exact arguments, and spend it once.** Otherwise "approve booking Paris" can become "booking Rio".
- **Limits are code.** A prompt that says "do not spend more than $1" is a request, not a limit.
- **Two PII mechanisms.** Logs need irreversible redaction; the model needs a reversible stand-in.

## 5. What it does not do

Taint is conversation-wide. There is no user interface for approvals (a terminal function is provided). The audit log is tamper-evident, not tamper-proof. PII detection is a floor, not a guarantee.

## 6. Running and testing

`python redteam.py` runs offline. `python agent.py "..."` needs an API key. `python -m pytest -q` runs 70 offline tests.
