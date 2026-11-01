# Lesson 06: trustworthy agents, enforced in code

Lesson 06 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) lists threats and mitigations in prose. This folder implements the mitigations as code and attacks them.

The principle: **assume the model will be fooled, and make that not matter.** Every control below is enforced by code outside the model. None depends on the prompt.

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q            # 70 tests, offline, no API key
python redteam.py              # the red-team scorecard (no model needed)
set ANTHROPIC_API_KEY=sk-ant-...
python agent.py "Look up booking BK1001 and tell me the customer's name"
```

## Files

| File | Control |
|---|---|
| `policy.py` | allow / ask / deny decided from **declared tool metadata**, user scopes, and taint, not from text the model wrote |
| `approvals.py` | Human approval bound to the exact tool and arguments (hash), single-use, expiring, fail-closed |
| `redact.py` | PII redaction for logs, and a per-run vault so the model sees `[EMAIL_1]`, never the real value |
| `audit.py` | Hash-chained, tamper-evident audit log, with PII redacted before hashing |
| `limits.py` | Per-run budget (steps, tool calls, dollars), per-user rate limit, kill switch |
| `agent.py` | The loop that applies all of them to every tool call |
| `demo_tools.py` | Stub tools with metadata (`side_effects`, `irreversible`, `cost_fn`, `required_scope`, ...) |
| `redteam.py` | Scorecard: decisions vs a keyword baseline, approval-gate attacks, PII attacks |

## The order every tool call passes through
rate limit, argument validation, policy, budget, human approval (if policy said `ask`), execute, tokenize PII in the result, audit entry.

Two details matter:
- **Denials are generic** (`"This action is not permitted."`). The reason goes only to the audit log. The lesson-06 notebook fed denial reasons back to the model so it could propose something different, which teaches it to reword requests until the gate lets them through.
- **Budget is checked before approval**, so a human is never asked to approve something that would be blocked anyway.

## Red-team scorecard

`python redteam.py` compares the notebook's keyword approach with this policy on 14 scenarios (10 dangerous, 4 harmless).

| | unsafe auto-run | over-block |
|---|---|---|
| keyword baseline (my transcription of the notebook's logic) | 10 / 10 | 2 / 4 |
| policy from tool metadata | 0 / 10 | 0 / 4 |

**Read this table with suspicion.** It is a demonstration, not a benchmark:
- **I wrote the scenarios after knowing the keyword approach's weaknesses.** That is selection bias.
- **Only 6 of the 10 dangerous scenarios are fair.** Those six are one action phrased in words the keywords miss ("dispatch", "wipe the target", "wire", "remove", "upload", "forward the list"). The other four need scopes, taint tracking, or an unknown-tool check, which a keyword classifier has no concept of. They flatter the policy.
- **The policy is only as good as its metadata.** `test_known_limit_policy_trusts_the_developers_declarations` shows a tool that really sends email, registered without `side_effects=True`, being auto-allowed. The risk moved from "the model's wording" to "whether a developer declared the tool correctly". That is a better place for it (you review declarations once), but it is not zero.
- **The baseline is my own re-typing of the notebook's logic**, not the notebook's code, which I did not run.

The approval-gate and PII sections of the scorecard are different in kind: each attack either is handled or is not, and all are handled.

## Bugs I found in my own code while building this
1. **The approval hash was computed after the approver ran, and the approver received the live arguments.** A tampering approver could have changed the arguments and the grant would have matched the change. The hash is now taken first and the approver gets a deep copy (`test_approver_cannot_change_what_it_is_approving`).
2. **Passing a `Budget` object reused it across runs**, so spend accumulated silently despite a comment saying otherwise. Budgets are now copied with `fresh()` for each run.
3. **The default budget is $1 per run**, which silently refuses any tool costing more. That is intended and safe, but not obvious. Set `Budget(max_usd=...)` for real workloads.

## Limits you should know
- **PII detection is regex heuristics.** It finds emails, SSNs, Luhn-valid cards and some phone formats. It misses names and addresses (a test documents this). Use a real PII service in production.
- **The audit chain cannot detect truncation of the end of the log** on its own. Anchor `head()` somewhere you do not control. It also cannot stop someone who can rewrite the file and recompute every hash.
- **Approval relies on the human reading the summary.** Approval fatigue is a human problem that code does not solve.
- **Taint is coarse:** after any untrusted read, all side-effect tools are denied for the rest of the run, even harmless ones.
- **No live run.** `agent.py` has never been called against the real API in this session, because I have no key. All tests use a scripted model, so they check the controls, not how a real model behaves against them.
- Scopes (`UserContext`) are assumed to come from a trustworthy auth system. This code does not provide one.

## Exercises
1. Register a tool without `side_effects=True` that sends something. Add a startup check that rejects tools whose names or descriptions contain words like "send", "delete", "pay", but have no declared side effects. How many false alarms do you get?
2. Make taint finer: allow side effects whose arguments contain no text derived from the untrusted source. Why is that hard to do correctly?
3. Add a `tool_calls_per_minute` cap per tool, not only per user.
4. Anchor `audit.head()` to a daily file and write a checker that detects end-of-log truncation.
5. Replace the regex PII detector with a better one and add the misses from `test_known_limit_names_and_free_form_identifiers_are_not_detected` as expected passes.
