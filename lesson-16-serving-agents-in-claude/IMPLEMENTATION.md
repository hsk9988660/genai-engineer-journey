# Lesson 16: implementation guide

This document explains **how the code works**: where a request starts, what each module does, where the request ends, what is shared between threads, and what you would change to extend it. The [README](README.md) explains *why* the lesson exists and what was measured. Read the README first if you want the argument, and this guide if you want to read, change or debug the code.

Every module name below is a file in this folder. Functions are named rather than numbered by line, so this guide does not go stale when a line moves.

---

## 1. The system on one page

```
 client ──HTTP──► http_app.py ──► service.AgentService.handle()
                  (auth, size limits,        │
                   status codes)             ▼
              ┌──────────────────────────────────────────────────────────────┐
              │ 1 validate        _validate()                                │
              │ 2 admit           admission.Admission.slot()    → 429 / 503  │
              │ 3 idempotency     resilience.IdempotencyStore   → replay     │
              │ 4 budget          cost.CostLedger.check()       → 429        │
              │ 5 pick version    release.Rollout.version_for()              │
              │ 6 load thread     threads.ThreadStore.messages()→ 404 / 409  │
              │ 7 cache           cache.ResponseCache  +  Singleflight       │
              │ 8 cascade         router.cascade()                           │
              │      └ each tier  retry (backoff+jitter) → CircuitBreaker    │
              │                    → agent_fn()  (claude_agent.make_agent_fn)│
              │                       └ tools (support_tools) with user_id   │
              │                         supplied by CODE                     │
              │ 9 persist, cost, metrics, log                                │
              └──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                                 Response(status, text, thread_id, version, tier, ...)
```

Two things are deliberately kept **outside** the request path: `release.gate()` (the evaluation gate, run before a version ships) and `Rollout.evaluate()` (the canary decision, run periodically). A release is a separate act from serving.

---

## 2. A request from start to end

### Where it starts
Two entry points:
- **Over HTTP:** `http_app.make_server(...)` returns a `ThreadingHTTPServer`. Its `Handler.do_POST` runs for `POST /v1/chat`.
- **In code:** `AgentService.handle(Request(...))`. Tests and the harness use this directly.

### What happens, in order

**HTTP layer (`http_app.py`), only when coming over HTTP**
1. Read `Content-Length`. Missing → 411. Over `MAX_BODY` (16 KB) → read up to `DRAIN_LIMIT` bytes, then 413 with `Connection: close`. (Answering without reading the body can reset the connection; this was a real bug in lesson 11.)
2. Read the body, then authenticate: `Authenticator.user_for(header)` maps a bearer token to a user id by comparing SHA-256 hashes in constant time. No match → 401. **The user id comes from the token. A `user_id` in the body is never read.**
3. Parse JSON. Not an object → 400.
4. Build `Request` and call `service.handle(...)`, wrapped in `try/except` so the connection is never dropped (500 instead).

**Service layer (`service.py`), `AgentService.handle`**
1. A trace id is taken from the request or generated (`observability.new_trace_id`).
2. `_handle` runs inside a `try/except Exception`. Anything unexpected becomes `Response("error")`. **`handle` never raises.** (The HTTP test suite found a case where it did.)
3. `_validate`: user id, text length (≤ 4000), thread and idempotency key lengths (≤ 64), deadline (0 to 300 s). Failure → `invalid` (400). No work has been done yet.
4. `Admission.slot(user)`: take a slot, wait in a bounded queue, or be refused → `overloaded` (429 + retry hint) or `shutting_down` (503).
5. If the request has an idempotency key, `IdempotencyStore.run("<user>:<key>", ...)` wraps `_serve`. A repeat returns the stored response with `detail="replayed"`; a duplicate arriving at the same moment waits for the first. Only `ok` responses are stored, so a failure can be retried.
6. `_serve`:
   1. `CostLedger.check(user)` → `budget` (429) if the user or the day is out.
   2. `_pick_version(user)`: the `Rollout` decides stable or candidate by a stable hash of the user id.
   3. If `thread_id` is given, `ThreadStore.messages(...)`. A thread owned by someone else raises `NotFound` → `not_found` (404). A thread near its message cap → `thread_full` (409), before any model call.
   4. Optional `user_context(user)` text. If present, the answer is personal and the shared cache is skipped.
   5. **Cache.** Only for a *first-turn* question with no user context. A hit returns immediately (`tier="cache"`, `cached=True`, cost 0, still saved to the asker's own new thread). A miss goes through `Singleflight.do(key, ...)`, so identical questions arriving together share one model call.
   6. `_compute`: `router.cascade(...)` starts at the cheap tier (or the top tier if a keyword hint says so) and, for each tier it tries, calls `call_with_retry(attempt)` where `attempt` checks the deadline, then calls `CircuitBreaker.call(lambda: enforce_tools(tier, agent_fn(call)))`. Tokens are charged to the ledger for every provider call that returns, even if the answer is then rejected.
   7. The cascade's `check` function (default: non-empty successful text) decides whether to accept the answer or escalate. If even the top tier fails the check → `error`.
   8. If the answer is `shareable(...)` (no user context, only public tools), it is put in the cache.
   9. `_persist`: create the thread if needed and save the user's message and the answer. A failure here is counted (`persist_failed`) but **does not discard the answer**.
7. Back in `handle`: metrics (`requests{status}`, latency, `tier_requests{tier}`), a `Rollout.observe(...)` for versions, and one log line with *no user content*.

### Where it ends
`handle` returns a `Response`. The HTTP layer turns it into a status code, headers (`X-Trace-Id`, `Retry-After`) and a JSON body. Nothing else happens after that: there are no background tasks.

### Status table

| `status` | HTTP | Meaning | Counts against a canary version? | What the client should do |
|---|---|---|---|---|
| `ok` | 200 | Answered | counts as success | use it |
| `invalid` | 400 | Bad input | no | fix the request |
| `not_found` | 404 | Thread is not yours or does not exist | no | start a new thread |
| `thread_full` | 409 | Thread at its message cap | no | start a new thread |
| `overloaded` | 429 | No room right now | no | wait `Retry-After`, retry |
| `budget` | 429 | User or day budget used | no | wait until tomorrow |
| `unavailable` | 503 | Provider failing or circuit open | **yes** | retry later |
| `shutting_down` | 503 | Draining | no | retry on another instance |
| `deadline` | 504 | Ran out of time | **yes** | retry, maybe with a longer deadline |
| `error` | 500 | Bug, permanent provider error, or no tier gave an acceptable answer | **yes** | report with the trace id |

The "counts against a version" column is `service.FAILURES`. Bad input, a busy server and an empty budget are not the new version's fault, so they must not trigger a rollback.

---

## 3. Module reference

### `agentdef.py`: what an agent *is*
- `ToolSpec(name, scope, description)`: scope is `public` (same answer for everyone), `user` (depends on who asks) or `effect` (changes the world).
- `Tier(name, model, tools, in_price, out_price)`: one rung of the model ladder. Prices are per 1,000 tokens and are **placeholders**.
- `AgentVersion(name, prompt, tiers, tools, routing)`: frozen dataclass. `version_id` is the first 12 hex characters of a SHA-256 over a canonical JSON of everything that affects behaviour.
- `__post_init__` enforces: at least one tier; every tool a tier names exists; **a tool with scope `effect` may only be on the last (top) tier**.
- `AgentResult` is what a model call returns. `AgentCall` is what it receives, including `user_id`, which is for *tools*.

### `threads.py`: conversation state outside the process
- `ThreadStore(path, max_messages, max_chars)` on SQLite. `PRAGMA secure_delete=ON`, so deleted text is overwritten (a test scans the raw file).
- `Scope(user_id)` is a validated value. Every query joins on `user_id`, so another user's thread id is indistinguishable from a missing one (`NotFound`).
- Thread ids come from `secrets.token_urlsafe(12)`.
- One `RLock` guards all access. SQLite is opened with `check_same_thread=False`.

### `admission.py`: decide before doing expensive work
- `Admission(max_concurrent, max_queue, per_user_max, queue_timeout_s)` with one `Condition`.
- `slot(user)` is a context manager. State: `running`, `waiting`, per-user counts, `closed`.
- The per-user count includes both waiting and running requests. On *any* exit (rejection, timeout, shutdown) the counts are put back and waiters are notified.
- `drain(timeout)` closes admission, wakes waiters (who get `ShuttingDown`), and waits for `running` to reach 0.
- `Overloaded.retry_after_s` is `ceil((waiting + 1) × average latency / max_concurrent)`, at least 1. It is a hint, not a promise.
- It uses real `time.monotonic()` for waiting, so tests for waiting use real (short) timeouts.

### `resilience.py`: failing well
- `Transient` / `Permanent` / `CircuitOpen` / `DeadlineExceeded`: `CircuitOpen` is *not* a `Transient`, so a retry loop never retries into an open circuit.
- `backoff_delay(attempt, policy, rng)`: full jitter, `rng() × min(cap, base × 2^attempt)`.
- `call_with_retry(fn, ...)`: retries only `Transient`. A provider's `retry_after_s` raises the delay (capped). It refuses to start a retry that would pass `deadline`.
- `CircuitBreaker`: closed → open after N consecutive transient failures → half-open after `reset_after_s` with **one probe at a time**. `Permanent` errors mean the provider *answered*, so they close the circuit rather than count as an outage. An unexpected exception during a probe releases the probe (otherwise the circuit would be stuck).
- `Singleflight.do(key, fn)` returns `(value, shared)`. Followers get the leader's value or its exception.
- `IdempotencyStore.run(key, fn, store_if)` returns `(value, replayed)`. It stores only results `store_if` accepts, expires them by TTL, and is size-bounded.

### `cache.py`: a cache that cannot leak
- `normalize(query)`, `cache_key(version_id, query)`: the version id is part of the key.
- `shareable(version, result, had_user_context)`: true only if the call succeeded, produced text, used no user context, and **every tool it called has scope `public`**.
- `ResponseCache`: LRU + TTL, bounded. Counters for hits, misses, evictions, expirations.

### `cost.py`: money
- `request_cost(tier, in, out)` is `Decimal` arithmetic.
- `CostLedger`: per-user and global daily spend (keyed by day string), per-tier counts, cache-hit count. `check(user)` runs *before* a request; `record(...)` runs *after* each provider call.

### `observability.py`: seeing without watching
- `Metrics`: counters (with labels) and latency samples (the most recent 10,000). `percentile` is nearest-rank.
- `RequestLog.emit` **raises** if a field is named `text`, `query`, `prompt`, `content`, `message(s)`, `answer`, `response`, `body`, `input`, `output`, or if a string value is longer than 100 characters. A careless log line fails in tests instead of leaking in production.
- `AlertRule` kinds: `p95`, `counter`, `ratio` (with a minimum number of events so three failures out of three do not page anyone).

### `release.py`: shipping safely
- `wilson_interval(passes, n)` gives a 95% interval for a pass rate.
- `gate(passes, n, required=0.8, min_cases=30)` returns `pass`, `fail` or `inconclusive`. Only `pass` ships.
- `naive_gate_error_rates(...)` computes, exactly, how often the lesson's 4-case rule blocks a good agent or ships a bad one.
- `bucket(user_id, salt)` is a stable 0 to 99 per user. `Rollout` moves through stages (1, 10, 50, 100%), needs `min_samples` per stage, rolls back when the Wilson **lower bound** of the candidate's error rate exceeds the stable error rate plus a margin, and resets counters at each stage.

### `router.py`: the cascade
- `start_index(version, text)`: a keyword **hint** for where to start.
- `enforce_tools(tier, result)`: raises `ToolNotAllowed` (a `Permanent`) if the result used a tool the tier was not offered.
- `cascade(version, run_tier, check, start)`: try each tier from `start`; return the first answer that passes `check`; the top tier's answer is returned (with `passed=False`) if it also fails. A `ToolNotAllowed` on a cheap tier escalates; on the top tier it propagates.

### `support_tools.py`: the example tools
- `build_tools(data, approver, threshold)` returns `{name: impl}`. **Every implementation takes `user_id` as its first argument**, supplied by code.
- `get_order_status` answers "No order found" identically for someone else's order and a non-existent one.
- `issue_refund` checks ownership, amount (positive and within what is left), threshold approval (`approver` returns `True`, `False` or `None` = queue it), and is idempotent on `(order, amount)`, all under one lock.
- `SCHEMAS` are what the model sees: no `user_id`, `additionalProperties: false`.

### `claude_agent.py`: the real model call
- `make_agent_fn(client, impls, schemas, max_rounds, max_tokens)` returns `agent_fn(call)`.
- It offers only the tools in `call.tier.tools`. User facts go in the *conversation*, labelled as data, so the system prompt is identical for everyone.
- A tool-use loop (up to `max_rounds`). A tool the tier was not offered → `ToolNotAllowed`. A `TypeError` from a tool (for example a `user_id` the model invented) becomes an error string for the model, not an execution.
- `classify_error(exc)` maps provider errors to `Transient` (429, 408, 409, 5xx, known class names) or `Permanent` (everything else, including unknown errors). It reads a `retry-after` header if present.

### `service.py`: the orchestrator
Described in section 2. Key design points: `handle` never raises; admission wraps everything; idempotency wraps `_serve`; the cache is used only for first-turn questions without user context; cost is charged per provider call, not per user-visible response.

### `http_app.py`: the front door
`POST /v1/chat`, `GET /healthz` (alive), `GET /readyz` (should receive traffic: not draining, circuit not open, store answering), `GET /metrics` (admin token). Error bodies are `{"status", "detail"}` or `{"error": code}`, never a traceback or user text.

### `harness.py`: the comparison
Six scenarios, each running the lesson's pattern and the served pattern. Test doubles: `Provider` (latency, an in-flight rate limit that answers 429, a call counter), `rule_based_agent` (calls the real tools by simple rules), and seeded failure rates. `FakeClock` makes time-dependent logic deterministic.

---

## 4. Concurrency model

The server is **thread-per-request**. There is no async code.

| Object | Guarded by | Notes |
|---|---|---|
| `Admission` | one `Condition` | the only place that blocks on purpose |
| `ThreadStore` | one `RLock` | serialises all SQLite access |
| `ResponseCache` | one `Lock` | short critical sections |
| `CircuitBreaker` | one `Lock` | the provider call happens **outside** the lock |
| `Singleflight` | one `Lock` | the leader's work happens outside the lock |
| `IdempotencyStore` | one `Lock` + a `Singleflight` | |
| `CostLedger`, `Metrics`, `RequestLog` | one `Lock` each | |
| `SupportData` (refunds) | one `Lock` | covers check-then-act |

**Lock order.** A request holds an `Admission` slot (not the condition lock) for its whole life. Inside it, each lock is taken briefly and released before the next is taken: `Singleflight` and `IdempotencyStore` run the user's function *outside* their own locks, and `CircuitBreaker` calls the provider outside its lock. I know of no cycle, but I have not proven there is none.

**One exception to know about.** `issue_refund` holds the `SupportData` lock while it calls the `approver` callback. A slow approver would therefore block every other refund. In this lesson the approver is a function that returns at once; a real approval step should queue the request and return, not wait.

---

## 5. Where state lives, and what a restart loses

| State | Lives in | Survives a restart? | If you run several instances |
|---|---|---|---|
| Threads | SQLite file | yes | needs a shared database, not a local file |
| Response cache | process memory | no | each instance has its own, so hit rate drops |
| Idempotency results | process memory | no | a retry that lands on another instance **runs again** |
| Circuit breaker | process memory | no | each instance learns about an outage on its own |
| Canary counters | process memory | no | each instance decides alone |
| Cost ledger | process memory | no | budgets are per instance, not global |
| Metrics, logs | process memory | no | export them |

This is the main gap between this lesson and a real multi-instance deployment: **threads are the only state that is genuinely external.** Moving the idempotency store, the cost ledger and the canary counters to a shared store is the first step toward running more than one instance. The interfaces are small on purpose.

---

## 6. Configuration and how to choose values

| Setting | Where | Guidance |
|---|---|---|
| `max_concurrent` | `Admission` | Set it **below** the provider's concurrent-request limit, so you queue instead of getting 429s |
| `max_queue` | `Admission` | A longer queue hides overload and adds latency; a shorter one refuses sooner. Pick it from your latency budget |
| `per_user_max` | `Admission` | Stops one caller taking every slot |
| `queue_timeout_s` | `Admission` | Should be shorter than the client's own timeout |
| `RetryPolicy(attempts, base_s, cap_s)` | `AgentService` | Fewer attempts per request plus a breaker beats many attempts |
| `CircuitBreaker(failure_threshold, reset_after_s)` | `AgentService` | Threshold small enough to react, large enough to ignore a blip |
| `ResponseCache(max_entries, ttl_s)` | `AgentService` | TTL = how long a stale public answer is acceptable |
| `CostLedger(user_daily_budget, global_daily_budget)` | `AgentService` | Check real prices before trusting the numbers |
| `gate(required, min_cases)` | release | `min_cases` should be large enough that a decision is possible |
| `Rollout(min_samples, max_extra_error, stages)` | release | More samples per stage means slower, safer rollouts |

All timing parameters in the defaults are guesses. **I have not tuned any of them against real traffic.**

---

## 7. How to extend it

**Add a tool.** (1) Write `impl(user_id, **args)`. (2) Add a `ToolSpec` with the correct scope. (3) Add a schema with `additionalProperties: false` and **no** `user_id`. (4) Name it in the right tier's `tools`. If its scope is `effect`, `AgentVersion` will refuse to put it on a cheap tier. (5) Add tests, including one with a *different* user.

**Add a tier.** Add a `Tier` to `AgentVersion.tiers` in cheapest-first order. Nothing else changes: the cascade and the effect-tool rule use the order.

**Change the prompt or the model.** Build a new `AgentVersion`. Its `version_id` changes, so caches, canaries and metrics separate automatically. Run `release.evaluate(...)` on the *whole serving path* before a `Rollout`.

**Use a different model function.** Anything with the signature `agent_fn(AgentCall) -> AgentResult` that raises `Transient` or `Permanent` works. `make_agent_fn` is the Claude one.

**Make the acceptance check real.** Pass `check=` to `AgentService`. Put schema validation, grounding checks and policy checks there. The default only checks for non-empty text.

**Add a real browser-style adapter, a real counter, a real approval UI.** The seams are `agent_fn`, `tokens.estimate_tokens`, and the `approver` callback.

---

## 8. Design decisions and what was rejected

- **Thread-per-request, no async.** Simple to read and to test deterministically. It limits how many idle connections one process can hold; a real deployment would likely use an async server.
- **Admission control in the service, not only at a load balancer.** The service knows about per-user fairness and about its provider's limit. A load balancer can still add its own.
- **Cascade instead of a difficulty classifier.** A classifier is a second model whose accuracy you must also measure. A cascade uses the answer's own check. Its cost is that a wrongly accepted cheap answer is never looked at again.
- **A cache that rejects more than it accepts.** Safe by default: anything personal, tool-using beyond public tools, or erroneous is never shared. The cost is a lower hit rate.
- **`handle` never raises.** A serving layer that throws drops connections. Every path ends in a `Response`.
- **Cost charged per provider call.** Money is spent when tokens are used, not when a user sees an answer.
- **Rollback needs confidence, promotion needs only an absence of evidence.** Rolling back on one bad answer in eight rolled back a *good* candidate in my first version. A promotion therefore means "no evidence of harm yet", not "proven equal".
- **Rejected: caching per user.** It would give a hit rate of nearly zero and a larger privacy surface for no benefit in this lesson.
- **Rejected: a hand-rolled web framework.** The standard library server is enough to show status codes, headers and limits.

---

## 9. Testing guide

Run everything: `python -m pytest -q` in this folder (about 12 seconds, no key, no network, nothing leaves `127.0.0.1`).

| File | Covers |
|---|---|
| `test_agentdef_threads.py` | version ids, the effect-tool rule, thread ownership and deletion |
| `test_admission.py` | concurrency, queue, per-user limit, timeout, drain (uses real threads and short real timeouts) |
| `test_resilience.py` | backoff, retry, breaker, singleflight, idempotency |
| `test_cache_cost_obs.py` | cache keys and sharing, TTL/LRU, ledger, metrics, the log guard, alerts |
| `test_release.py` | intervals, the gate, buckets, the canary |
| `test_router_tools_agent.py` | cascade, the support tools, the Claude adapter |
| `test_service.py` | the whole request path |
| `test_http.py` | real HTTP on `127.0.0.1` with an ephemeral port |
| `test_harness.py` | the six scenarios |

**Test doubles to know about.** `Provider` and `rule_based_agent` in `harness.py` are scripted. They have no real latency distribution and make no real decisions. `FakeClient` (from `fakes.py`) replays scripted Claude responses. Nothing here has touched the real API.

**Mutation checking.** Deliberately break a behaviour in a scratch copy and confirm a test fails. The README lists what was covered and what survived.

**Tests that use real time.** `test_admission.py`, the overload scenario and a few service tests use threads and sleeps of tens of milliseconds. They passed on repeated runs on my machine, but a heavily loaded machine could make them flaky.

---

## 10. Limits you should know before relying on this

See the README's "What is NOT known" and "Limits of the design". The short list: no real model or real provider limits were used; the cache, idempotency, budget and canary state is per process; the HTTP server is the standard library's; there is no TLS, no real auth system, and no persistence for anything except threads.
