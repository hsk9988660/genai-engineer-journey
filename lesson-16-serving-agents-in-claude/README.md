# Lesson 16: serving agents, with the failure cases handled

Lesson 16 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) is about taking an agent from a notebook to production: hosting, scaling, model routing, caching, evaluation gates, observability and human approval. It is a good checklist. Its notebook is a single-user script on a Foundry client, and several of its production pieces fail the moment two users or a failing provider are involved. This folder builds a provider-neutral **serving layer** for Claude agents and tests those failure cases.

> **Going to production is mostly about what happens when many requests, many users and many failures meet your agent at the same time.** A cache, a router and an evaluation gate that look reasonable for one user become a data leak, an unmeasured cost claim and a false sense of safety at scale.

For *how the code works* (request lifecycle, module reference, concurrency, state, extension), read [IMPLEMENTATION.md](IMPLEMENTATION.md). This README covers what was built, what was measured and what is not known.

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q          # 252 offline tests, no API key, no cloud account, about 12 seconds
python harness.py            # the six scenarios below (offline; labelled test doubles; a few real sleeps)
```

There is no live mode. Nothing here has called the real Claude API, and the "deployment" is a local process.

## What is built

| File | What it is |
|---|---|
| `service.py` | **The core.** One request in, one response out: validate, admit, deduplicate, budget, pick a version, load the thread, cache, cascade through model tiers with retries and a circuit breaker, persist, meter, log. It never raises. |
| `agentdef.py` | An agent as **data** (prompt, tiers, tools, routing) with a version id. Side-effect tools may only be offered to the top tier. |
| `admission.py` | Bounded concurrency, a bounded queue, per-user fairness, queue timeouts, and graceful drain. |
| `resilience.py` | Backoff with full jitter, a circuit breaker with one probe at a time, request deadlines, single-flight and idempotency keys. |
| `cache.py` | A response cache that cannot leak: only shareable answers, keyed by agent version, with TTL and an LRU bound. |
| `router.py` | Model routing as a **cascade** driven by a check on the answer, with the tool rule enforced. |
| `cost.py` | Per-request cost in `Decimal`, per-user and global daily budgets, a cost report. |
| `release.py` | An evaluation gate with a confidence interval and three outcomes, and a canary rollout that can roll itself back. |
| `observability.py` | Metrics (p50/p95), alert rules, and a request log that refuses user content. |
| `threads.py` | Conversation threads in SQLite, owned per user. |
| `support_tools.py`, `claude_agent.py` | The example tools (ownership-checked, idempotent refunds) and the Claude tool-use loop. |
| `http_app.py` | A thin standard-library HTTP front end: auth from the token, limits, 429/503 with Retry-After, health, readiness. |
| `harness.py` | The lesson's patterns versus the served ones on six scenarios. |

## The rules the code enforces
1. **The service never throws.** Any unexpected error becomes a clean 500 with no message in it, and the admission slot is always released.
2. **Work is refused before it is started.** A bounded number of requests run, a bounded number wait, one user cannot take every slot, and a waiter gives up after a timeout. The rest get 429 with a `Retry-After`.
3. **Retries are for transient errors only**, use full jitter, never start past the request's deadline, and sit behind a circuit breaker that probes with one request at a time.
4. **A repeated request with an idempotency key returns the first answer.** Two at once run once. Failures are not stored, so they can be retried.
5. **The cache shares only what is identical for everyone.** An answer is shareable only if it used no user context and only `public` tools. The agent version is part of the key. Entries expire, the cache is bounded, errors are never cached, and identical simultaneous questions make one model call.
6. **A side-effect tool can only ever be offered to the top model tier**, and a result that used a tool its tier was not offered is refused.
7. **Tools get the user id from code.** It is not in any tool schema, and one the model invents is rejected. Someone else's order answers exactly like a missing one.
8. **Refunds** check ownership, positive amount within what is left, human approval above a threshold, and are idempotent.
9. **Escalation is driven by a check on the answer**, not by guessing how hard the question is. A keyword only decides where to *start*.
10. **Money is `Decimal`**, a budget is checked *before* a request runs, and tokens are charged for every provider call even if the answer is then rejected.
11. **The evaluation gate has three outcomes**: pass, fail, or "inconclusive, run more cases". Only pass ships.
12. **A canary uses a stable slice of users**, advances only while the candidate is not *confidently* worse, and rolls everyone back otherwise. Each stage needs fresh traffic.
13. **The log refuses user content.** A field called `text`, `query`, `prompt`, `content` and so on raises, as does a long string.
14. **Identity comes from the token over HTTP.** A `user_id` in the body is ignored.

## Measured results

`python harness.py`. The "model" is a test double: fixed latency, an in-flight rate limit that answers 429, seeded failure rates, and a rule-based agent that calls the real tools.

| Scenario | The lesson's pattern | Served |
|---|---|---|
| **40 requests at once**, provider limit 5 | no admission control: **5 succeed, 35 fail**, 35 provider 429s, up to 40 in the provider at once | queue and bounded concurrency: **40 succeed**, 0 provider 429s, never more than 5 in the provider. With a small queue: 10 succeed, 30 get a polite 429 and none reach the provider |
| **Provider outage**, 20 requests | retry 5 times each, no breaker: **100 provider calls** | breaker: **5 provider calls**, the rest fail fast. Both recover on the first request after the pause |
| **Two customers ask the same question** | cache keyed by the query only: the second customer **sees the first customer's order** | the other customer sees "no order found" and nothing is cached. A public question asked 5 times costs **1 model call, 4 cache hits** |
| **A bad release** (candidate fails 30%) | shipped to everyone: **312 error responses in 1,000** | canary: **24 errors**, only 9 requests ever reached the candidate, then rolled back. A good candidate (2% errors) was promoted through all stages |
| **The evaluation gate** | 4 cases, ship at 80%: blocks a good (90%) agent **34% of the time** and ships a bad (60%) one **13%** | 100 cases with an interval: bad agent **never ships**, 99.7% fail. Good agent passes 80%, inconclusive 20%, never fails. With 4 cases it is **always inconclusive** |
| **Ten mixed queries** | no cost report | 6 small-tier calls, 3 large-tier, 4 cache hits, 3 escalations, total cost from the ledger |

**How to read this honestly.**
- The gate row for the *lesson's* rule is exact arithmetic (a binomial), not a simulation. The interval gate rows are simulations over 2,000 trials with a fixed seed.
- The cost row shows the **mechanism** (counting tiers, escalations and cache hits). Whether a cheap tier is good enough for your traffic is **not** measured. The prices are placeholders.
- The canary result depends on a seeded 30% failure rate. It shows the rollback logic, not how large a real regression has to be to be caught.
- The lesson's side of each row is a small stand-in for its pattern, not the lesson's code.

## Findings from building it
1. **An unexpected exception escaped the service.** My first `handle()` caught only the errors I had named. A test with a provider that raised a plain `RuntimeError` killed the HTTP connection with no response. It now catches everything and returns a clean 500, and the front end has its own catch as well.
2. **My first rollback rule rolled back a good release.** I rolled back whenever the candidate's *observed* error rate was 5 points above stable. With 8 samples, one unlucky error is 12.5% and the harness rolled back a candidate whose true error rate was 2%. The rule now needs the Wilson lower bound of the candidate's error rate to exceed the margin.
3. **Even 100 evaluation cases cannot always confirm a good agent.** A 90% agent against an 80% bar passes only about 80% of the time. The rest are "inconclusive", which is correct, and means a gate needs more cases than people expect.
4. **A metric label called `name` collided with the function's own parameter**, which raised a `TypeError` in the first harness run. I renamed it.
5. **Where the lesson's design fails and mine only moves the problem.** My cache, idempotency store, cost ledger, circuit breaker and canary counters are per process. See "Limits".
6. **Mutation checking found six real test gaps.** (a) My test for impossible pass counts accepted *any* `ValueError`, and the mutant with the check deleted still raised one, by accident, from a square root of a negative number. It now checks the message. (b) A `thread_full` (409) response was never shown to be excluded from a canary's error count. (c) Nothing proved the deadline is checked before the second model tier starts. (d) My log test used a user id longer than the 12 characters the log keeps, so logging the raw id slipped through. (e) Nothing proved an HTTP success body carries only the documented fields. (f) My concurrent-refund test, with 8 simultaneous callers, **still passed with the refund lock deleted** (8 of 8 runs): the window for the race is too small to hit by luck. I replaced it with a deterministic test that holds the first caller inside the approval step; with the lock removed it fails 5 of 5 runs.

## How I checked the tests can fail
I mutation-checked 122 behaviors in a scratch copy (each run in parallel, every mutant checked to compile): the agent-definition rules, thread ownership and deletion, every admission limit, backoff, retry and breaker rule, single-flight and idempotency, every cache-sharing rule, budgets, the log guard, the interval gate, canary rules, cascade and tool rules, the support tools' ownership/amount/approval/idempotency checks, the Claude adapter, the service's ordering and never-throw rule, and the HTTP limits and headers. **116 of the first 121 were caught on the first run. The 5 survivors, plus the refund-lock check I added afterwards, were real test gaps (listed above). I added tests, and all 122 are now caught.** That says these 122 changes are detected, not that the suite is complete. Race conditions in particular are hard to test by luck, so only the refund lock has a deterministic race test. The other locks (cache, ledger, metrics, breaker) are exercised by multi-thread tests but I did not try deleting each one.

## What is NOT known
- **Nothing ran against the real Claude API or a real provider's limits.** The 429 behaviour, latency and token counts are all test doubles. `make_agent_fn` is tested with scripted replies only, and its error classification works by status code and class name, not against real SDK exceptions.
- **No load test at realistic scale.** The overload scenario uses 40 threads and 30 ms of fake work. I do not know how this behaves at hundreds of requests per second or with real network latency.
- **I did not deploy anything.** There is no container, no Foundry or other cloud deployment, no TLS and no real authentication system. The lesson's Foundry-specific parts (Hosted Agents, Model Router, Foundry tracing, the smoke-test action) were read, not run.
- **The router's quality is unmeasured.** Nothing here says how often a cheap tier's answer passes your check, or whether the check catches bad answers.
- **Timing-based tests could be flaky on a loaded machine.** They passed on repeated runs here.
- **Windows only.** I ran everything on one Windows machine.

## Limits of the design
- **State is per process.** The cache, idempotency results, budgets, breaker and canary counters are in memory. With several instances, a retry can land on another instance and run again, budgets are per instance, and each instance decides its own rollback. Only threads are stored externally. [IMPLEMENTATION.md](IMPLEMENTATION.md) lists what would have to move.
- **A restart forgets the cache and the budgets.**
- **The default acceptance check only tests for non-empty text.** A real cascade needs a real check.
- **Promotion means "no evidence of harm yet".** A canary stage sees too little traffic to prove a small regression is absent.
- **The keyword hint is a guess.** It only saves a wasted cheap call; it is not a safety control.
- **A slow approver blocks other refunds**, because the lock is held while it runs.
- **Thread-per-request.** Fine here; a real deployment might want an async server.
- **The standard-library HTTP server** is not for the open internet.
- **Rate limits are per user and per process**, not global.
- **Persisted thread contents are not encrypted.**

## Exercises
1. Point `make_agent_fn` at the real API with a handful of requests. How often does `classify_error` treat a real error the way you would?
2. Move the idempotency store and the cost ledger into SQLite so two processes share them. What new race appears?
3. Replace the default acceptance check with a schema and grounding check. How many answers does the cheap tier now fail?
4. Run the overload scenario at 400 requests with 300 ms of fake latency. What do the queue timeout and `Retry-After` do?
5. Build a canary scenario where the candidate is only 5 points worse. How many requests does it take to catch?
