# Stage 15: running models in production

**Code:** [stage15_model_ops.py](../stage15_model_ops.py) | **Tests:** [test_stage15_model_ops.py](../test_stage15_model_ops.py) | **Run:** `python stage15_model_ops.py`

The model call is the slowest, costliest and least reliable part of an agent. LangChain gives you hooks at three levels.

| Level | Tool | What it can do |
|---|---|---|
| **Outside** the agent | callbacks (`BaseCallbackHandler`) | watch every model and tool call: tokens, cost, latency, errors |
| **On the model object** | `cache=`, `rate_limiter=` | skip a repeated call; slow calls down to a budget |
| **Inside the agent loop** | middleware | retry a failed call, switch to another model, summarise a long history |

## Level 1: callbacks that count

```python
class UsageTracker(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs): ...      # a model call finished: add its tokens
    def on_tool_start(self, serialized, input_str, **kwargs): ...
    def on_llm_error(self, error, **kwargs): ...
```

You pass one in the run config, and the agent does not change at all:

```python
agent.invoke(input, {"callbacks": [tracker]}, context=...)
```

```
model calls=2 tokens in/out=200/34 tools=['lookup_order'] cost=0.1110 cents (placeholder prices)
```

The numbers come from each `AIMessage.usage_metadata`. A real provider fills these in; the fake model attaches them when you write `say("...", tokens=(120, 14))`. Create **one tracker per run or per request**: the lists are per instance (`test_each_tracker_has_its_own_lists`), and a tracker that is not passed sees nothing (`test_a_tracker_that_is_not_passed_sees_nothing`).

The prices in the file are **made up**. Use your provider's current prices, and keep them in configuration, not in code. A failed model call is recorded *and* still raised (`test_a_failed_model_call_is_recorded_and_still_raised`): a callback observes, it does not hide.

Cost per request is only half the picture. Lesson 10 explains why **cost per successful outcome** is the number to watch.

## Level 2: cache and rate limit

### Cache

```python
model = ScriptedChatModel(script=[...], cache=InMemoryCache())     # any chat model accepts cache=
model.invoke("return policy?")      # a real call
model.invoke("return policy?")      # answered from memory; the model is not involved
```

The key is the **whole prompt**. A different question is a miss (`test_a_different_prompt_is_a_miss`), and so is a changed fact inside the prompt (`test_a_changed_fact_in_the_prompt_is_a_miss`). Several models can share one cache (`test_two_models_can_share_one_cache`).

Three things I found, because the first tests I wrote were wrong:

1. A cached reply is **stamped as free**. The returned `AIMessage` has `usage_metadata == {"total_cost": 0}` (`test_a_cached_reply_is_stamped_as_free`). That is useful for cost tracking.
2. In an **agent**, that stamp changes the next prompt. Run the same agent question twice. The first model call of the second run is identical, so it hits. The second call contains the cached reply, now carrying the stamp, so its key no longer matches the first run's, and it misses (`test_in_an_agent_only_the_first_call_of_a_repeated_run_is_replayed`: 4 lookups, 1 hit, the model really called 3 times). A model-level cache therefore saves less inside an agent loop than you might expect.
3. My first test for this passed for the wrong reason (everything missed anyway), until I counted hits with `CountingCache` instead of guessing.

When is a cache right? For questions whose answer does **not** depend on who asks or on a fact that changes, such as general policy wording. When is it wrong? Anything about a particular order or customer. Lesson 16 shows a cache that leaked one customer's order to another; the lesson is the same here.

### Rate limiting

```python
ScriptedChatModel(script=[...], rate_limiter=InMemoryRateLimiter(requests_per_second=..., max_bucket_size=...))
```

Before every provider call, the chat model asks its limiter for a token. `InMemoryRateLimiter` is a token bucket. Two details that surprised me, both tested:

- A new bucket **starts empty**: `acquire(blocking=False)` is `False` at once (`test_a_fresh_token_bucket_starts_empty`).
- Its clock starts at the **first use**, not at creation. Waiting before the first call does not fill it (`test_the_buckets_clock_starts_at_its_first_use_not_at_creation`). After the first call and a short wait it refills, and a burst is capped by `max_bucket_size` (`test_the_bucket_refills_over_time_and_a_burst_is_capped`).
- `acquire(blocking=True)` waits until a token is free.

`CountingLimiter` never waits and just counts, which shows when the limiter is consulted: once per model call (`test_each_model_call_asks_the_limiter_once`). A limiter is **per process**. For several servers you need a shared limit (lesson 16).

## Level 3: middleware around the model call

```python
SummarizationMiddleware(model=summarizer, trigger=("messages", 8), keep=("messages", 2))
ModelRetryMiddleware(max_retries=1, retry_on=(ConnectionError,), on_failure="error", ...)
ModelFallbackMiddleware(backup)
```

### Retry and fallback, and why order matters again

For model-call middleware, **the first in the list is the outermost layer** (the same rule as stage 6). That changes what happens when the primary model is down:

| Order | Primary tried | Result |
|---|---|---|
| `[fallback, retry]` (`retry_before_fallback=True`, the default here) | **2 times** (1 try + 1 retry) | then the backup answers |
| `[retry, fallback]` | **1 time** | the first failure goes straight to the backup, so the retry never fires |

(`test_a_dead_primary_is_retried_and_then_replaced`, `test_with_retry_on_the_outside_the_primary_is_tried_only_once`.) My first demo used the second order and the "retry" did nothing. Decide what you want and write a test for the order you chose.

Other cases: a healthy primary never touches the backup (`test_a_healthy_primary_never_touches_the_backup`); a single blip is retried on the primary and the backup stays unused; and when the backup fails too, the error **reaches the caller** and is not hidden (`test_when_the_backup_fails_too_the_error_reaches_the_caller`).

### Summarising a long conversation

When the history reaches the trigger (8 messages), the middleware asks a summariser model to condense the older part and keeps the newest messages (`keep`). The main model then sees the summary plus the recent turns (`test_a_long_conversation_is_summarised_before_the_model_is_called`). A short conversation never calls the summariser (`test_a_short_conversation_is_not_summarised`).

Compare with stage 13's `trim_history`, which drops old messages. Summarising keeps the gist but costs an extra model call and can be **wrong**. The old turns are gone from the prompt, so keep the raw transcript somewhere else. Lesson 12 shows how to verify a summary before it replaces anything.

## `init_chat_model`

```python
make_model("anthropic:claude-sonnet-5-5")                          # provider and model in one string
make_model("claude-haiku-5-5", model_provider="anthropic")         # or the provider separately
```

The rest of your code receives a `BaseChatModel` and does not care which provider it is. An unknown provider raises `ValueError`. The tests construct the models with a dummy key and make **no network call**.

## What can go wrong

- **Token counts you cannot trust.** `usage_metadata` is only as good as the provider's report.
- **A cache that serves stale or another user's answers.**
- **Retrying a call that already cost money.** A failed call can still be billed; set a total budget (lesson 16).
- **A limiter per process** behind a load balancer.
- **A summary that silently drops a fact** the next step needs.
- **A fallback model with different behaviour.** A cheaper backup may follow your rules less well. Evaluate the backup too.

## Try it

1. Make `UsageTracker` record how long each model call took (`on_llm_start`, `on_llm_end`). How do you keep it deterministic in a test?
2. Add a daily budget: raise an error from a middleware once `tracker.cost_cents` passes a limit.
3. Put `ModelCallLimitMiddleware(run_limit=3)` in front of `build_ops_agent`'s middleware list. What does it do to a loop?

## Check yourself

- Where would you count tokens without changing the agent?
- Why can a model-level cache replay only the first call of a repeated agent run?
- `[retry, fallback]` or `[fallback, retry]`: which tries the primary model twice?
- A token bucket is created and not used for ten minutes. How many tokens does it hold at first use?
