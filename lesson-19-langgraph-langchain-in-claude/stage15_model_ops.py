"""Stage 15: running models in production. Cost tracking, caching, rate limits, fallbacks and long conversations.

The model call is the slowest, costliest and least reliable part of an agent. LangChain gives you hooks at three levels:

1. CALLBACKS see every model call and tool call from outside, without changing the agent. Perfect for token and cost
   tracking, latency, and logging. You subclass `BaseCallbackHandler` and pass it in the run config.
2. MODEL SETTINGS belong to the chat model object: `cache=` skips a repeated identical call, `rate_limiter=` slows calls
   down to a budget before they reach the provider.
3. MIDDLEWARE (stage 6) wraps the model call inside the agent: `ModelRetryMiddleware` retries a failed call,
   `ModelFallbackMiddleware` switches to another model when one fails, and `SummarizationMiddleware` shrinks a long
   history by summarising the oldest part.

Two cautions that the tests pin down:
- A cache answers from memory, so the model (and its bill) is not involved. It is only right for questions whose answer
  does not depend on who asks or on a changing fact. The key is the whole prompt, so changed wording or a changed fact
  is a miss. Inside an agent it works less well than you would hope: a cached reply is stamped with
  `usage_metadata={"total_cost": 0}`, so the next prompt in the loop differs from the first run's, and only the first
  model call of a repeated run is replayed.
- A summary is written by a model, so it can be wrong. The history it replaces is gone from the prompt. Keep the raw
  transcript elsewhere (lesson 12 verifies summaries).

Run the offline demo:  python stage15_model_ops.py
"""

from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware, SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.caches import InMemoryCache
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

from desk import Desk
from fakes import FlakyChatModel, ScriptedChatModel, call, say
from tools import DeskContext, make_tools

# Placeholder prices in cents per million tokens. They are made up. Use your provider's real, current prices.
PRICE_PER_MILLION_CENTS = {"input": 300, "output": 1500}


@dataclass
class UsageTracker(BaseCallbackHandler):
    """Counts model calls, tokens, tool calls and failures for one run. Create one per run (or per request)."""

    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.model_calls += 1
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self.input_tokens += usage["input_tokens"]
                    self.output_tokens += usage["output_tokens"]

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        self.tool_calls.append(serialized.get("name", "?"))

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.failures.append(f"model: {type(error).__name__}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self.failures.append(f"tool: {type(error).__name__}")

    @property
    def cost_cents(self) -> float:
        return (self.input_tokens * PRICE_PER_MILLION_CENTS["input"] + self.output_tokens * PRICE_PER_MILLION_CENTS["output"]) / 1_000_000


class CountingLimiter(BaseRateLimiter):
    """A rate limiter that never waits and just counts. Shows when a chat model asks its limiter for permission."""

    def __init__(self) -> None:
        self.acquired = 0

    def acquire(self, *, blocking: bool = True) -> bool:
        self.acquired += 1
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        return self.acquire(blocking=blocking)


class CountingCache(InMemoryCache):
    """An in-memory cache that also counts how often it was asked and how often it could answer."""

    def __init__(self) -> None:
        super().__init__()
        self.lookups = 0
        self.hits = 0

    def lookup(self, prompt: str, llm_string: str):
        self.lookups += 1
        found = super().lookup(prompt, llm_string)
        self.hits += found is not None
        return found


def cached_model(script: list, cache: InMemoryCache | None = None) -> ScriptedChatModel:
    """A model with a cache. The same messages a second time are answered from memory."""
    return ScriptedChatModel(script=script, cache=cache if cache is not None else InMemoryCache())


def limited_model(script: list, limiter: BaseRateLimiter) -> ScriptedChatModel:
    return ScriptedChatModel(script=script, rate_limiter=limiter)


def fast_limiter(per_second: float = 1000) -> InMemoryRateLimiter:
    """A token bucket. `requests_per_second` is the refill rate; `max_bucket_size` is the largest burst."""
    return InMemoryRateLimiter(requests_per_second=per_second, check_every_n_seconds=0.001, max_bucket_size=1)


def make_model(spec: str, **kwargs: Any) -> BaseChatModel:
    """`"provider:model"` in, a chat model out. The rest of your code does not care which provider it is."""
    return init_chat_model(spec, **kwargs)


def build_ops_agent(primary: BaseChatModel, backup: BaseChatModel, summarizer: BaseChatModel, desk: Desk, *, summarize_after: int = 8,
                    keep: int = 2, retries: int = 1, retry_before_fallback: bool = True):
    """Retry the primary model, fall back to a backup, and summarise a long history.

    ORDER DECIDES THE BEHAVIOUR. The first middleware in the list is the outermost layer.
    - fallback OUTSIDE retry (the default here): the primary is retried `retries` times, and only then is the backup used.
    - retry OUTSIDE fallback: the first failure moves straight to the backup inside the first attempt, so the primary
      is tried only once and the retry never fires.
    """
    retry = ModelRetryMiddleware(max_retries=retries, retry_on=(ConnectionError,), on_failure="error", initial_delay=0.0, jitter=False)
    fallback = ModelFallbackMiddleware(backup)
    return create_agent(
        primary, make_tools(desk), context_schema=DeskContext,
        middleware=[SummarizationMiddleware(model=summarizer, trigger=("messages", summarize_after), keep=("messages", keep)),
                    *([fallback, retry] if retry_before_fallback else [retry, fallback])],
    )


def demo() -> None:
    tracker = UsageTracker()
    model = ScriptedChatModel(script=[call("lookup_order", {"order_id": "A1001"}, tokens=(80, 20)), say("It is a pair of Trail Shoes.", tokens=(120, 14))])
    agent = create_agent(model, make_tools(Desk()), context_schema=DeskContext)
    agent.invoke({"messages": [("user", "What is in A1001?")]}, {"callbacks": [tracker]}, context=DeskContext("C-100"))
    print(f"model calls={tracker.model_calls} tokens in/out={tracker.input_tokens}/{tracker.output_tokens} tools={tracker.tool_calls} "
          f"cost={tracker.cost_cents:.4f} cents (placeholder prices)")

    cache = InMemoryCache()
    model = cached_model([say("first answer"), say("second answer")], cache)
    print("\nsame prompt twice:", model.invoke("What is your return policy?").content, "/", model.invoke("What is your return policy?").content,
          "| model really called", model.cursor, "time")

    flaky = FlakyChatModel(script=[], failures=9)
    agent = build_ops_agent(flaky, ScriptedChatModel(script=[say("Answered by the backup model.")]), ScriptedChatModel(script=[]), Desk())
    result = agent.invoke({"messages": [("user", "hello")]}, context=DeskContext("C-100"))
    print("\nprimary was tried", flaky.attempts, "times (1 try + 1 retry); the customer got:", result["messages"][-1].content)


if __name__ == "__main__":
    demo()
