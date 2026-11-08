import time

import pytest
from langchain.agents import create_agent
from langchain_core.caches import InMemoryCache
from langchain_core.messages import AIMessage, HumanMessage

from desk import Desk
from fakes import FlakyChatModel, ScriptedChatModel, call, say
from stage15_model_ops import (CountingCache, CountingLimiter, UsageTracker, build_ops_agent, cached_model, fast_limiter, limited_model, make_model)
from tools import DeskContext, make_tools

LOOKUP = {"order_id": "A1001"}


def run_agent(model, desk, text="What is in A1001?", callbacks=None, **kwargs):
    agent = create_agent(model, make_tools(desk), context_schema=DeskContext, **kwargs)
    config = {"callbacks": callbacks} if callbacks else None
    return agent.invoke({"messages": [HumanMessage(text)]}, config, context=DeskContext("C-100"))


class TestUsageTracker:
    def tracked(self, desk):
        tracker = UsageTracker()
        model = ScriptedChatModel(script=[call("lookup_order", LOOKUP, tokens=(80, 20)), say("Trail Shoes.", tokens=(120, 14))])
        run_agent(model, desk, callbacks=[tracker])
        return tracker

    def test_it_counts_model_calls_and_tokens(self, desk):
        tracker = self.tracked(desk)
        assert (tracker.model_calls, tracker.input_tokens, tracker.output_tokens) == (2, 200, 34)

    def test_it_lists_the_tools_that_ran(self, desk):
        assert self.tracked(desk).tool_calls == ["lookup_order"]

    def test_cost_follows_the_price_table(self, desk):
        assert self.tracked(desk).cost_cents == pytest.approx((200 * 300 + 34 * 1500) / 1_000_000)

    def test_messages_without_usage_numbers_cost_nothing(self, desk):
        tracker = UsageTracker()
        run_agent(ScriptedChatModel(script=[say("hi")]), desk, callbacks=[tracker])
        assert (tracker.model_calls, tracker.input_tokens, tracker.cost_cents) == (1, 0, 0)

    def test_each_tracker_has_its_own_lists(self):
        first, second = UsageTracker(), UsageTracker()
        first.tool_calls.append("x")
        assert second.tool_calls == []

    def test_a_failed_model_call_is_recorded_and_still_raised(self):
        tracker = UsageTracker()
        with pytest.raises(ConnectionError):
            FlakyChatModel(script=[], failures=1).invoke("hi", {"callbacks": [tracker]})
        assert tracker.failures == ["model: ConnectionError"] and tracker.model_calls == 0

    def test_a_tracker_that_is_not_passed_sees_nothing(self, desk):
        tracker = UsageTracker()
        run_agent(ScriptedChatModel(script=[say("hi")]), desk)
        assert tracker.model_calls == 0


class TestCache:
    def test_the_same_prompt_twice_calls_the_model_once(self):
        model = cached_model([say("first"), say("second")])
        assert (model.invoke("return policy?").content, model.invoke("return policy?").content, model.cursor) == ("first", "first", 1)

    def test_a_different_prompt_is_a_miss(self):
        model = cached_model([say("first"), say("second")])
        assert (model.invoke("return policy?").content, model.invoke("shipping?").content) == ("first", "second")

    def test_without_a_cache_every_call_reaches_the_model(self):
        model = ScriptedChatModel(script=[say("first"), say("second")])
        assert (model.invoke("same").content, model.invoke("same").content) == ("first", "second")

    def test_two_models_can_share_one_cache(self):
        cache = InMemoryCache()
        first = cached_model([say("from the first")], cache)
        first.invoke("question")
        second = cached_model([say("from the second")], cache)
        assert second.invoke("question").content == "from the first" and second.cursor == 0

    def test_a_changed_fact_in_the_prompt_is_a_miss(self):
        model = cached_model([say("Trail Shoes."), say("Hiking Boots.")])
        assert model.invoke("The order contains Trail Shoes. What is in the order?").content == "Trail Shoes."
        assert model.invoke("The order contains Hiking Boots. What is in the order?").content == "Hiking Boots."

    def test_a_cached_reply_is_stamped_as_free(self):
        model = cached_model([say("first")])
        model.invoke("q")
        assert model.invoke("q").usage_metadata == {"total_cost": 0}

    def test_in_an_agent_only_the_first_call_of_a_repeated_run_is_replayed(self, desk):
        cache = CountingCache()
        model = ScriptedChatModel(script=[call("lookup_order", LOOKUP), say("one"), say("two")], cache=cache)
        run_agent(model, desk)
        run_agent(model, desk)
        # The second run's first prompt is identical, so it hits. Its second prompt contains the cached reply, which now carries
        # the zero-cost stamp, so it differs from the first run's and misses.
        assert (cache.lookups, cache.hits, model.cursor) == (4, 1, 3)

    def test_the_counting_cache_counts_a_plain_repeat_as_a_hit(self):
        cache = CountingCache()
        model = cached_model([say("a")], cache)
        model.invoke("q")
        model.invoke("q")
        assert (cache.lookups, cache.hits) == (2, 1)


class TestRateLimiting:
    def test_each_model_call_asks_the_limiter_once(self, desk):
        limiter = CountingLimiter()
        run_agent(limited_model([call("lookup_order", LOOKUP), say("done")], limiter), desk)
        assert limiter.acquired == 2

    def test_a_model_without_a_limiter_never_asks(self, desk):
        limiter = CountingLimiter()
        run_agent(ScriptedChatModel(script=[say("hi")]), desk)
        assert limiter.acquired == 0

    def test_a_fresh_token_bucket_starts_empty(self):
        assert fast_limiter().acquire(blocking=False) is False

    def test_the_buckets_clock_starts_at_its_first_use_not_at_creation(self):
        limiter = fast_limiter(1000)
        time.sleep(0.02)  # waiting before the first use does not fill the bucket
        assert limiter.acquire(blocking=False) is False

    def test_the_bucket_refills_over_time_and_a_burst_is_capped(self):
        limiter = fast_limiter(1000)
        assert limiter.acquire(blocking=False) is False  # the first call starts the clock, so the bucket is empty
        time.sleep(0.02)
        assert limiter.acquire(blocking=False) is True and limiter.acquire(blocking=False) is False

    def test_a_blocking_acquire_waits_for_a_token(self):
        assert fast_limiter(1000).acquire(blocking=True) is True

    def test_a_chat_model_with_a_real_limiter_still_answers(self):
        assert limited_model([say("fine")], fast_limiter(1000)).invoke("hi").content == "fine"


class TestInitChatModel:
    def test_a_provider_and_model_in_one_string(self):
        pytest.importorskip("langchain_anthropic")
        assert make_model("anthropic:claude-sonnet-5-5", api_key="test-not-real").model == "claude-sonnet-5-5"

    def test_the_provider_can_be_given_separately(self):
        pytest.importorskip("langchain_anthropic")
        assert make_model("claude-haiku-5-5", model_provider="anthropic", api_key="test-not-real").model == "claude-haiku-5-5"

    def test_an_unknown_provider_is_an_error(self):
        with pytest.raises(ValueError):
            make_model("nowhere:some-model")


class TestRetryFallbackAndSummarising:
    def agent(self, primary, backup=None, summarizer=None, **kwargs):
        return build_ops_agent(primary, backup or ScriptedChatModel(script=[say("backup answer")]), summarizer or ScriptedChatModel(script=[]), Desk(), **kwargs)

    def ask(self, agent, text="hello"):
        return agent.invoke({"messages": [HumanMessage(text)]}, context=DeskContext("C-100"))

    def test_a_healthy_primary_never_touches_the_backup(self):
        backup = ScriptedChatModel(script=[say("backup answer")])
        result = self.ask(self.agent(ScriptedChatModel(script=[say("primary answer")]), backup))
        assert result["messages"][-1].content == "primary answer" and backup.seen == []

    def test_one_blip_is_retried_on_the_primary(self):
        primary, backup = FlakyChatModel(script=[say("primary answer")], failures=1), ScriptedChatModel(script=[say("backup answer")])
        result = self.ask(self.agent(primary, backup))
        assert result["messages"][-1].content == "primary answer" and primary.attempts == 2 and backup.seen == []

    def test_a_dead_primary_is_retried_and_then_replaced(self):
        primary = FlakyChatModel(script=[], failures=9)
        result = self.ask(self.agent(primary))
        assert result["messages"][-1].content == "backup answer" and primary.attempts == 2

    def test_with_retry_on_the_outside_the_primary_is_tried_only_once(self):
        primary = FlakyChatModel(script=[], failures=9)
        result = self.ask(self.agent(primary, retry_before_fallback=False))
        assert result["messages"][-1].content == "backup answer" and primary.attempts == 1

    def test_when_the_backup_fails_too_the_error_reaches_the_caller(self):
        with pytest.raises(ConnectionError):
            self.ask(self.agent(FlakyChatModel(script=[], failures=9), FlakyChatModel(script=[], failures=9)))

    def long_history(self, turns):
        messages = []
        for n in range(turns):
            messages += [HumanMessage(f"question {n}"), AIMessage(f"answer {n}")]
        return messages

    def test_a_short_conversation_is_not_summarised(self):
        summarizer = ScriptedChatModel(script=[say("SUMMARY")])
        self.ask(self.agent(ScriptedChatModel(script=[say("ok")]), summarizer=summarizer, summarize_after=20))
        assert summarizer.seen == []

    def test_a_long_conversation_is_summarised_before_the_model_is_called(self):
        summarizer, primary = ScriptedChatModel(script=[say("The customer asked ten questions about boots.")]), ScriptedChatModel(script=[say("ok")])
        agent = self.agent(primary, summarizer=summarizer, summarize_after=8, keep=2)
        agent.invoke({"messages": [*self.long_history(6), HumanMessage("latest question")]}, context=DeskContext("C-100"))
        sent = " ".join(str(m.content) for m in primary.seen[0])
        assert len(summarizer.seen) == 1
        assert "ten questions about boots" in sent and "latest question" in sent
        assert "question 0" not in sent and len(primary.seen[0]) < 13
