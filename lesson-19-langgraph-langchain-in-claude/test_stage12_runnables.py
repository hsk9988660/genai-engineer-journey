import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence

from fakes import FlakyChatModel, ScriptedChatModel, ScriptExhausted, say
from stage12_runnables import (SIGNATURE, Sentiment, build_analysis_chain, build_drafting_chain, build_prompt, build_reliable_reply_chain,
                               build_reply_chain, draft_many, fallback_reply, redact, step_names)

PEOPLE = {"name": "Sam", "message": "My boots leak."}


class TestRedaction:
    @pytest.mark.parametrize("raw, clean", [
        ("write to sam@example.com", "write to [email]"),
        ("write to sam@example.com.", "write to [email]."),
        ("write to a.b+tag@mail.example.co.uk, thanks", "write to [email], thanks"),
        ("call +1 415 555 0100 now", "call [phone] now"),
        ("call (415) 555-0100.", "call [phone]."),
        ("call 415.555.0100 or 415-555-0100", "call [phone] or [phone]"),
        ("no personal data here", "no personal data here"),
        ("order A1001 costs 89 dollars", "order A1001 costs 89 dollars"),
    ])
    def test_redact(self, raw, clean):
        assert redact(raw) == clean


class TestKnownLimits:
    def test_a_date_looks_like_a_phone_number_a_known_false_positive(self):
        assert redact("delivered 2026-10-20") == "delivered [phone]"

    def test_names_are_not_found_at_all(self):
        assert redact("My name is Sam Carter") == "My name is Sam Carter"


class TestTheReplyChain:
    def test_the_chain_is_a_flat_sequence_of_five_steps(self):
        chain = build_reply_chain(ScriptedChatModel(script=[]))
        assert isinstance(chain, RunnableSequence)
        assert step_names(chain) == ["RunnableLambda", "ChatPromptTemplate", "ScriptedChatModel", "StrOutputParser", "RunnableLambda"]

    def test_personal_data_never_reaches_the_model(self):
        model = ScriptedChatModel(script=[say("ok")])
        build_reply_chain(model).invoke({"name": "Sam", "message": "Call +1 415 555 0100 or mail sam@example.com"})
        sent = " ".join(str(m.content) for m in model.seen[0])
        assert "415" not in sent and "sam@example.com" not in sent and "[phone]" in sent and "[email]" in sent

    def test_the_prompt_has_the_rules_first_and_the_customer_last(self):
        model = ScriptedChatModel(script=[say("ok")])
        build_reply_chain(model).invoke(PEOPLE)
        assert [m.type for m in model.seen[0]] == ["system", "human"] and model.seen[0][1].content == "Customer Sam writes:\nMy boots leak."

    def test_the_signature_is_added_by_code(self):
        reply = build_reply_chain(ScriptedChatModel(script=[say("  We are looking into it.  ")])).invoke(PEOPLE)
        assert reply == f"We are looking into it.\n\n{SIGNATURE}"

    def test_earlier_messages_go_where_the_placeholder_is(self):
        model = ScriptedChatModel(script=[say("ok")])
        history = [HumanMessage("hello"), AIMessage("hi, how can I help?")]
        build_reply_chain(model).invoke({**PEOPLE, "history": history})
        assert [m.type for m in model.seen[0]] == ["system", "human", "ai", "human"]

    def test_curly_braces_in_the_customers_text_are_not_treated_as_variables(self):
        model = ScriptedChatModel(script=[say("ok")])
        build_reply_chain(model).invoke({"name": "Sam", "message": "my code is {name} and {oops}"})
        assert "my code is {name} and {oops}" in model.seen[0][-1].content

    def test_a_missing_variable_is_an_error_not_a_blank(self):
        with pytest.raises(KeyError):
            build_prompt().invoke({"name": "Sam"})

    def test_a_chain_is_a_runnable_that_can_be_used_as_a_step_in_another(self):
        double = build_reply_chain(ScriptedChatModel(script=[say("a")])) | RunnableLambda(lambda text: text.upper())
        assert double.invoke(PEOPLE).startswith("A\n\n")


class TestParallelAnalysis:
    def chain(self, sentiment='{"label": "negative", "reason": "the boots leak"}'):
        first, second = ScriptedChatModel(script=[say(sentiment)]), ScriptedChatModel(script=[say("The boots leak.")])
        return build_analysis_chain(first, second), first, second

    def test_both_branches_answer_in_one_dict(self):
        chain, _, _ = self.chain()
        result = chain.invoke(PEOPLE)
        assert result == {"sentiment": Sentiment(label="negative", reason="the boots leak"), "summary": "The boots leak."}

    def test_the_typed_branch_returns_an_object_not_text(self):
        chain, _, _ = self.chain()
        assert isinstance(chain.invoke(PEOPLE)["sentiment"], Sentiment)

    def test_the_format_instructions_were_filled_into_the_prompt_once(self):
        chain, first, _ = self.chain()
        chain.invoke(PEOPLE)
        system = first.seen[0][0].content
        assert "label" in system and "{format_instructions}" not in system

    def test_both_branches_see_the_same_redacted_message(self):
        chain, first, second = self.chain()
        chain.invoke({"name": "Sam", "message": "mail sam@example.com"})
        assert "[email]" in first.seen[0][-1].content and "[email]" in second.seen[0][-1].content

    def test_a_label_outside_the_schema_is_a_parse_error(self):
        chain, _, _ = self.chain('{"label": "furious", "reason": "x"}')
        with pytest.raises(OutputParserException):
            chain.invoke(PEOPLE)

    def test_text_that_is_not_json_is_a_parse_error(self):
        chain, _, _ = self.chain("I think they are unhappy.")
        with pytest.raises(OutputParserException):
            chain.invoke(PEOPLE)


class TestPassthroughAssign:
    def test_the_input_is_kept_and_the_reply_is_added(self):
        result = build_drafting_chain(ScriptedChatModel(script=[say("ok")])).invoke(PEOPLE)
        assert set(result) == {"name", "message", "reply"} and result["name"] == "Sam" and result["reply"].endswith(SIGNATURE)


class TestResilience:
    def test_a_connection_error_is_retried(self):
        model = FlakyChatModel(script=[say("We are on it.")], failures=1)
        reply = build_reliable_reply_chain(model, attempts=2).invoke(PEOPLE)
        assert reply.startswith("We are on it.") and model.attempts == 2

    def test_when_every_attempt_fails_a_safe_canned_reply_is_sent(self):
        model = FlakyChatModel(script=[say("never used")], failures=5)
        reply = build_reliable_reply_chain(model, attempts=2).invoke(PEOPLE)
        assert reply == fallback_reply(PEOPLE) and "Hello Sam" in reply and model.attempts == 2

    def test_the_canned_reply_never_promises_anything(self):
        assert "refund" not in fallback_reply(PEOPLE).lower()

    def test_an_error_that_is_not_listed_is_not_hidden_by_the_fallback(self):
        model = ScriptedChatModel(script=[])
        with pytest.raises(ScriptExhausted):
            build_reliable_reply_chain(model).invoke(PEOPLE)


class TestBatch:
    def test_results_come_back_in_the_order_of_the_inputs(self):
        chain = RunnableLambda(lambda d: d["n"] * 10)
        assert chain.batch([{"n": n} for n in range(8)], config={"max_concurrency": 3}) == [0, 10, 20, 30, 40, 50, 60, 70]

    def test_draft_many_returns_a_failure_in_place_instead_of_raising(self):
        model = ScriptedChatModel(script=[say("first"), say("second")])
        results = draft_many(build_reply_chain(model), [PEOPLE, PEOPLE, PEOPLE], max_concurrency=1)
        assert [type(r).__name__ for r in results] == ["str", "str", "ScriptExhausted"]
        assert results[0].startswith("first") and results[1].startswith("second")


class TestStreaming:
    def test_a_plain_chain_streams_word_by_word_and_ends_with_an_empty_marker(self):
        chain = build_prompt() | ScriptedChatModel(script=[say("We are on it now")]) | StrOutputParser()
        chunks = list(chain.stream(PEOPLE))
        assert [c for c in chunks if c] == ["We ", "are ", "on ", "it ", "now"] and chunks[-1] == ""

    def test_a_function_at_the_end_holds_the_stream_until_the_text_is_complete(self):
        chain = build_reply_chain(ScriptedChatModel(script=[say("We are on it now")]))
        assert list(chain.stream(PEOPLE)) == [f"We are on it now\n\n{SIGNATURE}"]
