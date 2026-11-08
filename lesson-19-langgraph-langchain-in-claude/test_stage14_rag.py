import json
import math

import pytest
from langchain.agents import create_agent
from langchain_core.documents import Document

from conftest import tool_texts
from fakes import ScriptedChatModel, call, say
from stage14_rag import (ARTICLES, NOTHING_FOUND, HashingEmbeddings, KnowledgeBase, ask_knowledge_base, make_search_tool, split,
                         to_documents)
from tools import DeskContext


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


@pytest.fixture
def kb():
    return KnowledgeBase()


def ids(hits):
    return [doc.metadata["id"] for doc, _ in hits]


def articles(hits):
    return {doc.metadata["article"] for doc, _ in hits}


class TestEmbeddings:
    emb = HashingEmbeddings()

    def test_same_text_gives_the_same_vector(self):
        assert self.emb.embed_query("shipping takes days") == self.emb.embed_query("shipping takes days")

    def test_vectors_have_the_configured_size_and_unit_length(self):
        vector = self.emb.embed_query("standard shipping")
        assert len(vector) == 4096 and math.isclose(math.sqrt(dot(vector, vector)), 1.0)

    def test_shared_words_mean_a_closer_vector(self):
        query = self.emb.embed_query("how long does shipping take")
        near = self.emb.embed_query("Standard shipping takes 3 to 5 business days.")
        far = self.emb.embed_query("Every product has a one-year warranty.")
        assert dot(query, near) > dot(query, far)

    def test_common_words_are_ignored(self):
        assert self.emb.embed_query("the shipping") == self.emb.embed_query("shipping")

    def test_a_plural_matches_its_singular(self):
        assert self.emb.embed_query("days") == self.emb.embed_query("day")

    def test_empty_text_does_not_crash(self):
        assert self.emb.embed_query("") == [0.0] * 4096

    def test_documents_are_embedded_one_vector_each(self):
        assert len(self.emb.embed_documents(["a b", "c d", "e f"])) == 3


class TestKnownLimitOfWordMatching:
    def test_it_matches_words_not_meanings_so_a_synonym_is_missed(self, kb):
        # A real embedding model would link "watertight parka" to the care article about waterproof jackets.
        assert kb.search("will my parka stay watertight", "public") == []


class TestSplitting:
    def test_chunks_are_no_longer_than_the_limit(self):
        assert all(len(c.page_content) <= 140 for c in split(to_documents()))

    def test_every_chunk_keeps_its_metadata_and_gets_an_id(self):
        for chunk in split(to_documents()):
            assert chunk.metadata["audience"] in {"public", "staff"} and chunk.metadata["id"].startswith(chunk.metadata["article"] + "-")

    def test_ids_are_unique(self):
        chunk_ids = [c.metadata["id"] for c in split(to_documents())]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_neighbouring_chunks_overlap(self):
        first, second = [c for c in split(to_documents()) if c.metadata["article"] == "shipping"][:2]
        assert any(first.page_content.endswith(first.page_content[-n:]) and second.page_content.startswith(first.page_content[-n:]) for n in range(10, 26))

    def test_no_word_is_lost(self):
        for article in ARTICLES:
            chunks = [c.page_content for c in split(to_documents([article])) if c.metadata["article"] == article["id"]]
            assert set(article["text"].split()) <= set(" ".join(chunks).split())

    def test_a_short_document_stays_in_one_chunk(self):
        assert len(split([Document(page_content="Short.", metadata={"article": "x", "audience": "public"})])) == 1


class TestSearch:
    def test_the_best_match_comes_first(self, kb):
        assert kb.search("how long does shipping take", "public")[0][0].metadata["article"] == "shipping"

    def test_results_are_ordered_by_score_and_limited_by_k(self, kb):
        hits = kb.search("returns refund shipping warranty", "public", k=2)
        assert len(hits) <= 2 and [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)

    @pytest.mark.parametrize("query", ["what is the footwear margin", "agents may refund up to 200 dollars", "internal gross margin percent", "manager larger refunds"])
    def test_a_customer_never_gets_a_staff_article_even_with_a_huge_k(self, kb, query):
        assert "margins" not in articles(kb.search(query, "public", k=50))

    def test_staff_see_internal_and_public_articles(self, kb):
        assert "margins" in articles(kb.search("what is the footwear margin", "staff")) and "shipping" in articles(kb.search("shipping days", "staff"))

    def test_an_unknown_audience_sees_nothing(self, kb):
        assert kb.search("shipping days", "stranger") == []

    def test_the_filter_runs_before_ranking_so_a_hidden_chunk_cannot_take_a_slot(self, kb):
        query = "refund up to 200 dollars manager"
        top_unfiltered = kb.store.similarity_search_with_score(query, k=1)[0][0]
        assert top_unfiltered.metadata["audience"] == "staff"  # a post-filter would have nothing left after removing this
        assert all(doc.metadata["audience"] == "public" for doc, _ in kb.search(query, "public", k=1))

    def test_an_unrelated_question_finds_nothing(self, kb):
        assert kb.search("do you rent kayaks in the summer", "public") == []

    def test_without_a_threshold_the_nearest_chunks_come_back_however_far(self):
        assert len(KnowledgeBase(min_score=-1.0).search("do you rent kayaks in the summer", "public", k=3)) == 3

    def test_a_higher_threshold_removes_weak_matches(self):
        loose, strict = KnowledgeBase(min_score=0.1), KnowledgeBase(min_score=0.6)
        assert len(strict.search("shipping", "public")) < len(loose.search("shipping", "public"))


class TestWithdrawingAnArticle:
    def test_forget_removes_every_chunk_of_it(self, kb):
        assert kb.forget("shipping") == 2
        assert kb.search("how long does shipping take", "public") == []

    def test_other_articles_are_untouched(self, kb):
        kb.forget("shipping")
        assert "returns" in articles(kb.search("return an item within 30 days", "public"))

    def test_forgetting_an_unknown_article_removes_nothing(self, kb):
        assert kb.forget("nope") == 0


class TestAnswersYouCanCheck:
    def test_a_real_citation_makes_the_answer_grounded(self, kb):
        model = ScriptedChatModel(script=[say("It takes 3 to 5 business days [shipping-1].")])
        answer = ask_knowledge_base(model, kb, "How long does shipping take?")
        assert answer.grounded and answer.cited == ["shipping-1"] and answer.invalid == []

    def test_an_invented_citation_is_caught(self, kb):
        model = ScriptedChatModel(script=[say("It takes 3 days [shipping-9].")])
        answer = ask_knowledge_base(model, kb, "How long does shipping take?")
        assert not answer.grounded and answer.invalid == ["shipping-9"] and answer.cited == []

    def test_one_real_and_one_invented_citation_is_still_not_grounded(self, kb):
        model = ScriptedChatModel(script=[say("3 to 5 days [shipping-1]. Free for all [promo-1].")])
        answer = ask_knowledge_base(model, kb, "How long does shipping take?")
        assert not answer.grounded and answer.cited == ["shipping-1"] and answer.invalid == ["promo-1"]

    def test_an_answer_with_no_citation_is_not_grounded(self, kb):
        answer = ask_knowledge_base(ScriptedChatModel(script=[say("About a week.")]), kb, "How long does shipping take?")
        assert not answer.grounded and answer.cited == []

    def test_a_source_cited_twice_is_listed_once(self, kb):
        model = ScriptedChatModel(script=[say("3 to 5 days [shipping-1]. Express is faster [shipping-1].")])
        assert ask_knowledge_base(model, kb, "How long does shipping take?").cited == ["shipping-1"]

    def test_a_citation_of_a_chunk_that_exists_but_was_not_retrieved_is_invalid(self, kb):
        model = ScriptedChatModel(script=[say("Warranty is one year [warranty-1].")])
        answer = ask_knowledge_base(model, kb, "How long does shipping take?")
        assert answer.invalid == ["warranty-1"] and not answer.grounded

    def test_when_nothing_is_relevant_the_model_is_not_called(self, kb):
        model = ScriptedChatModel(script=[say("I made something up.")])
        answer = ask_knowledge_base(model, kb, "Do you rent kayaks?")
        assert model.seen == [] and answer.text == NOTHING_FOUND and answer.sources == [] and not answer.grounded

    def test_the_prompt_lists_the_sources_with_their_ids_and_the_rule(self, kb):
        model = ScriptedChatModel(script=[say("ok [shipping-1]")])
        ask_knowledge_base(model, kb, "How long does shipping take?")
        system, human = model.seen[0]
        assert "ONLY" in system.content and "[shipping-1] Standard shipping takes" in system.content
        assert human.content == "How long does shipping take?"

    def test_a_staff_only_article_never_reaches_the_prompt_for_a_customer(self, kb):
        model = ScriptedChatModel(script=[say("ok")])
        ask_knowledge_base(model, kb, "How many business days does standard shipping take, and what is the footwear margin?")
        assert model.seen, "the question should have found the public shipping article"
        assert "42 percent" not in " ".join(str(m.content) for m in model.seen[0])

    def test_a_question_only_staff_could_answer_gets_the_nothing_found_reply(self, kb):
        model = ScriptedChatModel(script=[say("42 percent")])
        assert ask_knowledge_base(model, kb, "what is the footwear margin", "public").text == NOTHING_FOUND and model.seen == []

    def test_staff_get_the_internal_answer(self, kb):
        model = ScriptedChatModel(script=[say("42 percent [margins-1]")])
        answer = ask_knowledge_base(model, kb, "what is the footwear margin", "staff")
        assert answer.grounded and "margins-1" in answer.sources

    def test_the_sources_are_reported_in_rank_order(self, kb):
        model = ScriptedChatModel(script=[say("ok")])
        answer = ask_knowledge_base(model, kb, "returns refund within 30 days", k=3)
        assert answer.sources[0].startswith("returns")


class TestAsAnAgentTool:
    def agent_result(self, kb, context, query="footwear margin refund"):
        model = ScriptedChatModel(script=[call("search_knowledge_base", {"query": query}), say("ok")])
        agent = create_agent(model, [make_search_tool(kb)], context_schema=DeskContext)
        return agent.invoke({"messages": [("user", "x")]}, context=context)

    def test_the_model_sees_only_a_query_argument(self, kb):
        assert set(make_search_tool(kb).tool_call_schema.model_json_schema()["properties"]) == {"query"}

    def test_a_customer_context_gets_public_passages_only(self, kb):
        body = json.loads(tool_texts(self.agent_result(kb, DeskContext("C-100")))[0])
        assert all(not p["id"].startswith("margins") for p in body["passages"])

    def test_a_staff_context_gets_the_internal_passage(self, kb):
        body = json.loads(tool_texts(self.agent_result(kb, DeskContext("S-1", audience="staff")))[0])
        assert any(p["id"].startswith("margins") for p in body["passages"])

    def test_an_audience_argument_from_the_model_is_ignored(self, kb):
        model = ScriptedChatModel(script=[call("search_knowledge_base", {"query": "footwear margin", "audience": "staff"}), say("ok")])
        agent = create_agent(model, [make_search_tool(kb)], context_schema=DeskContext)
        result = agent.invoke({"messages": [("user", "x")]}, context=DeskContext("C-100"))
        assert json.loads(tool_texts(result)[0]) == {"passages": []}

    def test_nothing_relevant_is_an_empty_list_not_an_error(self, kb):
        result = self.agent_result(kb, DeskContext("C-100"), "do you rent kayaks")
        assert json.loads(tool_texts(result)[0]) == {"passages": []}
