"""Chunking, search, permissions and retrieval quality. All offline and deterministic."""

import math

import pytest

from chunker import chunk_document
from corpus import CASES, DOCUMENTS, build_chunks
from evals import retrieval_report
from index import BM25, HashingEmbedder, HybridIndex, reciprocal_rank_fusion, tokenize


@pytest.fixture(scope="module")
def index():
    return HybridIndex(build_chunks())


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# --- chunker -------------------------------------------------------------------------------
LONG = "First fact is here. Second fact is here. Third fact is here. Fourth fact is here. Fifth fact is here."


def test_chunks_respect_max_chars_and_ids_are_sequential():
    chunks = chunk_document("d", "Doc", LONG, {"public"}, max_chars=50, overlap_sentences=0)
    assert len(chunks) > 1 and all(len(c.text) <= 50 for c in chunks)
    assert [c.id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]
    assert " ".join(c.text for c in chunks) == LONG  # nothing lost without overlap


def test_overlap_repeats_the_boundary_sentence():
    chunks = chunk_document("d", "Doc", LONG, {"public"}, max_chars=50, overlap_sentences=1)
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = a.text.split(". ")[-1].rstrip(".") + "."
        assert b.text.startswith(last_sentence)


def test_single_overlong_sentence_becomes_its_own_chunk_and_whitespace_is_normalized():
    chunks = chunk_document("d", "Doc", "A  very\nlong   sentence without any end that exceeds the limit.", {"public"}, max_chars=10)
    assert len(chunks) == 1 and "  " not in chunks[0].text and "\n" not in chunks[0].text


def test_empty_text_gives_no_chunks_and_access_is_kept():
    assert chunk_document("d", "Doc", "   ", {"public"}) == []
    assert chunk_document("d", "Doc", "Hello there.", {"staff"})[0].access == frozenset({"staff"})


def test_corpus_exercises_the_chunker():
    chunks = build_chunks()
    assert len(chunks) > len(DOCUMENTS)  # at least one document split in two
    assert all(c.id.split("#")[0] in {d[0] for d in DOCUMENTS} for c in chunks)


# --- tokenizer, BM25, embedder, fusion ------------------------------------------------------
def test_tokenizer_keeps_codes_drops_stopwords_and_stems_plurals():
    assert "fare-204" in tokenize("What does the FARE-204 flag mean?")
    assert "the" not in tokenize("the refunds") and tokenize("refunds") == tokenize("refund")


def test_bm25_prefers_the_rarer_term():
    docs = [["common", "rare"], ["common", "common"], ["common"]]
    bm = BM25(docs)
    assert bm.score(["rare"], 0) > bm.score(["common"], 0)
    assert bm.score(["rare"], 1) == 0.0


def test_hashing_embedder_is_deterministic_unit_length_and_shape_sensitive():
    e1, e2 = HashingEmbedder(), HashingEmbedder()
    v = e1.embed(["cancellation fee"])[0]
    assert v == e2.embed(["cancellation fee"])[0]
    assert math.isclose(cosine(v, v), 1.0, rel_tol=1e-9)
    a, b, c = (e1.embed([w])[0] for w in ("cancellation", "cancel", "baggage"))
    assert cosine(a, b) > cosine(a, c)  # similar word shapes are closer; this is NOT semantic similarity


def test_rrf_rewards_agreement_between_rankers():
    fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]]))
    assert fused[1] > fused[2] and fused[3] > fused[2] and fused[4] < fused[1]
    assert [i for i, _ in reciprocal_rank_fusion([[5, 6]])] == [5, 6]


# --- search and permissions -----------------------------------------------------------------
@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_public_users_never_receive_staff_chunks_in_any_mode(index, mode):
    for q in ("approval code MGR-77 goodwill credits", "FARE-204 manual review", "INTERNAL fraud escalation risk team"):
        hits = index.search(q, k=10, roles=("public",), mode=mode)
        assert all("staff" not in h.chunk.access for h in hits)
        assert all("MGR-77" not in h.chunk.text for h in hits)


def test_staff_users_do_receive_staff_chunks(index):
    hits = index.search("approval code for goodwill credits above $200", k=3, roles=("public", "staff"))
    assert any("MGR-77" in h.chunk.text for h in hits)


def test_unknown_mode_and_no_match(index):
    with pytest.raises(ValueError):
        index.search("x", mode="magic")
    assert index.search("zzzqqq", k=3, mode="bm25") == []


def test_exact_code_lookup_works_with_keywords(index):
    top = index.search("INS-200", k=1, mode="bm25")[0].chunk
    assert top.doc_id == "insurance"


# --- retrieval quality (floors from observed results, with documented limits) ---------------
def test_retrieval_floors(index):
    r = retrieval_report(index, k=3)
    assert r["hybrid"]["recall@k"] >= 0.9 and r["hybrid"]["mrr"] >= 0.85
    assert r["bm25"]["recall@k"] >= 0.9


def test_hybrid_returns_no_more_noise_than_keywords_alone(index):
    """At min_cosine=0.2 the trigram embedder let irrelevant chunks in and hybrid precision fell below BM25."""
    r = retrieval_report(index, k=4)
    assert r["hybrid"]["precision"] >= r["bm25"]["precision"] - 0.01


def test_known_gap_vocabulary_mismatch_is_fixed_by_rewriting_the_query(index):
    """'8 year old' never appears in the documents ('unaccompanied minors aged 5 to 11' does). No search mode
    bridges that gap, which is what query rewriting by an agent is for."""
    original = {h.chunk.doc_id for h in index.search("Can an 8 year old fly alone?", k=3)}
    rewritten = {h.chunk.doc_id for h in index.search("unaccompanied minor", k=3)}
    assert "children" not in original and "children" in rewritten


def test_every_answerable_case_has_gold_docs_that_exist():
    doc_ids = {d[0] for d in DOCUMENTS}
    assert all(c.gold_docs <= doc_ids for c in CASES)
    assert any(not c.gold_docs for c in CASES)  # unanswerable cases exist
