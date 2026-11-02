import json

import pytest

from retrieval import BM25Index, Hit, chunk_text, search_text, tokenize

DOCS = {"auth": "The login() function checks credentials. It currently compares passwords in plain text, which should be hashed.",
        "sessions": "Sessions are cleared on logout via session.clear(). Sessions are stored in memory and lost on restart.",
        "utils": "clamp(value, low, high) constrains a number to a range. Used throughout the UI layer for bounds checking.",
        "style": "This project follows PEP 8. Functions use snake_case and modules include a docstring at the top."}


@pytest.fixture
def idx():
    i = BM25Index()
    for k, v in DOCS.items():
        i.add(k, v)
    return i


# --- tokens and chunks ------------------------------------------------------------------------------------------------------------------
def test_tokens_are_lowercase_without_stopwords_and_plurals_are_folded():
    assert tokenize("How are the Passwords handled?") == ["password", "handled"]
    assert tokenize("a an the of") == []


def test_words_ending_in_double_s_or_short_words_keep_their_s():
    assert tokenize("class pass bus gas") == ["class", "pass", "bus", "gas"]


def test_identifiers_with_underscores_stay_whole():
    assert "snake_case" in tokenize("use snake_case names")


def test_paragraphs_become_chunks_and_whitespace_is_collapsed():
    assert chunk_text("First   para\nsame para.\n\n\nSecond para.") == ["First para same para.", "Second para."]


def test_a_long_paragraph_is_split_and_no_chunk_is_longer_than_the_limit():
    text = ". ".join(f"Sentence number {i} is here" for i in range(60)) + "."
    chunks = chunk_text(text, max_chars=100)
    assert len(chunks) > 5 and all(len(c) <= 100 for c in chunks) and " ".join(chunks).count("Sentence") == 60


def test_text_with_no_break_points_is_cut_by_length():
    chunks = chunk_text("x" * 250, max_chars=100)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == [] and chunk_text("   \n\n  ") == []


# --- searching -----------------------------------------------------------------------------------------------------------------------
def test_a_question_finds_the_document_that_answers_it(idx):
    hits = idx.search("how are passwords handled?")
    assert [h.doc_id for h in hits] == ["auth"] and isinstance(hits[0], Hit) and hits[0].coverage >= 0.5


def test_a_question_the_documents_never_mention_returns_nothing(idx):
    for q in ("what is the capital of France?", "how do I bake sourdough bread?", "who won the world cup?"):
        assert idx.search(q) == []


def test_without_a_threshold_the_lessons_behavior_returns_documents_for_anything_that_shares_a_single_word(idx):
    """min_coverage=0 is how the lesson's 'top 2, always' behaves: any chunk comes back, relevant or not."""
    assert len(idx.search("what is the capital of France?", k=2, min_coverage=0.0)) == 2


def test_partial_matches_pass_only_at_or_above_the_coverage_threshold(idx):
    q = "passwords stored memory"  # 'password' is in auth, 'stored' and 'memory' are in sessions
    assert [h.doc_id for h in idx.search(q, min_coverage=0.6)] == ["sessions"]
    assert {h.doc_id for h in idx.search(q, min_coverage=0.3)} == {"auth", "sessions"}


def test_results_are_ranked_by_score_and_limited_to_k(idx):
    idx.add("more", "Sessions sessions sessions are everywhere in this memory.")
    hits = idx.search("sessions memory", k=2)
    assert len(hits) == 2 and hits[0].score >= hits[1].score


def test_a_rare_word_counts_for_more_than_a_common_one():
    i = BM25Index()
    i.add("a", "alpha common common common")
    i.add("b", "beta common")
    i.add("c", "gamma common")
    assert i.search("alpha common", min_coverage=0.5)[0].doc_id == "a"


def test_equal_scores_are_ordered_by_document_id_so_results_are_deterministic():
    i = BM25Index()
    for name in ("zeta", "alpha", "mid"):
        i.add(name, "identical sentence about widgets")
    assert [h.doc_id for h in i.search("widgets")] == ["alpha", "mid", "zeta"]


def test_an_empty_query_a_stopword_only_query_and_an_empty_index_return_nothing(idx):
    assert idx.search("") == [] and idx.search("the of and") == [] and BM25Index().search("anything") == []


def test_add_reports_how_many_chunks_it_stored_and_skips_text_with_no_words():
    i = BM25Index()
    assert i.add("x", "The the of.") == 0 and i.add("y", "Real content here.\n\nMore real content.") == 2 and len(i) == 2


# --- saving and loading -----------------------------------------------------------------------------------------------------------------
def test_an_index_saved_to_disk_and_loaded_back_answers_the_same(idx, tmp_path):
    idx.save(tmp_path / "idx.json")
    again = BM25Index.load(tmp_path / "idx.json")
    assert [(h.doc_id, h.score) for h in again.search("passwords")] == [(h.doc_id, h.score) for h in idx.search("passwords")]


def test_loading_something_that_is_not_an_index_is_refused(tmp_path):
    p = tmp_path / "bad.json"
    for payload in ('{"a": 1}', '[{"doc": "x"}]', "[1, 2]", "not json"):
        p.write_text(payload)
        with pytest.raises(ValueError):
            BM25Index.load(p)


# --- the text the model receives ----------------------------------------------------------------------------------------------------
def test_matching_documents_are_returned_with_their_source(idx):
    assert search_text(idx, "how are passwords handled?").startswith("[auth] The login() function")


def test_nothing_relevant_is_an_explicit_message_that_tells_the_model_not_to_guess(idx):
    out = search_text(idx, "what is the capital of France?")
    assert out.startswith("No relevant documentation found.") and "Do not guess" in out


def test_a_rare_word_outranks_a_word_that_is_in_most_documents_even_when_the_common_one_repeats():
    i = BM25Index()
    i.add("a", "alpha")
    i.add("b", "common common common common")
    for n in "cde":
        i.add(n, "common")
    assert i.search("alpha common", min_coverage=0.5)[0].doc_id == "a"


def test_with_equal_term_counts_the_shorter_chunk_ranks_first():
    i = BM25Index()
    i.add("a_long", "alpha " + "filler words that add length to this one " * 6)
    i.add("z_short", "alpha beta")
    assert [h.doc_id for h in i.search("alpha")][0] == "z_short"
