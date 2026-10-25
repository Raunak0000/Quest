"""Unit tests for Rankers (TF-IDF, etc.)."""

import os
import tempfile
import pytest
from backend.rankers import TfidfRanker, Bm25Ranker, identity_analyzer


def test_identity_analyzer():
    tokens = ["quick", "brown", "fox"]
    assert identity_analyzer(tokens) == tokens


def test_tfidf_ranker_small():
    # Toy corpus with 3 documents
    corpus_tokens = {
        101: ["appl", "pie", "recip", "dessert"],
        102: ["banana", "bread", "recip", "bake"],
        103: ["chocol", "cake", "sweet", "dessert"],
    }
    
    ranker = TfidfRanker()
    ranker.fit(corpus_tokens)
    
    assert ranker.tfidf_matrix.shape == (3, 10)
    assert len(ranker.pids) == 3

    # Search for apple pie
    results = ranker.search("apple pie", k=2)
    assert len(results) == 2
    # First result should be doc 101 with highest score
    assert results[0][0] == 101
    assert results[0][1] > results[1][1]

    # Scores should be sorted descending
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_tfidf_ranker_save_load():
    corpus_tokens = [
        ["run", "fast", "marathon"],
        ["swim", "pool", "water"],
    ]
    pids = [10, 20]

    ranker = TfidfRanker()
    ranker.fit(corpus_tokens, pids=pids)

    results_before = ranker.search("swimming pool", k=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "tfidf.joblib")
        ranker.save(save_path)
        assert os.path.exists(save_path)

        loaded_ranker = TfidfRanker.load(save_path)
        results_after = loaded_ranker.search("swimming pool", k=2)

        assert results_before == results_after
        assert results_after[0][0] == 20


def test_tfidf_ranker_oov():
    corpus_tokens = [["cat", "dog"]]
    ranker = TfidfRanker()
    ranker.fit(corpus_tokens, pids=[1])

    # Out of vocabulary term
    results = ranker.search("spaceship", k=5)
    assert results == []


def test_bm25_ranker_small():
    corpus_tokens = {
        101: ["appl", "pie", "recip", "dessert"],
        102: ["banana", "bread", "recip", "bake"],
        103: ["chocol", "cake", "sweet", "dessert"],
    }

    ranker = Bm25Ranker()
    ranker.fit(corpus_tokens)

    results = ranker.search("apple pie", k=2)
    assert len(results) == 2
    assert results[0][0] == 101
    assert results[0][1] >= results[1][1]

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_ranker_save_load():
    corpus_tokens = [
        ["run", "fast", "marathon"],
        ["swim", "pool", "water"],
    ]
    pids = [10, 20]

    ranker = Bm25Ranker()
    ranker.fit(corpus_tokens, pids=pids)

    results_before = ranker.search("swimming pool", k=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "bm25.joblib")
        ranker.save(save_path)
        assert os.path.exists(save_path)

        loaded_ranker = Bm25Ranker.load(save_path)
        results_after = loaded_ranker.search("swimming pool", k=2)

        assert results_before == results_after
        assert results_after[0][0] == 20


def test_bm25_vs_tfidf_overlap_not_identical():
    # Documents with different term frequencies and lengths
    corpus_tokens = {
        1: ["coffe", "bean", "roast", "dark", "brew", "coffe"],
        2: ["coffe", "espresso", "latte", "milk"],
        3: ["tea", "green", "leaf", "brew"],
        4: ["water", "drink", "cold"],
    }

    tfidf = TfidfRanker().fit(corpus_tokens)
    bm25 = Bm25Ranker().fit(corpus_tokens)

    tfidf_results = tfidf.search("coffee brew", k=3)
    bm25_results = bm25.search("coffee brew", k=3)

    tfidf_pids = [pid for pid, _ in tfidf_results]
    bm25_pids = [pid for pid, _ in bm25_results]

    # Both should retrieve document 1 in top results
    assert 1 in tfidf_pids
    assert 1 in bm25_pids
    # Scores are not identical
    assert tfidf_results[0][1] != bm25_results[0][1]

