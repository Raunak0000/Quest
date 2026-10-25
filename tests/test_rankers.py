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


def test_dense_ranker_fit_save_load():
    import numpy as np
    from backend.rankers import DenseRanker

    # 3 dummy normalized embeddings of dim 4
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    embeddings = np.vstack([v1, v2, v3])
    pids = [101, 202, 303]

    ranker = DenseRanker()
    ranker.fit(embeddings, pids=pids)

    assert ranker.index.ntotal == 3
    assert len(ranker.pids) == 3

    # Test save and load roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "dense.joblib")
        ranker.save(save_path)
        assert os.path.exists(save_path)

        loaded_ranker = DenseRanker.load(save_path)
        assert loaded_ranker.index.ntotal == 3
        assert list(loaded_ranker.pids) == pids

        # Direct FAISS search on loaded index with vector closest to v2
        query_vec = np.array([[0.1, 0.9, 0.0, 0.0]], dtype=np.float32)
        scores, indices = loaded_ranker.index.search(query_vec, 1)
        assert loaded_ranker.pids[indices[0][0]] == 202


def test_rrf_fuse_hand_computed():
    from backend.rrf import rrf_fuse

    # 2-list hand-computed test case
    # k = 60
    # List 1: [100, 200, 300]
    # List 2: [100, 300, 400]
    list1 = [100, 200, 300]
    list2 = [100, 300, 400]

    # Hand computation:
    # 100: rank 1 (L1) + rank 1 (L2) -> 1/(60+1) + 1/(60+1) = 2/61
    # 300: rank 3 (L1) + rank 2 (L2) -> 1/(60+3) + 1/(60+2) = 1/63 + 1/62 = 125/3906
    # 200: rank 2 (L1) -> 1/(60+2) = 1/62
    # 400: rank 3 (L2) -> 1/(60+3) = 1/63
    expected_100 = 2.0 / 61.0
    expected_300 = 1.0 / 63.0 + 1.0 / 62.0
    expected_200 = 1.0 / 62.0
    expected_400 = 1.0 / 63.0

    results = rrf_fuse([list1, list2], k=60, top_n=4)

    assert len(results) == 4
    # Check exact order
    assert [pid for pid, _ in results] == [100, 300, 200, 400]

    # Check exact scores
    assert results[0] == (100, pytest.approx(expected_100, rel=1e-7))
    assert results[1] == (300, pytest.approx(expected_300, rel=1e-7))
    assert results[2] == (200, pytest.approx(expected_200, rel=1e-7))
    assert results[3] == (400, pytest.approx(expected_400, rel=1e-7))


def test_hybrid_ranker():
    from backend.rankers import HybridRanker

    # Mock components with known search results
    class DummySparse:
        def search(self, query, k=100):
            return [(1, 10.0), (2, 8.0), (3, 6.0)]

    class DummyDense:
        def search(self, query, k=100):
            return [(2, 0.9), (1, 0.7), (4, 0.5)]

    class DummyTfidf:
        def search(self, query, k=100):
            return [(3, 0.8), (2, 0.5), (5, 0.3)]

    # 1. 2-way Hybrid (Sparse BM25 + Dense)
    hybrid_2way = HybridRanker(
        sparse=DummySparse(),
        dense=DummyDense(),
        use_tfidf=False,
        rrf_k=60,
    )
    res_2way = hybrid_2way.search("test query", k=3)
    pids_2way = [pid for pid, _ in res_2way]
    # In 2-way:
    # PID 1: rank 1 (sparse) + rank 2 (dense) -> 1/61 + 1/62 = 0.032522
    # PID 2: rank 2 (sparse) + rank 1 (dense) -> 1/62 + 1/61 = 0.032522
    # Both 1 and 2 are top ranked
    assert set(pids_2way[:2]) == {1, 2}

    # 2. 3-way Hybrid ablation (Sparse BM25 + Dense + TF-IDF)
    hybrid_3way = HybridRanker(
        sparse=DummySparse(),
        dense=DummyDense(),
        tfidf=DummyTfidf(),
        use_tfidf=True,
        rrf_k=60,
    )
    res_3way = hybrid_3way.search("test query", k=3)
    pids_3way = [pid for pid, _ in res_3way]
    # With TF-IDF included:
    # PID 2 gets bonus from rank 2 in TF-IDF -> 1/62 + 1/61 + 1/62 = 0.04865
    # PID 1 gets nothing from TF-IDF -> 0.032522
    # PID 2 MUST be #1 unambiguously!
    assert pids_3way[0] == 2



