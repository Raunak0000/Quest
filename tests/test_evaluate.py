"""Unit tests for backend.evaluate: run_ranker, evaluate, success_at_k."""

from __future__ import annotations

import pytest
from backend.evaluate import run_ranker, evaluate, success_at_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOISE_PIDS = list(range(200, 210))


def _make_run_with_relevant_at_rank1(qrels: dict) -> dict:
    """Return a run where the single relevant passage is always ranked first."""
    run = {}
    for qid, rel_pids in qrels.items():
        rel_pid = next(iter(rel_pids))
        scores = {rel_pid: 10.0}
        for i, np_ in enumerate(_NOISE_PIDS):
            scores[np_] = float(9 - i)
        run[qid] = scores
    return run


# ---------------------------------------------------------------------------
# run_ranker
# ---------------------------------------------------------------------------


class _MockRanker:
    """Deterministic mock ranker that returns the same ranked list for any query."""

    def __init__(self, results: list[tuple[int, float]]) -> None:
        self._results = results

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        return self._results[:k]


def test_run_ranker_basic():
    queries = {1: "hello world", 2: "foo bar"}
    ranker = _MockRanker([(101, 5.0), (102, 3.0)])
    run = run_ranker(ranker, queries, k=2)

    assert set(run.keys()) == {1, 2}
    for qid in (1, 2):
        assert run[qid] == {101: 5.0, 102: 3.0}


def test_run_ranker_empty_results():
    """Queries that get no results should map to an empty dict."""
    queries = {99: "obscure query"}
    ranker = _MockRanker([])
    run = run_ranker(ranker, queries, k=10)

    assert run == {99: {}}


def test_run_ranker_k_respected():
    queries = {1: "query"}
    ranker = _MockRanker([(i, float(100 - i)) for i in range(50)])
    run = run_ranker(ranker, queries, k=5)

    assert len(run[1]) == 5


# ---------------------------------------------------------------------------
# evaluate – perfect run
# ---------------------------------------------------------------------------


def test_evaluate_perfect_run():
    """Feeding qrels as the run should give MAP=1.0, R@100=1.0, nDCG@10=1.0."""
    qrels = {1: {101: 1}, 2: {102: 1}, 3: {103: 1}}
    run = _make_run_with_relevant_at_rank1(qrels)
    agg, per_query = evaluate(run, qrels)

    assert abs(agg["MAP"] - 1.0) < 1e-6
    assert abs(agg["R@100"] - 1.0) < 1e-6
    assert abs(agg["nDCG@10"] - 1.0) < 1e-6

    # per_query keys are strings (JSON-compat)
    assert all(isinstance(k, str) for k in per_query)
    # Each query has all four metrics
    for qid_str, metrics in per_query.items():
        assert set(metrics.keys()) == {"P@10", "R@100", "MAP", "nDCG@10"}


def test_evaluate_returns_aggregate_and_per_query():
    qrels = {1: {101: 1}}
    run = {1: {101: 5.0, 202: 3.0}}
    agg, per_query = evaluate(run, qrels)

    assert isinstance(agg, dict)
    assert isinstance(per_query, dict)
    assert set(agg.keys()) == {"P@10", "R@100", "MAP", "nDCG@10"}


def test_evaluate_missed_queries_score_zero():
    """A query in qrels but not in run should not crash; missing qid gets 0 metrics."""
    qrels = {1: {101: 1}, 2: {102: 1}}
    run = {1: {101: 5.0}}  # query 2 is missing from run
    agg, per_query = evaluate(run, qrels)

    # query 1 scores perfectly
    assert abs(per_query.get("1", {}).get("MAP", -1) - 1.0) < 1e-6
    # aggregate MAP is degraded below 1.0 because query 2 got 0
    assert agg["MAP"] < 1.0


def test_evaluate_random_below_perfect():
    """Random ranking should score strictly below a perfect ranking."""
    import random
    random.seed(0)

    qrels = {i: {i * 10: 1} for i in range(1, 6)}  # 5 queries
    perfect_run = _make_run_with_relevant_at_rank1(qrels)

    all_pids = list(range(10, 60))
    random_run = {}
    for qid in qrels:
        pids = random.sample(all_pids, 10)
        random_run[qid] = {pid: random.random() for pid in pids}

    agg_perfect, _ = evaluate(perfect_run, qrels)
    agg_rand, _ = evaluate(random_run, qrels)

    assert agg_rand["MAP"] < agg_perfect["MAP"]
    assert agg_rand["nDCG@10"] <= agg_perfect["nDCG@10"]


# ---------------------------------------------------------------------------
# success_at_k
# ---------------------------------------------------------------------------


def test_success_at_k_perfect():
    """Relevant doc always at rank 1 => success@k = 1.0 for any k >= 1."""
    qrels = {1: {101: 1}, 2: {102: 1}, 3: {103: 1}}
    run = _make_run_with_relevant_at_rank1(qrels)

    assert abs(success_at_k(run, qrels, k=1) - 1.0) < 1e-9
    assert abs(success_at_k(run, qrels, k=5) - 1.0) < 1e-9


def test_success_at_k_no_hits():
    """Run that never retrieves any relevant passage => 0.0."""
    qrels = {1: {999: 1}}
    run = {1: {1: 5.0, 2: 4.0, 3: 3.0}}

    assert success_at_k(run, qrels, k=5) == 0.0


def test_success_at_k_partial():
    """2 out of 4 queries have relevant at top k=3."""
    qrels = {
        1: {101: 1},
        2: {102: 1},
        3: {103: 1},
        4: {104: 1},
    }
    run = {
        1: {101: 10.0, 200: 5.0, 201: 3.0},   # hit at rank 1
        2: {200: 10.0, 201: 5.0, 102: 3.0},   # hit at rank 3
        3: {200: 10.0, 201: 5.0, 202: 3.0},   # no hit in top 3
        4: {200: 10.0, 201: 5.0, 202: 3.0},   # no hit in top 3
    }
    rate = success_at_k(run, qrels, k=3)
    assert abs(rate - 0.5) < 1e-9  # 2/4


def test_success_at_k_empty_run():
    """No common query IDs => 0.0."""
    qrels = {1: {101: 1}}
    run = {99: {200: 5.0}}
    assert success_at_k(run, qrels, k=5) == 0.0


def test_success_at_k_cutoff_matters():
    """Relevant doc at rank 6 should NOT count for k=5 but SHOULD for k=6."""
    qrels = {1: {106: 1}}

    # Build run: pids 1-5 with scores 5.0-1.0, then 106 at score 0.5 (rank 6)
    # Top-5: pids [1,2,3,4,5]; Top-6: pids [1,2,3,4,5,106]
    run = {1: {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.0, 106: 0.5}}

    assert success_at_k(run, qrels, k=5) == 0.0   # 106 not in top-5
    assert success_at_k(run, qrels, k=6) == 1.0   # 106 in top-6
