"""Unit tests for backend.rrf.rrf_fuse.

Covers
------
* Hand-computed exact scores for a 2-list case
* Empty inputs / single-list edge cases
* Tie-breaking determinism
* top_n truncation
* Rank-1 item always wins (RRF property)
* Tuple (pid, score) items as well as plain pid items
"""

from __future__ import annotations

import pytest
from backend.rrf import rrf_fuse


# ---------------------------------------------------------------------------
# Hand-computed exact scores
# ---------------------------------------------------------------------------

def test_rrf_two_lists_exact():
    """Exact arithmetic check from the spec (k=60).

    List 1: [100, 200, 300]
    List 2: [100, 300, 400]

    Scores:
      100 -> 1/(60+1) + 1/(60+1)  = 2/61
      300 -> 1/(60+3) + 1/(60+2)  = 1/63 + 1/62
      200 -> 1/(60+2)             = 1/62
      400 -> 1/(60+3)             = 1/63
    """
    list1 = [100, 200, 300]
    list2 = [100, 300, 400]

    expected_100 = 2.0 / 61.0
    expected_300 = 1.0 / 63.0 + 1.0 / 62.0
    expected_200 = 1.0 / 62.0
    expected_400 = 1.0 / 63.0

    results = rrf_fuse([list1, list2], k=60, top_n=4)

    assert len(results) == 4
    pids = [pid for pid, _ in results]
    assert pids == [100, 300, 200, 400]

    scores = {pid: score for pid, score in results}
    assert scores[100] == pytest.approx(expected_100, rel=1e-7)
    assert scores[300] == pytest.approx(expected_300, rel=1e-7)
    assert scores[200] == pytest.approx(expected_200, rel=1e-7)
    assert scores[400] == pytest.approx(expected_400, rel=1e-7)


def test_rrf_single_list():
    """With one ranked list, score(d) = 1/(k+rank)."""
    ranked = [10, 20, 30]
    results = rrf_fuse([ranked], k=60, top_n=3)

    pids = [pid for pid, _ in results]
    assert pids == [10, 20, 30]

    scores = {pid: s for pid, s in results}
    assert scores[10] == pytest.approx(1.0 / 61.0, rel=1e-7)
    assert scores[20] == pytest.approx(1.0 / 62.0, rel=1e-7)
    assert scores[30] == pytest.approx(1.0 / 63.0, rel=1e-7)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_rrf_empty_input():
    """No lists -> empty result."""
    assert rrf_fuse([], k=60, top_n=10) == []


def test_rrf_empty_lists_inside():
    """Lists that contain no items -> empty result."""
    assert rrf_fuse([[], []], k=60, top_n=10) == []


def test_rrf_single_item_lists():
    pid_a, pid_b = 7, 8
    results = rrf_fuse([[pid_a], [pid_b]], k=60, top_n=2)
    pids = [pid for pid, _ in results]
    # Both should appear; tie in score -> deterministic by pid asc
    assert set(pids) == {7, 8}
    # Both get score 1/(60+1); tie broken by ascending pid
    assert pids == [7, 8]


# ---------------------------------------------------------------------------
# top_n truncation
# ---------------------------------------------------------------------------

def test_rrf_top_n_truncation():
    """Only top_n results should be returned."""
    ranked = list(range(100))
    results = rrf_fuse([ranked], k=60, top_n=5)
    assert len(results) == 5
    # First 5 should be the highest-scored (lowest ranks)
    pids = [pid for pid, _ in results]
    assert pids == [0, 1, 2, 3, 4]


def test_rrf_top_n_larger_than_pool():
    """If top_n > number of unique docs, return all docs."""
    results = rrf_fuse([[1, 2, 3]], k=60, top_n=100)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Score monotonicity: higher rank -> higher score
# ---------------------------------------------------------------------------

def test_rrf_score_monotone_single_list():
    """Rank 1 always beats rank 2 etc. in a single list."""
    results = rrf_fuse([[1, 2, 3, 4, 5]], k=60, top_n=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_rrf_rank1_item_always_wins():
    """The item ranked #1 in all lists must have the highest fused score."""
    list1 = [99, 1, 2, 3]
    list2 = [99, 4, 5, 6]
    results = rrf_fuse([list1, list2], k=60, top_n=7)
    assert results[0][0] == 99


# ---------------------------------------------------------------------------
# Three-list fusion
# ---------------------------------------------------------------------------

def test_rrf_three_lists():
    """Item appearing in all 3 lists at rank 1 beats all."""
    list1 = [42, 1, 2]
    list2 = [42, 3, 4]
    list3 = [42, 5, 6]
    results = rrf_fuse([list1, list2, list3], k=60, top_n=7)

    # 42 is at rank 1 in all three lists
    assert results[0][0] == 42
    expected_42 = 3.0 / 61.0
    assert results[0][1] == pytest.approx(expected_42, rel=1e-7)


# ---------------------------------------------------------------------------
# Tuple-item input  (pid, score) pairs, not plain ints
# ---------------------------------------------------------------------------

def test_rrf_accepts_tuple_items():
    """rrf_fuse should accept (pid, score) tuples as list items."""
    list1 = [(10, 5.0), (20, 4.0), (30, 3.0)]
    list2 = [(10, 0.9), (30, 0.7), (40, 0.5)]
    results = rrf_fuse([list1, list2], k=60, top_n=4)

    # pid=10 is rank-1 in both lists -> highest score
    pids = [pid for pid, _ in results]
    assert pids[0] == 10


# ---------------------------------------------------------------------------
# k parameter sensitivity
# ---------------------------------------------------------------------------

def test_rrf_larger_k_compresses_scores():
    """Doubling k should reduce all individual scores."""
    ranked = [1, 2, 3]
    r60 = rrf_fuse([ranked], k=60, top_n=3)
    r120 = rrf_fuse([ranked], k=120, top_n=3)

    for (pid60, s60), (pid120, s120) in zip(r60, r120):
        assert pid60 == pid120
        assert s120 < s60


# ---------------------------------------------------------------------------
# Tie-breaking is deterministic (ascending pid)
# ---------------------------------------------------------------------------

def test_rrf_tie_break_by_pid_ascending():
    """When two docs have the same fused score, lower pid comes first."""
    # Both pids appear at rank 1 in separate lists -> identical scores
    results = rrf_fuse([[5], [3]], k=60, top_n=2)
    pids = [pid for pid, _ in results]
    assert pids == [3, 5]  # ascending pid tie-break
