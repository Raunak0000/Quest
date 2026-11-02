"""Reciprocal Rank Fusion (RRF).

Combines ranked lists from multiple retrieval systems into a single fused ranking.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence, Any


def rrf_fuse(
    rank_lists: Sequence[Sequence[Any]],
    k: int = 60,
    top_n: int = 100,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Formula
    -------
    score(d) = sum_{m in M} 1 / (k + rank_m(d))
    where rank_m(d) is 1-based rank (1, 2, 3, ...).

    Parameters
    ----------
    rank_lists : Sequence[Sequence[int | tuple[int, Any]]]
        A sequence of ranked lists. Each item can be a list of passage IDs (pids)
        or a list of (pid, score) tuples. Order implies rank (first element is rank 1).
    k : int, default=60
        Smoothing constant added to the rank to penalize lower-ranked results.
    top_n : int, default=100
        Maximum number of fused results to return.

    Returns
    -------
    list of (pid, score)
        Merged passage IDs and their fused scores, sorted descending by score.
    """
    scores: dict[int, float] = defaultdict(float)

    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list, start=1):
            pid = item[0] if isinstance(item, (tuple, list)) else item
            scores[int(pid)] += 1.0 / (k + rank)

    # Sort descending by score, tie-break by pid for deterministic order
    sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [(pid, score) for pid, score in sorted_items[:top_n]]
