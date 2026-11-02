"""Evaluation metrics: Precision@10, Recall@100, MAP, nDCG@10, Success@5.

Public API
----------
run_ranker(ranker, queries, k=100)
    -> run : dict[int, dict[int, float]]   {qid: {pid: score}}

evaluate(run, qrels)
    -> aggregate : dict[str, float]        mean metric values over all queries
    -> per_query : dict[str, dict[str, float]]  per-query metric values

success_at_k(run, qrels, k=5)
    -> float  fraction of queries with at least one relevant passage in top k
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import ir_measures
from ir_measures import AP, P, R, ScoredDoc, nDCG
from ir_measures import Qrel as IrQrel


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A run is: {qid: {pid: score}}
Run = dict[int, dict[int, float]]

# Qrels are: {qid: {pid: relevance_int}}  (as returned by data_loader)
Qrels = dict[int, dict[int, int]]


@runtime_checkable
class Ranker(Protocol):
    """Structural protocol for any ranker with a .search() method."""

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        ...


# ---------------------------------------------------------------------------
# Measures used throughout
# ---------------------------------------------------------------------------

_MEASURES = [P @ 10, R @ 100, AP, nDCG @ 10]

# Human-readable names for aggregate output keys
_MEASURE_NAMES: dict[Any, str] = {
    P @ 10: "P@10",
    R @ 100: "R@100",
    AP: "MAP",
    nDCG @ 10: "nDCG@10",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def run_ranker(
    ranker: Ranker,
    queries: dict[int, str],
    k: int = 100,
) -> Run:
    """Run *ranker* over all queries and return a scored run dict.

    Parameters
    ----------
    ranker :
        Any object with a ``search(query: str, k: int) -> list[tuple[int, float]]``
        method (e.g. TfidfRanker, Bm25Ranker, DenseRanker, HybridRanker).
    queries :
        Mapping from query-id (int) to query text, as returned by
        ``data_loader.load_lotte``.
    k :
        Maximum number of passages to retrieve per query.

    Returns
    -------
    run : dict[int, dict[int, float]]
        ``{qid: {pid: score}}`` -- outer key is query-id, inner key is
        passage-id, value is the ranker's relevance score.
    """
    run: Run = {}
    for qid, query_text in queries.items():
        results = ranker.search(query_text, k=k)
        if results:
            run[qid] = {pid: score for pid, score in results}
        else:
            run[qid] = {}
    return run


def evaluate(
    run: Run,
    qrels: Qrels,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute P@10, R@100, MAP (AP), and nDCG@10 using *ir_measures*.

    Parameters
    ----------
    run :
        Scored run as returned by :func:`run_ranker`:
        ``{qid: {pid: score}}``.
    qrels :
        Relevance judgements as returned by ``data_loader.load_lotte``:
        ``{qid: {pid: 1}}``.

    Returns
    -------
    aggregate : dict[str, float]
        Mean metric values over all queries:
        ``{"P@10": ..., "R@100": ..., "MAP": ..., "nDCG@10": ...}``.
    per_query : dict[str, dict[str, float]]
        Per-query metric values:
        ``{qid_str: {"P@10": ..., "R@100": ..., "MAP": ..., "nDCG@10": ...}}``.
        Query IDs are returned as strings for JSON-compatibility.
    """
    ir_qrels = _to_ir_qrels(qrels)
    ir_run = _to_ir_run(run)

    # Accumulate per-query scores
    per_query_raw: dict[str, dict[str, float]] = {}
    for metric in ir_measures.iter_calc(_MEASURES, ir_qrels, ir_run):
        qid_str = str(metric.query_id)
        name = _MEASURE_NAMES.get(metric.measure, str(metric.measure))
        per_query_raw.setdefault(qid_str, {})[name] = float(metric.value)

    # Compute aggregates (mean over queries)
    aggregate: dict[str, float] = {}
    for measure, name in _MEASURE_NAMES.items():
        values = [
            pq[name]
            for pq in per_query_raw.values()
            if name in pq
        ]
        aggregate[name] = float(sum(values) / len(values)) if values else 0.0

    return aggregate, per_query_raw


def success_at_k(
    run: Run,
    qrels: Qrels,
    k: int = 5,
) -> float:
    """Fraction of queries that have at least one relevant passage in the top k.

    Parameters
    ----------
    run :
        Scored run as returned by :func:`run_ranker`:
        ``{qid: {pid: score}}``.
    qrels :
        Relevance judgements: ``{qid: {pid: 1}}``.
    k :
        Cutoff -- only the top-k passages (by score) for each query are checked.

    Returns
    -------
    float
        Success@k rate in [0.0, 1.0].
        Returns 0.0 if there are no queries in common between run and qrels.
    """
    query_ids = set(run.keys()) & set(qrels.keys())
    if not query_ids:
        return 0.0

    hits = 0
    for qid in query_ids:
        pid_scores = run[qid]
        if not pid_scores:
            continue
        # Top-k pids by descending score
        top_pids = sorted(pid_scores, key=lambda p: -pid_scores[p])[:k]
        relevant = qrels[qid]
        if any(pid in relevant for pid in top_pids):
            hits += 1

    return hits / len(query_ids)


# ---------------------------------------------------------------------------
# ir_measures conversion helpers
# ---------------------------------------------------------------------------


def _to_ir_qrels(qrels: Qrels) -> list[IrQrel]:
    """Convert project-native qrels dict to a list of ir_measures.Qrel."""
    ir_qrels: list[IrQrel] = []
    for qid, pid_rel in qrels.items():
        qid_str = str(qid)
        for pid, relevance in pid_rel.items():
            ir_qrels.append(IrQrel(qid_str, str(pid), int(relevance)))
    return ir_qrels


def _to_ir_run(run: Run) -> list[ScoredDoc]:
    """Convert project-native run dict to a list of ir_measures.ScoredDoc."""
    ir_run: list[ScoredDoc] = []
    for qid, pid_scores in run.items():
        qid_str = str(qid)
        for pid, score in pid_scores.items():
            ir_run.append(ScoredDoc(qid_str, str(pid), float(score)))
    return ir_run
