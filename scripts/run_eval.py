"""Full evaluation run: TF-IDF, BM25, Dense, Hybrid RRF, and 3-way Hybrid.

Outputs
-------
results/results.json          – all metrics for every ranker
results/comparison_table.md   – markdown table, best value per column bold
results/<ranker>_per_query.csv – per-query metric CSV (one per ranker)
results/figures/pr_curves.png – Precision-Recall curves at 11 standard recall levels
results/config.json           – dataset / model / hyperparameter log

Usage
-----
    python scripts/run_eval.py [--domain DOMAIN] [--split SPLIT]
                               [--k K] [--rrf-k RRF_K]
                               [--model-name MODEL_NAME]
                               [--seed SEED] [--output-dir OUTPUT_DIR]
                               [--skip-dense]

Prerequisites
-------------
* indexes/corpus_tokens.pkl   (run scripts/build_token_cache.py first)
* indexes/embeddings.npy + indexes/pids.json  (optional; dense & hybrid)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if sys.platform == "win32":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from backend.data_loader import load_lotte
from backend.evaluate import evaluate, run_ranker, success_at_k
from backend.rankers import Bm25Ranker, DenseRanker, HybridRanker, TfidfRanker

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate all rankers on a LoTTE domain/split."
    )
    p.add_argument("--domain", default="lifestyle", help="LoTTE domain (default: lifestyle)")
    p.add_argument("--split", default="dev", help="Dataset split: dev or test (default: dev)")
    p.add_argument("--query-type", default="search", help="Query type: search or forum (default: search)")
    p.add_argument("--k", type=int, default=100, help="Retrieval depth k (default: 100)")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant (default: 60)")
    p.add_argument("--model-name", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    p.add_argument(
        "--output-dir",
        default=os.path.join(_REPO_ROOT, "results"),
        help="Output directory (default: results/)",
    )
    p.add_argument(
        "--indexes-dir",
        default=os.path.join(_REPO_ROOT, "indexes"),
        help="Directory containing corpus_tokens.pkl and optional embeddings (default: indexes/)",
    )
    p.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip Dense and Hybrid rankers even if embeddings are available.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Precision-Recall helpers
# ---------------------------------------------------------------------------

_RECALL_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _interpolated_pr_curve(
    run: dict[int, dict[int, float]],
    qrels: dict[int, dict[int, int]],
    recall_levels: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Compute macro-averaged interpolated Precision-Recall curve.

    For each query, builds the full P-R curve sorted by score descending,
    then applies standard 11-point interpolation (max precision at or above
    each recall level).  Returns mean precision at each recall level.
    """
    if recall_levels is None:
        recall_levels = _RECALL_LEVELS

    per_query_prec: list[list[float]] = []

    query_ids = set(run.keys()) & set(qrels.keys())
    for qid in query_ids:
        rel_pids = set(qrels[qid].keys())
        if not rel_pids:
            continue

        pid_scores = run[qid]
        ranked = sorted(pid_scores.items(), key=lambda x: -x[1])

        # Build raw P-R curve
        n_rel = len(rel_pids)
        n_ret_rel = 0
        curve: list[tuple[float, float]] = []  # (recall, precision)
        for rank, (pid, _) in enumerate(ranked, start=1):
            if pid in rel_pids:
                n_ret_rel += 1
            prec = n_ret_rel / rank
            rec = n_ret_rel / n_rel
            curve.append((rec, prec))

        if not curve:
            per_query_prec.append([0.0] * len(recall_levels))
            continue

        # 11-point interpolation: max precision for recall >= r
        interp_prec: list[float] = []
        for r in recall_levels:
            max_p = max((p for rec, p in curve if rec >= r), default=0.0)
            interp_prec.append(max_p)
        per_query_prec.append(interp_prec)

    if not per_query_prec:
        return recall_levels, [0.0] * len(recall_levels)

    arr = np.array(per_query_prec)  # (n_queries, n_levels)
    mean_prec = arr.mean(axis=0).tolist()
    return recall_levels, mean_prec


# ---------------------------------------------------------------------------
# Markdown table helpers
# ---------------------------------------------------------------------------

_METRIC_COLS = ["P@10", "R@100", "MAP", "nDCG@10", "Success@5"]


def _build_markdown_table(
    results: dict[str, dict[str, float]],
) -> str:
    """Build a Markdown comparison table with best value per column bolded."""
    ranker_names = list(results.keys())
    cols = _METRIC_COLS

    # Best value per metric (highest = best)
    best: dict[str, float] = {}
    for col in cols:
        values = [results[r].get(col, float("-inf")) for r in ranker_names]
        best[col] = max(v for v in values if v != float("-inf"))

    # Header
    header = "| Ranker | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(cols)) + " |"

    rows = [header, sep]
    for name in ranker_names:
        cells = []
        for col in cols:
            val = results[name].get(col, float("nan"))
            if not isinstance(val, float) or val != val:  # NaN check
                formatted = "–"
            else:
                formatted = f"{val:.4f}"
                if abs(val - best.get(col, float("nan"))) < 1e-9:
                    formatted = f"**{formatted}**"
            cells.append(formatted)
        rows.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    print("=" * 72)
    print("Quest — Full Evaluation Run")
    print(f"  Domain: {args.domain} | Split: {args.split} | Query type: {args.query_type}")
    print(f"  k={args.k} | RRF-k={args.rrf_k} | model={args.model_name} | seed={args.seed}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("\n[1/N] Loading dataset...")
    t0 = time.time()
    passages, queries, qrels = load_lotte(args.domain, args.split, args.query_type)
    print(
        f"      {len(passages):,} passages | {len(queries):,} queries | "
        f"{len(qrels):,} qrel entries  ({time.time()-t0:.2f}s)"
    )

    # ------------------------------------------------------------------
    # 2. Load token cache
    # ------------------------------------------------------------------
    token_cache_path = os.path.join(args.indexes_dir, "corpus_tokens.pkl")
    if not os.path.exists(token_cache_path):
        print(
            f"\nERROR: Token cache not found at {token_cache_path}\n"
            "Run:  python scripts/build_token_cache.py\n"
        )
        sys.exit(1)

    print("\n[2/N] Loading token cache...")
    t1 = time.time()
    with open(token_cache_path, "rb") as fh:
        corpus_tokens: dict[int, list[str]] = pickle.load(fh)
    print(f"      {len(corpus_tokens):,} tokenised passages loaded ({time.time()-t1:.2f}s)")

    # ------------------------------------------------------------------
    # 3. Check for dense embeddings
    # ------------------------------------------------------------------
    emb_path = os.path.join(args.indexes_dir, "embeddings.npy")
    pids_path = os.path.join(args.indexes_dir, "pids.json")
    dense_available = (
        not args.skip_dense
        and os.path.exists(emb_path)
        and os.path.exists(pids_path)
    )
    if not dense_available:
        if args.skip_dense:
            print("\n[3/N] Skipping dense/hybrid rankers (--skip-dense flag).")
        else:
            print(
                f"\n[3/N] Dense embeddings not found at {emb_path}.\n"
                "      Dense and Hybrid rankers will be skipped.\n"
                "      Run:  python scripts/embed.py --domain {args.domain} --split {args.split}"
            )
    else:
        print(f"\n[3/N] Dense embeddings found at {emb_path}.")

    # ------------------------------------------------------------------
    # 4. Build rankers
    # ------------------------------------------------------------------
    rankers_to_eval: list[tuple[str, Any]] = []

    print("\n[4/N] Building sparse rankers...")

    # TF-IDF
    t2 = time.time()
    tfidf = TfidfRanker().fit(corpus_tokens)
    print(f"      TF-IDF fitted ({time.time()-t2:.2f}s,  matrix={tfidf.tfidf_matrix.shape})")
    rankers_to_eval.append(("TF-IDF", tfidf))

    gc.collect()

    # BM25
    t3 = time.time()
    bm25 = Bm25Ranker(k1=1.5, b=0.75).fit(corpus_tokens)
    print(f"      BM25   fitted ({time.time()-t3:.2f}s)")
    rankers_to_eval.append(("BM25", bm25))

    gc.collect()

    dense: DenseRanker | None = None
    if dense_available:
        print("\n[4b/N] Building dense ranker from pre-computed embeddings...")
        t4 = time.time()
        dense = DenseRanker.from_files(emb_path, pids_path, model_name=args.model_name)
        print(f"       DenseRanker loaded ({time.time()-t4:.2f}s, {dense.index.ntotal:,} vectors)")
        rankers_to_eval.append(("Dense", dense))

        # 2-way Hybrid (BM25 + Dense)
        hybrid = HybridRanker(sparse=bm25, dense=dense, rrf_k=args.rrf_k)
        rankers_to_eval.append(("Hybrid-RRF", hybrid))

        # 3-way ablation (BM25 + Dense + TF-IDF)
        hybrid3 = HybridRanker(sparse=bm25, dense=dense, tfidf=tfidf, use_tfidf=True, rrf_k=args.rrf_k)
        rankers_to_eval.append(("Hybrid-3way", hybrid3))

    # ------------------------------------------------------------------
    # 5. Evaluate each ranker
    # ------------------------------------------------------------------
    print(f"\n[5/N] Evaluating {len(rankers_to_eval)} ranker(s) (k={args.k})...")

    all_results: dict[str, dict[str, float]] = {}
    all_per_query: dict[str, dict[str, dict[str, float]]] = {}
    pr_curves: dict[str, tuple[list[float], list[float]]] = {}

    for ranker_name, ranker in rankers_to_eval:
        print(f"\n  → {ranker_name}")
        t_start = time.time()

        run = run_ranker(ranker, queries, k=args.k)
        agg, per_query = evaluate(run, qrels)
        sak = success_at_k(run, qrels, k=5)
        agg["Success@5"] = sak

        elapsed = time.time() - t_start
        print(
            f"     P@10={agg['P@10']:.4f}  R@100={agg['R@100']:.4f}  "
            f"MAP={agg['MAP']:.4f}  nDCG@10={agg['nDCG@10']:.4f}  "
            f"Success@5={sak:.4f}  ({elapsed:.1f}s)"
        )

        all_results[ranker_name] = agg
        all_per_query[ranker_name] = per_query

        # PR curve
        recalls, precisions = _interpolated_pr_curve(run, qrels)
        pr_curves[ranker_name] = (recalls, precisions)

        gc.collect()

    # ------------------------------------------------------------------
    # 6. Save results/results.json
    # ------------------------------------------------------------------
    results_json_path = os.path.join(args.output_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n[6/N] Saved aggregate results → {results_json_path}")

    # ------------------------------------------------------------------
    # 7. Save per-query CSV files
    # ------------------------------------------------------------------
    print("[7/N] Saving per-query CSV files...")
    for ranker_name, per_query in all_per_query.items():
        safe_name = ranker_name.lower().replace("-", "_").replace(" ", "_")
        csv_path = os.path.join(args.output_dir, f"{safe_name}_per_query.csv")
        fieldnames = ["qid"] + _METRIC_COLS
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for qid_str, metrics in per_query.items():
                row: dict[str, Any] = {"qid": qid_str}
                for col in _METRIC_COLS:
                    row[col] = metrics.get(col, "")
                writer.writerow(row)
        print(f"     {csv_path}")

    # ------------------------------------------------------------------
    # 8. Save comparison_table.md
    # ------------------------------------------------------------------
    table_md = _build_markdown_table(all_results)
    table_path = os.path.join(args.output_dir, "comparison_table.md")
    with open(table_path, "w", encoding="utf-8") as fh:
        fh.write("# Retrieval Evaluation Results\n\n")
        fh.write(f"**Domain:** {args.domain} | **Split:** {args.split} | "
                 f"**Query type:** {args.query_type} | **k:** {args.k}\n\n")
        fh.write(table_md)
    print(f"[8/N] Saved comparison table → {table_path}")
    print()
    print(table_md)

    # ------------------------------------------------------------------
    # 9. Plot Precision-Recall curves
    # ------------------------------------------------------------------
    print("[9/N] Plotting PR curves...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

        for i, (ranker_name, (recalls, precisions)) in enumerate(pr_curves.items()):
            ax.plot(
                recalls,
                precisions,
                label=ranker_name,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2,
                marker="o",
                markersize=5,
            )

        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title(
            f"Precision-Recall Curves\n"
            f"LoTTE {args.domain}/{args.split} ({args.query_type} queries, k={args.k})",
            fontsize=13,
        )
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xticks(_RECALL_LEVELS)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        pr_path = os.path.join(args.output_dir, "figures", "pr_curves.png")
        fig.savefig(pr_path, dpi=300)
        plt.close(fig)
        print(f"     Saved → {pr_path}")
    except Exception as exc:  # pragma: no cover
        print(f"     WARNING: Could not save PR plot: {exc}")

    # ------------------------------------------------------------------
    # 9b. Save PR curve data as JSON (consumed by GET /pr-curves)
    # ------------------------------------------------------------------
    pr_curves_json: dict[str, dict[str, list[float]]] = {}
    for ranker_name, (recalls, precisions) in pr_curves.items():
        pr_curves_json[ranker_name] = {
            "recall": recalls,
            "precision": precisions,
        }
    pr_curves_json_path = os.path.join(args.output_dir, "pr_curves.json")
    with open(pr_curves_json_path, "w", encoding="utf-8") as fh:
        json.dump(pr_curves_json, fh, indent=2)
    print(f"[9b/N] Saved PR curve data  → {pr_curves_json_path}")

        # ------------------------------------------------------------------
    # 10. Save config.json
    # ------------------------------------------------------------------
    config = {
        "domain": args.domain,
        "split": args.split,
        "query_type": args.query_type,
        "k": args.k,
        "rrf_k": args.rrf_k,
        "model_name": args.model_name,
        "seed": args.seed,
        "dense_available": dense_available,
        "rankers_evaluated": [name for name, _ in rankers_to_eval],
        "token_cache_path": token_cache_path,
        "embeddings_path": emb_path if dense_available else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"[10/N] Saved run config → {config_path}")

    print("\n" + "=" * 72)
    print("Evaluation complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
