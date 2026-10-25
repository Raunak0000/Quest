"""Dataset statistics and figures for report Section 2.

Computes:
  - Number of passages and queries
  - Average and median passage length in words
  - Average number of relevant passages per query
  - Figure 1: Histogram of passage lengths (words)
  - Figure 2: Histogram of number of relevant passages per query

Also prints one worked example:
  - A query, its relevant passage(s), and one irrelevant passage

Output:
  results/figures/fig1_passage_lengths.png
  results/figures/fig2_relevant_per_query.png

Usage:
    python3 scripts/dataset_stats.py [domain] [split]

Defaults: lifestyle / dev
"""

from __future__ import annotations

import os
import sys
import random

import matplotlib; matplotlib.use("Agg")  # must precede pyplot
import matplotlib.pyplot as plt
import numpy as np

# ── project imports ────────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from backend.data_loader import load_lotte

# ── output directory ───────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(_REPO, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
ACCENT   = "#4C72B0"
ACCENT2  = "#DD8452"
DPI      = 300
FIGSIZE  = (8, 4.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(domain: str = "lifestyle", split: str = "dev") -> None:
    print(f"\n{'='*60}")
    print(f"  LoTTE Dataset Statistics — {domain} / {split}")
    print(f"{'='*60}\n")

    passages, queries, qrels = load_lotte(domain, split, query_type="search")

    # ------------------------------------------------------------------ #
    # 1. Basic counts
    # ------------------------------------------------------------------ #
    n_passages = len(passages)
    n_queries  = len(queries)
    print(f"  Passages : {n_passages:>10,}")
    print(f"  Queries  : {n_queries:>10,}")

    # ------------------------------------------------------------------ #
    # 2. Passage lengths (in words)
    # ------------------------------------------------------------------ #
    lengths = [word_count(t) for t in passages.values()]
    avg_len    = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    print(f"\n  Passage length (words)")
    print(f"    Mean   : {avg_len:>8.1f}")
    print(f"    Median : {median_len:>8.1f}")
    print(f"    Min    : {min(lengths):>8,}")
    print(f"    Max    : {max(lengths):>8,}")

    # ------------------------------------------------------------------ #
    # 3. Relevant passages per query
    # ------------------------------------------------------------------ #
    rel_counts = [len(v) for v in qrels.values()]
    avg_rel    = float(np.mean(rel_counts))
    median_rel = float(np.median(rel_counts))
    print(f"\n  Relevant passages per query")
    print(f"    Mean   : {avg_rel:>8.2f}")
    print(f"    Median : {median_rel:>8.1f}")
    print(f"    Min    : {min(rel_counts):>8,}")
    print(f"    Max    : {max(rel_counts):>8,}")

    # ================================================================== #
    # Figure 1 — Histogram of passage lengths
    # ================================================================== #
    cap = 500  # clip extreme tails for readability
    clipped = [min(wc, cap) for wc in lengths]

    fig1, ax1 = plt.subplots(figsize=FIGSIZE)
    ax1.hist(clipped, bins=60, color=ACCENT, edgecolor="white", linewidth=0.4)
    ax1.axvline(avg_len,    color="tomato",     linestyle="--", linewidth=1.5,
                label=f"Mean {avg_len:.0f} w")
    ax1.axvline(median_len, color="darkorange",  linestyle=":",  linewidth=1.5,
                label=f"Median {median_len:.0f} w")
    ax1.set_xlabel("Passage length (words)", fontsize=12)
    ax1.set_ylabel("Number of passages",     fontsize=12)
    ax1.set_title(f"Passage length distribution — {domain} / {split}", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    if max(lengths) > cap:
        ax1.set_xlim(0, cap)
        ax1.annotate(f"x-axis capped at {cap} words for readability",
                     xy=(0.99, 0.96), xycoords="axes fraction",
                     ha="right", fontsize=8, color="grey")
    fig1.tight_layout()
    out1 = os.path.join(FIGURES_DIR, "fig1_passage_lengths.png")
    fig1.savefig(out1, dpi=DPI)
    plt.close(fig1)
    print(f"\n  [OK] Saved: {out1}")

    # ================================================================== #
    # Figure 2 — Histogram of relevant passages per query
    # ================================================================== #
    max_rel = max(rel_counts)
    bins2   = range(0, max_rel + 2)

    fig2, ax2 = plt.subplots(figsize=FIGSIZE)
    ax2.hist(rel_counts, bins=bins2, color=ACCENT2, edgecolor="white", linewidth=0.4,
             align="left")
    ax2.axvline(avg_rel,    color="steelblue",  linestyle="--", linewidth=1.5,
                label=f"Mean {avg_rel:.1f}")
    ax2.axvline(median_rel, color="navy",        linestyle=":",  linewidth=1.5,
                label=f"Median {median_rel:.0f}")
    ax2.set_xlabel("Number of relevant passages per query", fontsize=12)
    ax2.set_ylabel("Number of queries",                     fontsize=12)
    ax2.set_title(f"Relevant passages per query — {domain} / {split}", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(range(0, max_rel + 1))
    fig2.tight_layout()
    out2 = os.path.join(FIGURES_DIR, "fig2_relevant_per_query.png")
    fig2.savefig(out2, dpi=DPI)
    plt.close(fig2)
    print(f"  [OK] Saved: {out2}")

    # ================================================================== #
    # Worked example
    # ================================================================== #
    print(f"\n{'='*60}")
    print("  Worked example")
    print(f"{'='*60}")

    # Pick a query that has at least 1 relevant passage
    random.seed(42)
    example_qid = random.choice([qid for qid in qrels if len(qrels[qid]) >= 1])
    example_query = queries[example_qid]
    relevant_pids = list(qrels[example_qid].keys())

    print(f"\n  Query (qid={example_qid}):")
    print(f"    {example_query}")

    print(f"\n  Relevant passage(s) (pids: {relevant_pids[:3]}):")
    for pid in relevant_pids[:2]:
        text = passages[pid]
        preview = text[:300] + ("…" if len(text) > 300 else "")
        print(f"\n    [pid={pid}]")
        print(f"    {preview}")

    # One irrelevant passage — pick a random pid not in qrels[example_qid]
    all_pids = list(passages.keys())
    irrel_pid = None
    for candidate in random.sample(all_pids, 20):
        if candidate not in qrels[example_qid]:
            irrel_pid = candidate
            break

    if irrel_pid is not None:
        text = passages[irrel_pid]
        preview = text[:300] + ("…" if len(text) > 300 else "")
        print(f"\n  Irrelevant passage (pid={irrel_pid}):")
        print(f"    {preview}")

    print()


if __name__ == "__main__":
    domain = sys.argv[1] if len(sys.argv) > 1 else "lifestyle"
    split  = sys.argv[2] if len(sys.argv) > 2 else "dev"
    run(domain, split)
