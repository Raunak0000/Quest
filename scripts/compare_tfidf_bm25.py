"""Compare TF-IDF vs BM25 top-5 retrieval on LoTTE lifestyle queries."""

from __future__ import annotations

import os
import sys
import time
import pickle

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if sys.platform == "win32":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from backend.data_loader import load_lotte
from backend.rankers import TfidfRanker, Bm25Ranker


def main() -> None:
    cache_path = os.path.join(_REPO_ROOT, "indexes", "corpus_tokens.pkl")
    if not os.path.exists(cache_path):
        print(f"Error: Token cache not found at {cache_path}. Run build_token_cache.py first.")
        sys.exit(1)

    print("=" * 80)
    print("TF-IDF vs BM25 Retrieval Comparison")
    print("=" * 80)

    # 1. Load token cache
    t0 = time.time()
    with open(cache_path, "rb") as fh:
        corpus_tokens = pickle.load(fh)
    pids = list(corpus_tokens.keys())
    print(f"[1/4] Loaded token cache ({len(corpus_tokens):,} passages) in {time.time()-t0:.2f}s")

    # 2. Fit TF-IDF Ranker
    print("[2/4] Fitting TfidfRanker...")
    t1 = time.time()
    tfidf = TfidfRanker().fit(corpus_tokens, pids=pids)
    print(f"      TfidfRanker fitted in {time.time()-t1:.2f}s (matrix shape: {tfidf.tfidf_matrix.shape})")

    import gc
    gc.collect()

    # 3. Fit BM25 Ranker
    print("[3/4] Fitting Bm25Ranker...")
    t2 = time.time()
    bm25 = Bm25Ranker(k1=1.5, b=0.75).fit(corpus_tokens, pids=pids)
    print(f"      Bm25Ranker fitted in {time.time()-t2:.2f}s")

    # 4. Load raw passages for snippet display
    passages, _, _ = load_lotte("lifestyle", "dev")

    # 3 representative queries
    queries = [
        "healthy breakfast foods",
        "how to brew pour over coffee",
        "dog leash training tips",
    ]

    print("\n[4/4] Comparing top-5 results across 3 queries:")

    for query in queries:
        print("\n" + "=" * 80)
        print(f"Query: \"{query}\"")
        print("=" * 80)

        t_search0 = time.time()
        tfidf_top5 = tfidf.search(query, k=5)
        t_tfidf = time.time() - t_search0

        t_search1 = time.time()
        bm25_top5 = bm25.search(query, k=5)
        t_bm25 = time.time() - t_search1

        tfidf_pids = [pid for pid, _ in tfidf_top5]
        bm25_pids = [pid for pid, _ in bm25_top5]
        overlap = set(tfidf_pids).intersection(set(bm25_pids))

        print(f"\n--- TF-IDF (time: {t_tfidf*1000:.1f}ms) ---")
        for rank, (pid, score) in enumerate(tfidf_top5, 1):
            text = passages.get(pid, "").replace("\n", " ").strip()[:90]
            print(f"  {rank}. PID: {pid:7d} | Score: {score:.4f} | {text}...")

        print(f"\n--- BM25 (time: {t_bm25*1000:.1f}ms) ---")
        for rank, (pid, score) in enumerate(bm25_top5, 1):
            text = passages.get(pid, "").replace("\n", " ").strip()[:90]
            print(f"  {rank}. PID: {pid:7d} | Score: {score:.4f} | {text}...")

        print(f"\nOverlap in top-5: {len(overlap)}/5 passages shared {sorted(overlap)}")
        if 0 < len(overlap) < 5 or tfidf_pids != bm25_pids:
            print("Status: Overlapping but not identical [OK]")
        else:
            print("Status: Identical or disjoint")

    print("\n" + "=" * 80)
    print("Comparison complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
