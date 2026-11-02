"""Build the preprocessing token cache (indexes/corpus_tokens.pkl).

The optimised preprocess_corpus() now automatically caps worker count at 4
on Windows (to avoid RAM crashes from NLTK re-import in each spawned process)
and uses a fast regex tokeniser inside workers instead of NLTK word_tokenize.

Usage:
    python scripts/build_token_cache.py          # auto workers (<=4 on Windows)
    python scripts/build_token_cache.py 1        # single-process (safest)
    python scripts/build_token_cache.py 4        # explicit worker count
"""
import sys
import os

# Ensure repo root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from backend.data_loader import load_lotte
    from backend.preprocess import preprocess_corpus

    # Optional CLI arg: number of workers (default -1 = auto)
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else -1

    print("[build_token_cache] Loading LoTTE lifestyle/dev ...")
    passages, _, _ = load_lotte("lifestyle", "dev")
    print(f"[build_token_cache] Loaded {len(passages):,} passages.")

    print(f"[build_token_cache] Preprocessing (n_jobs={n_jobs}) ...")
    tokens = preprocess_corpus(passages, n_jobs=n_jobs)
    print(f"[build_token_cache] Done. Corpus tokens dict has {len(tokens):,} entries.")
