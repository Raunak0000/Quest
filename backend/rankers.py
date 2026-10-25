"""Rankers for information retrieval.

Classes
-------
TfidfRanker
    Classical vector-space baseline using scikit-learn TfidfVectorizer
    on pre-tokenized passages and cosine similarity via sparse dot product.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence, Any

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Ensure OpenBLAS doesn't crash on Windows due to thread pool memory limits
if sys.platform == "win32":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.preprocess import preprocess


def identity_analyzer(tokens: list[str]) -> list[str]:
    """Identity analyzer for scikit-learn TfidfVectorizer on pre-tokenized inputs.

    Must be defined as a top-level module function to be serializable by joblib/pickle.
    """
    return tokens


class TfidfRanker:
    """TF-IDF vector space ranker.

    Computes TF-IDF weights over pre-tokenized documents and ranks passages
    for a query using cosine similarity computed via sparse matrix-vector product.
    """

    def __init__(self, dtype: Any = np.float32) -> None:
        self.dtype = dtype
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any = None
        self.pids: np.ndarray | None = None

    def fit(
        self,
        corpus_tokens: Sequence[Sequence[str]] | dict[int, list[str]],
        pids: Sequence[int] | None = None,
    ) -> TfidfRanker:
        """Fit the TF-IDF vectorizer and build document-term matrix.

        Parameters
        ----------
        corpus_tokens : Sequence[Sequence[str]] or dict[int, list[str]]
            Pre-tokenized documents. If a dict of {pid: tokens} is given and
            pids is None, keys are used as passage IDs.
        pids : Sequence[int], optional
            Passage IDs corresponding to corpus_tokens.

        Returns
        -------
        self : TfidfRanker
        """
        if isinstance(corpus_tokens, dict):
            if pids is None:
                pids = list(corpus_tokens.keys())
            token_docs = (corpus_tokens[pid] for pid in pids)
        else:
            token_docs = corpus_tokens
            if pids is None:
                pids = list(range(len(corpus_tokens)))

        self.pids = np.asarray(pids, dtype=np.int64)

        self.vectorizer = TfidfVectorizer(
            analyzer=identity_analyzer,
            dtype=self.dtype,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(token_docs)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Retrieve the top-k passage IDs and cosine similarity scores for a query.

        Parameters
        ----------
        query : str
            Raw text query string (preprocessed via backend.preprocess.preprocess).
        k : int, default=10
            Number of top results to return.

        Returns
        -------
        list of (pid, score)
            Top-k passages with scores sorted descending.
        """
        if self.vectorizer is None or self.tfidf_matrix is None or self.pids is None:
            raise RuntimeError("TfidfRanker has not been fitted. Call fit() or load() first.")

        query_tokens = preprocess(query)
        if not query_tokens:
            return []

        query_vec = self.vectorizer.transform([query_tokens])
        if query_vec.nnz == 0:
            return []

        # Cosine similarity via sparse dot product: (N, V) @ (V, 1) -> (N, 1)
        scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()

        if len(scores) == 0:
            return []

        k = min(k, len(scores))
        if k >= len(scores):
            top_idx = np.argsort(-scores)[:k]
        else:
            top_idx = np.argpartition(-scores, k)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(int(self.pids[idx]), float(scores[idx])) for idx in top_idx]

    def save(self, path: str | os.PathLike) -> None:
        """Serialize and save the ranker to disk using joblib."""
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> TfidfRanker:
        """Load a saved TfidfRanker from disk using joblib."""
        ranker = joblib.load(path)
        if not isinstance(ranker, cls):
            raise TypeError(f"Loaded object is of type {type(ranker)}, expected {cls.__name__}")
        return ranker


if __name__ == "__main__":
    import pickle
    import time
    from backend.data_loader import load_lotte

    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "indexes",
        "corpus_tokens.pkl",
    )
    if not os.path.exists(cache_path):
        print(f"Token cache not found at {cache_path}. Run build_token_cache.py first.")
        sys.exit(1)

    print("Loading preprocessed token cache...")
    t0 = time.time()
    with open(cache_path, "rb") as f:
        corpus_tokens = pickle.load(f)
    print(f"Loaded {len(corpus_tokens):,} passages in {time.time()-t0:.2f}s")

    ranker = TfidfRanker()
    print("Fitting TfidfRanker...")
    t1 = time.time()
    ranker.fit(corpus_tokens)
    print(f"Fitted in {time.time()-t1:.2f}s. Matrix shape: {ranker.tfidf_matrix.shape}")

    # Test hand-picked query
    test_query = "healthy breakfast foods"
    print(f"\nSearching for query: '{test_query}' (top 10)...")
    results = ranker.search(test_query, k=10)

    # Load passages to display text
    passages, _, _ = load_lotte("lifestyle", "dev")

    print("\nTop 10 Results:")
    print("-" * 70)
    for rank, (pid, score) in enumerate(results, 1):
        snippet = passages.get(pid, "").replace("\n", " ")[:100]
        print(f"Rank {rank:2d} | PID: {pid:7d} | Score: {score:.4f} | {snippet}...")
    print("-" * 70)
    print("Verification complete.")
