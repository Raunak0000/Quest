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


class Bm25Ranker:
    """BM25 probabilistic ranker using the bm25s library.

    Computes BM25 scores over pre-tokenized documents with parameters k1=1.5, b=0.75.
    Preserves original passage ID (PID) order and mapping.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.retriever: Any = None
        self.pids: np.ndarray | None = None

    def fit(
        self,
        corpus_tokens: Sequence[Sequence[str]] | dict[int, list[str]],
        pids: Sequence[int] | None = None,
    ) -> Bm25Ranker:
        """Fit BM25 index on pre-tokenized documents.

        Parameters
        ----------
        corpus_tokens : Sequence[Sequence[str]] or dict[int, list[str]]
            Pre-tokenized documents.
        pids : Sequence[int], optional
            Passage IDs corresponding to corpus_tokens.

        Returns
        -------
        self : Bm25Ranker
        """
        import bm25s
        import scipy.sparse as sp

        # Memory-efficient CSC construction using scipy to prevent MemoryError on Windows
        def _scipy_csc(data, rows, cols, shape):
            csc = sp.csc_matrix((data, (rows, cols)), shape=shape)
            return csc.data, csc.indices, csc.indptr

        bm25s.BM25._np_csc = staticmethod(_scipy_csc)

        if isinstance(corpus_tokens, dict):
            if pids is None:
                pids = list(corpus_tokens.keys())
            token_docs = [corpus_tokens[pid] for pid in pids]
        else:
            token_docs = list(corpus_tokens)
            if pids is None:
                pids = list(range(len(corpus_tokens)))

        self.pids = np.asarray(pids, dtype=np.int64)
        self.retriever = bm25s.BM25(k1=self.k1, b=self.b)
        self.retriever.index(token_docs)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Retrieve top-k passage IDs and BM25 scores for a query.

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
        if self.retriever is None or self.pids is None:
            raise RuntimeError("Bm25Ranker has not been fitted. Call fit() or load() first.")

        query_tokens = preprocess(query)
        if not query_tokens:
            return []

        k = min(k, len(self.pids))
        docs, scores = self.retriever.retrieve([query_tokens], k=k)

        return [(int(self.pids[doc_idx]), float(score)) for doc_idx, score in zip(docs[0], scores[0])]

    def save(self, path: str | os.PathLike) -> None:
        """Serialize and save the ranker to disk using joblib or directory."""
        path_str = str(path)
        if os.path.isdir(path_str):
            self.retriever.save(path_str)
            if self.pids is not None:
                joblib.dump(self.pids, os.path.join(path_str, "pids.joblib"))
        else:
            dir_name = os.path.dirname(os.path.abspath(path_str))
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            joblib.dump(self, path_str)

    @classmethod
    def load(cls, path: str | os.PathLike) -> Bm25Ranker:
        """Load a saved Bm25Ranker from disk."""
        path_str = str(path)
        if os.path.isdir(path_str) and os.path.exists(os.path.join(path_str, "params.json")):
            import bm25s
            retriever = bm25s.BM25.load(path_str)
            ranker = cls(k1=retriever.k1, b=retriever.b)
            ranker.retriever = retriever
            pids_file = os.path.join(path_str, "pids.joblib")
            if os.path.exists(pids_file):
                ranker.pids = joblib.load(pids_file)
            return ranker

        ranker = joblib.load(path_str)
        if not isinstance(ranker, cls):
            raise TypeError(f"Loaded object is of type {type(ranker)}, expected {cls.__name__}")
        return ranker


class DenseRanker:
    """Dense semantic retriever using SentenceTransformer and FAISS IndexFlatIP.

    Default model: all-MiniLM-L6-v2 (dim=384).
    Embeddings are L2-normalized so faiss.IndexFlatIP computes cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self.index: Any = None
        self.pids: np.ndarray | None = None

    def _get_model(self):
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def fit(
        self,
        embeddings: np.ndarray,
        pids: Sequence[int] | None = None,
    ) -> DenseRanker:
        """Build FAISS IndexFlatIP from pre-computed normalized embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Array of shape (N, dim), dtype float32.
        pids : Sequence[int], optional
            Passage IDs corresponding to embeddings rows.

        Returns
        -------
        self : DenseRanker
        """
        import faiss

        emb_array = np.ascontiguousarray(embeddings, dtype=np.float32)
        dim = emb_array.shape[1]

        if pids is None:
            pids = list(range(len(emb_array)))
        self.pids = np.asarray(pids, dtype=np.int64)

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb_array)
        return self

    @classmethod
    def from_files(
        cls,
        embeddings_path: str | os.PathLike,
        pids_path: str | os.PathLike,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> DenseRanker:
        """Load embeddings.npy and pids.json and build DenseRanker."""
        import json

        embeddings = np.load(embeddings_path)
        with open(pids_path, "r", encoding="utf-8") as f:
            pids = json.load(f)

        ranker = cls(model_name=model_name, device=device)
        ranker.fit(embeddings, pids=pids)
        return ranker

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Retrieve top-k passage IDs and cosine similarity scores for a semantic query.

        Parameters
        ----------
        query : str
            Raw query text (encoded with SentenceTransformer; not stemmed).
        k : int, default=10
            Number of top results to return.

        Returns
        -------
        list of (pid, score)
            Top-k passages with scores sorted descending.
        """
        if self.index is None or self.pids is None:
            raise RuntimeError("DenseRanker has not been fitted or loaded. Call fit() or from_files().")

        model = self._get_model()
        q_emb = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        k = min(k, len(self.pids))
        scores, indices = self.index.search(q_emb, k)

        return [(int(self.pids[idx]), float(score)) for idx, score in zip(indices[0], scores[0])]

    def save(self, path: str | os.PathLike) -> None:
        """Serialize and save the ranker to disk using joblib."""
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> DenseRanker:
        """Load a saved DenseRanker from disk."""
        ranker = joblib.load(path)
        if not isinstance(ranker, cls):
            raise TypeError(f"Loaded object is of type {type(ranker)}, expected {cls.__name__}")
        return ranker

    def __getstate__(self):
        import faiss

        state = self.__dict__.copy()
        if self.index is not None:
            state["_index_bytes"] = faiss.serialize_index(self.index)
            state["index"] = None
        state["_model"] = None
        return state

    def __setstate__(self, state):
        import faiss

        index_bytes = state.pop("_index_bytes", None)
        self.__dict__.update(state)
        if index_bytes is not None:
            self.index = faiss.deserialize_index(index_bytes)


class HybridRanker:
    """Hybrid ranker combining sparse and dense retrievers via Reciprocal Rank Fusion (RRF).

    Combines:
      - Sparse baseline (default: Bm25Ranker)
      - Dense semantic retriever (default: DenseRanker)
      - Optional 3-way ablation baseline (TfidfRanker if use_tfidf=True)
    """

    def __init__(
        self,
        sparse: Bm25Ranker | None = None,
        dense: DenseRanker | None = None,
        tfidf: TfidfRanker | None = None,
        use_tfidf: bool = False,
        rrf_k: int = 60,
        component_top_k: int = 100,
    ) -> None:
        self.sparse = sparse
        self.dense = dense
        self.tfidf = tfidf
        self.use_tfidf = use_tfidf
        self.rrf_k = rrf_k
        self.component_top_k = component_top_k

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Retrieve top-k passages by fusing sparse and dense ranked lists via RRF.

        Parameters
        ----------
        query : str
            Raw text query string.
        k : int, default=10
            Number of top results to return.

        Returns
        -------
        list of (pid, score)
            Fused top-k passages sorted descending by RRF score.
        """
        from backend.rrf import rrf_fuse

        rank_lists: list[list[int]] = []

        # 1. Sparse retrieval (BM25)
        if self.sparse is not None:
            sparse_results = self.sparse.search(query, k=self.component_top_k)
            sparse_pids = [pid for pid, _ in sparse_results]
            if sparse_pids:
                rank_lists.append(sparse_pids)

        # 2. Dense retrieval (SentenceTransformer + FAISS)
        if self.dense is not None:
            dense_results = self.dense.search(query, k=self.component_top_k)
            dense_pids = [pid for pid, _ in dense_results]
            if dense_pids:
                rank_lists.append(dense_pids)

        # 3. Optional 3-way ablation (TF-IDF)
        if self.use_tfidf and self.tfidf is not None:
            tfidf_results = self.tfidf.search(query, k=self.component_top_k)
            tfidf_pids = [pid for pid, _ in tfidf_results]
            if tfidf_pids:
                rank_lists.append(tfidf_pids)

        if not rank_lists:
            return []

        return rrf_fuse(rank_lists, k=self.rrf_k, top_n=k)

    def save(self, path: str | os.PathLike) -> None:
        """Serialize HybridRanker to disk using joblib."""
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> HybridRanker:
        """Load a saved HybridRanker from disk."""
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
