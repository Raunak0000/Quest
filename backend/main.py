"""FastAPI application — loads all indexes at startup and exposes:
  GET  /health
  GET  /search?q=...&ranker=...&top_k=...
  GET  /compare?q=...&top_k=...
  POST /rag  (streaming, Server-Sent Events)
  GET  /metrics
"""

from __future__ import annotations

import os
import sys
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Any

# ---------------------------------------------------------------------------
# Windows thread-pool fix — must come before numpy/sklearn imports
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure repo root is on sys.path so `backend.*` imports work regardless of cwd
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.rankers import TfidfRanker, Bm25Ranker, DenseRanker, HybridRanker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quest.api")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_INDEXES_DIR = os.path.join(_REPO_ROOT, "indexes")

_TFIDF_PATH         = os.path.join(_INDEXES_DIR, "tfidf_ranker.joblib")
_CORPUS_TOKENS_PATH = os.path.join(_INDEXES_DIR, "corpus_tokens.pkl")
_BM25_PATH          = os.path.join(_INDEXES_DIR, "bm25_ranker.joblib")
_DENSE_EMBEDDINGS   = os.path.join(_INDEXES_DIR, "embeddings.npy")
_DENSE_PIDS         = os.path.join(_INDEXES_DIR, "pids.json")
_HYBRID_PATH        = os.path.join(_INDEXES_DIR, "hybrid_ranker.joblib")

# ---------------------------------------------------------------------------
# Shared app state (populated at startup)
# ---------------------------------------------------------------------------

class AppState:
    tfidf:  TfidfRanker  | None = None
    bm25:   Bm25Ranker   | None = None
    dense:  DenseRanker  | None = None
    hybrid: HybridRanker | None = None

    # Number of indexed documents per ranker
    tfidf_size:  int = 0
    bm25_size:   int = 0
    dense_size:  int = 0
    hybrid_size: int = 0


_state = AppState()

# ---------------------------------------------------------------------------
# Lifespan — load indexes once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all available indexes into app state before serving requests.

    Load order is chosen to minimise peak RAM:
      1. TF-IDF  (compact sparse matrix — loads fast)
      2. BM25    (load pre-built joblib  OR  load corpus tokens as a LOCAL
                  variable, fit BM25, then immediately del + gc.collect so
                  the ~2 GB Python dict is freed before anything else loads)
      3. Dense   (optional — only if embeddings.npy / pids.json exist)
      4. Hybrid  (composed from whichever components are available)
    """
    import gc

    # 1. TF-IDF ranker — load first; compact sparse matrix, low overhead
    if os.path.exists(_TFIDF_PATH):
        log.info("Loading TF-IDF ranker from %s ...", _TFIDF_PATH)
        _state.tfidf = TfidfRanker.load(_TFIDF_PATH)
        _state.tfidf_size = (
            _state.tfidf.tfidf_matrix.shape[0]
            if _state.tfidf.tfidf_matrix is not None
            else 0
        )
        log.info("  -> %d documents indexed", _state.tfidf_size)
    else:
        log.warning("tfidf_ranker.joblib not found — TF-IDF ranker unavailable.")

    # 2. BM25 ranker
    if os.path.exists(_BM25_PATH):
        log.info("Loading BM25 ranker from %s ...", _BM25_PATH)
        _state.bm25 = Bm25Ranker.load(_BM25_PATH)
        _state.bm25_size = (
            len(_state.bm25.pids) if _state.bm25.pids is not None else 0
        )
        log.info("  -> %d documents indexed", _state.bm25_size)
    elif os.path.exists(_CORPUS_TOKENS_PATH):
        # corpus_tokens is kept as a LOCAL variable — never stored in _state —
        # so we can free it immediately after BM25 is fitted.
        log.info("No bm25_ranker.joblib — loading corpus tokens to fit BM25 ...")
        with open(_CORPUS_TOKENS_PATH, "rb") as fh:
            corpus_tokens = pickle.load(fh)
        log.info("  -> %d passages loaded", len(corpus_tokens))

        log.info("Fitting BM25 ...")
        _state.bm25 = Bm25Ranker()
        _state.bm25.fit(corpus_tokens)
        _state.bm25_size = (
            len(_state.bm25.pids) if _state.bm25.pids is not None else 0
        )
        log.info("  -> %d documents indexed", _state.bm25_size)

        # Free the ~2 GB Python dict so Dense / Hybrid don't OOM.
        del corpus_tokens
        gc.collect()
        log.info("Corpus tokens freed from memory.")
    else:
        log.warning("BM25 ranker unavailable (no index file and no corpus tokens).")

    # 3. Dense ranker
    if os.path.exists(_DENSE_EMBEDDINGS) and os.path.exists(_DENSE_PIDS):
        log.info("Loading Dense ranker from %s ...", _DENSE_EMBEDDINGS)
        _state.dense = DenseRanker.from_files(_DENSE_EMBEDDINGS, _DENSE_PIDS)
        _state.dense_size = (
            len(_state.dense.pids) if _state.dense.pids is not None else 0
        )
        log.info("  -> %d documents indexed", _state.dense_size)
    else:
        log.warning(
            "Dense ranker unavailable (embeddings.npy / pids.json not found)."
        )

    # 4. Hybrid ranker — load pre-built, or compose from available components
    if os.path.exists(_HYBRID_PATH):
        log.info("Loading Hybrid ranker from %s ...", _HYBRID_PATH)
        _state.hybrid = HybridRanker.load(_HYBRID_PATH)
        _state.hybrid_size = max(_state.bm25_size, _state.dense_size)
        log.info("  -> hybrid loaded")
    elif _state.bm25 is not None or _state.dense is not None:
        log.info("Composing Hybrid ranker from available components ...")
        _state.hybrid = HybridRanker(
            sparse=_state.bm25,
            dense=_state.dense,
            tfidf=_state.tfidf,
        )
        _state.hybrid_size = max(_state.bm25_size, _state.dense_size)
        log.info(
            "  -> hybrid composed (sparse=%s, dense=%s)",
            _state.bm25 is not None,
            _state.dense is not None,
        )
    else:
        log.warning("Hybrid ranker unavailable — need at least BM25 or Dense.")

    log.info("All indexes loaded. Server ready.")
    yield  # application runs here

    log.info("Shutting down — releasing index memory.")
    _state.tfidf  = None
    _state.bm25   = None
    _state.dense  = None
    _state.hybrid = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Quest IR API",
    description=(
        "Information-retrieval API for the Quest project. "
        "Supports TF-IDF, BM25, Dense, and Hybrid rankers over the LoTTE corpus."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow the Vite dev-server on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class RankerIndexInfo(BaseModel):
    """Index size and availability for a single ranker."""

    available: bool = Field(..., description="Whether this ranker was loaded successfully.")
    num_docs: int   = Field(..., description="Number of indexed documents.")


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str = Field("ok", description="Always 'ok' when the server is running.")
    indexes: dict[str, RankerIndexInfo] = Field(
        ...,
        description="Per-ranker index sizes.",
        examples=[{
            "tfidf":  {"available": True,  "num_docs": 269597},
            "bm25":   {"available": True,  "num_docs": 269597},
            "dense":  {"available": False, "num_docs": 0},
            "hybrid": {"available": False, "num_docs": 0},
        }],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check — returns index sizes for all rankers.",
    tags=["Diagnostics"],
)
def health() -> HealthResponse:
    """Return server status and the number of indexed documents per ranker.

    This endpoint is cheap and performs no retrieval.
    It is intended for readiness probes and the Swagger UI smoke-test.
    """
    return HealthResponse(
        status="ok",
        indexes={
            "tfidf":  RankerIndexInfo(available=_state.tfidf  is not None, num_docs=_state.tfidf_size),
            "bm25":   RankerIndexInfo(available=_state.bm25   is not None, num_docs=_state.bm25_size),
            "dense":  RankerIndexInfo(available=_state.dense  is not None, num_docs=_state.dense_size),
            "hybrid": RankerIndexInfo(available=_state.hybrid is not None, num_docs=_state.hybrid_size),
        },
    )
