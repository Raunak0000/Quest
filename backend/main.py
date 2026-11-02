"""FastAPI application — loads all indexes at startup and exposes:
  GET  /health
  GET  /search?q=...&mode=tfidf|bm25|dense|hybrid&k=10
  GET  /compare?q=...&k=10
  GET  /queries/sample
  POST /rag  (streaming, Server-Sent Events)
  GET  /metrics
"""

from __future__ import annotations

import os
import sys
import pickle
import logging
import time
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

from fastapi import FastAPI, HTTPException, Query
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

_INDEXES_DIR        = os.path.join(_REPO_ROOT, "indexes")
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

    # Passages dict for text lookup in search results: {pid: text}
    passages: dict[int, str] | None = None

    # Sample queries built at startup for /queries/sample
    sample_queries: list[dict] = []


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
      5. Passages + sample queries (for /search text lookup and /queries/sample)
    """
    import gc

    # 1. TF-IDF ranker
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
        log.warning("Dense ranker unavailable (embeddings.npy / pids.json not found).")

    # 4. Hybrid ranker
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

    # 5. Load passages + build sample query list for /queries/sample
    try:
        from backend.data_loader import load_lotte
        log.info("Loading LoTTE lifestyle/dev passages and queries ...")
        passages, queries, qrels = load_lotte("lifestyle", "dev")
        _state.passages = passages

        sample: list[dict] = []
        for qid, text in queries.items():
            if qid in qrels and qrels[qid]:
                sample.append({
                    "qid": qid,
                    "text": text,
                    "relevant_pids": sorted(qrels[qid].keys()),
                })
            if len(sample) >= 10:
                break
        _state.sample_queries = sample
        log.info(
            "  -> %d passages, %d queries, %d sample queries loaded",
            len(passages), len(queries), len(sample),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load passages/queries for text lookup: %s", exc)

    log.info("All indexes loaded. Server ready.")
    yield  # application runs here

    log.info("Shutting down — releasing index memory.")
    _state.tfidf    = None
    _state.bm25     = None
    _state.dense    = None
    _state.hybrid   = None
    _state.passages = None


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


class SearchHit(BaseModel):
    """A single ranked result returned by /search or /compare."""

    rank:  int   = Field(..., description="1-based rank position.")
    pid:   int   = Field(..., description="Passage ID.")
    score: float = Field(..., description="Retrieval score (ranker-dependent).")
    text:  str   = Field(..., description="Passage text (empty if passages not loaded).")


class SearchResponse(BaseModel):
    """Response model for GET /search."""

    query:      str             = Field(..., description="The query string that was searched.")
    mode:       str             = Field(..., description="Ranker used: tfidf | bm25 | dense | hybrid.")
    k:          int             = Field(..., description="Number of results requested.")
    latency_ms: float           = Field(..., description="End-to-end retrieval latency in milliseconds.")
    results:    list[SearchHit] = Field(..., description="Ranked list of results.")


class CompareResponse(BaseModel):
    """Response model for GET /compare — four parallel result lists."""

    query:      str             = Field(..., description="The query string.")
    k:          int             = Field(..., description="Number of results per ranker.")
    latency_ms: float           = Field(..., description="Total wall-clock latency across all rankers (ms).")
    tfidf:      list[SearchHit] = Field(..., description="TF-IDF results.")
    bm25:       list[SearchHit] = Field(..., description="BM25 results.")
    dense:      list[SearchHit] = Field(..., description="Dense results.")
    hybrid:     list[SearchHit] = Field(..., description="Hybrid RRF results.")


class SampleQuery(BaseModel):
    """A single sample query for the frontend demo dropdown."""

    qid:           int       = Field(..., description="Query ID from the evaluation set.")
    text:          str       = Field(..., description="Query text.")
    relevant_pids: list[int] = Field(..., description="Known relevant passage IDs.")


class SampleQueriesResponse(BaseModel):
    """Response model for GET /queries/sample."""

    queries: list[SampleQuery] = Field(..., description="Up to 10 sample evaluation queries.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_MODES: frozenset[str] = frozenset({"tfidf", "bm25", "dense", "hybrid"})


def _ranker_for_mode(mode: str) -> Any:
    """Return the loaded ranker for *mode*, raising HTTPException if unavailable."""
    mapping: dict[str, Any] = {
        "tfidf":  _state.tfidf,
        "bm25":   _state.bm25,
        "dense":  _state.dense,
        "hybrid": _state.hybrid,
    }
    ranker = mapping.get(mode)
    if ranker is None:
        raise HTTPException(
            status_code=503,
            detail=f"Ranker '{mode}' is not available (index not loaded at startup).",
        )
    return ranker


def _hits_from_results(
    results: list[tuple[int, float]],
    passages: dict[int, str] | None,
) -> list[SearchHit]:
    """Convert [(pid, score), ...] into SearchHit objects with passage text."""
    hits: list[SearchHit] = []
    for rank, (pid, score) in enumerate(results, start=1):
        text = passages.get(pid, "") if passages else ""
        hits.append(SearchHit(rank=rank, pid=pid, score=score, text=text))
    return hits


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


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Single-mode search — ranked results with passage text and latency.",
    tags=["Search"],
)
def search(
    q: str = Query(..., min_length=1, description="Query string (must be non-empty)."),
    mode: str = Query(
        "bm25",
        description="Ranker to use. One of: tfidf, bm25, dense, hybrid.",
    ),
    k: int = Query(10, ge=1, le=100, description="Number of results to return (1-100)."),
) -> SearchResponse:
    """Retrieve the top-*k* passages for query *q* using the specified *mode*.

    Each hit in the response includes its 1-based rank, passage ID, retrieval
    score, and full passage text. ``latency_ms`` measures only the retrieval
    step, not network time.

    **Error codes**
    - **400** — *mode* is not one of ``tfidf``, ``bm25``, ``dense``, ``hybrid``.
    - **422** — *q* is empty or *k* is out of range (validated by Pydantic/FastAPI).
    - **503** — The requested ranker index was not loaded at startup.
    """
    if mode not in _VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid mode '{mode}'. "
                f"Must be one of: {sorted(_VALID_MODES)}."
            ),
        )

    ranker = _ranker_for_mode(mode)

    t0 = time.perf_counter()
    raw = ranker.search(q, k=k)
    latency_ms = (time.perf_counter() - t0) * 1_000

    return SearchResponse(
        query=q,
        mode=mode,
        k=k,
        latency_ms=round(latency_ms, 3),
        results=_hits_from_results(raw, _state.passages),
    )


@app.get(
    "/compare",
    response_model=CompareResponse,
    summary="Four-way comparison — runs all rankers for one query.",
    tags=["Search"],
)
def compare(
    q: str = Query(..., min_length=1, description="Query string (must be non-empty)."),
    k: int = Query(10, ge=1, le=100, description="Number of results per ranker (1-100)."),
) -> CompareResponse:
    """Run *q* through all four rankers (TF-IDF, BM25, Dense, Hybrid) and return
    their result lists side-by-side.

    Rankers not loaded at startup return an empty list instead of an error, so
    you can still compare whichever subset is available.

    **Error codes**
    - **422** — *q* is empty or *k* is out of range.
    """
    t0 = time.perf_counter()

    def _safe(ranker: Any, query: str, top_k: int) -> list[SearchHit]:
        if ranker is None:
            return []
        try:
            return _hits_from_results(ranker.search(query, k=top_k), _state.passages)
        except Exception as exc:  # noqa: BLE001
            log.warning("Ranker error during /compare: %s", exc)
            return []

    tfidf_hits  = _safe(_state.tfidf,  q, k)
    bm25_hits   = _safe(_state.bm25,   q, k)
    dense_hits  = _safe(_state.dense,  q, k)
    hybrid_hits = _safe(_state.hybrid, q, k)
    latency_ms  = (time.perf_counter() - t0) * 1_000

    return CompareResponse(
        query=q,
        k=k,
        latency_ms=round(latency_ms, 3),
        tfidf=tfidf_hits,
        bm25=bm25_hits,
        dense=dense_hits,
        hybrid=hybrid_hits,
    )


@app.get(
    "/queries/sample",
    response_model=SampleQueriesResponse,
    summary="Return up to 10 sample queries from the evaluation set.",
    tags=["Demo"],
)
def queries_sample() -> SampleQueriesResponse:
    """Return a fixed list of up to 10 queries drawn from the LoTTE lifestyle/dev
    evaluation set, each with its known relevant passage IDs.

    Built once at startup from the evaluation qrels; intended for the frontend
    demo dropdown so users can quickly try realistic queries without typing.
    """
    return SampleQueriesResponse(
        queries=[SampleQuery(**q) for q in _state.sample_queries],
    )


# ---------------------------------------------------------------------------
# POST /rag — RAG streaming endpoint (Server-Sent Events)
# ---------------------------------------------------------------------------

from fastapi import Request
from fastapi.responses import StreamingResponse


class RagRequest(BaseModel):
    """Request body for POST /rag."""

    question: str = Field(..., min_length=1, description="The user's question.")
    k: int = Field(5, ge=1, le=20, description="Number of passages to retrieve (1-20).")


@app.post(
    "/rag",
    summary="RAG answer with citations — streams Server-Sent Events.",
    tags=["RAG"],
    response_class=StreamingResponse,
)
async def rag(body: RagRequest) -> StreamingResponse:
    """Retrieve the top-*k* passages with the hybrid ranker, build a prompt,
    then stream the LLM answer as Server-Sent Events in three phases:

    1. **passages** event — JSON array ``[{rank, pid, score, text}, ...]`` so
       the client can display citations immediately.
    2. **token** events — one per LLM output token, value is the token text.
    3. **done** event — signals end of stream (value ``"[DONE]"``).

    On any error (missing key, unsupported provider, API failure) an **error**
    SSE event is emitted and the stream ends; the server stays up.

    Requires the hybrid ranker to be loaded (falls back to BM25 if not).
    """
    import json as _json
    from backend.rag import build_prompt, generate

    # Choose ranker: prefer hybrid, fall back to bm25, then tfidf
    ranker = _state.hybrid or _state.bm25 or _state.tfidf
    if ranker is None:
        async def _no_ranker():
            yield "event: error\ndata: No ranker is loaded — cannot retrieve passages.\n\n"
        return StreamingResponse(_no_ranker(), media_type="text/event-stream")

    async def _stream():
        # --- Phase 1: retrieve passages ---
        try:
            raw = ranker.search(body.question, k=body.k)
        except Exception as exc:
            yield f"event: error\ndata: Retrieval failed: {exc}\n\n"
            return

        passages: list[tuple[int, str]] = []
        hits_payload: list[dict] = []
        for rank, (pid, score) in enumerate(raw, start=1):
            text = (_state.passages or {}).get(pid, "")
            passages.append((pid, text))
            hits_payload.append({"rank": rank, "pid": pid, "score": score, "text": text})

        yield f"event: passages\ndata: {_json.dumps(hits_payload)}\n\n"

        # --- Phase 2: stream LLM tokens ---
        prompt = build_prompt(body.question, passages)
        try:
            async for token in generate(prompt):
                # Escape newlines so each SSE data line is single-line
                safe = token.replace("\n", "\\n")
                yield f"event: token\ndata: {safe}\n\n"
        except (EnvironmentError, ValueError) as exc:
            yield f"event: error\ndata: {exc}\n\n"
            return
        except Exception as exc:
            yield f"event: error\ndata: LLM error: {exc}\n\n"
            return

        # --- Phase 3: done ---
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /metrics — serve results/results.json produced by scripts/run_eval.py
# ---------------------------------------------------------------------------

_RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
_RESULTS_JSON = os.path.join(_RESULTS_DIR, "results.json")
_PR_CURVES_JSON = os.path.join(_RESULTS_DIR, "pr_curves.json")
_RUN_EVAL_HINT = (
    "Run  python scripts/run_eval.py  to generate evaluation results, "
    "then restart the server."
)


@app.get(
    "/metrics",
    summary="Aggregate evaluation metrics for all rankers.",
    tags=["Evaluation"],
)
def metrics() -> dict:
    """Return the contents of ``results/results.json`` produced by
    ``scripts/run_eval.py``.

    The response is a JSON object keyed by ranker name, each value being a
    dict of metric names to float values, e.g.:

    ```json
    {
      "TF-IDF": {"P@10": 0.083, "R@100": 0.518, "MAP": 0.184, ...},
      "BM25":   {"P@10": 0.105, "R@100": 0.632, "MAP": 0.253, ...}
    }
    ```

    **404** is returned with a helpful hint if the file does not exist.
    """
    import json as _json

    if not os.path.exists(_RESULTS_JSON):
        raise HTTPException(
            status_code=404,
            detail=f"results/results.json not found. {_RUN_EVAL_HINT}",
        )
    try:
        with open(_RESULTS_JSON, encoding="utf-8") as fh:
            data = _json.load(fh)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read results.json: {exc}",
        ) from exc
    return data


@app.get(
    "/pr-curves",
    summary="Precision-Recall curve points for all evaluated rankers.",
    tags=["Evaluation"],
)
def pr_curves() -> dict:
    """Return interpolated PR curve data from ``results/pr_curves.json``
    produced by ``scripts/run_eval.py``.

    The response is keyed by ranker name; each value has:
    - **recall** — list of 11 standard recall levels ``[0.0, 0.1, …, 1.0]``
    - **precision** — corresponding mean interpolated precision values

    ```json
    {
      "TF-IDF": {"recall": [0.0, 0.1, ...], "precision": [0.45, 0.38, ...]},
      "BM25":   {"recall": [0.0, 0.1, ...], "precision": [0.57, 0.49, ...]}
    }
    ```

    **404** is returned with a hint to run ``scripts/run_eval.py`` if the
    file does not exist (it is written by the evaluation script alongside
    ``results.json``).
    """
    import json as _json

    if not os.path.exists(_PR_CURVES_JSON):
        raise HTTPException(
            status_code=404,
            detail=f"results/pr_curves.json not found. {_RUN_EVAL_HINT}",
        )
    try:
        with open(_PR_CURVES_JSON, encoding="utf-8") as fh:
            data = _json.load(fh)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read pr_curves.json: {exc}",
        ) from exc
    return data
