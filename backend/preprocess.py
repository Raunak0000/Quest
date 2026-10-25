"""Sparse-path text preprocessing: lowercase, strip HTML/URLs,
remove punctuation, tokenise, remove English stop-words, apply PorterStemmer.

- Used ONLY for TF-IDF and BM25.
  The dense encoder (Sentence Transformer) must receive raw text.

Public API
----------
preprocess(text: str) -> list[str]
    Clean and stem a single string (uses NLTK word_tokenize for quality).

preprocess_corpus(passages: dict[int, str], n_jobs: int = -1) -> dict[int, list[str]]
    Preprocess every passage; cache result to
    indexes/corpus_tokens.pkl so the second call loads in < 5 s.

Optimisation notes (Windows / Python 3.14)
------------------------------------------
* Windows multiprocessing uses 'spawn': every worker re-imports all
  modules including NLTK, causing huge RAM spikes. With 16 workers
  this crashes the machine.
* Fix: cap workers at 4 on Windows + fast regex tokeniser in workers
  (no NLTK punkt loading per process) + large batches to cut IPC overhead.
* Single-process fallback (n_jobs=1) also has progress reporting.
"""

from __future__ import annotations

import os
import pickle
import re
import string
import sys
import multiprocessing

import nltk

# -- NLTK data: download quietly if not already present ----------------
for _resource, _path in [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords",    "stopwords"),
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_path, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# -- constants ---------------------------------------------------------
_STOP_WORDS  = set(stopwords.words("english"))
_STEMMER     = PorterStemmer()
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Path for the cached corpus tokens
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_PATH = os.path.join(_REPO_ROOT, "indexes", "corpus_tokens.pkl")

# Regex to strip HTML tags and bare URLs
_HTML_RE = re.compile(r"<[^>]+"+">")
_URL_RE  = re.compile(r"https?://\S+|www\.\S+")
# Fast alpha-only tokeniser for bulk corpus work.
# Avoids loading NLTK punkt model in every spawned worker process.
_TOK_RE  = re.compile(r"[a-z]{2,}")


# ---------------------------------------------------------------------------
# Core function  (query-time -- uses NLTK word_tokenize for quality)
# ---------------------------------------------------------------------------

def preprocess(text: str) -> list[str]:
    """Clean and stem a single piece of text.

    Uses NLTK word_tokenize; best for short query strings.

    Steps
    -----
    1. Lowercase
    2. Strip HTML tags
    3. Strip URLs
    4. Remove punctuation
    5. Tokenise with NLTK word_tokenize
    6. Remove English stop-words
    7. Porter-stem each token
    8. Drop empty / single-character tokens

    Returns
    -------
    list[str]
        Stemmed, filtered tokens.
    """
    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    tokens = word_tokenize(text)
    return [
        _STEMMER.stem(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and len(tok) > 1
    ]


# ---------------------------------------------------------------------------
# Fast single-text variant  (used inside corpus workers -- no NLTK punkt)
# ---------------------------------------------------------------------------

def _preprocess_fast(text: str) -> list[str]:
    """Like preprocess() but uses a regex tokeniser instead of NLTK punkt.

    Loading the punkt model in every spawned Windows worker is the primary
    source of memory blowup. This variant avoids it entirely.
    """
    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    tokens = _TOK_RE.findall(text)   # only [a-z]{2,} runs
    return [
        _STEMMER.stem(tok)
        for tok in tokens
        if tok not in _STOP_WORDS
    ]


# ---------------------------------------------------------------------------
# Batch worker  (module-level so it is picklable under Windows spawn)
# ---------------------------------------------------------------------------

def _process_batch(batch: list) -> list:
    """Process a list of (pid, text) pairs; return list of (pid, tokens).

    Large batches reduce IPC round-trips and keep each worker busy.
    """
    return [(pid, _preprocess_fast(text)) for pid, text in batch]


# ---------------------------------------------------------------------------
# Corpus-level preprocessing (parallel + cached)
# ---------------------------------------------------------------------------

def preprocess_corpus(
    passages: dict[int, str],
    n_jobs: int = -1,
    cache_path: str = _CACHE_PATH,
    force: bool = False,
) -> dict[int, list[str]]:
    """Preprocess all passages, with disk caching.

    Parameters
    ----------
    passages:
        Mapping from pid -> raw passage text (from data_loader).
    n_jobs:
        Number of worker processes.
        -1  = auto (capped at 4 on Windows to prevent RAM crashes).
        1   = single-process, safest on Windows (~3-4 min for 268k).
    cache_path:
        Where to store / load the pickle cache.
    force:
        If True, re-compute even if cache exists.

    Returns
    -------
    dict[int, list[str]]
        Mapping from pid -> list of stemmed tokens.
    """
    # -- Cache hit -------------------------------------------------------
    if not force and os.path.exists(cache_path):
        print(f"[preprocess] Loading cached tokens from {cache_path} ...")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    n_passages = len(passages)
    print(f"[preprocess] Preprocessing {n_passages:,} passages ...")

    # -- Determine worker count ------------------------------------------
    max_cpu = multiprocessing.cpu_count()
    if n_jobs == -1:
        # Cap at 4 workers on Windows: each spawned process re-imports NLTK
        # (~200-300 MB each); 16 workers = ~4 GB RAM spike = crash.
        n_workers = 1 if sys.platform == "win32" else max_cpu  # spawn + OpenBLAS OOM on Windows
    else:
        n_workers = max(1, n_jobs)

    print(f"[preprocess] Using {n_workers} worker(s) ({max_cpu} cores available).")

    items = list(passages.items())

    if n_workers == 1:
        # -- Single-process path: zero spawn overhead, max stability ------
        results = []
        report_every = max(1, n_passages // 20)   # log every 5%
        for i, (pid, text) in enumerate(items, 1):
            results.append((pid, _preprocess_fast(text)))
            if i % report_every == 0 or i == n_passages:
                pct = i / n_passages * 100
                print(f"  [{pct:5.1f}%]  {i:,} / {n_passages:,}", flush=True)
    else:
        # -- Multi-process path ------------------------------------------
        # Large batches keep workers busy; small enough the first result
        # arrives quickly.
        batch_size = max(500, n_passages // (n_workers * 8))
        batches = [items[i: i + batch_size] for i in range(0, n_passages, batch_size)]
        n_batches = len(batches)
        print(f"[preprocess] {n_batches} batches x ~{batch_size} passages each ...")

        results = []
        completed = 0
        report_every = max(1, n_batches // 20)

        # Explicit spawn context -- required on Windows, harmless elsewhere.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for batch_result in pool.imap_unordered(_process_batch, batches):
                results.extend(batch_result)
                completed += 1
                if completed % report_every == 0 or completed == n_batches:
                    done = len(results)
                    pct = done / n_passages * 100
                    print(f"  [{pct:5.1f}%]  {done:,} / {n_passages:,}", flush=True)

    corpus_tokens: dict[int, list[str]] = dict(results)

    # -- Persist cache ---------------------------------------------------
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(corpus_tokens, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[preprocess] Cached {len(corpus_tokens):,} entries -> {cache_path}")

    return corpus_tokens
