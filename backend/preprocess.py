"""Sparse-path text preprocessing: lowercase, strip HTML/URLs,
remove punctuation, tokenise (NLTK word_tokenize), remove English
stop-words, apply PorterStemmer.

⚠️  Used ONLY for TF-IDF and BM25.
    The dense encoder (Sentence Transformer) must receive raw text.

Public API
----------
preprocess(text: str) -> list[str]
    Clean and stem a single string.

preprocess_corpus(passages: dict[int, str], n_jobs: int = -1) -> list[list[str]]
    Preprocess every passage in parallel; cache result to
    indexes/corpus_tokens.pkl so the second call loads in < 5 s.
"""

from __future__ import annotations

import os
import pickle
import re
import string
import multiprocessing
from functools import partial

import nltk

# ── NLTK data: download quietly if not already present ─────────────────────
for _resource, _path in [
    ("tokenizers/punkt_tab",       "punkt_tab"),
    ("corpora/stopwords",          "stopwords"),
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_path, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ── constants ───────────────────────────────────────────────────────────────
_STOP_WORDS  = set(stopwords.words("english"))
_STEMMER     = PorterStemmer()
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Path for the cached corpus tokens
_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_PATH  = os.path.join(_REPO_ROOT, "indexes", "corpus_tokens.pkl")

# Regex to strip HTML tags and bare URLs
_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE  = re.compile(r"https?://\S+|www\.\S+")


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def preprocess(text: str) -> list[str]:
    """Clean and stem a single piece of text.

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
    # 1. Lowercase
    text = text.lower()
    # 2. Strip HTML
    text = _HTML_RE.sub(" ", text)
    # 3. Strip URLs
    text = _URL_RE.sub(" ", text)
    # 4. Remove punctuation
    text = text.translate(_PUNCT_TABLE)
    # 5. Tokenise
    tokens = word_tokenize(text)
    # 6 & 7. Stop-words + stemming (skip tokens ≤ 1 char)
    return [
        _STEMMER.stem(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and len(tok) > 1
    ]


# ---------------------------------------------------------------------------
# Corpus-level preprocessing (parallel + cached)
# ---------------------------------------------------------------------------

def _preprocess_item(item: tuple[int, str]) -> tuple[int, list[str]]:
    """Worker function: preprocess one (pid, text) pair."""
    pid, text = item
    return pid, preprocess(text)


def preprocess_corpus(
    passages: dict[int, str],
    n_jobs: int = -1,
    cache_path: str = _CACHE_PATH,
    force: bool = False,
) -> dict[int, list[str]]:
    """Preprocess all passages in parallel, with disk caching.

    Parameters
    ----------
    passages:
        Mapping from pid → raw passage text (from data_loader).
    n_jobs:
        Number of worker processes. -1 = all CPU cores.
    cache_path:
        Where to store / load the pickle cache.
    force:
        If True, re-compute even if cache exists.

    Returns
    -------
    dict[int, list[str]]
        Mapping from pid → list of stemmed tokens.
    """
    if not force and os.path.exists(cache_path):
        print(f"[preprocess] Loading cached tokens from {cache_path} …")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    print(f"[preprocess] Preprocessing {len(passages):,} passages …")

    n_workers = (
        multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    )

    items = list(passages.items())

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(_preprocess_item, items, chunksize=500)

    corpus_tokens: dict[int, list[str]] = dict(results)

    # Ensure the indexes/ directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(corpus_tokens, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[preprocess] Cached to {cache_path}")

    return corpus_tokens
