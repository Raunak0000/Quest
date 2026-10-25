"""Load LoTTE passages, search queries, and relevance labels from disk.

Public API
----------
load_lotte(domain, split, query_type='search')
    -> passages : dict[int, str]          {pid: text}
    -> queries  : dict[int, str]          {qid: text}
    -> qrels    : dict[int, dict[int,int]] {qid: {pid: 1}}
"""

from __future__ import annotations

import json
import os
from typing import Literal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Resolve from this file's location so it works regardless of cwd
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT  = os.path.join(_REPO_ROOT, "lotte")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require(path: str) -> str:
    """Return *path* if it exists, otherwise raise a descriptive FileNotFoundError."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            "Make sure the LoTTE dataset is extracted to lotte/<domain>/<split>/."
        )
    return path


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_lotte(
    domain: str,
    split: Literal["dev", "test"],
    query_type: Literal["search", "forum"] = "search",
) -> tuple[dict[int, str], dict[int, str], dict[int, dict[int, int]]]:
    """Load a single LoTTE domain/split into memory.

    Parameters
    ----------
    domain:
        One of ``lifestyle``, ``recreation``, ``science``, ``technology``,
        ``writing``, or ``pooled``.
    split:
        ``"dev"`` or ``"test"``.
    query_type:
        ``"search"`` (default) or ``"forum"``.
        The project uses the *search* query set for evaluation.

    Returns
    -------
    passages : dict[int, str]
        Mapping from passage-id (int) to passage text.
    queries : dict[int, str]
        Mapping from query-id (int) to query text.
    qrels : dict[int, dict[int, int]]
        Mapping from query-id to a dict of relevant passage-ids → 1.
    """
    split_dir = os.path.join(DATA_ROOT, domain, split)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Domain/split directory not found: {split_dir}\n"
            f"Available domains: lifestyle, recreation, science, technology, writing, pooled.\n"
            f"Available splits:  dev, test."
        )

    # ------------------------------------------------------------------
    # 1. Passages — collection.tsv  (pid<TAB>text)
    # ------------------------------------------------------------------
    collection_path = _require(os.path.join(split_dir, "collection.tsv"))
    passages: dict[int, str] = {}
    with open(collection_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            pid_str, text = line.split("\t", 1)
            passages[int(pid_str)] = text

    # ------------------------------------------------------------------
    # 2. Queries — questions.<query_type>.tsv  (qid<TAB>text)
    # ------------------------------------------------------------------
    questions_path = _require(
        os.path.join(split_dir, f"questions.{query_type}.tsv")
    )
    queries: dict[int, str] = {}
    with open(questions_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            qid_str, text = line.split("\t", 1)
            queries[int(qid_str)] = text

    # ------------------------------------------------------------------
    # 3. Qrels — qas.<query_type>.jsonl
    #    Each row: {"qid": int, "query": str, "answer_pids": [int, ...]}
    # ------------------------------------------------------------------
    qas_path = _require(
        os.path.join(split_dir, f"qas.{query_type}.jsonl")
    )
    qrels: dict[int, dict[int, int]] = {}
    with open(qas_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid  = int(record["qid"])
            pids = {int(pid): 1 for pid in record["answer_pids"]}
            qrels[qid] = pids

    return passages, queries, qrels
