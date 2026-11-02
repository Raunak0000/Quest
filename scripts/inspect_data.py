"""Walk data/lotte/<domain>/<split>/, print file names, sizes,
the first 3 lines of each text/TSV/JSONL file, and counts of
passages and queries.

Usage:
    python scripts/inspect_data.py [domain] [split]

Defaults to lifestyle / dev.
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_base_lotte = os.path.join(os.path.dirname(__file__), "..", "lotte")
_nested_lotte = os.path.join(_base_lotte, "lotte")
DATA_ROOT = _nested_lotte if os.path.exists(_nested_lotte) else _base_lotte


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def first_lines(path: str, n: int = 3) -> list[str]:
    lines = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            lines.append(line.rstrip("\n"))
            if len(lines) == n:
                break
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def inspect(domain: str = "lifestyle", split: str = "dev") -> None:
    split_dir = os.path.join(DATA_ROOT, domain, split)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Directory not found: {split_dir}\n"
            f"Check that the LoTTE dataset is extracted to lotte/<domain>/<split>/."
        )

    print(f"\n{'='*70}")
    print(f"  Domain: {domain}   Split: {split}")
    print(f"  Path:   {split_dir}")
    print(f"{'='*70}\n")

    files = sorted(os.listdir(split_dir))

    for fname in files:
        fpath = os.path.join(split_dir, fname)
        size = os.path.getsize(fpath)
        ext = os.path.splitext(fname)[1].lower()

        print(f"  [{human_size(size):>9}]  {fname}")

        if ext in (".tsv", ".jsonl"):
            lines = first_lines(fpath, 3)
            for i, line in enumerate(lines, 1):
                preview = line[:120] + ("…" if len(line) > 120 else "")
                print(f"              row {i}: {preview}")
        print()

    # -----------------------------------------------------------------------
    # Counts
    # -----------------------------------------------------------------------
    collection_path = os.path.join(split_dir, "collection.tsv")
    search_q_path   = os.path.join(split_dir, "questions.search.tsv")
    forum_q_path    = os.path.join(split_dir, "questions.forum.tsv")

    def count_lines(path: str) -> int:
        if not os.path.exists(path):
            return 0
        with open(path, encoding="utf-8") as fh:
            return sum(1 for _ in fh)

    n_passages     = count_lines(collection_path)
    n_search_qs    = count_lines(search_q_path)
    n_forum_qs     = count_lines(forum_q_path)

    print(f"  Passages (collection.tsv):        {n_passages:>8,}")
    print(f"  Search queries:                   {n_search_qs:>8,}")
    print(f"  Forum queries:                    {n_forum_qs:>8,}")
    print()

    # -----------------------------------------------------------------------
    # One worked example from qas.search.jsonl
    # -----------------------------------------------------------------------
    qas_path = os.path.join(split_dir, "qas.search.jsonl")
    if os.path.exists(qas_path):
        with open(qas_path, encoding="utf-8") as fh:
            example = json.loads(next(iter(fh)))
        qid   = example["qid"]
        query = example["query"]
        pids  = example["answer_pids"]
        print(f"  Example QA (search):")
        print(f"    qid   : {qid}")
        print(f"    query : {query}")
        print(f"    pids  : {pids}")
        print()


if __name__ == "__main__":
    domain = sys.argv[1] if len(sys.argv) > 1 else "lifestyle"
    split  = sys.argv[2] if len(sys.argv) > 2 else "dev"
    inspect(domain, split)
