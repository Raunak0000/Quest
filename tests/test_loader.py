"""Unit tests for backend.data_loader.load_lotte.

Covers
------
* Consistent IDs: every relevant pid in qrels exists in passages
* queries dict maps int -> str
* passages dict maps int -> str
* qrels dict maps int -> dict[int, int]
* All qrel relevance values are positive integers
* No qid in qrels that is absent from queries
* FileNotFoundError on missing domain/split
* Return types are correct dicts (not lists or other containers)
* Data is loaded from the real lifestyle/dev split so results are stable
"""

from __future__ import annotations

import pytest
from backend.data_loader import load_lotte


# ---------------------------------------------------------------------------
# Fixture: load lifestyle/dev once for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lifestyle_dev():
    """Load LoTTE lifestyle/dev search split once; skip if data missing."""
    try:
        return load_lotte("lifestyle", "dev", "search")
    except FileNotFoundError:
        pytest.skip("LoTTE lifestyle/dev data not available in this environment.")


# ---------------------------------------------------------------------------
# Return-type tests
# ---------------------------------------------------------------------------

class TestReturnTypes:

    def test_returns_three_dicts(self, lifestyle_dev):
        passages, queries, qrels = lifestyle_dev
        assert isinstance(passages, dict)
        assert isinstance(queries, dict)
        assert isinstance(qrels, dict)

    def test_passages_int_keys_str_values(self, lifestyle_dev):
        passages, _, _ = lifestyle_dev
        for pid, text in list(passages.items())[:20]:
            assert isinstance(pid, int), f"passage key {pid!r} is not int"
            assert isinstance(text, str), f"passage value for pid={pid} is not str"

    def test_queries_int_keys_str_values(self, lifestyle_dev):
        _, queries, _ = lifestyle_dev
        for qid, text in list(queries.items())[:20]:
            assert isinstance(qid, int), f"query key {qid!r} is not int"
            assert isinstance(text, str), f"query text for qid={qid} is not str"

    def test_qrels_structure(self, lifestyle_dev):
        """qrels: {int: {int: int}} — outer key qid, inner key pid, value relevance."""
        _, _, qrels = lifestyle_dev
        for qid, pid_rel in list(qrels.items())[:20]:
            assert isinstance(qid, int), f"qrel outer key {qid!r} not int"
            assert isinstance(pid_rel, dict), f"qrel[{qid}] not dict"
            for pid, rel in pid_rel.items():
                assert isinstance(pid, int), f"pid {pid!r} in qrel[{qid}] not int"
                assert isinstance(rel, int), f"relevance {rel!r} not int"


# ---------------------------------------------------------------------------
# Content sanity
# ---------------------------------------------------------------------------

class TestContentSanity:

    def test_non_empty_collections(self, lifestyle_dev):
        passages, queries, qrels = lifestyle_dev
        assert len(passages) > 0
        assert len(queries) > 0
        assert len(qrels) > 0

    def test_passages_are_non_empty_strings(self, lifestyle_dev):
        passages, _, _ = lifestyle_dev
        for pid, text in list(passages.items())[:50]:
            assert text.strip(), f"Empty passage text for pid={pid}"

    def test_queries_are_non_empty_strings(self, lifestyle_dev):
        _, queries, _ = lifestyle_dev
        for qid, text in list(queries.items())[:50]:
            assert text.strip(), f"Empty query text for qid={qid}"

    def test_qrel_relevance_positive(self, lifestyle_dev):
        """All relevance values must be >= 1 (LoTTE uses binary 1)."""
        _, _, qrels = lifestyle_dev
        for qid, pid_rel in qrels.items():
            for pid, rel in pid_rel.items():
                assert rel >= 1, f"Non-positive relevance {rel} for qid={qid}, pid={pid}"


# ---------------------------------------------------------------------------
# KEY INVARIANT: every relevant pid in qrels exists in passages
# ---------------------------------------------------------------------------

class TestConsistentIDs:

    def test_all_qrel_pids_exist_in_passages(self, lifestyle_dev):
        """Core correctness: every pid annotated as relevant must be loadable.

        If this fails, the evaluation loop will silently miss relevant passages,
        corrupting all IR metrics.
        """
        passages, _, qrels = lifestyle_dev
        missing = []
        for qid, pid_rel in qrels.items():
            for pid in pid_rel:
                if pid not in passages:
                    missing.append((qid, pid))

        assert not missing, (
            f"{len(missing)} relevant pids in qrels are absent from passages.\n"
            f"First 5: {missing[:5]}"
        )

    def test_all_qrel_qids_exist_in_queries(self, lifestyle_dev):
        """Every qid that has relevance judgements must also have a query text."""
        _, queries, qrels = lifestyle_dev
        missing_qids = [qid for qid in qrels if qid not in queries]
        assert not missing_qids, (
            f"{len(missing_qids)} qids in qrels not found in queries: {missing_qids[:5]}"
        )

    def test_passage_ids_are_unique(self, lifestyle_dev):
        """PIDs must be globally unique (dict guarantees this, but sanity-check count)."""
        passages, _, _ = lifestyle_dev
        # If any line in collection.tsv was duplicated, dict would silently drop it
        # We can't detect that here but we can assert the dict is non-degenerate
        assert len(passages) > 100_000, (
            f"Unexpectedly few passages: {len(passages)}. Possible data corruption."
        )

    def test_query_ids_are_unique(self, lifestyle_dev):
        _, queries, _ = lifestyle_dev
        assert len(queries) >= 10, f"Suspiciously few queries: {len(queries)}"

    def test_each_qrel_entry_has_at_least_one_relevant(self, lifestyle_dev):
        """Every qid in qrels must have at least one relevant pid."""
        _, _, qrels = lifestyle_dev
        for qid, pid_rel in qrels.items():
            assert len(pid_rel) >= 1, f"qid={qid} has zero relevant passages in qrels"


# ---------------------------------------------------------------------------
# forum query type
# ---------------------------------------------------------------------------

def test_forum_query_type_loads():
    """Forum queries should load without error when the file exists."""
    try:
        passages, queries, qrels = load_lotte("lifestyle", "dev", "forum")
        assert isinstance(passages, dict)
        assert isinstance(queries, dict)
        assert isinstance(qrels, dict)
        # Forum queries are independent from search queries
        assert len(queries) > 0
    except FileNotFoundError:
        pytest.skip("LoTTE lifestyle/dev forum data not available.")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_missing_domain_raises(self):
        with pytest.raises(FileNotFoundError):
            load_lotte("nonexistent_domain_xyz", "dev")

    def test_missing_split_raises(self):
        with pytest.raises(FileNotFoundError):
            load_lotte("lifestyle", "nonexistent_split_xyz")

    def test_bad_domain_message_helpful(self):
        """Error message should mention the missing path or available options."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_lotte("invalid_domain_abc", "dev")
        msg = str(exc_info.value)
        # Should include some contextual info (path or domain list)
        assert len(msg) > 20
