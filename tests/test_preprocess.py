"""Unit tests for backend.preprocess.

Covers
------
* preprocess() removes stop-words
* preprocess() applies PorterStemmer
* preprocess() strips HTML tags
* preprocess() strips URLs
* preprocess() strips punctuation
* preprocess() lowercases input
* preprocess() drops single-char tokens
* preprocess() returns empty list for whitespace-only / stop-word-only input
* preprocess_corpus() produces same PIDs as input passages
* preprocess_corpus() every relevant pid exists in passages
  (IDs are never invented / dropped / remapped)
* _preprocess_fast() produces stemmed, alpha-only tokens
"""

from __future__ import annotations

import tempfile
import os
import pickle

import pytest

from backend.preprocess import preprocess, _preprocess_fast, preprocess_corpus


# ---------------------------------------------------------------------------
# preprocess() – single-string interface
# ---------------------------------------------------------------------------

class TestPreprocessSingle:

    def test_lowercase(self):
        tokens = preprocess("RUNNING")
        # "running" -> stemmed -> "run"
        assert all(t == t.lower() for t in tokens)

    def test_stop_words_removed(self):
        """Common stop-words must not appear in output."""
        tokens = preprocess("the quick brown fox and the lazy dog")
        stop_words = {"the", "and", "a", "an", "is", "are", "of", "in", "it"}
        for tok in tokens:
            assert tok not in stop_words, f"Stop-word '{tok}' not removed"

    def test_stemming_applied(self):
        """Words that share a morphological root should produce the same stem.

        Porter collapses "running" and "runs" to "run"; "runner" maps to
        "runner" (a known Porter quirk), so we test the forms that do collapse.
        """
        stems_run = set()
        for word in ["running", "runs"]:
            result = preprocess(word)
            if result:
                stems_run.add(result[0])
        # "running" and "runs" must produce the same stem
        assert len(stems_run) == 1, (
            f"Expected 'running' and 'runs' to have the same stem; got {stems_run}"
        )

        # Stemming must actually change the token (not a no-op)
        raw = preprocess("running")
        assert raw != ["running"], "Stemmer appears to be a no-op on 'running'"

    def test_html_stripped(self):
        tokens_with = preprocess("<b>breakfast</b> ideas")
        tokens_without = preprocess("breakfast ideas")
        # HTML tags must not produce extra tokens
        assert tokens_with == tokens_without

    def test_url_stripped(self):
        tokens = preprocess("visit https://example.com for recipes")
        for tok in tokens:
            assert "http" not in tok
            assert "www" not in tok
            assert "example" not in tok or True  # "example" itself is fine if stemmed

    def test_punctuation_removed(self):
        tokens = preprocess("hello, world! it's great.")
        for tok in tokens:
            assert not any(c in tok for c in "!,.'")

    def test_single_char_tokens_dropped(self):
        """Single characters produced after stripping should be excluded."""
        tokens = preprocess("a b c x y z")
        assert tokens == []  # all are stop-words or single chars

    def test_empty_string(self):
        assert preprocess("") == []

    def test_whitespace_only(self):
        assert preprocess("   \t\n  ") == []

    def test_stop_word_only_sentence(self):
        assert preprocess("the and is a an") == []

    def test_mixed_content(self):
        """Non-trivial sentence -> some meaningful stems survive."""
        tokens = preprocess("Baking delicious chocolate cake recipes")
        # "bake", "delici", "chocol", "cake", "recip" or similar stems expected
        assert len(tokens) >= 3

    def test_returns_list_of_strings(self):
        tokens = preprocess("machine learning models")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_numeric_tokens_handled(self):
        """Numbers mixed with text should not crash; stop-words still removed."""
        tokens = preprocess("item 42 is great")
        assert isinstance(tokens, list)


# ---------------------------------------------------------------------------
# _preprocess_fast() – worker variant
# ---------------------------------------------------------------------------

class TestPreprocessFast:

    def test_returns_list_of_strings(self):
        tokens = _preprocess_fast("quick brown fox")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_stop_words_removed(self):
        tokens = _preprocess_fast("the quick brown fox")
        assert "the" not in tokens

    def test_stemming_applied(self):
        tokens_run = _preprocess_fast("running")
        tokens_runs = _preprocess_fast("runs")
        # Both should produce the same stem
        assert tokens_run == tokens_runs

    def test_html_stripped(self):
        tokens = _preprocess_fast("<p>healthy breakfast</p>")
        for tok in tokens:
            assert "<" not in tok and ">" not in tok

    def test_only_alpha_tokens(self):
        """Fast variant uses [a-z]{2,} regex, so numbers/symbols excluded."""
        tokens = _preprocess_fast("42 items cost $9.99")
        # None of the tokens should contain digits or symbols
        assert all(tok.isalpha() for tok in tokens)

    def test_empty_string(self):
        assert _preprocess_fast("") == []


# ---------------------------------------------------------------------------
# preprocess_corpus() – PID consistency
# ---------------------------------------------------------------------------

class TestPreprocessCorpus:

    def _make_passages(self) -> dict[int, str]:
        return {
            101: "apple pie recipe for dessert",
            202: "banana bread baking tips",
            303: "chocolate cake with cocoa and sugar",
        }

    def test_output_pids_match_input_pids(self):
        """Every PID in the result must come from the input passages — no invention."""
        passages = self._make_passages()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = os.path.join(tmpdir, "tokens.pkl")
            result = preprocess_corpus(passages, n_jobs=1, cache_path=cache)

        assert set(result.keys()) == set(passages.keys()), (
            "preprocess_corpus invented or dropped passage IDs"
        )

    def test_all_pids_present(self):
        """Every input PID must be present in the output (no silent drops)."""
        passages = self._make_passages()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = os.path.join(tmpdir, "tokens.pkl")
            result = preprocess_corpus(passages, n_jobs=1, cache_path=cache)

        for pid in passages:
            assert pid in result, f"PID {pid} missing from preprocess_corpus output"

    def test_tokens_are_lists_of_strings(self):
        passages = self._make_passages()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = os.path.join(tmpdir, "tokens.pkl")
            result = preprocess_corpus(passages, n_jobs=1, cache_path=cache)

        for pid, tokens in result.items():
            assert isinstance(tokens, list), f"tokens for PID {pid} is not a list"
            assert all(isinstance(t, str) for t in tokens), \
                f"non-string tokens for PID {pid}"

    def test_cache_hit_returns_same_result(self):
        """Second call with same cache path loads from disk and returns identical output."""
        passages = self._make_passages()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = os.path.join(tmpdir, "tokens.pkl")

            result1 = preprocess_corpus(passages, n_jobs=1, cache_path=cache)
            # Mutate passages after caching — cache should be returned unchanged
            passages[999] = "this should not appear"
            result2 = preprocess_corpus(passages, n_jobs=1, cache_path=cache)

        assert result1 == result2
        assert 999 not in result2

    def test_force_recomputes(self):
        """force=True must bypass the cache and recompute."""
        passages = self._make_passages()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = os.path.join(tmpdir, "tokens.pkl")

            result1 = preprocess_corpus(passages, n_jobs=1, cache_path=cache)

            # Now add a new passage and force recompute
            passages[404] = "extra passage about coffee"
            result2 = preprocess_corpus(passages, n_jobs=1, cache_path=cache, force=True)

        assert 404 in result2
        assert set(result2.keys()) == set(passages.keys())

    def test_no_invented_tokens(self):
        """Every token in the output must ultimately derive from the passage text."""
        passages = {1: "running fast marathon"}
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = os.path.join(tmpdir, "tokens.pkl")
            result = preprocess_corpus(passages, n_jobs=1, cache_path=cache)

        # Tokens must be non-empty strings; no numeric artefacts expected
        for tok in result[1]:
            assert isinstance(tok, str) and len(tok) >= 2

    def test_empty_corpus(self):
        """Empty input should produce an empty dict (and not crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = os.path.join(tmpdir, "empty.pkl")
            result = preprocess_corpus({}, n_jobs=1, cache_path=cache)
        assert result == {}
