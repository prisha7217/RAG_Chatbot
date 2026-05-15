"""
tests/test_conflict_resolver.py — Unit tests for the ConflictResolver.

Tests that:
1. Composite score re-ranks results higher when entity overlap is strong
2. Sentiment spread contradiction is flagged correctly
3. Negation clash contradiction is flagged correctly
4. Results with no shared entities are not flagged
5. ResolvedContext.contradiction_summary() is non-empty when flags exist
6. to_context_string() marks conflicting chunks with ⚠CONFLICT
7. No crashes on empty result lists
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.retriever import RetrievalContext, RetrievalResult
from retrieval.conflict_resolver import ConflictResolver, ResolvedContext


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_result(
    doc_id: str,
    text: str,
    score: float = 0.80,
    conv_id: int = 0,
    source: str = "chunk",
) -> RetrievalResult:
    return RetrievalResult(
        source=source,
        doc_id=doc_id,
        text=text,
        score=score,
        metadata={"conversation_id": conv_id},
    )


def _make_ctx(chunks: list[RetrievalResult], query: str = "test query") -> RetrievalContext:
    return RetrievalContext(
        query=query,
        topic_results=[],
        fixed_results=[],
        chunk_results=chunks,
    )


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def resolver():
    return ConflictResolver()


@pytest.fixture
def positive_chunk():
    return _make_result(
        "chunk_pos",
        "User 1 absolutely loves dogs and has a wonderful dog named Max.",
        score=0.75, conv_id=100,
    )


@pytest.fixture
def negative_chunk():
    return _make_result(
        "chunk_neg",
        "User 1 does not have a dog and never liked pets at all.",
        score=0.70, conv_id=200,
    )


@pytest.fixture
def neutral_chunk():
    return _make_result(
        "chunk_neu",
        "User 2 enjoys reading books and going to the library.",
        score=0.80, conv_id=50,
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestCompositeScoring:

    def test_entity_rich_chunk_ranks_higher(self, resolver):
        """A chunk with strong entity overlap with query should outscore one without."""
        dog_query = "does user 1 have a dog"
        dog_chunk = _make_result("dog", "User 1 loves dogs and has a dog named Max.", score=0.70, conv_id=1)
        off_topic = _make_result("off", "User 2 enjoys cooking pasta and baking.", score=0.80, conv_id=2)

        ctx = _make_ctx([dog_chunk, off_topic], query=dog_query)
        resolved = resolver.resolve(ctx, dog_query)

        dog_result = next(r for r in resolved.chunk_results if r.doc_id == "dog")
        off_result  = next(r for r in resolved.chunk_results if r.doc_id == "off")

        assert dog_result.score >= off_result.score, (
            f"Dog chunk ({dog_result.score:.3f}) should score >= off-topic ({off_result.score:.3f})"
        )

    def test_original_cosine_preserved_in_metadata(self, resolver, positive_chunk):
        ctx = _make_ctx([positive_chunk])
        resolved = resolver.resolve(ctx)
        result = resolved.chunk_results[0]
        assert "original_cosine" in result.metadata
        assert result.metadata["original_cosine"] == pytest.approx(0.75, abs=0.01)

    def test_empty_results_no_crash(self, resolver):
        ctx = _make_ctx([])
        resolved = resolver.resolve(ctx, "anything")
        assert resolved.chunk_results == []
        assert resolved.contradiction_flags == []


class TestContradictionDetection:

    def test_sentiment_spread_flagged(self, resolver, positive_chunk, negative_chunk):
        """
        positive_chunk: 'loves dogs' (positive VADER)
        negative_chunk: 'does not have a dog, never liked pets' (negative VADER)
        Both share entity 'dogs'/'dog'/'pets' — should trigger sentiment_spread flag.
        """
        ctx = _make_ctx([positive_chunk, negative_chunk], query="does user 1 have a dog")
        resolved = resolver.resolve(ctx)

        spread_flags = [f for f in resolved.contradiction_flags if f.conflict_type == "sentiment_spread"]
        assert len(spread_flags) >= 1, "Expected at least one sentiment_spread contradiction"

    def test_negation_clash_flagged(self, resolver, positive_chunk, negative_chunk):
        """
        'does not have a dog' should trigger a negation clash against 'loves dogs'.
        """
        ctx = _make_ctx([positive_chunk, negative_chunk], query="does user 1 have a dog")
        resolved = resolver.resolve(ctx)

        negation_flags = [f for f in resolved.contradiction_flags if f.conflict_type == "negation"]
        assert len(negation_flags) >= 1, "Expected at least one negation clash flag"

    def test_no_contradiction_when_no_shared_entities(self, resolver, positive_chunk, neutral_chunk):
        """
        positive_chunk talks about dogs, neutral_chunk talks about reading.
        No shared entities → no contradiction even if sentiments differ.
        """
        ctx = _make_ctx([positive_chunk, neutral_chunk], query="books and reading")
        resolved = resolver.resolve(ctx)
        # May or may not have flags depending on entity overlap — but dogs vs books should not clash
        dog_book_flags = [
            f for f in resolved.contradiction_flags
            if "dog" in f.entity or "book" in f.entity
        ]
        # They don't share entities so no dog/book contradiction
        assert all("dog" not in f.entity for f in dog_book_flags)

    def test_has_contradictions_property(self, resolver, positive_chunk, negative_chunk):
        ctx = _make_ctx([positive_chunk, negative_chunk], query="does user 1 have a dog")
        resolved = resolver.resolve(ctx)
        if resolved.contradiction_flags:
            assert resolved.has_contradictions is True
        else:
            assert resolved.has_contradictions is False

    def test_contradiction_ids_reference_real_chunks(self, resolver, positive_chunk, negative_chunk):
        ctx = _make_ctx([positive_chunk, negative_chunk], query="dog")
        resolved = resolver.resolve(ctx)
        known_ids = {"chunk_pos", "chunk_neg"}
        for flag in resolved.contradiction_flags:
            assert flag.result_a_id in known_ids, f"Unexpected result_a_id: {flag.result_a_id}"
            assert flag.result_b_id in known_ids, f"Unexpected result_b_id: {flag.result_b_id}"


class TestResolvedContext:

    def test_contradiction_summary_non_empty_when_flags_exist(self, resolver, positive_chunk, negative_chunk):
        ctx = _make_ctx([positive_chunk, negative_chunk], query="does user 1 have a dog")
        resolved = resolver.resolve(ctx)
        if resolved.has_contradictions:
            summary = resolved.contradiction_summary()
            assert len(summary) > 0
            assert "contradiction" in summary.lower()

    def test_to_context_string_marks_conflicting_chunks(self, resolver, positive_chunk, negative_chunk):
        ctx = _make_ctx([positive_chunk, negative_chunk], query="does user 1 have a dog")
        resolved = resolver.resolve(ctx)
        ctx_str = resolved.to_context_string()
        if resolved.has_contradictions:
            assert "⚠CONFLICT" in ctx_str

    def test_to_context_string_contains_query(self, resolver, positive_chunk):
        ctx = _make_ctx([positive_chunk], query="unique test query xyz")
        resolved = resolver.resolve(ctx)
        assert "unique test query xyz" in resolved.to_context_string()

    def test_all_results_sorted_descending(self, resolver, positive_chunk, negative_chunk, neutral_chunk):
        ctx = _make_ctx([positive_chunk, negative_chunk, neutral_chunk])
        resolved = resolver.resolve(ctx)
        scores = [r.score for r in resolved.all_results]
        assert scores == sorted(scores, reverse=True)


class TestEntityExtraction:
    """Test the _extract_entities helper directly."""

    def test_stopwords_excluded(self, resolver):
        entities = resolver._extract_entities("the user is not happy")
        assert "the" not in entities
        assert "not" not in entities
        assert "is" not in entities

    def test_meaningful_words_included(self, resolver):
        entities = resolver._extract_entities("User 1 loves dogs and hiking")
        assert "loves" in entities or "dogs" in entities or "hiking" in entities

    def test_empty_text(self, resolver):
        entities = resolver._extract_entities("")
        assert entities == set()


class TestNegationDetection:
    """Test the _is_negated helper directly."""

    def test_negation_detected(self, resolver):
        assert resolver._is_negated("dog", "user does not have a dog") is True

    def test_no_negation_when_absent(self, resolver):
        assert resolver._is_negated("dog", "user loves their dog") is False

    def test_negation_window_respected(self, resolver):
        # Negation word is 10 tokens away — should NOT trigger (window is ±3)
        text = "never mind that other thing because user has a wonderful dog"
        assert resolver._is_negated("dog", text) is False
