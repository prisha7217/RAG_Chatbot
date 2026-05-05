"""
tests/test_retrieval.py — Validate ChromaDB index and retrieval quality.

Tests:
    1. All 3 collections exist in the index
    2. Topic collection has expected number of entries (≈ 38,932)
    3. Fixed collection has expected number of entries (≈ 1,916)
    4. Chunk collection has entries
    5. Query returns results from all 3 collections
    6. Retrieval scores are in valid range (0.0 – 1.0)
    7. Known-fact retrieval: querying "radiology student" retrieves relevant content
    8. Known-fact retrieval: querying "1964 Impala" retrieves relevant content
    9. Results are sorted by score descending
    10. to_context_string() produces non-empty output

Run with:
    python -m pytest tests/test_retrieval.py -v

Note: Requires the index to be built first (python main.py build then index).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import chromadb

from retrieval.embedder import Embedder
from retrieval.retriever import Retriever, RetrievalContext
from config import (
    INDEX_DIR,
    CHROMA_COLLECTION_TOPICS,
    CHROMA_COLLECTION_FIXED,
    CHROMA_COLLECTION_CHUNKS,
)


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture(scope="module")
def retriever(embedder):
    return Retriever(embedder)


@pytest.fixture(scope="module")
def chroma_client():
    return chromadb.PersistentClient(path=str(INDEX_DIR))


# ─── Collection Existence ─────────────────────────────────────────────────────

def test_topic_collection_exists(chroma_client):
    names = [c.name for c in chroma_client.list_collections()]
    assert CHROMA_COLLECTION_TOPICS in names, f"Collection '{CHROMA_COLLECTION_TOPICS}' not found"


def test_fixed_collection_exists(chroma_client):
    names = [c.name for c in chroma_client.list_collections()]
    assert CHROMA_COLLECTION_FIXED in names, f"Collection '{CHROMA_COLLECTION_FIXED}' not found"


def test_chunk_collection_exists(chroma_client):
    names = [c.name for c in chroma_client.list_collections()]
    assert CHROMA_COLLECTION_CHUNKS in names, f"Collection '{CHROMA_COLLECTION_CHUNKS}' not found"


# ─── Collection Sizes ─────────────────────────────────────────────────────────

def test_topic_collection_has_entries(chroma_client):
    col = chroma_client.get_collection(CHROMA_COLLECTION_TOPICS)
    count = col.count()
    assert count >= 10000, f"Expected ≥10,000 topic entries, got {count:,}"


def test_fixed_collection_has_entries(chroma_client):
    col = chroma_client.get_collection(CHROMA_COLLECTION_FIXED)
    count = col.count()
    assert count >= 1000, f"Expected ≥1,000 fixed entries, got {count:,}"


def test_chunk_collection_has_entries(chroma_client):
    col = chroma_client.get_collection(CHROMA_COLLECTION_CHUNKS)
    count = col.count()
    assert count >= 10000, f"Expected ≥10,000 chunk entries, got {count:,}"


# ─── Retrieval Functionality ──────────────────────────────────────────────────

def test_retrieve_returns_results_from_all_collections(retriever):
    context = retriever.retrieve("What hobbies do people enjoy?")
    assert len(context.topic_results) > 0, "No topic results returned"
    assert len(context.fixed_results) > 0, "No fixed results returned"
    assert len(context.chunk_results) > 0, "No chunk results returned"


def test_retrieval_scores_in_valid_range(retriever):
    context = retriever.retrieve("Does anyone have a pet dog?")
    for r in context.all_results:
        assert 0.0 <= r.score <= 1.0, (
            f"Score {r.score} out of range for {r.doc_id}"
        )


def test_results_sorted_by_score_descending(retriever):
    context = retriever.retrieve("cooking and food")
    all_results = context.all_results
    for i in range(1, len(all_results)):
        assert all_results[i].score <= all_results[i - 1].score, (
            f"Results not sorted: {all_results[i-1].score} → {all_results[i].score}"
        )


# ─── Known-Fact Retrieval ─────────────────────────────────────────────────────

def test_radiology_student_retrieved(retriever):
    """
    Conv 1 has 'User 1: I'm a fulltime student studying radiology'.
    Querying for this should retrieve relevant content with a reasonable score.
    """
    context = retriever.retrieve("radiology student college")
    top = context.all_results[0] if context.all_results else None
    assert top is not None, "No results returned"
    assert top.score >= 0.3, f"Low score for known fact: {top.score:.3f}"

    # The text should mention radiology somewhere in top results
    all_text = " ".join(r.text.lower() for r in context.all_results[:10])
    assert "radiology" in all_text or "student" in all_text, (
        "Expected 'radiology' or 'student' in top 10 results"
    )


def test_impala_classic_car_retrieved(retriever):
    """
    Conv 1 has '1964 Impala' conversation.
    """
    context = retriever.retrieve("1964 Impala classic car")
    assert context.all_results, "No results returned"

    all_text = " ".join(r.text.lower() for r in context.all_results[:10])
    assert "impala" in all_text or "classic" in all_text, (
        "Expected 'impala' or 'classic' in top 10 results"
    )


# ─── Context String ───────────────────────────────────────────────────────────

def test_context_string_is_non_empty(retriever):
    context = retriever.retrieve("What do people like to do on weekends?")
    ctx_str = context.to_context_string()
    assert len(ctx_str) > 100, "Context string is too short"
    assert "Query:" in ctx_str, "Context string missing Query header"
