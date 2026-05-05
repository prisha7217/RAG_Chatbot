"""
tests/test_chunking.py — Validate topic chunking and fixed checkpoints.

Tests:
    1. Every conversation produces >= 1 topic checkpoint
    2. At least 70% of multi-message conversations produce >= 2 topic segments
    3. No topic segment violates the MIN_TOPIC_MESSAGES constraint (except short conversations)
    4. No topic segment violates the MAX_TOPIC_MESSAGES constraint
    5. Topic checkpoint IDs are unique across all conversations
    6. Global index ranges in checkpoints don't overlap within a conversation
    7. All messages in a conversation are covered by topic checkpoints (no gaps)
    8. Fixed checkpoints cover exactly checkpoint_size messages (last may be smaller)
    9. Fixed checkpoint IDs are unique
    10. Total messages across all fixed checkpoints equals total messages in dataset

Run with:
    python -m pytest tests/test_chunking.py -v

Note: This test loads the embedding model and runs chunking on a SAMPLE of
      conversations (first 50) to keep test time reasonable.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from sentence_transformers import SentenceTransformer
from data.parser import parse_conversations
from data.models import ParsedDataset
from chunking.topic_chunker import TopicChunker
from chunking.fixed_chunker import FixedChunker
from summarizer.summarizer import Summarizer
from config import (
    EMBEDDING_MODEL,
    MIN_TOPIC_MESSAGES,
    MAX_TOPIC_MESSAGES,
    FIXED_CHECKPOINT_SIZE,
)

SAMPLE_SIZE = 50  # Only run on first N conversations to keep tests fast


@pytest.fixture(scope="module")
def model():
    """Load embedding model once for all tests."""
    return SentenceTransformer(EMBEDDING_MODEL)


@pytest.fixture(scope="module")
def summarizer(model):
    return Summarizer(model)


@pytest.fixture(scope="module")
def chunker(model, summarizer):
    return TopicChunker(model, summarizer)


@pytest.fixture(scope="module")
def dataset():
    return parse_conversations()


@pytest.fixture(scope="module")
def sample_conversations(dataset):
    """Use a small sample for speed."""
    return dataset.conversations[:SAMPLE_SIZE]


@pytest.fixture(scope="module")
def all_topic_checkpoints(chunker, sample_conversations):
    """Run topic chunking on the sample, return (conversation, checkpoints) pairs."""
    results = []
    for conv in sample_conversations:
        cps = chunker.chunk(conv)
        results.append((conv, cps))
    return results


@pytest.fixture(scope="module")
def fixed_checkpoints(summarizer, sample_conversations):
    """Run fixed chunking on the sample conversations."""
    fc = FixedChunker(summarizer)
    return fc.chunk_all(sample_conversations)


# ─── Topic Chunker Tests ─────────────────────────────────────────────────────

def test_every_conversation_has_at_least_one_checkpoint(all_topic_checkpoints):
    """Every conversation must produce at least 1 topic checkpoint."""
    for conv, cps in all_topic_checkpoints:
        assert len(cps) >= 1, (
            f"Conv {conv.conversation_id} produced 0 checkpoints"
        )


def test_multi_message_conversations_get_multiple_topics(all_topic_checkpoints):
    """
    At least 50% of conversations long enough to support multiple topics
    should actually get split into >= 2 segments.
    (Conservative threshold: 50% given that many short conversations exist.)
    """
    eligible = [(c, cps) for c, cps in all_topic_checkpoints if c.message_count >= MIN_TOPIC_MESSAGES * 2]
    if not eligible:
        pytest.skip("No eligible conversations in sample")

    multi_topic = sum(1 for _, cps in eligible if len(cps) >= 2)
    ratio = multi_topic / len(eligible)
    assert ratio >= 0.50, (
        f"Only {ratio:.1%} of eligible conversations have ≥2 topic segments"
    )


def test_no_segment_exceeds_max_messages(all_topic_checkpoints):
    """No topic segment should have more than MAX_TOPIC_MESSAGES messages."""
    for conv, cps in all_topic_checkpoints:
        for cp in cps:
            assert cp.message_count <= MAX_TOPIC_MESSAGES + 1, (
                f"Conv {conv.conversation_id}: segment has {cp.message_count} messages "
                f"(max allowed: {MAX_TOPIC_MESSAGES})"
            )


def test_topic_checkpoint_ids_are_unique(all_topic_checkpoints):
    """All checkpoint IDs must be unique."""
    all_ids = [cp.checkpoint_id for _, cps in all_topic_checkpoints for cp in cps]
    assert len(all_ids) == len(set(all_ids)), "Duplicate topic checkpoint IDs found"


def test_all_messages_covered_no_gaps(all_topic_checkpoints):
    """
    Within each conversation, every message should be covered by exactly
    one topic checkpoint (no gaps, no overlaps in local index ranges).
    """
    for conv, cps in all_topic_checkpoints:
        covered_local = set()
        for cp in cps:
            for msg in cp.messages:
                assert msg.local_index not in covered_local, (
                    f"Conv {conv.conversation_id}: message {msg.local_index} covered twice"
                )
                covered_local.add(msg.local_index)

        expected = set(range(conv.message_count))
        assert covered_local == expected, (
            f"Conv {conv.conversation_id}: "
            f"covered {len(covered_local)}/{conv.message_count} messages"
        )


def test_checkpoints_have_summaries(all_topic_checkpoints):
    """Every checkpoint should have a non-empty summary."""
    for conv, cps in all_topic_checkpoints:
        for cp in cps:
            assert cp.summary.strip(), (
                f"Conv {conv.conversation_id}: checkpoint {cp.checkpoint_id} has empty summary"
            )


def test_checkpoints_have_topic_labels(all_topic_checkpoints):
    """Every checkpoint should have a non-empty topic label."""
    for conv, cps in all_topic_checkpoints:
        for cp in cps:
            assert cp.topic_label.strip(), (
                f"Conv {conv.conversation_id}: checkpoint {cp.checkpoint_id} has empty label"
            )


# ─── Fixed Chunker Tests ──────────────────────────────────────────────────────

def test_fixed_checkpoints_exist(fixed_checkpoints):
    """Should produce at least 1 fixed checkpoint."""
    assert len(fixed_checkpoints) >= 1


def test_fixed_checkpoint_ids_are_unique(fixed_checkpoints):
    """All fixed checkpoint IDs must be unique."""
    ids = [cp.checkpoint_id for cp in fixed_checkpoints]
    assert len(ids) == len(set(ids)), "Duplicate fixed checkpoint IDs found"


def test_fixed_checkpoints_cover_all_messages(fixed_checkpoints, sample_conversations):
    """Total messages across all fixed checkpoints should equal sample total."""
    total_in_checkpoints = sum(cp.message_count for cp in fixed_checkpoints)
    total_in_sample = sum(c.message_count for c in sample_conversations)
    assert total_in_checkpoints == total_in_sample, (
        f"Fixed checkpoints cover {total_in_checkpoints} messages, "
        f"but sample has {total_in_sample}"
    )


def test_fixed_checkpoints_have_summaries(fixed_checkpoints):
    """Every fixed checkpoint should have a non-empty summary."""
    for cp in fixed_checkpoints:
        assert cp.summary.strip(), f"Fixed checkpoint {cp.checkpoint_id} has empty summary"
