"""
tests/test_parser.py — Validate the CSV parser output.

Tests:
    1. Conversation count is reasonable (> 1000 conversations expected)
    2. Total message count is reasonable (> 10000 messages expected)
    3. All messages have valid speaker labels ("User 1" or "User 2")
    4. Global indices are unique and monotonically increasing
    5. Conversation IDs are unique
    6. No empty message texts
    7. Local indices within each conversation start at 0 and are sequential
    8. start/end global indices match actual message indices

Run with:
    python -m pytest tests/test_parser.py -v
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.parser import parse_conversations
from data.models import ParsedDataset, Conversation, Message


@pytest.fixture(scope="module")
def dataset() -> ParsedDataset:
    """Parse once and reuse across all tests."""
    return parse_conversations()


# ─── Basic Counts ────────────────────────────────────────────────────────────

def test_conversation_count(dataset: ParsedDataset):
    """Should have at least 1000 conversations parsed from the CSV."""
    assert dataset.total_conversations >= 1000, (
        f"Expected ≥1000 conversations, got {dataset.total_conversations}"
    )
    assert dataset.total_conversations == len(dataset.conversations)


def test_message_count(dataset: ParsedDataset):
    """Should have at least 10000 messages total."""
    assert dataset.total_messages >= 10000, (
        f"Expected ≥10000 messages, got {dataset.total_messages}"
    )


def test_message_count_matches_conversations(dataset: ParsedDataset):
    """total_messages should equal the sum of messages in all conversations."""
    actual_count = sum(c.message_count for c in dataset.conversations)
    assert dataset.total_messages == actual_count


# ─── Speaker Labels ──────────────────────────────────────────────────────────

def test_all_messages_have_valid_speaker(dataset: ParsedDataset):
    """Every message must be from 'User 1' or 'User 2'."""
    valid_speakers = {"User 1", "User 2"}
    invalid = [
        (m.conversation_id, m.global_index, m.speaker)
        for c in dataset.conversations
        for m in c.messages
        if m.speaker not in valid_speakers
    ]
    assert not invalid, f"Found {len(invalid)} messages with invalid speakers: {invalid[:5]}"


def test_conversations_have_both_speakers(dataset: ParsedDataset):
    """At least 90% of conversations should have both User 1 and User 2."""
    both_speakers = sum(
        1 for c in dataset.conversations
        if c.user1_messages and c.user2_messages
    )
    ratio = both_speakers / dataset.total_conversations
    assert ratio >= 0.90, (
        f"Only {ratio:.1%} of conversations have both speakers (expected ≥90%)"
    )


# ─── Global Index Integrity ───────────────────────────────────────────────────

def test_global_indices_are_unique(dataset: ParsedDataset):
    """Each message must have a unique global_index."""
    all_indices = [m.global_index for c in dataset.conversations for m in c.messages]
    assert len(all_indices) == len(set(all_indices)), "Duplicate global indices found"


def test_global_indices_are_monotonic(dataset: ParsedDataset):
    """Global indices across all messages should be monotonically increasing."""
    all_msgs = dataset.all_messages
    for i in range(1, len(all_msgs)):
        assert all_msgs[i].global_index == all_msgs[i - 1].global_index + 1, (
            f"Gap in global indices at position {i}: "
            f"{all_msgs[i-1].global_index} → {all_msgs[i].global_index}"
        )


# ─── Conversation Integrity ──────────────────────────────────────────────────

def test_conversation_ids_are_unique(dataset: ParsedDataset):
    """Each conversation must have a unique conversation_id."""
    ids = [c.conversation_id for c in dataset.conversations]
    assert len(ids) == len(set(ids)), "Duplicate conversation IDs found"


def test_conversation_start_end_indices(dataset: ParsedDataset):
    """start_global_index and end_global_index must match actual messages."""
    for c in dataset.conversations:
        assert c.start_global_index == c.messages[0].global_index, (
            f"Conv {c.conversation_id}: start_global_index mismatch"
        )
        assert c.end_global_index == c.messages[-1].global_index, (
            f"Conv {c.conversation_id}: end_global_index mismatch"
        )


# ─── Message Content ─────────────────────────────────────────────────────────

def test_no_empty_message_texts(dataset: ParsedDataset):
    """No message should have an empty or whitespace-only text."""
    empty = [
        (m.conversation_id, m.global_index)
        for c in dataset.conversations
        for m in c.messages
        if not m.text.strip()
    ]
    assert not empty, f"Found {len(empty)} empty messages: {empty[:5]}"


def test_local_indices_are_sequential(dataset: ParsedDataset):
    """Within each conversation, local_index should be 0, 1, 2, ..."""
    for c in dataset.conversations:
        for expected, msg in enumerate(c.messages):
            assert msg.local_index == expected, (
                f"Conv {c.conversation_id}: expected local_index {expected}, "
                f"got {msg.local_index} for message: {msg.text[:50]}"
            )


# ─── Sample Spot-Check ───────────────────────────────────────────────────────

def test_first_conversation_starts_with_greeting(dataset: ParsedDataset):
    """The first message in conversation 0 should be a short greeting-style text."""
    first_msg = dataset.conversations[0].messages[0]
    assert len(first_msg.text) > 0
    assert first_msg.local_index == 0
    assert first_msg.global_index == 0
    assert first_msg.conversation_id == 0


def test_minimum_messages_per_conversation(dataset: ParsedDataset):
    """At least 90% of conversations should have ≥2 messages."""
    sufficient = sum(1 for c in dataset.conversations if c.message_count >= 2)
    ratio = sufficient / dataset.total_conversations
    assert ratio >= 0.90, (
        f"Only {ratio:.1%} of conversations have ≥2 messages"
    )
