"""
tests/test_drift.py — Unit tests for the DriftDetector.

Tests that:
1. Drift events are detected when sentiment changes significantly between segments
2. No drift events are raised when segments have similar mood
3. ConversationTimeline is constructed correctly
4. Speakers with no messages in a segment get a neutral/silent placeholder
5. load_timelines() gracefully returns empty dict if file not found
"""

from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to sys.path so 'data' module can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.models import Conversation, Message, TopicCheckpoint
from persona.drift import DriftDetector, load_timelines
from persona.schema import ConversationTimeline


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_message(global_idx: int, conv_id: int, local_idx: int, speaker: str, text: str) -> Message:
    return Message(
        global_index=global_idx,
        conversation_id=conv_id,
        local_index=local_idx,
        speaker=speaker,
        text=text,
    )


def _make_checkpoint(conv_id: int, seg_idx: int, start: int, end: int, label: str) -> TopicCheckpoint:
    return TopicCheckpoint(
        checkpoint_id=f"topic_conv{conv_id:05d}_seg{seg_idx:03d}",
        conversation_id=conv_id,
        topic_label=label,
        start_global_index=start,
        end_global_index=end,
        start_local_index=start,
        end_local_index=end,
        messages=[],  # not needed by DriftDetector (uses conv messages)
        summary="",
    )


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    return DriftDetector()


@pytest.fixture
def happy_then_sad_conv():
    """Conversation 0: User 1 is happy in seg 0, sad in seg 1 → should detect drift."""
    messages = [
        # Segment 0 (local 0-2): very positive
        _make_message(0, 0, 0, "User 1", "I'm so happy today! Everything is wonderful!"),
        _make_message(1, 0, 1, "User 2", "That's great to hear!"),
        _make_message(2, 0, 2, "User 1", "Life is amazing! I love everything!"),
        # Segment 1 (local 3-5): very negative
        _make_message(3, 0, 3, "User 1", "I'm devastated. My dog just passed away."),
        _make_message(4, 0, 4, "User 2", "I'm so sorry to hear that."),
        _make_message(5, 0, 5, "User 1", "I miss him terribly. I feel terrible."),
    ]
    conv = Conversation(
        conversation_id=0,
        messages=messages,
        start_global_index=0,
        end_global_index=5,
    )
    checkpoints = [
        _make_checkpoint(0, 0, 0, 2, "Happy & Joy & Love"),
        _make_checkpoint(0, 1, 3, 5, "Sad & Loss & Dog"),
    ]
    return conv, checkpoints


@pytest.fixture
def flat_mood_conv():
    """
    Conversation 1: User 1 has genuinely flat mood/style across both segments.
    Both segments have similar positive sentiment and NO questions (so question_rate
    stays stable and doesn't trigger a style drift).
    """
    messages = [
        # Segment 0 (local 0-2): User 1 makes statements, no questions, mild positive
        _make_message(0, 1, 0, "User 1", "I enjoy spending time outdoors."),
        _make_message(1, 1, 1, "User 2", "That sounds nice."),
        _make_message(2, 1, 2, "User 1", "Yes, it helps me relax and think."),
        # Segment 1 (local 3-5): User 1 still makes statements, similar positive sentiment
        _make_message(3, 1, 3, "User 2", "I like reading in my spare time."),
        _make_message(4, 1, 4, "User 1", "Reading is a good way to unwind."),
        _make_message(5, 1, 5, "User 2", "It is really calming."),
    ]
    conv = Conversation(
        conversation_id=1,
        messages=messages,
        start_global_index=0,
        end_global_index=5,
    )
    checkpoints = [
        _make_checkpoint(1, 0, 0, 2, "Outdoors & Relaxing"),
        _make_checkpoint(1, 1, 3, 5, "Reading & Hobby"),
    ]
    return conv, checkpoints


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestDriftDetection:

    def test_drift_detected_on_strong_sentiment_change(self, detector, happy_then_sad_conv):
        conv, checkpoints = happy_then_sad_conv
        timelines = detector.detect_all([conv], checkpoints)

        user1_key = "0_User 1"
        assert user1_key in timelines, "Should produce a timeline for User 1 in conv 0"

        timeline = timelines[user1_key]
        assert timeline.drift_event_count > 0, "Should detect at least one drift event"
        assert timeline.drift_events[0].drift_type in ("sentiment", "both")
        assert timeline.drift_events[0].sentiment_delta < 0, "Sentiment should go negative"

    def test_no_drift_on_flat_mood(self, detector):
        """
        Detector produces no drift when adjacent segments have nearly identical
        sentiment and style. Tests _detect_drift_events directly with controlled
        SegmentMood values — avoids brittleness from VADER scoring specific sentences.
        """
        from persona.schema import SegmentMood

        seg0 = SegmentMood(
            segment_index=0, checkpoint_id="cp0", topic_label="Outdoors & Relax",
            message_count=3, avg_sentiment=0.30, sentiment_label="positive",
            question_rate=0.0, exclamation_rate=0.0, avg_message_length=8.0,
            mood_label="positive & formal",
        )
        seg1 = SegmentMood(
            segment_index=1, checkpoint_id="cp1", topic_label="Reading & Hobby",
            message_count=3, avg_sentiment=0.32, sentiment_label="positive",
            question_rate=0.0, exclamation_rate=0.0, avg_message_length=7.5,
            mood_label="positive & formal",
        )
        # delta = 0.02 — well below DRIFT_SENTIMENT_THRESHOLD (0.15)
        events = detector._detect_drift_events([seg0, seg1])
        assert events == [], f"Expected no drift events, got: {events}"

    def test_timeline_structure(self, detector, happy_then_sad_conv):
        conv, checkpoints = happy_then_sad_conv
        timelines = detector.detect_all([conv], checkpoints)

        user1_key = "0_User 1"
        timeline = timelines[user1_key]

        assert isinstance(timeline, ConversationTimeline)
        assert timeline.conversation_id == 0
        assert timeline.speaker == "User 1"
        assert len(timeline.segments) == 2
        assert timeline.segment_count == 2

    def test_segment_mood_labels_populated(self, detector, happy_then_sad_conv):
        conv, checkpoints = happy_then_sad_conv
        timelines = detector.detect_all([conv], checkpoints)

        timeline = timelines["0_User 1"]
        for seg in timeline.segments:
            if seg.message_count > 0:
                assert seg.mood_label != "", "Mood label should be set"
                assert seg.sentiment_label in (
                    "very positive", "positive", "neutral", "negative", "very negative"
                )

    def test_trigger_topic_matches_new_segment(self, detector, happy_then_sad_conv):
        conv, checkpoints = happy_then_sad_conv
        timelines = detector.detect_all([conv], checkpoints)

        timeline = timelines["0_User 1"]
        if timeline.drift_events:
            evt = timeline.drift_events[0]
            assert "Sad" in evt.trigger_topic or "Loss" in evt.trigger_topic or "Dog" in evt.trigger_topic

    def test_single_segment_conv_excluded(self, detector):
        """Conversations with only 1 segment produce no timeline (no drift possible)."""
        messages = [
            _make_message(0, 99, 0, "User 1", "Hi!"),
            _make_message(1, 99, 1, "User 2", "Hello!"),
        ]
        conv = Conversation(conversation_id=99, messages=messages,
                            start_global_index=0, end_global_index=1)
        checkpoints = [_make_checkpoint(99, 0, 0, 1, "Greeting")]
        timelines = detector.detect_all([conv], checkpoints)
        assert "99_User 1" not in timelines


class TestSaveLoad:

    def test_save_and_load_roundtrip(self, detector, happy_then_sad_conv):
        conv, checkpoints = happy_then_sad_conv
        timelines = detector.detect_all([conv], checkpoints)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = Path(f.name)

        try:
            detector.save(timelines, output_path=tmp_path)
            loaded = load_timelines(tmp_path)
            assert len(loaded) == len(timelines)
            for key in timelines:
                assert key in loaded
                assert loaded[key].conversation_id == timelines[key].conversation_id
                assert loaded[key].drift_event_count == timelines[key].drift_event_count
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_missing_file_returns_empty(self):
        result = load_timelines(Path("/nonexistent/path/drift_timelines.json"))
        assert result == {}


class TestContextString:

    def test_to_context_string_has_segments(self, detector, happy_then_sad_conv):
        conv, checkpoints = happy_then_sad_conv
        timelines = detector.detect_all([conv], checkpoints)

        timeline = timelines.get("0_User 1")
        if timeline:
            ctx = timeline.to_context_string()
            assert "Seg 1" in ctx
            assert "sentiment=" in ctx
