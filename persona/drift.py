"""
persona/drift.py — Adaptive Persona Drift Detector.

Detects how a user's mood and communication style shifts across topic segments
within a conversation. Each topic segment is treated as a "time step" — the
natural temporal proxy when no timestamps are available.

For each conversation and speaker, produces a ConversationTimeline:
    Segment 1 [Moving & City]     → curious & positive  | sentiment +0.42
    Segment 2 [Books & Powell]    → warm & expressive    | sentiment +0.61
    Segment 3 [Goodbye & Thanks]  → friendly & neutral   | sentiment +0.18
    ⟶ Drift at seg 2→3: warm & expressive → friendly & neutral
       (trigger: "Goodbye & Thanks")

Usage (standalone):
    from persona.drift import DriftDetector
    detector = DriftDetector()
    timelines = detector.detect_all(conversations, topic_checkpoints)
    detector.save(timelines)

Usage via main.py:
    python main.py drift
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data.models import Conversation, Message, TopicCheckpoint
from persona.schema import (
    SegmentMood,
    DriftEvent,
    ConversationTimeline,
)
from config import (
    DRIFT_SENTIMENT_THRESHOLD,
    DRIFT_STYLE_THRESHOLD,
    DRIFT_TIMELINES_FILE,
    CHECKPOINTS_DIR,
)

logger = logging.getLogger(__name__)

# Mood label building blocks
_SENTIMENT_LABELS = [
    (0.40,  "very positive"),
    (0.15,  "positive"),
    (-0.05, "neutral"),
    (-0.25, "negative"),
    (-1.01, "very negative"),
]

_STYLE_ADJECTIVES = {
    "curious":    lambda seg: seg.question_rate >= 0.25,
    "expressive": lambda seg: seg.exclamation_rate >= 0.20,
    "talkative":  lambda seg: seg.avg_message_length >= 18,
    "reserved":   lambda seg: seg.avg_message_length <= 5,
    "formal":     lambda seg: seg.question_rate < 0.10 and seg.exclamation_rate < 0.05,
}

SPEAKERS = ["User 1", "User 2"]


def _sentiment_label(compound: float) -> str:
    for threshold, label in _SENTIMENT_LABELS:
        if compound >= threshold:
            return label
    return "very negative"


def _style_adjective(seg: SegmentMood) -> str:
    """Pick the most prominent style descriptor for this segment."""
    for adj, condition in _STYLE_ADJECTIVES.items():
        if condition(seg):
            return adj
    return "conversational"


def _mood_label(seg: SegmentMood) -> str:
    """Combine sentiment and style into a human-readable mood label."""
    return f"{_sentiment_label(seg.avg_sentiment)} & {_style_adjective(seg)}"


def _extract_keywords(topic_label: str) -> list[str]:
    """Split a topic label like 'Pets & Dog & Best & Cat' into keywords."""
    return [kw.strip().lower() for kw in topic_label.split("&") if kw.strip()]


class DriftDetector:
    """
    Detects mood/tone drift across topic segments within conversations.

    Pipeline:
    1. For each topic segment: slice messages by local_index range
    2. Compute VADER sentiment + style stats over those messages
    3. Build SegmentMood objects
    4. Compare adjacent segments per speaker → detect DriftEvents
    5. Assemble ConversationTimeline per (conversation_id, speaker)
    """

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()
        logger.info("DriftDetector initialised with VADER sentiment analyser.")

    # ─── Public API ───────────────────────────────────────────────────────────

    def detect_all(
        self,
        conversations: list[Conversation],
        topic_checkpoints: list[TopicCheckpoint],
    ) -> dict[str, ConversationTimeline]:
        """
        Run drift detection over all conversations.

        Args:
            conversations:      Parsed conversations (contains all messages).
            topic_checkpoints:  TopicCheckpoint objects with local_index ranges.

        Returns:
            Dict mapping "{conversation_id}_{speaker}" → ConversationTimeline.
            (e.g. "42_User 1", "42_User 2")
        """
        # Build fast lookup: conv_id → Conversation
        conv_map: dict[int, Conversation] = {c.conversation_id: c for c in conversations}

        # Group checkpoints by conversation_id, sorted by start_local_index
        cp_by_conv: dict[int, list[TopicCheckpoint]] = defaultdict(list)
        for cp in topic_checkpoints:
            cp_by_conv[cp.conversation_id].append(cp)
        for cid in cp_by_conv:
            cp_by_conv[cid].sort(key=lambda c: c.start_local_index)

        timelines: dict[str, ConversationTimeline] = {}
        total = len(cp_by_conv)

        for i, (conv_id, checkpoints) in enumerate(cp_by_conv.items()):
            conv = conv_map.get(conv_id)
            if conv is None or len(checkpoints) < 2:
                # Need at least 2 segments to detect drift
                continue

            for speaker in SPEAKERS:
                timeline = self._build_timeline(conv, checkpoints, speaker)
                if timeline:
                    key = f"{conv_id}_{speaker}"
                    timelines[key] = timeline

            if (i + 1) % 2000 == 0:
                logger.info(f"  Drift detection: {i+1:,}/{total:,} conversations processed...")

        logger.info(
            f"Drift detection complete: "
            f"{len(timelines):,} timelines produced "
            f"({sum(1 for t in timelines.values() if t.drift_event_count > 0):,} with drift events)."
        )
        return timelines

    def save(
        self,
        timelines: dict[str, ConversationTimeline],
        output_path: Path = DRIFT_TIMELINES_FILE,
    ) -> Path:
        """Serialise all timelines to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {key: tl.model_dump() for key, tl in timelines.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Drift timelines saved to: {output_path} ({len(timelines):,} entries)")
        return output_path

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _build_timeline(
        self,
        conv: Conversation,
        checkpoints: list[TopicCheckpoint],
        speaker: str,
    ) -> Optional[ConversationTimeline]:
        """Build a ConversationTimeline for one (conversation, speaker) pair."""
        # Slice messages per segment for this speaker
        segment_moods: list[SegmentMood] = []

        for seg_idx, cp in enumerate(checkpoints):
            # Filter messages in this segment's local index range for this speaker
            seg_messages = [
                m for m in conv.messages
                if (cp.start_local_index <= m.local_index <= cp.end_local_index
                    and m.speaker == speaker)
            ]
            if not seg_messages:
                # Speaker has no messages in this segment — create a neutral placeholder
                # so segment indices stay aligned
                placeholder = self._empty_mood(seg_idx, cp)
                segment_moods.append(placeholder)
                continue

            mood = self._compute_segment_mood(seg_idx, cp, seg_messages)
            segment_moods.append(mood)

        # Detect drift events between adjacent segments
        drift_events = self._detect_drift_events(segment_moods)

        # Only return timelines where the speaker actually participated in ≥2 segments
        active_segments = [s for s in segment_moods if s.message_count > 0]
        if len(active_segments) < 2:
            return None

        return ConversationTimeline(
            conversation_id=conv.conversation_id,
            speaker=speaker,
            segment_count=len(segment_moods),
            drift_event_count=len(drift_events),
            segments=segment_moods,
            drift_events=drift_events,
        )

    def _compute_segment_mood(
        self,
        seg_idx: int,
        cp: TopicCheckpoint,
        messages: list[Message],
    ) -> SegmentMood:
        """Compute VADER sentiment + style stats for one speaker's messages in a segment."""
        texts = [m.text for m in messages]
        n = len(texts)

        # VADER compound per message, then average
        compounds = [
            self._vader.polarity_scores(t)["compound"]
            for t in texts
        ]
        avg_sentiment = sum(compounds) / n if compounds else 0.0

        # Style stats
        question_rate = sum(1 for t in texts if "?" in t) / n
        exclamation_rate = sum(1 for t in texts if "!" in t) / n
        avg_message_length = sum(len(t.split()) for t in texts) / n

        # Build the SegmentMood (mood_label filled in after construction)
        partial = SegmentMood(
            segment_index=seg_idx,
            checkpoint_id=cp.checkpoint_id,
            topic_label=cp.topic_label or f"Segment {seg_idx}",
            message_count=n,
            avg_sentiment=round(avg_sentiment, 4),
            sentiment_label=_sentiment_label(avg_sentiment),
            question_rate=round(question_rate, 4),
            exclamation_rate=round(exclamation_rate, 4),
            avg_message_length=round(avg_message_length, 2),
            mood_label="",  # filled below
        )
        # Mood label needs the object itself for the style adjective
        mood = _mood_label(partial)
        return partial.model_copy(update={"mood_label": mood})

    def _empty_mood(self, seg_idx: int, cp: TopicCheckpoint) -> SegmentMood:
        """Neutral placeholder for a segment where this speaker had no messages."""
        return SegmentMood(
            segment_index=seg_idx,
            checkpoint_id=cp.checkpoint_id,
            topic_label=cp.topic_label or f"Segment {seg_idx}",
            message_count=0,
            avg_sentiment=0.0,
            sentiment_label="neutral",
            question_rate=0.0,
            exclamation_rate=0.0,
            avg_message_length=0.0,
            mood_label="silent",
        )

    def _detect_drift_events(
        self,
        segments: list[SegmentMood],
    ) -> list[DriftEvent]:
        """
        Compare adjacent SegmentMoods and emit DriftEvents where significant
        mood or style changes are detected.

        A drift event is emitted when:
        - |sentiment_delta| > DRIFT_SENTIMENT_THRESHOLD, OR
        - |question_rate change| or |exclamation_rate change| > DRIFT_STYLE_THRESHOLD
        """
        events: list[DriftEvent] = []

        for i in range(len(segments) - 1):
            prev = segments[i]
            curr = segments[i + 1]

            # Skip if either segment has no speaker messages
            if prev.message_count == 0 or curr.message_count == 0:
                continue

            sentiment_delta = curr.avg_sentiment - prev.avg_sentiment
            question_delta = abs(curr.question_rate - prev.question_rate)
            exclamation_delta = abs(curr.exclamation_rate - prev.exclamation_rate)

            sentiment_drift = abs(sentiment_delta) > DRIFT_SENTIMENT_THRESHOLD
            style_drift = (
                question_delta > DRIFT_STYLE_THRESHOLD
                or exclamation_delta > DRIFT_STYLE_THRESHOLD
            )

            if not (sentiment_drift or style_drift):
                continue

            if sentiment_drift and style_drift:
                drift_type = "both"
            elif sentiment_drift:
                drift_type = "sentiment"
            else:
                drift_type = "style"

            direction = "more positive" if sentiment_delta > 0 else "more negative"
            description = (
                f"Mood shifted from '{prev.mood_label}' → '{curr.mood_label}' "
                f"({direction} by {abs(sentiment_delta):.2f})"
            )

            events.append(DriftEvent(
                from_segment_index=i,
                to_segment_index=i + 1,
                from_mood=prev.mood_label,
                to_mood=curr.mood_label,
                sentiment_delta=round(sentiment_delta, 4),
                drift_type=drift_type,
                trigger_topic=curr.topic_label,
                trigger_keywords=_extract_keywords(curr.topic_label),
                description=description,
            ))

        return events


# ─── Convenience loader (for serve phase) ─────────────────────────────────────

def load_timelines(
    path: Path = DRIFT_TIMELINES_FILE,
) -> dict[str, ConversationTimeline]:
    """
    Load drift timelines from JSON at serve startup.
    Returns empty dict if file not found (graceful degradation).
    """
    if not path.exists():
        logger.warning(
            f"drift_timelines.json not found at {path}. "
            "Run 'python main.py drift' to generate it."
        )
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    timelines = {}
    for key, data in raw.items():
        try:
            timelines[key] = ConversationTimeline(**data)
        except Exception as e:
            logger.warning(f"Failed to parse timeline {key}: {e}")
    logger.info(f"Loaded {len(timelines):,} drift timelines from {path}")
    return timelines
