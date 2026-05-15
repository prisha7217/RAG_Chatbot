"""
persona/schema.py — Pydantic models for the UserPersona system.

A UserPersona is built for each speaker (User 1, User 2) across ALL
conversations in the dataset — not per-conversation. The full history
(~95k messages per speaker) is used for maximum signal.

Captures:
    - facts:          Extracted SVO triples with categories and evidence
    - interests:      LDA-discovered interest topics
    - traits:         Rule-based personality traits
    - style:          Detailed communication style statistics
    - life_events:    Detected life milestones
    - values:         Recurring value themes
    - named_entities: Most-mentioned named entities (places, orgs, people)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict


class PersonaFact(BaseModel):
    """A single extracted factual SVO triple about the speaker."""
    model_config = ConfigDict(frozen=True)

    category: str           # e.g. "occupation", "hobby", "location", "pet", "education"
    subject: str            # Normalized subject ("I", "my family", etc.)
    predicate: str          # Verb/relation ("study", "have", "live in")
    obj: str                # Object ("radiology", "a dog", "Portland")
    raw_text: str           # Original sentence
    speaker: str            # "User 1" or "User 2"
    conversation_id: int    # Source conversation
    confidence: float = 1.0


class PersonaInterest(BaseModel):
    """An interest topic discovered by LDA across all messages."""
    model_config = ConfigDict(frozen=True)

    topic_id: int
    label: str              # Human-readable label (top keywords)
    keywords: list[str]
    weight: float           # Mean topic weight across all messages (0–1)


class PersonaTrait(BaseModel):
    """A personality trait with confidence and evidence count."""
    model_config = ConfigDict(frozen=True)

    trait: str              # e.g. "positive", "curious", "agreeable", "humorous"
    confidence: float       # 0–1
    evidence_count: int     # Number of messages supporting this trait


class PersonaStyle(BaseModel):
    """
    Detailed communication style statistics computed across all messages.

    Covers: message length, punctuation, sentiment, formality, vocabulary
    richness, self-disclosure, hedging, certainty, emoji usage, and more.
    """
    model_config = ConfigDict(frozen=True)

    # ── Message length ───────────────────────────────────────────────────
    avg_message_length: float       # Average words per message
    median_message_length: float
    max_message_length: int         # Longest message (words)
    pct_short_messages: float       # % messages with < 5 words (replies/filler)
    pct_long_messages: float        # % messages with > 30 words (detailed)

    # ── Punctuation / expression ─────────────────────────────────────────
    question_rate: float            # Fraction of messages ending with '?'
    exclamation_rate: float         # Fraction of messages ending with '!'
    ellipsis_rate: float            # Fraction containing '...' (trailing thought)
    emoji_rate: float               # Fraction containing emoji characters
    caps_rate: float                # Fraction of all-caps words (emphasis)

    # ── Sentiment ────────────────────────────────────────────────────────
    avg_sentiment: float            # VADER compound score (-1 to 1)
    positive_rate: float
    negative_rate: float
    neutral_rate: float
    sentiment_variance: float       # How much mood varies (high = emotionally variable)

    # ── Language style ───────────────────────────────────────────────────
    formality_score: float          # 0 = casual, 1 = formal
    vocabulary_richness: float      # Type-token ratio (unique words / total words)
    self_disclosure_rate: float     # Fraction of messages with "I" statements
    hedging_rate: float             # Use of uncertain language ("maybe", "I think")
    certainty_rate: float           # Use of certain language ("definitely", "I know")
    humor_rate: float               # Use of humor indicators ("lol", "haha", "funny")
    agreement_rate: float           # Use of agreement words ("same", "agree", "exactly")
    empathy_rate: float             # Use of empathy language ("sorry", "I understand")

    total_messages: int


class PersonaLifeEvent(BaseModel):
    """A detected life milestone mentioned in the conversations."""
    model_config = ConfigDict(frozen=True)

    event_type: str         # "marriage", "birth", "graduation", "job_change", "moving", "loss"
    description: str        # The triggering sentence
    conversation_id: int
    confidence: float


class PersonaNamedEntity(BaseModel):
    """A frequently mentioned named entity."""
    model_config = ConfigDict(frozen=True)

    entity_type: str        # "GPE" (location), "ORG", "PERSON", "WORK_OF_ART", etc.
    text: str               # The entity text
    mention_count: int      # How many times mentioned


class PersonaValues(BaseModel):
    """Recurring value themes detected from keyword frequency."""
    model_config = ConfigDict(frozen=True)

    family_focus: float     # 0–1: how much they talk about family
    career_focus: float     # 0–1: how much they talk about work/career
    health_focus: float     # 0–1: how much they talk about health/fitness
    social_focus: float     # 0–1: how much they talk about friends/socializing
    creative_focus: float   # 0–1: how much they talk about art/music/creativity
    nature_focus: float     # 0–1: how much they talk about outdoors/nature
    spiritual_focus: float  # 0–1: religious/spiritual language
    intellectual_focus: float  # 0–1: reading/learning/education language

class ConversationPersona(BaseModel):
    """
    Lightweight persona extracted from a SINGLE conversation.

    Because each conversation is one distinct pair of people, facts here
    are internally consistent — no cross-person contradictions.
    Indexed by conversation_id for fast lookup during retrieval.
    """

    conversation_id: int

    # Per-speaker facts within this conversation
    user1_facts: list[PersonaFact] = []
    user2_facts: list[PersonaFact] = []

    # Per-speaker life events within this conversation
    user1_events: list[PersonaLifeEvent] = []
    user2_events: list[PersonaLifeEvent] = []

    # Key named entities mentioned (shared across speakers)
    named_entities: list[PersonaNamedEntity] = []

    def facts_for(self, speaker: str) -> list[PersonaFact]:
        """Return facts for the given speaker ('User 1' or 'User 2')."""
        return self.user1_facts if speaker == "User 1" else self.user2_facts

    def to_context_string(self) -> str:
        """Format conversation-specific persona context for the LLM."""
        lines = [f"=== CONVERSATION {self.conversation_id} PERSONAS ==="]

        for speaker, facts in [("User 1", self.user1_facts), ("User 2", self.user2_facts)]:
            if not facts:
                continue
            lines.append(f"\n  {speaker} facts in this conversation:")
            by_cat: dict[str, list[str]] = {}
            for f in facts[:10]:
                by_cat.setdefault(f.category, []).append(f"{f.predicate} {f.obj}".strip())
            for cat, items in by_cat.items():
                lines.append(f"    [{cat}] {' | '.join(items[:3])}")

        for speaker, events in [("User 1", self.user1_events), ("User 2", self.user2_events)]:
            if events:
                lines.append(f"\n  {speaker} life events:")
                for ev in events[:2]:
                    lines.append(f"    [{ev.event_type}] {ev.description[:80]}")

        if self.named_entities:
            places = [e.text for e in self.named_entities if e.entity_type == "GPE"][:3]
            if places:
                lines.append(f"\n  Places mentioned: {', '.join(places)}")

        return "\n".join(lines)


# ─── Round 2: Drift Detection ─────────────────────────────────────────────────

class SegmentMood(BaseModel):
    """
    Mood snapshot for a single topic segment within a conversation.
    Computed from VADER sentiment + style stats on the segment's messages.
    """
    model_config = ConfigDict(frozen=True)

    segment_index: int          # Position of this segment within the conversation (0-based)
    checkpoint_id: str          # Links back to TopicCheckpoint.checkpoint_id
    topic_label: str            # Topic label from the chunker (e.g. "Pets & Animals")
    message_count: int

    # Sentiment
    avg_sentiment: float        # VADER compound, averaged across segment messages (-1 to 1)
    sentiment_label: str        # "very_positive" | "positive" | "neutral" | "negative" | "very_negative"

    # Style stats
    question_rate: float        # Fraction of messages with '?'
    exclamation_rate: float     # Fraction of messages with '!'
    avg_message_length: float   # Avg words per message

    # Human-readable mood summary (e.g. "curious & formal")
    mood_label: str


class DriftEvent(BaseModel):
    """
    A detected mood/tone shift between two adjacent topic segments.

    Produced when either:
    - Sentiment delta between segment[i] and segment[i+1] > DRIFT_SENTIMENT_THRESHOLD
    - A key style metric (question rate, exclamation rate) changes beyond threshold
    """
    model_config = ConfigDict(frozen=True)

    from_segment_index: int     # Segment where the drift originates
    to_segment_index: int       # Segment where the new mood begins
    from_mood: str              # Mood label of the earlier segment
    to_mood: str                # Mood label of the later segment
    sentiment_delta: float      # Signed change in avg_sentiment (positive = more positive)
    drift_type: str             # "sentiment" | "style" | "both"

    # What caused the drift — the topic label + keywords of the new segment
    trigger_topic: str
    trigger_keywords: list[str] = []

    # Human-readable summary
    description: str            # e.g. "Mood shifted from curious & formal → warm & expressive"


class ConversationTimeline(BaseModel):
    """
    Full drift timeline for a single conversation.

    Contains:
    - One SegmentMood per topic segment (the temporal sequence)
    - Zero or more DriftEvents (detected mood shifts between segments)

    Stored per conversation_id in outputs/persona/drift_timelines.json.
    """

    conversation_id: int
    speaker: str                # "User 1" or "User 2"
    segment_count: int
    drift_event_count: int

    segments: list[SegmentMood]
    drift_events: list[DriftEvent]

    def to_context_string(self) -> str:
        """Format drift timeline as a readable string for LLM context."""
        if not self.segments:
            return ""

        lines = [
            f"  Mood timeline for {self.speaker} "
            f"({self.segment_count} segments, {self.drift_event_count} drift events):"
        ]
        for seg in self.segments:
            lines.append(
                f"    Seg {seg.segment_index + 1} [{seg.topic_label}]: "
                f"{seg.mood_label} | sentiment={seg.avg_sentiment:+.2f}"
            )
        for evt in self.drift_events:
            lines.append(
                f"    ⟶ Drift at seg {evt.from_segment_index + 1}→{evt.to_segment_index + 1}: "
                f"{evt.from_mood} → {evt.to_mood} "
                f"(trigger: {evt.trigger_topic})"
            )
        return "\n".join(lines)



class UserPersona(BaseModel):
    """
    Aggregate persona for a speaker label across ALL conversations.

    Contains only signals that aggregate meaningfully across different people:
    - Style statistics (computed on all ~95k messages)
    - Personality traits (computed from aggregated style stats)
    - Value focus scores (keyword frequency across population)
    - LDA interest topics (valid across population)

    Facts and life events are intentionally excluded here — they live in
    ConversationPersona, since each conversation is a different person.
    """

    speaker: str
    total_conversations: int
    total_messages: int

    # Aggregate signals only (no facts — those are per-conversation)
    interests: list[PersonaInterest] = []
    traits: list[PersonaTrait] = []
    style: Optional[PersonaStyle] = None
    values: Optional[PersonaValues] = None

    @property
    def top_traits(self) -> list[PersonaTrait]:
        return sorted(self.traits, key=lambda t: t.confidence, reverse=True)

    def to_context_string(self) -> str:
        """Format aggregate persona context for the LLM."""
        lines = [f"=== AGGREGATE PERSONA: {self.speaker} ==="]
        lines.append(f"Analyzed {self.total_messages:,} messages across {self.total_conversations:,} conversations.\n")

        if self.traits:
            top_t = [f"{t.trait} ({t.confidence:.0%})" for t in self.top_traits[:5]]
            lines.append(f"Personality tendencies: {', '.join(top_t)}")

        if self.values:
            v = self.values
            value_scores = {
                "family": v.family_focus, "career": v.career_focus,
                "health": v.health_focus, "social": v.social_focus,
                "creative": v.creative_focus, "nature": v.nature_focus,
            }
            top_values = sorted(value_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(f"Core values: {', '.join(f'{k} ({sv:.0%})' for k, sv in top_values if sv > 0.1)}")

        if self.interests:
            top = sorted(self.interests, key=lambda x: x.weight, reverse=True)[:4]
            lines.append(f"Common interests: {', '.join(i.label for i in top)}")

        if self.style:
            s = self.style
            lines.append(f"Avg message: {s.avg_message_length:.0f} words | "
                         f"Sentiment: {s.avg_sentiment:+.2f} | "
                         f"Questions: {s.question_rate:.0%}")

        return "\n".join(lines)
