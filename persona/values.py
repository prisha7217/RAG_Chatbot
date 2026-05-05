"""
persona/values.py — Value theme extraction from keyword frequency analysis.

Computes 8 value focus scores (0–1) by measuring how much of the speaker's
vocabulary falls into each thematic keyword cluster.

Values computed:
    family_focus      — family, children, parents, relationships
    career_focus      — work, job, career, business, money
    health_focus      — health, fitness, exercise, diet, wellness
    social_focus      — friends, socializing, parties, community
    creative_focus    — art, music, writing, creativity, design
    nature_focus      — outdoors, hiking, nature, animals, environment
    spiritual_focus   — religion, god, faith, spirituality, prayer
    intellectual_focus — reading, learning, education, books, knowledge

Usage:
    from persona.values import ValuesAnalyzer
    values = ValuesAnalyzer().analyze(messages)
"""

from __future__ import annotations

import re
from data.models import Message
from persona.schema import PersonaValues

# ─── Keyword clusters per value dimension ────────────────────────────────────

VALUE_KEYWORDS: dict[str, list[str]] = {
    "family": [
        "family", "mom", "dad", "mother", "father", "sister", "brother",
        "son", "daughter", "kids", "children", "child", "baby", "husband",
        "wife", "parents", "grandma", "grandpa", "aunt", "uncle", "nephew",
        "niece", "spouse", "partner", "home", "household",
    ],
    "career": [
        "work", "job", "career", "boss", "office", "business", "company",
        "salary", "money", "income", "profession", "employer", "employee",
        "promotion", "interview", "resume", "startup", "client", "project",
        "deadline", "meeting", "colleague", "coworker", "manager",
    ],
    "health": [
        "health", "healthy", "fitness", "exercise", "workout", "gym",
        "diet", "nutrition", "eat", "food", "sleep", "rest", "doctor",
        "hospital", "medicine", "medication", "sick", "illness", "weight",
        "running", "yoga", "mental health", "therapy", "wellness",
    ],
    "social": [
        "friend", "friends", "party", "hang out", "hangout", "social",
        "people", "everyone", "together", "group", "event", "gathering",
        "bar", "club", "night out", "dating", "relationship", "meet",
        "community", "neighbor", "network",
    ],
    "creative": [
        "art", "music", "painting", "drawing", "writing", "poetry",
        "creative", "design", "craft", "photography", "dance", "sing",
        "song", "guitar", "piano", "instrument", "sculpture", "film",
        "movie", "theater", "theatre", "sketch", "novel", "story",
    ],
    "nature": [
        "nature", "outdoors", "outdoor", "hiking", "hike", "camping",
        "forest", "mountain", "beach", "ocean", "park", "garden",
        "gardening", "plant", "tree", "animal", "wildlife", "environment",
        "eco", "sustainable", "bird", "dog", "cat", "pet",
    ],
    "spiritual": [
        "god", "faith", "pray", "prayer", "church", "religion", "religious",
        "spiritual", "spirituality", "bible", "mosque", "temple", "worship",
        "belief", "believe", "soul", "blessing", "meditation", "mindful",
        "universe", "grateful", "gratitude",
    ],
    "intellectual": [
        "read", "reading", "book", "books", "learn", "learning", "study",
        "knowledge", "education", "school", "university", "college",
        "research", "science", "history", "philosophy", "documentary",
        "podcast", "curious", "think", "theory", "debate", "news",
    ],
}

# Pre-compile patterns for speed
_COMPILED = {
    dim: re.compile(
        r"\b(" + "|".join(re.escape(kw) for kw in kws) + r")\b",
        re.IGNORECASE,
    )
    for dim, kws in VALUE_KEYWORDS.items()
}


class ValuesAnalyzer:
    """
    Computes 8 value focus scores from a speaker's complete message history.
    Scores are normalized so they represent relative emphasis, not absolute frequency.
    """

    def analyze(self, messages: list[Message]) -> PersonaValues:
        if not messages:
            return PersonaValues(
                family_focus=0.0, career_focus=0.0, health_focus=0.0,
                social_focus=0.0, creative_focus=0.0, nature_focus=0.0,
                spiritual_focus=0.0, intellectual_focus=0.0,
            )

        full_text = " ".join(m.text for m in messages)
        n_words = max(len(full_text.split()), 1)

        # Count hits per dimension per 100 words (normalized frequency)
        raw_scores: dict[str, float] = {}
        for dim, pattern in _COMPILED.items():
            hits = len(pattern.findall(full_text))
            raw_scores[dim] = hits / n_words * 100  # hits per 100 words

        # Scale to 0–1: divide by a reasonable max (5 hits/100 words = 1.0)
        MAX_RATE = 5.0
        scaled = {dim: min(1.0, score / MAX_RATE) for dim, score in raw_scores.items()}

        return PersonaValues(
            family_focus=round(scaled["family"], 4),
            career_focus=round(scaled["career"], 4),
            health_focus=round(scaled["health"], 4),
            social_focus=round(scaled["social"], 4),
            creative_focus=round(scaled["creative"], 4),
            nature_focus=round(scaled["nature"], 4),
            spiritual_focus=round(scaled["spiritual"], 4),
            intellectual_focus=round(scaled["intellectual"], 4),
        )
