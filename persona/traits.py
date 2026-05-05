"""
persona/traits.py — Rule-based personality trait extraction.

Derives 12 personality traits from style statistics and keyword patterns.
All rule-based — no training data or LLM required.

Traits:
    Sentiment:     positive / negative
    Engagement:    curious, expressive, talkative, reserved
    Style:         formal, casual, humorous
    Interpersonal: empathetic, agreeable, supportive
    Cognition:     analytical (hedging+certainty patterns)

Usage:
    from persona.traits import TraitClassifier
    traits = TraitClassifier().classify(style, messages)
"""

from __future__ import annotations

import re
from data.models import Message
from persona.schema import PersonaTrait, PersonaStyle

SUPPORTIVE_PATTERN = re.compile(
    r"\b(you can do it|believe in you|proud of you|here for you|rooting for|"
    r"you got this|keep going|hang in there|you're doing great|don't give up)\b",
    re.IGNORECASE,
)

ANALYTICAL_PATTERN = re.compile(
    r"\b(because|therefore|however|although|whereas|since|considering|"
    r"in fact|on the other hand|which means|that said|as a result|"
    r"for example|for instance|in contrast|nevertheless)\b",
    re.IGNORECASE,
)

SOCIAL_PATTERN = re.compile(
    r"\b(friend|party|hang out|going out|people|everyone|together|"
    r"group|meet up|social|crowd|community|network)\b",
    re.IGNORECASE,
)

SOLO_PATTERN = re.compile(
    r"\b(alone|by myself|on my own|quiet|solo|peaceful|solitude|"
    r"introvert|prefer staying home|like being alone)\b",
    re.IGNORECASE,
)


class TraitClassifier:
    """
    Classifies personality traits from style statistics and keyword patterns.
    Produces PersonaTrait objects with confidence and evidence counts.
    """

    def classify(
        self, style: PersonaStyle, messages: list[Message]
    ) -> list[PersonaTrait]:
        traits: list[PersonaTrait] = []
        texts = [m.text for m in messages]
        full_text = " ".join(texts)
        n = style.total_messages or 1

        # ── Sentiment ────────────────────────────────────────────────────
        if style.avg_sentiment >= 0.15:
            traits.append(PersonaTrait(
                trait="positive",
                confidence=min(1.0, 0.5 + style.positive_rate * 0.5),
                evidence_count=int(style.positive_rate * n),
            ))
        elif style.avg_sentiment <= -0.12:
            traits.append(PersonaTrait(
                trait="negative",
                confidence=min(1.0, 0.5 + style.negative_rate * 0.5),
                evidence_count=int(style.negative_rate * n),
            ))

        # ── Curiosity (high question rate) ───────────────────────────────
        if style.question_rate >= 0.12:
            traits.append(PersonaTrait(
                trait="curious",
                confidence=min(1.0, style.question_rate * 2.5),
                evidence_count=int(style.question_rate * n),
            ))

        # ── Expressiveness (high exclamation rate) ────────────────────────
        if style.exclamation_rate >= 0.15:
            traits.append(PersonaTrait(
                trait="expressive",
                confidence=min(1.0, style.exclamation_rate * 2),
                evidence_count=int(style.exclamation_rate * n),
            ))

        # ── Talkativeness / reserve ───────────────────────────────────────
        if style.avg_message_length >= 20:
            traits.append(PersonaTrait(
                trait="talkative",
                confidence=min(1.0, style.avg_message_length / 40),
                evidence_count=n,
            ))
        elif style.avg_message_length <= 7:
            traits.append(PersonaTrait(
                trait="reserved",
                confidence=min(1.0, 1 - style.avg_message_length / 14),
                evidence_count=n,
            ))

        # ── Formality ────────────────────────────────────────────────────
        if style.formality_score >= 0.62:
            traits.append(PersonaTrait(
                trait="formal",
                confidence=min(1.0, style.formality_score),
                evidence_count=n,
            ))
        elif style.formality_score <= 0.38:
            traits.append(PersonaTrait(
                trait="casual",
                confidence=min(1.0, 1 - style.formality_score),
                evidence_count=n,
            ))

        # ── Humor ────────────────────────────────────────────────────────
        if style.humor_rate >= 0.08:
            traits.append(PersonaTrait(
                trait="humorous",
                confidence=min(1.0, style.humor_rate * 3),
                evidence_count=int(style.humor_rate * n),
            ))

        # ── Empathy ──────────────────────────────────────────────────────
        if style.empathy_rate >= 0.08:
            traits.append(PersonaTrait(
                trait="empathetic",
                confidence=min(1.0, style.empathy_rate * 3),
                evidence_count=int(style.empathy_rate * n),
            ))

        # ── Agreeableness ────────────────────────────────────────────────
        if style.agreement_rate >= 0.05:
            traits.append(PersonaTrait(
                trait="agreeable",
                confidence=min(1.0, style.agreement_rate * 4),
                evidence_count=int(style.agreement_rate * n),
            ))

        # ── Supportiveness (keyword pattern) ─────────────────────────────
        support_hits = len(SUPPORTIVE_PATTERN.findall(full_text))
        support_rate = support_hits / n
        if support_rate >= 0.03:
            traits.append(PersonaTrait(
                trait="supportive",
                confidence=min(1.0, support_rate * 10),
                evidence_count=support_hits,
            ))

        # ── Analytical (complex sentence connectors) ──────────────────────
        analytical_hits = len(ANALYTICAL_PATTERN.findall(full_text))
        analytical_rate = analytical_hits / n
        if analytical_rate >= 0.15:
            traits.append(PersonaTrait(
                trait="analytical",
                confidence=min(1.0, analytical_rate / 2),
                evidence_count=analytical_hits,
            ))

        # ── Social vs introverted ────────────────────────────────────────
        social_hits = len(SOCIAL_PATTERN.findall(full_text))
        solo_hits = len(SOLO_PATTERN.findall(full_text))
        if social_hits > solo_hits * 3 and social_hits / n >= 0.1:
            traits.append(PersonaTrait(
                trait="social",
                confidence=min(1.0, social_hits / n * 2),
                evidence_count=social_hits,
            ))
        elif solo_hits > social_hits and solo_hits / n >= 0.05:
            traits.append(PersonaTrait(
                trait="introverted",
                confidence=min(1.0, solo_hits / n * 5),
                evidence_count=solo_hits,
            ))

        return sorted(traits, key=lambda t: t.confidence, reverse=True)
