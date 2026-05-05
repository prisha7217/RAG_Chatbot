"""
persona/style.py — Detailed communication style analysis per speaker.

Computes 18 statistics across the speaker's complete message history:
    Message length: avg, median, max, pct_short, pct_long
    Punctuation:    question_rate, exclamation_rate, ellipsis_rate, emoji_rate, caps_rate
    Sentiment:      avg, positive_rate, negative_rate, neutral_rate, sentiment_variance
    Language:       formality, vocabulary_richness, self_disclosure, hedging,
                    certainty, humor, agreement, empathy

Usage:
    from persona.style import StyleAnalyzer
    analyzer = StyleAnalyzer()
    style = analyzer.analyze(messages)
"""

from __future__ import annotations

import re
import statistics

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data.models import Message
from persona.schema import PersonaStyle

# ─── Pattern definitions ──────────────────────────────────────────────────────

EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)

INFORMAL_WORDS = re.compile(
    r"\b(lol|omg|haha|hehe|yeah|yep|nope|gonna|wanna|kinda|sorta|dunno|gotta|"
    r"tbh|idk|imo|btw|ngl|fyi|brb|lmao|u\b|ur\b|r\b|thx|ty|gr8|b4)\b",
    re.IGNORECASE,
)

FORMAL_WORDS = re.compile(
    r"\b(furthermore|however|therefore|nevertheless|consequently|"
    r"regarding|additionally|whereas|albeit|notwithstanding|"
    r"accordingly|henceforth|hereby|pursuant|therein)\b",
    re.IGNORECASE,
)

HEDGING_WORDS = re.compile(
    r"\b(maybe|perhaps|possibly|probably|might|could|guess|think|suppose|"
    r"seem|appears|roughly|around|sort of|kind of|i think|i believe|"
    r"not sure|uncertain|i wonder)\b",
    re.IGNORECASE,
)

CERTAINTY_WORDS = re.compile(
    r"\b(definitely|absolutely|certainly|always|never|clearly|obviously|"
    r"for sure|without doubt|i know|i'm sure|guaranteed|exactly|of course)\b",
    re.IGNORECASE,
)

HUMOR_WORDS = re.compile(
    r"\b(lol|haha|hehe|lmao|funny|hilarious|joke|jk|just kidding|"
    r"😂|😄|😆|😁|🤣)\b",
    re.IGNORECASE,
)

AGREEMENT_WORDS = re.compile(
    r"\b(same|agree|agreed|exactly|right|totally|absolutely|yes|yep|"
    r"i know|me too|same here|for sure|true|that's true|definitely)\b",
    re.IGNORECASE,
)

EMPATHY_WORDS = re.compile(
    r"\b(sorry|understand|feel|tough|hard|difficult|hang in|you okay|"
    r"here for you|that must|must be|sounds like|i hear you|that sucks|"
    r"hope you feel|take care|thinking of you)\b",
    re.IGNORECASE,
)

SELF_DISCLOSURE_PATTERN = re.compile(r"\bI\b", re.UNICODE)


class StyleAnalyzer:
    """
    Computes 18 communication style statistics from a speaker's messages.
    Uses VADER for sentiment (rule-based, no model download needed).
    """

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    def analyze(self, messages: list[Message]) -> PersonaStyle:
        if not messages:
            return self._empty_style()

        texts = [m.text for m in messages]
        n = len(texts)
        full_text = " ".join(texts)

        # ── Message length ───────────────────────────────────────────────
        lengths = [len(t.split()) for t in texts]
        avg_len = statistics.mean(lengths)
        median_len = statistics.median(lengths)
        max_len = max(lengths)
        pct_short = sum(1 for l in lengths if l < 5) / n
        pct_long = sum(1 for l in lengths if l > 30) / n

        # ── Punctuation ──────────────────────────────────────────────────
        question_rate = sum(1 for t in texts if t.strip().endswith("?")) / n
        exclamation_rate = sum(1 for t in texts if t.strip().endswith("!")) / n
        ellipsis_rate = sum(1 for t in texts if "..." in t) / n
        emoji_rate = sum(1 for t in texts if EMOJI_PATTERN.search(t)) / n

        # CAPS: fraction of all words that are fully uppercase (len > 1 to exclude "I")
        all_words = full_text.split()
        caps_words = [w for w in all_words if w.isupper() and len(w) > 1]
        caps_rate = len(caps_words) / max(len(all_words), 1)

        # ── Sentiment ────────────────────────────────────────────────────
        scores = [self._vader.polarity_scores(t)["compound"] for t in texts]
        avg_sentiment = statistics.mean(scores)
        sentiment_variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        positive_rate = sum(1 for s in scores if s >= 0.05) / n
        negative_rate = sum(1 for s in scores if s <= -0.05) / n
        neutral_rate = 1.0 - positive_rate - negative_rate

        # ── Formality ────────────────────────────────────────────────────
        total_words = max(len(all_words), 1)
        informal_rate = len(INFORMAL_WORDS.findall(full_text)) / total_words * 100
        formal_rate = len(FORMAL_WORDS.findall(full_text)) / total_words * 100
        formality_score = min(1.0, max(0.0, 0.5 + (formal_rate - informal_rate) / 10))

        # ── Vocabulary richness (type-token ratio on sample) ─────────────
        # Use first 10,000 words to avoid TTR dropping artificially on huge corpora
        sample_words = [w.lower() for w in all_words[:10_000] if w.isalpha()]
        vocabulary_richness = len(set(sample_words)) / max(len(sample_words), 1)

        # ── Language feature rates ────────────────────────────────────────
        self_disclosure_rate = sum(
            1 for t in texts if SELF_DISCLOSURE_PATTERN.search(t)
        ) / n
        hedging_rate = len(HEDGING_WORDS.findall(full_text)) / total_words * 10
        certainty_rate = len(CERTAINTY_WORDS.findall(full_text)) / total_words * 10
        humor_rate = sum(1 for t in texts if HUMOR_WORDS.search(t)) / n
        agreement_rate = len(AGREEMENT_WORDS.findall(full_text)) / total_words * 10
        empathy_rate = sum(1 for t in texts if EMPATHY_WORDS.search(t)) / n

        return PersonaStyle(
            avg_message_length=round(avg_len, 2),
            median_message_length=round(median_len, 2),
            max_message_length=max_len,
            pct_short_messages=round(pct_short, 3),
            pct_long_messages=round(pct_long, 3),
            question_rate=round(question_rate, 3),
            exclamation_rate=round(exclamation_rate, 3),
            ellipsis_rate=round(ellipsis_rate, 3),
            emoji_rate=round(emoji_rate, 3),
            caps_rate=round(caps_rate, 4),
            avg_sentiment=round(avg_sentiment, 4),
            positive_rate=round(positive_rate, 3),
            negative_rate=round(negative_rate, 3),
            neutral_rate=round(max(0.0, neutral_rate), 3),
            sentiment_variance=round(sentiment_variance, 4),
            formality_score=round(formality_score, 3),
            vocabulary_richness=round(vocabulary_richness, 4),
            self_disclosure_rate=round(self_disclosure_rate, 3),
            hedging_rate=round(min(1.0, hedging_rate), 4),
            certainty_rate=round(min(1.0, certainty_rate), 4),
            humor_rate=round(humor_rate, 3),
            agreement_rate=round(min(1.0, agreement_rate), 4),
            empathy_rate=round(empathy_rate, 3),
            total_messages=n,
        )

    def _empty_style(self) -> PersonaStyle:
        return PersonaStyle(
            avg_message_length=0.0, median_message_length=0.0,
            max_message_length=0, pct_short_messages=0.0, pct_long_messages=0.0,
            question_rate=0.0, exclamation_rate=0.0, ellipsis_rate=0.0,
            emoji_rate=0.0, caps_rate=0.0,
            avg_sentiment=0.0, positive_rate=0.0, negative_rate=0.0,
            neutral_rate=0.0, sentiment_variance=0.0,
            formality_score=0.5, vocabulary_richness=0.0,
            self_disclosure_rate=0.0, hedging_rate=0.0, certainty_rate=0.0,
            humor_rate=0.0, agreement_rate=0.0, empathy_rate=0.0,
            total_messages=0,
        )
