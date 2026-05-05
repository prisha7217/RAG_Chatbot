"""
summarizer/summarizer.py — IDF-weighted extractive summarization.

Uses IDF (Inverse Document Frequency) to score each message by how many
rare/informative words it contains. High-IDF messages (e.g. "radiology",
"1964 Impala", "punk band") are selected over generic filler
("hey", "doing great", "that's awesome").

Topic labels are generated from the same IDF scores — highest-IDF keywords
in the segment become the label.

Usage:
    from summarizer.summarizer import Summarizer
    s = Summarizer(embedding_model)
    summary = s.summarize(messages, top_k=3)
    label   = s.topic_label(messages, n_keywords=4)
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data.models import Message
from config import SUMMARY_TOP_K, TOP_K_KEYWORDS

logger = logging.getLogger(__name__)

# Conversational filler words to exclude from topic labels and scoring.
# These are common across almost ALL conversations and carry no topic signal.
CONVERSATIONAL_STOPWORDS = {
    "hey", "hi", "hello", "yeah", "yep", "nope", "okay", "ok",
    "really", "great", "awesome", "nice", "cool", "good", "doing",
    "thanks", "thank", "sure", "thing", "things", "just", "got",
    "like", "love", "want", "know", "think", "feel", "make", "little",
    "lot", "bit", "going", "come", "went", "way", "pretty", "kind",
    "time", "day", "week", "year", "life", "right", "definitely",
    "actually", "basically", "guess", "maybe", "probably", "sounds",
}


class Summarizer:
    """
    IDF-weighted extractive summarizer.

    Scores each message by the sum of IDF values of its words.
    Messages containing rare, informative words (e.g. "radiology",
    "impala", "yoga", "band") score higher than generic ones
    ("hey", "doing well", "that's awesome").

    Args:
        model: A loaded SentenceTransformer model instance (kept for
               interface compatibility with TopicChunker — not used
               for summarization itself).
    """

    def __init__(self, model):
        self.model = model  # Kept for compatibility; used by TopicChunker

    def _build_tfidf(self, texts: list[str]):
        """
        Fit a TF-IDF vectorizer on the segment's messages.
        Returns the fitted vectorizer, or None if fitting fails
        (e.g. all words filtered out by stopwords/min-length).
        """
        import sklearn.feature_extraction.text as sft
        english_stop = set(sft.ENGLISH_STOP_WORDS)
        combined_stop = list(english_stop | CONVERSATIONAL_STOPWORDS)

        tfidf = TfidfVectorizer(
            max_features=300,
            stop_words=combined_stop,
            ngram_range=(1, 1),
            token_pattern=r"[a-zA-Z]{3,}",  # Min 3 chars — no 'll', 've', 're'
        )
        try:
            tfidf.fit(texts)
            # Verify vocabulary is non-empty
            if not tfidf.vocabulary_:
                return None
            return tfidf
        except Exception:
            return None

    def summarize(self, messages: list[Message], top_k: int = SUMMARY_TOP_K) -> str:
        """
        Select top-k messages by IDF-weighted informativeness score.
        Falls back to first top_k messages if TF-IDF fitting fails.
        """
        texts = [m.text for m in messages]
        n = len(texts)

        if n == 0:
            return ""
        if n <= top_k:
            return " | ".join(f"{messages[i].speaker}: {texts[i]}" for i in range(n))

        tfidf = self._build_tfidf(texts)

        if tfidf is not None:
            try:
                tfidf_matrix = tfidf.transform(texts).toarray()
                idf_values = tfidf.idf_
                message_scores = tfidf_matrix.dot(idf_values)
                top_indices = sorted(np.argsort(message_scores)[-top_k:].tolist())
                return " | ".join(f"{messages[i].speaker}: {texts[i]}" for i in top_indices)
            except Exception:
                pass

        # Fallback: return messages at even intervals (covers start, middle, end)
        step = max(1, n // top_k)
        fallback_indices = list(range(0, n, step))[:top_k]
        return " | ".join(f"{messages[i].speaker}: {texts[i]}" for i in fallback_indices)

    def topic_label(self, messages: list[Message], n_keywords: int = 4) -> str:
        """
        Generate a topic label from the highest-IDF keywords in the segment.
        Falls back to 'General Conversation' if TF-IDF fitting fails.
        """
        texts = [m.text for m in messages]
        if not texts:
            return "Unknown Topic"

        tfidf = self._build_tfidf(texts)

        if tfidf is None:
            return "General Conversation"

        try:
            tfidf_matrix = tfidf.transform(texts).toarray()
            feature_names = tfidf.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0)
            top_indices = np.argsort(scores)[-n_keywords:][::-1]
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]

            if not keywords:
                return "General Conversation"

            return " & ".join(kw.title() for kw in keywords[:n_keywords])

        except Exception as e:
            logger.warning(f"TF-IDF topic label failed: {e}")
            return "General Conversation"

