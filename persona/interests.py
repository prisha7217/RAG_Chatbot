"""
persona/interests.py — LDA-based interest topic discovery.

Usage:
    from persona.interests import InterestExtractor
    extractor = InterestExtractor()
    interests = extractor.extract(messages)
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from data.models import Message
from persona.schema import PersonaInterest
from config import LDA_N_TOPICS, LDA_MAX_FEATURES

logger = logging.getLogger(__name__)

INTEREST_STOPWORDS = {
    "hey", "hi", "hello", "yeah", "yep", "nope", "okay", "ok",
    "really", "great", "awesome", "nice", "cool", "good", "doing",
    "thanks", "thank", "sure", "thing", "things", "just", "got",
    "like", "love", "want", "know", "think", "feel", "make", "little",
    "lot", "bit", "going", "come", "went", "way", "pretty", "kind",
    "time", "day", "week", "year", "life", "right", "definitely",
    "actually", "basically", "guess", "maybe", "probably", "sounds",
    "that", "this", "yes", "no", "well", "also", "even", "still",
    # Sentiment/filler words that pollute LDA topics
    "glad", "fun", "appreciate", "wonderful", "amazing", "fantastic",
    "happy", "enjoy", "enjoying", "enjoyed", "favorite", "favourite",
    "love", "loving", "liked", "hoping", "hope", "wish", "feel",
    "try", "trying", "tried", "new", "getting", "hear", "heard",
    "learn", "learned", "learning", "play", "played", "playing",
    "help", "helped", "helping", "want", "wanted", "need", "needed",
    "look", "looks", "looking", "make", "makes", "making",
}


class InterestExtractor:
    def __init__(self, n_topics: int = LDA_N_TOPICS, max_features: int = LDA_MAX_FEATURES):
        self._n_topics = n_topics
        self._max_features = max_features

    def extract(self, messages: list[Message], top_words: int = 5) -> list[PersonaInterest]:
        texts = [m.text for m in messages]
        if len(texts) < 5:
            return []
        try:
            return self._fit_lda(texts, top_words)
        except Exception as e:
            logger.warning(f"LDA fitting failed: {e}")
            return []

    def _fit_lda(self, texts: list[str], top_words: int) -> list[PersonaInterest]:
        import sklearn.feature_extraction.text as sft
        combined_stop = list(set(sft.ENGLISH_STOP_WORDS) | INTEREST_STOPWORDS)

        vectorizer = CountVectorizer(
            max_features=self._max_features,
            stop_words=combined_stop,
            token_pattern=r"[a-zA-Z]{3,}",
            min_df=2,
        )
        try:
            doc_term_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return []

        actual_topics = min(self._n_topics, max(2, doc_term_matrix.shape[1] // 2))
        lda = LatentDirichletAllocation(
            n_components=actual_topics, random_state=42, max_iter=10, learning_method="online"
        )
        try:
            doc_topics = lda.fit_transform(doc_term_matrix)
        except Exception as e:
            logger.warning(f"LDA transform failed: {e}")
            return []

        feature_names = vectorizer.get_feature_names_out()
        mean_weights = doc_topics.mean(axis=0)

        interests = []
        for topic_idx in range(actual_topics):
            top_indices = lda.components_[topic_idx].argsort()[-top_words:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            label = " & ".join(kw.title() for kw in keywords[:3])
            interests.append(PersonaInterest(
                topic_id=topic_idx,
                label=label,
                keywords=list(keywords),
                weight=round(float(mean_weights[topic_idx]), 4),
            ))

        return sorted(
            [i for i in interests if i.weight > 0.01],
            key=lambda x: x.weight, reverse=True
        )
