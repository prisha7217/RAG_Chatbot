"""
chunking/topic_chunker.py — Hybrid topic boundary detection.

Algorithm:
    Primary signal  (70%): Cosine similarity between a sliding window of
                           past messages and the next message.
    Secondary signal(30%): TF-IDF keyword overlap (Jaccard) between the
                           same window and the next message.

Safeguards:
    - Adaptive threshold per conversation (mean - 0.5 * std of scores)
    - Min segment size: >= MIN_TOPIC_MESSAGES (default 3)
    - Max segment size: <= MAX_TOPIC_MESSAGES (default 15), forces a split
    - Validation log: warns if a conversation produced only 1 topic segment

Usage:
    from chunking.topic_chunker import TopicChunker
    chunker = TopicChunker(embedding_model, summarizer)
    checkpoints = chunker.chunk(conversation)
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from chunking.base import BaseChunker
from data.models import Conversation, Message, TopicCheckpoint
from summarizer.summarizer import Summarizer
from config import (
    WINDOW_SIZE,
    COSINE_WEIGHT,
    KEYWORD_WEIGHT,
    ADAPTIVE_THRESHOLD_FACTOR,
    MIN_TOPIC_MESSAGES,
    MAX_TOPIC_MESSAGES,
    TOP_K_KEYWORDS,
)

logger = logging.getLogger(__name__)


class TopicChunker(BaseChunker):
    """
    Hybrid cosine-similarity + TF-IDF keyword shift topic chunker.

    Args:
        model: A loaded SentenceTransformer model (shared with Summarizer).
        summarizer: A Summarizer instance for generating summaries and labels.
        window_size: Sliding window size in messages.
    """

    def __init__(self, model, summarizer: Summarizer, window_size: int = WINDOW_SIZE):
        self.model = model
        self.summarizer = summarizer
        self.window_size = window_size

    def chunk(self, conversation: Conversation) -> list[TopicCheckpoint]:
        """
        Detect topic boundaries in a single conversation and return
        a list of TopicCheckpoint objects, one per topic segment.
        """
        messages = conversation.messages
        n = len(messages)

        # Very short conversations: return as single segment
        if n < self.window_size + 1:
            return [self._make_checkpoint(conversation, messages, seg_index=0)]

        texts = [m.text for m in messages]

        # ── Step 1: Embed all messages ────────────────────────────────────
        embeddings = self.model.encode(texts, show_progress_bar=False)

        # ── Step 2: Compute combined scores at each position ──────────────
        scores = self._compute_boundary_scores(texts, embeddings)

        # ── Step 3: Compute adaptive threshold ───────────────────────────
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        threshold = mean_score - ADAPTIVE_THRESHOLD_FACTOR * std_score
        # Clamp: never go below 0.1 (avoids splitting on noise in very uniform convos)
        threshold = max(threshold, 0.10)

        # ── Step 4: Detect boundaries with min/max guards ─────────────────
        boundaries = self._detect_boundaries(scores, threshold, n)

        # ── Step 5: Build segments from boundaries ────────────────────────
        segments = self._boundaries_to_segments(messages, boundaries)

        # ── Step 6: Create TopicCheckpoint for each segment ───────────────
        checkpoints = []
        for seg_idx, seg_messages in enumerate(segments):
            cp = self._make_checkpoint(conversation, seg_messages, seg_index=seg_idx)
            checkpoints.append(cp)

        # ── Validation log ────────────────────────────────────────────────
        if len(checkpoints) == 1 and n >= MIN_TOPIC_MESSAGES * 2:
            logger.warning(
                f"Conv {conversation.conversation_id} ({n} messages) "
                f"produced only 1 topic segment. "
                f"Threshold was {threshold:.3f}. Consider lowering ADAPTIVE_THRESHOLD_FACTOR."
            )

        return checkpoints

    # ─────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────

    def _compute_boundary_scores(
        self, texts: list[str], embeddings: np.ndarray
    ) -> list[float]:
        """
        Compute a combined boundary score at each position i >= window_size.
        Lower score = higher probability of a topic break.
        """
        scores: list[float] = []
        W = self.window_size
        n = len(texts)

        # Pre-fit TF-IDF on all messages in this conversation
        tfidf = TfidfVectorizer(max_features=200, stop_words="english")
        try:
            tfidf.fit(texts)
            tfidf_fitted = True
        except Exception:
            tfidf_fitted = False

        for i in range(W, n):
            # ── Cosine similarity signal ──────────────────────────────────
            window_embeddings = embeddings[i - W : i]
            window_centroid = np.mean(window_embeddings, axis=0, keepdims=True)
            next_embedding = embeddings[i : i + 1]
            cos_score = float(cosine_similarity(window_centroid, next_embedding)[0][0])

            # ── Keyword overlap (Jaccard) signal ──────────────────────────
            if tfidf_fitted:
                kw_score = self._keyword_jaccard(tfidf, texts[i - W : i], texts[i])
            else:
                kw_score = cos_score  # fallback: use cosine only

            # ── Combined weighted score ───────────────────────────────────
            combined = COSINE_WEIGHT * cos_score + KEYWORD_WEIGHT * kw_score
            scores.append(combined)

        return scores

    def _keyword_jaccard(
        self, tfidf: TfidfVectorizer, window_texts: list[str], next_text: str
    ) -> float:
        """
        Jaccard similarity of top-k TF-IDF keywords between window and next message.
        """
        try:
            window_vec = tfidf.transform(window_texts).toarray().sum(axis=0)
            next_vec = tfidf.transform([next_text]).toarray()[0]

            # Get top-k keyword indices for each
            top_window = set(np.argsort(window_vec)[-TOP_K_KEYWORDS:])
            top_next = set(np.argsort(next_vec)[-TOP_K_KEYWORDS:])

            # Filter out zero-score entries
            top_window = {idx for idx in top_window if window_vec[idx] > 0}
            top_next = {idx for idx in top_next if next_vec[idx] > 0}

            if not top_window and not top_next:
                return 0.0

            intersection = len(top_window & top_next)
            union = len(top_window | top_next)
            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _detect_boundaries(
        self, scores: list[float], threshold: float, n_messages: int
    ) -> list[int]:
        """
        Return a list of message indices where topic breaks occur.
        Enforces MIN and MAX segment size constraints.

        Returns:
            List of boundary indices (exclusive end of each segment).
            Always includes n_messages as the final boundary.
        """
        W = self.window_size
        boundaries: list[int] = []
        last_boundary = 0  # Start of current segment (in message index space)

        for score_idx, score in enumerate(scores):
            msg_idx = score_idx + W  # Actual message index

            segment_length = msg_idx - last_boundary

            # Force split if segment is too long (MAX guard)
            if segment_length >= MAX_TOPIC_MESSAGES:
                boundaries.append(msg_idx)
                last_boundary = msg_idx
                continue

            # Don't split if segment is too short (MIN guard)
            if segment_length < MIN_TOPIC_MESSAGES:
                continue

            # Split if score is below threshold (topic break detected)
            if score < threshold:
                boundaries.append(msg_idx)
                last_boundary = msg_idx

        # Always include the end of the conversation
        if not boundaries or boundaries[-1] != n_messages:
            boundaries.append(n_messages)

        return boundaries

    def _boundaries_to_segments(
        self, messages: list[Message], boundaries: list[int]
    ) -> list[list[Message]]:
        """Convert boundary indices into lists of Message objects."""
        segments = []
        start = 0
        for end in boundaries:
            segment = list(messages[start:end])
            if segment:
                segments.append(segment)
            start = end
        return segments

    def _make_checkpoint(
        self,
        conversation: Conversation,
        seg_messages: list[Message],
        seg_index: int,
    ) -> TopicCheckpoint:
        """Build a TopicCheckpoint from a segment of messages."""
        summary = self.summarizer.summarize(seg_messages)
        label = self.summarizer.topic_label(seg_messages)

        return TopicCheckpoint(
            checkpoint_id=f"topic_conv{conversation.conversation_id:05d}_seg{seg_index:03d}",
            conversation_id=conversation.conversation_id,
            topic_label=label,
            start_global_index=seg_messages[0].global_index,
            end_global_index=seg_messages[-1].global_index,
            start_local_index=seg_messages[0].local_index,
            end_local_index=seg_messages[-1].local_index,
            messages=list(seg_messages),
            summary=summary,
            embedding=None,  # Set by embedder in Phase 3
        )
