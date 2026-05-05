"""
retrieval/embedder.py — Embedding wrapper using sentence-transformers.

Provides a thin, reusable wrapper around SentenceTransformer so the
same model instance is shared across indexing and retrieval.

Usage:
    from retrieval.embedder import Embedder
    embedder = Embedder()
    vector = embedder.embed("User 1: I love hiking in the mountains")
    vectors = embedder.embed_batch(["text1", "text2"])
"""

from __future__ import annotations

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class Embedder:
    """
    Thin wrapper around SentenceTransformer for consistent embedding.

    Args:
        model_name: HuggingFace model name. Defaults to config.EMBEDDING_MODEL.
        model: Optional pre-loaded SentenceTransformer instance.
               If provided, model_name is ignored (avoids double-loading).
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, model=None):
        if model is not None:
            self._model = model
            logger.info("Embedder: using pre-loaded model")
        else:
            logger.info(f"Embedder: loading model {model_name}")
            self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Returns:
            List of floats (embedding vector).
        """
        vector = self._model.encode(text, show_progress_bar=False)
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Embed a batch of texts efficiently.

        Args:
            texts: List of strings to embed.
            batch_size: Encoding batch size.

        Returns:
            List of embedding vectors (list of list of float).
        """
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return vectors.tolist()

    @property
    def model(self):
        """Access the underlying SentenceTransformer model."""
        return self._model
