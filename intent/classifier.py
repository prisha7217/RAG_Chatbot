"""
intent/classifier.py — Offline intent classifier (serve-phase).

Loads the trained SVM model at startup and classifies user queries
into 5 intent categories using the already-loaded sentence-transformer
embedder. Zero additional API calls. Inference: ~5ms on CPU.

Intents:
    reminder          — recall/lookup queries
    emotional-support — queries about emotional/sad content
    action-item       — list/enumerate/summarise requests
    small-talk        — greetings and filler
    unknown           — everything else (default pipeline)

Usage:
    from intent.classifier import IntentClassifier
    classifier = IntentClassifier()                         # loads SVM
    label, confidence = classifier.predict(query, embedder) # ~5ms
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from config import INTENT_MODEL_FILE, INTENT_CONFIDENCE_FLOOR

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Classifies user queries into 5 intent categories using a pre-trained SVM.

    The SVM was trained on sentence-transformer embeddings (all-MiniLM-L6-v2).
    At inference time, the same embedder (already loaded in the serve phase)
    is passed to predict() so we don't load a second copy.

    Graceful degradation: if the model file doesn't exist, every call returns
    ("unknown", 0.0) so the serve pipeline continues without intent-routing.
    """

    def __init__(self, model_path: Path = INTENT_MODEL_FILE):
        self._model = None
        self._model_path = model_path
        self._load()

    # ─── Public API ──────────────────────────────────────────────────────────

    def predict(self, query: str, embedder) -> tuple[str, float]:
        """
        Classify a query into an intent category.

        Args:
            query:    The raw user query string.
            embedder: A SentenceTransformer instance (already loaded at serve time).

        Returns:
            (intent_label, confidence) — e.g. ("reminder", 0.94)
            Falls back to ("unknown", 0.0) if model not loaded or
            confidence is below INTENT_CONFIDENCE_FLOOR.
        """
        if self._model is None:
            return "unknown", 0.0

        try:
            embedding = embedder.encode([query], normalize_embeddings=True)
            proba = self._model.predict_proba(embedding)[0]   # shape: (n_classes,)
            class_idx = int(np.argmax(proba))
            confidence = float(proba[class_idx])
            label = self._model.classes_[class_idx]

            if confidence < INTENT_CONFIDENCE_FLOOR:
                logger.debug(
                    f"Intent '{label}' confidence {confidence:.2f} below floor "
                    f"{INTENT_CONFIDENCE_FLOOR} → falling back to 'unknown'"
                )
                return "unknown", confidence

            logger.debug(f"Intent: {label} (confidence={confidence:.2f})")
            return label, confidence

        except Exception as e:
            logger.warning(f"Intent prediction failed: {e}")
            return "unknown", 0.0

    @property
    def is_loaded(self) -> bool:
        """True if the SVM model loaded successfully."""
        return self._model is not None

    def classes(self) -> list[str]:
        """Return the list of intent class labels the model knows."""
        if self._model is None:
            return []
        return list(self._model.classes_)

    # ─── Private helpers ────────────────────────────────────────────────────

    def _load(self):
        if not self._model_path.exists():
            logger.warning(
                f"Intent model not found at {self._model_path}. "
                "Run 'python main.py intent-train' to generate it. "
                "All queries will be classified as 'unknown' until then."
            )
            return
        try:
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)
            logger.info(
                f"Intent classifier loaded: {len(self.classes())} classes — "
                + ", ".join(self.classes())
            )
        except Exception as e:
            logger.error(f"Failed to load intent model from {self._model_path}: {e}")
            self._model = None
