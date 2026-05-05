"""
chunking/base.py — Abstract base class for all chunkers.

All chunkers follow the same interface:
    chunker.chunk(conversation) -> list of checkpoints
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from data.models import Conversation, TopicCheckpoint, FixedCheckpoint


class BaseChunker(ABC):
    """Abstract interface for conversation chunkers."""

    @abstractmethod
    def chunk(self, conversation: Conversation) -> list[TopicCheckpoint]:
        """
        Chunk a single conversation into topic segments.

        Args:
            conversation: A fully parsed Conversation object.

        Returns:
            List of TopicCheckpoint objects for this conversation.
        """
        raise NotImplementedError


class BaseFixedChunker(ABC):
    """Abstract interface for fixed-size chunkers (positional, not semantic)."""

    @abstractmethod
    def chunk_all(self, conversations: list[Conversation]) -> list[FixedCheckpoint]:
        """
        Create fixed-size checkpoints across all conversations chronologically.

        Args:
            conversations: All parsed conversations, in order.

        Returns:
            List of FixedCheckpoint objects.
        """
        raise NotImplementedError
