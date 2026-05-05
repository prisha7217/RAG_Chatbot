"""
chunking/fixed_chunker.py — Fixed 100-message checkpoints.

Creates positional checkpoints every N messages (default 100) across ALL
conversations in chronological order. Independent of topic boundaries.

These are used for timeline-based retrieval: "what happened around message 500?"

Usage:
    from chunking.fixed_chunker import FixedChunker
    chunker = FixedChunker(summarizer)
    checkpoints = chunker.chunk_all(conversations)
"""

from __future__ import annotations

import logging
from data.models import Conversation, Message, FixedCheckpoint
from chunking.base import BaseFixedChunker
from summarizer.summarizer import Summarizer
from config import FIXED_CHECKPOINT_SIZE

logger = logging.getLogger(__name__)


class FixedChunker(BaseFixedChunker):
    """
    Creates fixed-size checkpoints (default: every 100 messages) across
    all conversations in global chronological order.

    Args:
        summarizer: A Summarizer instance for generating summaries.
        checkpoint_size: Number of messages per checkpoint.
    """

    def __init__(self, summarizer: Summarizer, checkpoint_size: int = FIXED_CHECKPOINT_SIZE):
        self.summarizer = summarizer
        self.checkpoint_size = checkpoint_size

    def chunk_all(self, conversations: list[Conversation]) -> list[FixedCheckpoint]:
        """
        Iterate all messages in global chronological order and create a
        FixedCheckpoint every `checkpoint_size` messages.

        Returns:
            List of FixedCheckpoint objects.
        """
        # Collect all messages sorted by global_index
        all_messages: list[Message] = sorted(
            (m for c in conversations for m in c.messages),
            key=lambda m: m.global_index,
        )

        total = len(all_messages)
        logger.info(
            f"Creating fixed checkpoints every {self.checkpoint_size} messages "
            f"across {total:,} total messages..."
        )

        checkpoints: list[FixedCheckpoint] = []
        batch_idx = 0

        for start in range(0, total, self.checkpoint_size):
            end = min(start + self.checkpoint_size, total)
            batch = all_messages[start:end]

            if not batch:
                continue

            summary = self.summarizer.summarize(batch, top_k=5)

            cp = FixedCheckpoint(
                checkpoint_id=f"fixed_{batch[0].global_index:06d}_{batch[-1].global_index:06d}",
                start_global_index=batch[0].global_index,
                end_global_index=batch[-1].global_index,
                message_count=len(batch),
                summary=summary,
                embedding=None,  # Set by embedder in Phase 3
            )
            checkpoints.append(cp)
            batch_idx += 1

        logger.info(f"Created {len(checkpoints)} fixed checkpoints.")
        return checkpoints
