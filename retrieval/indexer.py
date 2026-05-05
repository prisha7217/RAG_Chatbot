"""
retrieval/indexer.py — Build and populate the ChromaDB vector index.

Creates 3 collections:
    1. topic_summaries   — One entry per topic segment (summary + label)
    2. fixed_summaries   — One entry per 100-message fixed checkpoint
    3. message_chunks    — Raw message chunks (MESSAGE_CHUNK_SIZE messages each)

This runs during the build phase only. The serve phase reads from the
already-built index.

Usage:
    from retrieval.indexer import Indexer
    indexer = Indexer(embedder)
    indexer.build_from_checkpoints(topic_cps, fixed_cps, conversations)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb

from retrieval.embedder import Embedder
from data.models import Conversation, TopicCheckpoint, FixedCheckpoint
from config import (
    INDEX_DIR,
    CHROMA_COLLECTION_TOPICS,
    CHROMA_COLLECTION_FIXED,
    CHROMA_COLLECTION_CHUNKS,
    MESSAGE_CHUNK_SIZE,
)

logger = logging.getLogger(__name__)


class Indexer:
    """
    Builds the ChromaDB vector index from parsed checkpoints.

    Args:
        embedder: An Embedder instance for generating embeddings.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=str(INDEX_DIR))

    def build_from_checkpoints(
        self,
        topic_checkpoints: list[TopicCheckpoint],
        fixed_checkpoints: list[FixedCheckpoint],
        conversations: list[Conversation],
    ) -> None:
        """
        Build all 3 ChromaDB collections from checkpoint model objects.
        Used in full build mode (python main.py build).
        """
        logger.info("Building ChromaDB index...")
        self._build_topic_collection(topic_checkpoints)
        self._build_fixed_collection(fixed_checkpoints)
        self._build_chunks_collection(conversations)
        logger.info("Index build complete.")

    def build_from_raw(
        self,
        topic_dicts: list[dict],
        fixed_dicts: list[dict],
        conversations: list[Conversation],
    ) -> None:
        """
        Build all 3 ChromaDB collections from raw JSON dicts loaded from disk.
        Used in --index-only mode (python main.py build --index-only).

        Topic and fixed dicts come directly from topic_checkpoints.json and
        fixed_checkpoints.json — they don't have model objects, just plain dicts.
        """
        logger.info("Building ChromaDB index from raw checkpoint dicts...")
        self._build_topic_collection_from_dicts(topic_dicts)
        self._build_fixed_collection_from_dicts(fixed_dicts)
        self._build_chunks_collection(conversations)
        logger.info("Index build complete.")

    def _build_topic_collection(self, checkpoints: list[TopicCheckpoint]) -> None:
        """Embed and store topic summaries."""
        logger.info(f"Indexing {len(checkpoints):,} topic summaries...")

        # Delete and recreate for clean rebuild
        try:
            self._client.delete_collection(CHROMA_COLLECTION_TOPICS)
        except Exception:
            pass
        collection = self._client.create_collection(
            name=CHROMA_COLLECTION_TOPICS,
            metadata={"hnsw:space": "cosine"},
        )

        # Batch process
        BATCH = 512
        for i in range(0, len(checkpoints), BATCH):
            batch = checkpoints[i : i + BATCH]
            texts = [cp.summary for cp in batch]
            ids = [cp.checkpoint_id for cp in batch]
            metadatas = [
                {
                    "conversation_id": cp.conversation_id,
                    "topic_label": cp.topic_label,
                    "start_global_index": cp.start_global_index,
                    "end_global_index": cp.end_global_index,
                    "start_local_index": cp.start_local_index,
                    "end_local_index": cp.end_local_index,
                }
                for cp in batch
            ]
            embeddings = self.embedder.embed_batch(texts)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"  Topic summaries: {min(i + BATCH, len(checkpoints)):,}/{len(checkpoints):,}")

    def _build_fixed_collection(self, checkpoints: list[FixedCheckpoint]) -> None:
        """Embed and store fixed checkpoint summaries."""
        logger.info(f"Indexing {len(checkpoints):,} fixed summaries...")

        try:
            self._client.delete_collection(CHROMA_COLLECTION_FIXED)
        except Exception:
            pass
        collection = self._client.create_collection(
            name=CHROMA_COLLECTION_FIXED,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [cp.summary for cp in checkpoints]
        ids = [cp.checkpoint_id for cp in checkpoints]
        metadatas = [
            {
                "start_global_index": cp.start_global_index,
                "end_global_index": cp.end_global_index,
                "message_count": cp.message_count,
            }
            for cp in checkpoints
        ]
        embeddings = self.embedder.embed_batch(texts)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"  Fixed summaries: {len(checkpoints):,} indexed.")

    def _build_chunks_collection(self, conversations: list[Conversation]) -> None:
        """
        Split all conversations into raw message chunks and index them.
        Each chunk is MESSAGE_CHUNK_SIZE consecutive messages, formatted as
        'Speaker: text' lines joined by newlines.
        """
        logger.info(f"Building raw message chunks (size={MESSAGE_CHUNK_SIZE})...")

        try:
            self._client.delete_collection(CHROMA_COLLECTION_CHUNKS)
        except Exception:
            pass
        collection = self._client.create_collection(
            name=CHROMA_COLLECTION_CHUNKS,
            metadata={"hnsw:space": "cosine"},
        )

        all_chunks_text = []
        all_chunks_ids = []
        all_chunks_meta = []

        for conv in conversations:
            messages = conv.messages
            for chunk_start in range(0, len(messages), MESSAGE_CHUNK_SIZE):
                chunk = messages[chunk_start : chunk_start + MESSAGE_CHUNK_SIZE]
                if not chunk:
                    continue

                chunk_text = "\n".join(f"{m.speaker}: {m.text}" for m in chunk)
                chunk_id = f"chunk_conv{conv.conversation_id:05d}_{chunk[0].local_index:04d}"

                all_chunks_text.append(chunk_text)
                all_chunks_ids.append(chunk_id)
                all_chunks_meta.append(
                    {
                        "conversation_id": conv.conversation_id,
                        "start_local_index": chunk[0].local_index,
                        "end_local_index": chunk[-1].local_index,
                        "start_global_index": chunk[0].global_index,
                        "end_global_index": chunk[-1].global_index,
                    }
                )

        logger.info(f"  Total message chunks: {len(all_chunks_text):,}")

        # Batch embed and add
        BATCH = 512
        for i in range(0, len(all_chunks_text), BATCH):
            batch_texts = all_chunks_text[i : i + BATCH]
            batch_ids = all_chunks_ids[i : i + BATCH]
            batch_meta = all_chunks_meta[i : i + BATCH]
            embeddings = self.embedder.embed_batch(batch_texts)
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )
            if (i // BATCH + 1) % 10 == 0:
                logger.info(f"  Message chunks: {min(i + BATCH, len(all_chunks_text)):,}/{len(all_chunks_text):,}")

        logger.info(f"  Message chunks indexed: {len(all_chunks_text):,}")

    def _build_topic_collection_from_dicts(self, dicts: list[dict]) -> None:
        """Embed and store topic summaries from raw JSON dicts (--index-only mode)."""
        logger.info(f"Indexing {len(dicts):,} topic summaries from dicts...")

        try:
            self._client.delete_collection(CHROMA_COLLECTION_TOPICS)
        except Exception:
            pass
        collection = self._client.create_collection(
            name=CHROMA_COLLECTION_TOPICS,
            metadata={"hnsw:space": "cosine"},
        )

        BATCH = 512
        for i in range(0, len(dicts), BATCH):
            batch = dicts[i : i + BATCH]
            texts = [d["summary"] for d in batch]
            ids = [d["checkpoint_id"] for d in batch]
            metadatas = [
                {
                    "conversation_id": d["conversation_id"],
                    "topic_label": d.get("topic_label", ""),
                    "start_global_index": d["start_global_index"],
                    "end_global_index": d["end_global_index"],
                    "start_local_index": d.get("start_local_index", 0),
                    "end_local_index": d.get("end_local_index", 0),
                }
                for d in batch
            ]
            embeddings = self.embedder.embed_batch(texts)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"  Topic summaries: {min(i + BATCH, len(dicts)):,}/{len(dicts):,}")

    def _build_fixed_collection_from_dicts(self, dicts: list[dict]) -> None:
        """Embed and store fixed summaries from raw JSON dicts (--index-only mode)."""
        logger.info(f"Indexing {len(dicts):,} fixed summaries from dicts...")

        try:
            self._client.delete_collection(CHROMA_COLLECTION_FIXED)
        except Exception:
            pass
        collection = self._client.create_collection(
            name=CHROMA_COLLECTION_FIXED,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [d["summary"] for d in dicts]
        ids = [d["checkpoint_id"] for d in dicts]
        metadatas = [
            {
                "start_global_index": d["start_global_index"],
                "end_global_index": d["end_global_index"],
                "message_count": d.get("message_count", 0),
            }
            for d in dicts
        ]
        embeddings = self.embedder.embed_batch(texts)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"  Fixed summaries: {len(dicts):,} indexed.")
