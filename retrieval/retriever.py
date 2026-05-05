"""
retrieval/retriever.py — Query the ChromaDB index and return ranked results.

Searches all 3 collections (topic summaries, fixed summaries, message chunks),
combines the results, and returns the top-k most relevant items.

The retriever returns a structured context dict that the answer generator
uses to build its response.

Usage:
    from retrieval.retriever import Retriever
    retriever = Retriever(embedder)
    context = retriever.retrieve("Does User 1 have any pets?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import chromadb

from retrieval.embedder import Embedder
from config import (
    INDEX_DIR,
    CHROMA_COLLECTION_TOPICS,
    CHROMA_COLLECTION_FIXED,
    CHROMA_COLLECTION_CHUNKS,
    RETRIEVAL_TOP_K_TOPICS,
    RETRIEVAL_TOP_K_CHUNKS,
    RETRIEVAL_TOP_K_FIXED,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved item from the index."""
    source: str          # "topic", "fixed", or "chunk"
    doc_id: str
    text: str            # The stored document text (summary or chunk)
    score: float         # Cosine similarity score (0–1, higher = more relevant)
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"RetrievalResult(source={self.source}, score={self.score:.3f}, id={self.doc_id})"


@dataclass
class RetrievalContext:
    """The full context returned to the answer generator."""
    query: str
    topic_results: list[RetrievalResult]
    fixed_results: list[RetrievalResult]
    chunk_results: list[RetrievalResult]

    @property
    def all_results(self) -> list[RetrievalResult]:
        """All results sorted by score descending."""
        combined = self.topic_results + self.fixed_results + self.chunk_results
        return sorted(combined, key=lambda r: r.score, reverse=True)

    def to_context_string(self) -> str:
        """
        Format retrieved results into a single context string for the LLM.
        Uses a structured format: source type + content.
        """
        lines = [f"Query: {self.query}\n"]

        if self.topic_results:
            lines.append("=== TOPIC SUMMARIES (most relevant conversation segments) ===")
            for r in self.topic_results:
                label = r.metadata.get("topic_label", "")
                conv_id = r.metadata.get("conversation_id", "")
                lines.append(f"[Conv {conv_id} | {label} | score={r.score:.2f}]")
                lines.append(r.text)
                lines.append("")

        if self.chunk_results:
            lines.append("=== RAW MESSAGE CHUNKS (exact conversation excerpts) ===")
            for r in self.chunk_results:
                conv_id = r.metadata.get("conversation_id", "")
                lines.append(f"[Conv {conv_id} | score={r.score:.2f}]")
                lines.append(r.text)
                lines.append("")

        if self.fixed_results:
            lines.append("=== POSITIONAL SUMMARIES (timeline context) ===")
            for r in self.fixed_results:
                start = r.metadata.get("start_global_index", "")
                end = r.metadata.get("end_global_index", "")
                lines.append(f"[Messages {start}–{end} | score={r.score:.2f}]")
                lines.append(r.text)
                lines.append("")

        return "\n".join(lines)


class Retriever:
    """
    Searches all 3 ChromaDB collections and returns a RetrievalContext.

    Args:
        embedder: An Embedder instance for query embedding.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self._topic_col = self._client.get_collection(CHROMA_COLLECTION_TOPICS)
        self._fixed_col = self._client.get_collection(CHROMA_COLLECTION_FIXED)
        self._chunk_col = self._client.get_collection(CHROMA_COLLECTION_CHUNKS)
        logger.info("Retriever initialized with all 3 collections.")

    def retrieve(
        self,
        query: str,
        top_k_topics: int = RETRIEVAL_TOP_K_TOPICS,
        top_k_fixed: int = RETRIEVAL_TOP_K_FIXED,
        top_k_chunks: int = RETRIEVAL_TOP_K_CHUNKS,
    ) -> RetrievalContext:
        """
        Embed the query and search all 3 collections.

        Args:
            query: The user's question.
            top_k_topics: How many topic summaries to retrieve.
            top_k_fixed: How many fixed summaries to retrieve.
            top_k_chunks: How many raw message chunks to retrieve.

        Returns:
            RetrievalContext with all results.
        """
        query_embedding = self.embedder.embed(query)

        topic_results = self._query_collection(
            self._topic_col, query_embedding, top_k_topics, source="topic"
        )
        fixed_results = self._query_collection(
            self._fixed_col, query_embedding, top_k_fixed, source="fixed"
        )
        chunk_results = self._query_collection(
            self._chunk_col, query_embedding, top_k_chunks, source="chunk"
        )

        logger.debug(
            f"Retrieved: {len(topic_results)} topics, "
            f"{len(fixed_results)} fixed, {len(chunk_results)} chunks"
        )

        return RetrievalContext(
            query=query,
            topic_results=topic_results,
            fixed_results=fixed_results,
            chunk_results=chunk_results,
        )

    def retrieve_conversation(self, conversation_id: int) -> RetrievalContext:
        """
        Fetch all stored chunks for a specific conversation by ID.
        Uses ChromaDB metadata filtering instead of semantic search.
        """
        where = {"conversation_id": conversation_id}
        topic_results, chunk_results = [], []

        try:
            res = self._topic_col.get(where=where, include=["documents", "metadatas"])
            for doc_id, text, meta in zip(
                res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
            ):
                topic_results.append(RetrievalResult(
                    source="topic", doc_id=doc_id, text=text or "",
                    score=1.0, metadata=meta or {},
                ))
        except Exception as e:
            logger.warning(f"Topic lookup for conv {conversation_id} failed: {e}")

        try:
            res = self._chunk_col.get(where=where, include=["documents", "metadatas"])
            for doc_id, text, meta in zip(
                res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
            ):
                chunk_results.append(RetrievalResult(
                    source="chunk", doc_id=doc_id, text=text or "",
                    score=1.0, metadata=meta or {},
                ))
        except Exception as e:
            logger.warning(f"Chunk lookup for conv {conversation_id} failed: {e}")

        query_str = f"conversation {conversation_id}"
        logger.debug(
            f"Direct lookup conv {conversation_id}: "
            f"{len(topic_results)} topics, {len(chunk_results)} chunks"
        )
        return RetrievalContext(
            query=query_str,
            topic_results=topic_results,
            fixed_results=[],
            chunk_results=chunk_results,
        )

    def _query_collection(
        self,
        collection,
        query_embedding: list[float],
        top_k: int,
        source: str,
    ) -> list[RetrievalResult]:
        """Query a single ChromaDB collection and parse results."""
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Collection query failed ({source}): {e}")
            return []

        items = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (dist / 2)
            score = 1.0 - (dist / 2.0)
            items.append(
                RetrievalResult(
                    source=source,
                    doc_id=doc_id,
                    text=text,
                    score=score,
                    metadata=meta or {},
                )
            )

        return items
