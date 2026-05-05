"""
data/models.py — Pydantic data models for the RAG Chatbot system.
All core data structures are defined here and shared across modules.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class Message(BaseModel):
    """A single message within a conversation."""

    global_index: int = Field(..., description="Chronological position across ALL messages in the dataset")
    conversation_id: int = Field(..., description="Which conversation block this message belongs to")
    local_index: int = Field(..., description="Position within its conversation (0-based)")
    speaker: str = Field(..., description="'User 1' or 'User 2'")
    text: str = Field(..., description="Raw message content")

    model_config = ConfigDict(frozen=True)  # Messages are immutable once created


class Conversation(BaseModel):
    """A single conversation block between User 1 and User 2."""

    conversation_id: int = Field(..., description="Unique conversation identifier (0-based)")
    messages: list[Message] = Field(default_factory=list)
    start_global_index: int = Field(..., description="Global index of the first message")
    end_global_index: int = Field(..., description="Global index of the last message")

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def user1_messages(self) -> list[Message]:
        return [m for m in self.messages if m.speaker == "User 1"]

    @property
    def user2_messages(self) -> list[Message]:
        return [m for m in self.messages if m.speaker == "User 2"]

    @property
    def text_only(self) -> list[str]:
        """All message texts in order."""
        return [m.text for m in self.messages]


class TopicCheckpoint(BaseModel):
    """A topic segment detected within a conversation."""

    checkpoint_id: str = Field(..., description="Unique ID e.g. 'topic_conv0042_seg2'")
    conversation_id: int
    topic_label: str = Field(default="", description="Auto-generated topic label from TF-IDF keywords")
    start_global_index: int
    end_global_index: int
    start_local_index: int = Field(..., description="Start position within its conversation")
    end_local_index: int = Field(..., description="End position within its conversation")
    messages: list[Message]
    summary: str = Field(default="", description="Extractive summary of this segment")
    embedding: Optional[list[float]] = Field(default=None, description="Embedding of the summary text")

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def text_only(self) -> list[str]:
        return [m.text for m in self.messages]


class FixedCheckpoint(BaseModel):
    """A positional checkpoint every N messages (chronological, independent of topics)."""

    checkpoint_id: str = Field(..., description="Unique ID e.g. 'fixed_0500_0599'")
    start_global_index: int
    end_global_index: int
    message_count: int
    summary: str = Field(default="", description="Extractive summary of this batch")
    embedding: Optional[list[float]] = Field(default=None)

    @property
    def messages_range(self) -> str:
        return f"{self.start_global_index}–{self.end_global_index}"


class ParsedDataset(BaseModel):
    """The full parsed result of conversations.csv."""

    conversations: list[Conversation]
    total_messages: int
    total_conversations: int

    @property
    def all_messages(self) -> list[Message]:
        """All messages sorted by global_index."""
        msgs = [m for c in self.conversations for m in c.messages]
        return sorted(msgs, key=lambda m: m.global_index)
