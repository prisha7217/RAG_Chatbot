"""
data/parser.py — Parse conversations.csv into structured Conversation objects.

CSV format:
    Each conversation block is enclosed in double quotes and separated by \\r\\n.
    Within a block, each line starts with "User 1: " or "User 2: ".

Usage:
    from data.parser import parse_conversations
    dataset = parse_conversations()
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

from data.models import Message, Conversation, ParsedDataset
from config import CONVERSATIONS_CSV

logger = logging.getLogger(__name__)


def parse_conversations(csv_path: Path = CONVERSATIONS_CSV) -> ParsedDataset:
    """
    Parse the raw conversations.csv into a ParsedDataset.

    Steps:
        1. Read entire file as text.
        2. Split on the block delimiter (closing quote + \\r\\n + opening quote).
        3. For each block, parse individual "User 1: " / "User 2: " lines.
        4. Assign global_index (chronological across all conversations) and conversation_id.

    Returns:
        ParsedDataset with all conversations and metadata.
    """
    logger.info(f"Parsing conversations from: {csv_path}")

    raw_text = csv_path.read_text(encoding="utf-8", errors="replace")

    # Split the file into individual conversation blocks.
    # Each block is enclosed in double quotes; blocks are separated by "\r\n
    # We split on the pattern: quote, optional whitespace, CRLF or LF, quote
    blocks = _split_blocks(raw_text)
    logger.info(f"Found {len(blocks)} raw conversation blocks")

    conversations: list[Conversation] = []
    global_index = 0
    skipped = 0
    conv_id = 0  # Sequential ID — only increments for valid conversations

    for block in blocks:
        block = block.strip().strip('"').strip()
        if not block:
            skipped += 1
            continue

        messages = _parse_block(block, conv_id, global_index)
        if not messages:
            skipped += 1
            continue

        conversation = Conversation(
            conversation_id=conv_id,
            messages=messages,
            start_global_index=messages[0].global_index,
            end_global_index=messages[-1].global_index,
        )
        conversations.append(conversation)
        global_index += len(messages)
        conv_id += 1  # Only increment for valid, non-empty conversations

    total_messages = sum(c.message_count for c in conversations)
    logger.info(
        f"Parsed {len(conversations)} conversations, "
        f"{total_messages} total messages. "
        f"Skipped {skipped} empty blocks."
    )

    return ParsedDataset(
        conversations=conversations,
        total_messages=total_messages,
        total_conversations=len(conversations),
    )


def _split_blocks(raw_text: str) -> list[str]:
    """
    Split the raw file text into individual conversation blocks.

    The file format has each block as a CSV cell (enclosed in double quotes),
    separated by \\r\\n between records. We split on the record boundary.
    """
    # Normalize line endings
    raw_text = raw_text.replace("\r\n", "\n")

    # Each conversation block is a quoted CSV field.
    # Split on: newline followed by optional whitespace then a double-quote
    # that starts a new block (i.e., at the beginning of a new record).
    # We use a pattern that finds block boundaries.
    blocks = re.split(r'\n"', raw_text)

    # The first block may or may not start with a quote
    cleaned = []
    for i, block in enumerate(blocks):
        # Strip leading quote from all blocks after the first split
        block = block.strip()
        if block.startswith('"'):
            block = block[1:]
        # Strip trailing quote
        if block.endswith('"'):
            block = block[:-1]
        cleaned.append(block)

    return cleaned


def _parse_block(block: str, conv_id: int, start_global_index: int) -> list[Message]:
    """
    Parse a single conversation block into a list of Message objects.

    Args:
        block: Raw block text (already stripped of outer quotes).
        conv_id: Conversation identifier.
        start_global_index: Global index offset for the first message.

    Returns:
        List of Message objects, empty if no valid messages found.
    """
    messages: list[Message] = []
    local_index = 0

    # Split block into lines
    lines = block.split("\n")

    # Buffer for multi-line messages (a message can span multiple lines)
    current_speaker: str | None = None
    current_text_parts: list[str] = []

    # Speaker pattern: "User 1: " or "User 2: " at the start of a line
    speaker_pattern = re.compile(r"^(User [12]):\s*(.*)", re.IGNORECASE)

    def flush_message():
        """Flush buffered speaker+text into a Message object."""
        nonlocal local_index
        if current_speaker and current_text_parts:
            text = " ".join(current_text_parts).strip()
            if text:
                messages.append(
                    Message(
                        global_index=start_global_index + local_index,
                        conversation_id=conv_id,
                        local_index=local_index,
                        speaker=current_speaker,
                        text=text,
                    )
                )
                local_index += 1

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = speaker_pattern.match(line)
        if match:
            # New speaker detected — flush previous message first
            flush_message()
            current_speaker = match.group(1)
            current_text_parts = [match.group(2).strip()] if match.group(2).strip() else []
        else:
            # Continuation of previous message
            if current_speaker is not None:
                current_text_parts.append(line)

    # Flush the last message
    flush_message()

    return messages


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dataset = parse_conversations()
    print(f"\n✓ Total conversations : {dataset.total_conversations}")
    print(f"✓ Total messages      : {dataset.total_messages}")
    print(f"\nSample — Conversation 0:")
    for msg in dataset.conversations[0].messages:
        print(f"  [{msg.local_index}] {msg.speaker}: {msg.text[:80]}")
