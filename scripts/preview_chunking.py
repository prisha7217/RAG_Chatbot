"""
scripts/preview_chunking.py — Visual preview of the end-to-end chunking pipeline.

Runs the full pipeline on a small sample and saves a structured JSON to:
    outputs/preview_chunking.json

Each entry shows: raw messages → detected segments → summaries → topic labels.

Run with:
    python scripts/preview_chunking.py
    python scripts/preview_chunking.py --conv 5       # Preview a specific conversation
    python scripts/preview_chunking.py --n 5          # Preview first 5 conversations
    python scripts/preview_chunking.py --conv 0 3 7   # Preview multiple specific ones
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from data.parser import parse_conversations
from summarizer.summarizer import Summarizer
from chunking.topic_chunker import TopicChunker
from config import EMBEDDING_MODEL, OUTPUTS_DIR


OUTPUT_FILE = OUTPUTS_DIR / "preview_chunking.json"


def preview(conv_ids: list[int], n: int):
    print("Loading model and parsing data...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    summarizer = Summarizer(model)
    chunker = TopicChunker(model, summarizer)
    dataset = parse_conversations()

    if conv_ids:
        conversations = [c for c in dataset.conversations if c.conversation_id in conv_ids]
    else:
        conversations = dataset.conversations[:n]

    print(f"Running chunker on {len(conversations)} conversation(s)...")

    output = []

    for conv in conversations:
        checkpoints = chunker.chunk(conv)

        entry = {
            "conversation_id": conv.conversation_id,
            "total_messages": conv.message_count,
            "segments_detected": len(checkpoints),
            "raw_messages": [
                {
                    "local_index": msg.local_index,
                    "global_index": msg.global_index,
                    "speaker": msg.speaker,
                    "text": msg.text,
                }
                for msg in conv.messages
            ],
            "topic_segments": [
                {
                    "segment_id": cp.checkpoint_id,
                    "local_range": f"{cp.start_local_index}–{cp.end_local_index}",
                    "global_range": f"{cp.start_global_index}–{cp.end_global_index}",
                    "message_count": cp.message_count,
                    "topic_label": cp.topic_label,
                    "summary": cp.summary,
                    "messages": [
                        {
                            "local_index": msg.local_index,
                            "speaker": msg.speaker,
                            "text": msg.text,
                        }
                        for msg in cp.messages
                    ],
                }
                for cp in checkpoints
            ],
        }
        output.append(entry)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Preview saved to: {OUTPUT_FILE}")
    print(f"  Conversations previewed : {len(conversations)}")
    for entry in output:
        print(f"  Conv {entry['conversation_id']:>4} : {entry['total_messages']} messages → {entry['segments_detected']} segments")


def main():
    parser = argparse.ArgumentParser(description="Preview topic chunking results")
    parser.add_argument("--conv", type=int, nargs="+", help="Specific conversation ID(s) to preview")
    parser.add_argument("--n", type=int, default=2, help="Number of conversations to preview (default: 2)")
    args = parser.parse_args()

    preview(conv_ids=args.conv or [], n=args.n)


if __name__ == "__main__":
    main()
