"""
scripts/inspect_single_topic.py — Find conversations with only 1 topic segment.

These are conversations where the chunker couldn't detect a topic shift strong
enough to split — either because:
  (a) The conversation is very short (< 6 messages)
  (b) The conversation is topically very uniform (same subject throughout)
  (c) The adaptive threshold was too high for that conversation

Reads from: outputs/checkpoints/topic_checkpoints.json
Saves to:   outputs/single_topic_conversations.json

Run with:
    python scripts/inspect_single_topic.py
    python scripts/inspect_single_topic.py --limit 20   # Show only first 20
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHECKPOINTS_DIR, OUTPUTS_DIR

TOPIC_CHECKPOINTS_FILE = CHECKPOINTS_DIR / "topic_checkpoints.json"
OUTPUT_FILE = OUTPUTS_DIR / "single_topic_conversations.json"


def main(limit: int | None = None):
    print(f"Reading: {TOPIC_CHECKPOINTS_FILE}")
    with open(TOPIC_CHECKPOINTS_FILE, "r", encoding="utf-8") as f:
        checkpoints = json.load(f)

    # Group checkpoints by conversation_id
    by_conv: dict[int, list[dict]] = defaultdict(list)
    for cp in checkpoints:
        by_conv[cp["conversation_id"]].append(cp)

    # Find conversations with only 1 segment
    single_segment = {
        conv_id: segs
        for conv_id, segs in by_conv.items()
        if len(segs) == 1
    }

    total_convs = len(by_conv)
    single_count = len(single_segment)
    ratio = single_count / total_convs * 100

    print(f"\n  Total conversations   : {total_convs:,}")
    print(f"  Single-segment convos : {single_count:,} ({ratio:.1f}%)")
    print(f"  Multi-segment convos  : {total_convs - single_count:,} ({100 - ratio:.1f}%)")

    # Sort by message count (largest single-segment convos are most suspicious)
    # message_count is a @property so it's not in the JSON — compute from indices
    def get_msg_count(item):
        seg = item[1][0]
        return seg["end_global_index"] - seg["start_global_index"] + 1

    sorted_singles = sorted(single_segment.items(), key=get_msg_count, reverse=True)

    if limit:
        sorted_singles = sorted_singles[:limit]

    # Build output
    output = {
        "summary": {
            "total_conversations": total_convs,
            "single_segment_count": single_count,
            "single_segment_pct": round(ratio, 2),
        },
        "conversations": [
            {
                "conversation_id": conv_id,
                "message_count": get_msg_count((conv_id, segs)),
                "topic_label": segs[0]["topic_label"],
                "summary": segs[0]["summary"],
                "global_range": f"{segs[0]['start_global_index']}–{segs[0]['end_global_index']}",
            }
            for conv_id, segs in sorted_singles
        ],
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved to: {OUTPUT_FILE}")
    print(f"\n  Single-segment conversations (sorted by size):")
    print(f"  {'Conv ID':>8}  {'Msgs':>5}  {'Topic Label'}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*40}")
    for conv_id, segs in sorted_singles:
        msg_count = get_msg_count((conv_id, segs))
        print(f"  {conv_id:>8}  {msg_count:>5}  {segs[0]['topic_label']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit output to N conversations")
    args = parser.parse_args()
    main(limit=args.limit)
