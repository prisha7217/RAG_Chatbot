"""
scripts/query.py — Interactive retrieval tester.

Lets you type queries and see exactly what the retriever pulls back from the
index — topic summaries, fixed summaries, and raw message chunks — along with
scores and metadata. Results saved to outputs/query_results.json.

Run with:
    python scripts/query.py
    python scripts/query.py --query "does anyone have a dog"
    python scripts/query.py --query "radiology student" --top 10
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.embedder import Embedder
from retrieval.retriever import Retriever
from config import OUTPUTS_DIR

OUTPUT_FILE = OUTPUTS_DIR / "query_results.json"


def run_query(query: str, top_k: int = 5):
    print(f"\nLoading retriever...")
    embedder = Embedder()
    retriever = Retriever(embedder)

    print(f'\nQuerying: "{query}"\n')

    context = retriever.retrieve(
        query,
        top_k_topics=top_k,
        top_k_fixed=max(2, top_k // 2),
        top_k_chunks=top_k,
    )

    # Build structured output
    result = {
        "query": query,
        "top_k": top_k,
        "topic_results": [
            {
                "rank": i + 1,
                "score": round(r.score, 4),
                "conversation_id": r.metadata.get("conversation_id"),
                "topic_label": r.metadata.get("topic_label", ""),
                "global_range": f"{r.metadata.get('start_global_index')}–{r.metadata.get('end_global_index')}",
                "summary": r.text,
            }
            for i, r in enumerate(context.topic_results)
        ],
        "chunk_results": [
            {
                "rank": i + 1,
                "score": round(r.score, 4),
                "conversation_id": r.metadata.get("conversation_id"),
                "global_range": f"{r.metadata.get('start_global_index')}–{r.metadata.get('end_global_index')}",
                "text": r.text,
            }
            for i, r in enumerate(context.chunk_results)
        ],
        "fixed_results": [
            {
                "rank": i + 1,
                "score": round(r.score, 4),
                "global_range": f"{r.metadata.get('start_global_index')}–{r.metadata.get('end_global_index')}",
                "summary": r.text,
            }
            for i, r in enumerate(context.fixed_results)
        ],
    }

    # Print summary to terminal
    print("=" * 60)
    print(f"  TOPIC SEGMENTS  (top {len(context.topic_results)})")
    print("=" * 60)
    for r in result["topic_results"]:
        print(f"\n  #{r['rank']} | score={r['score']} | Conv {r['conversation_id']} | {r['topic_label']}")
        for line in r["summary"].split(" | "):
            print(f"    > {line}")

    print("\n" + "=" * 60)
    print(f"  RAW CHUNKS  (top {len(context.chunk_results)})")
    print("=" * 60)
    for r in result["chunk_results"]:
        print(f"\n  #{r['rank']} | score={r['score']} | Conv {r['conversation_id']}")
        for line in r["text"].split("\n")[:4]:
            print(f"    > {line}")

    # Save full output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Full results saved to: {OUTPUT_FILE}")


def interactive_loop(top_k: int):
    print("Loading retriever (one-time)...")
    embedder = Embedder()
    retriever = Retriever(embedder)
    print("Ready. Type a query and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            query = input("Query > ").strip()
            if not query:
                continue

            context = retriever.retrieve(query, top_k_topics=top_k, top_k_chunks=top_k)

            print(f"\n  Top topic results:")
            for i, r in enumerate(context.topic_results[:3]):
                label = r.metadata.get("topic_label", "")
                conv = r.metadata.get("conversation_id", "?")
                print(f"  [{i+1}] score={r.score:.3f} | Conv {conv} | {label}")
                for part in r.text.split(" | ")[:2]:
                    print(f"       > {part}")

            print(f"\n  Top chunk results:")
            for i, r in enumerate(context.chunk_results[:2]):
                conv = r.metadata.get("conversation_id", "?")
                print(f"  [{i+1}] score={r.score:.3f} | Conv {conv}")
                for line in r.text.split("\n")[:3]:
                    print(f"       > {line}")

            print()

        except KeyboardInterrupt:
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Manual retrieval tester")
    parser.add_argument("--query", type=str, default=None, help="Single query to run")
    parser.add_argument("--top", type=int, default=5, help="Number of results per collection (default: 5)")
    args = parser.parse_args()

    if args.query:
        run_query(args.query, top_k=args.top)
    else:
        interactive_loop(top_k=args.top)


if __name__ == "__main__":
    main()
