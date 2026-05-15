"""
main.py — CLI entry point for the RAG Chatbot system.

Usage:
    python main.py parse              # Phase 1: Parse CSV and report stats
    python main.py build              # Phases 2-3: Chunk, summarize, index
    python main.py build --index-only # Phase 3 only: Re-index from saved checkpoints
    python main.py persona            # Phase 4: Extract user personas
    python main.py serve              # Phase 5: Launch the Gradio chatbot
"""

import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_parse():
    """Phase 1: Parse conversations.csv and report statistics."""
    from data.parser import parse_conversations

    logger.info("Starting parse phase...")
    dataset = parse_conversations()

    print("\n" + "=" * 50)
    print("  PARSE RESULTS")
    print("=" * 50)
    print(f"  Total conversations : {dataset.total_conversations:,}")
    print(f"  Total messages      : {dataset.total_messages:,}")

    avg_msgs = dataset.total_messages / dataset.total_conversations
    print(f"  Avg msgs/convo      : {avg_msgs:.1f}")

    # Spot-check first 3 conversations
    print("\n  Sample conversations:")
    for convo in dataset.conversations[:3]:
        print(f"\n  [Conv {convo.conversation_id}] — {convo.message_count} messages")
        for msg in convo.messages[:4]:
            print(f"    {msg.speaker}: {msg.text[:70]}{'...' if len(msg.text) > 70 else ''}")
        if convo.message_count > 4:
            print(f"    ... ({convo.message_count - 4} more messages)")
    print("\n" + "=" * 50)


def cmd_build(index_only: bool = False):
    """Phases 2-3: Full build pipeline, or --index-only to skip Phase 2."""
    import json
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODEL, CHECKPOINTS_DIR, INDEX_DIR
    from retrieval.embedder import Embedder
    from retrieval.indexer import Indexer

    if index_only:
        # ── Index-only mode: load checkpoints from disk ───────────────────
        logger.info("--index-only mode: loading checkpoints from disk...")
        from data.parser import parse_conversations
        from data.models import TopicCheckpoint, FixedCheckpoint

        topic_file = CHECKPOINTS_DIR / "topic_checkpoints.json"
        fixed_file = CHECKPOINTS_DIR / "fixed_checkpoints.json"

        if not topic_file.exists() or not fixed_file.exists():
            logger.error(
                f"Checkpoint files not found in {CHECKPOINTS_DIR}. "
                "Run 'python main.py build' first to generate them."
            )
            return

        with open(topic_file, "r", encoding="utf-8") as f:
            raw_topics = json.load(f)
        with open(fixed_file, "r", encoding="utf-8") as f:
            raw_fixed = json.load(f)

        logger.info(f"Loaded {len(raw_topics):,} topic segments, {len(raw_fixed):,} fixed checkpoints from disk.")

        # Parse conversations for the message chunk collection
        logger.info("Parsing conversations for message chunk collection...")
        dataset = parse_conversations()

        # Load model and build index
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        embedder = Embedder(model=model)
        indexer = Indexer(embedder)

        # For index-only, we use the raw dicts directly for topics/fixed
        # and re-generate chunks from conversations
        indexer.build_from_raw(
            topic_dicts=raw_topics,
            fixed_dicts=raw_fixed,
            conversations=dataset.conversations,
        )

        print(f"\n✓ Index rebuilt from existing checkpoints → {INDEX_DIR}")
        return

    # ── Full build mode ───────────────────────────────────────────────────
    from data.parser import parse_conversations
    from summarizer.summarizer import Summarizer
    from chunking.topic_chunker import TopicChunker
    from chunking.fixed_chunker import FixedChunker

    # Step 1: Parse
    logger.info("Step 1/4 — Parsing conversations...")
    dataset = parse_conversations()
    logger.info(f"Parsed {dataset.total_conversations:,} conversations, {dataset.total_messages:,} messages")

    # Step 2: Load model
    logger.info(f"Step 2/4 — Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    summarizer = Summarizer(model)

    # Step 3: Topic chunking
    logger.info("Step 3/4 — Running topic chunker on all conversations...")
    topic_chunker = TopicChunker(model, summarizer)
    all_topic_checkpoints = []

    for i, conv in enumerate(dataset.conversations):
        cps = topic_chunker.chunk(conv)
        all_topic_checkpoints.extend(cps)
        if (i + 1) % 500 == 0:
            logger.info(f"  Chunked {i+1:,}/{dataset.total_conversations:,} conversations...")

    logger.info(f"Topic chunking complete: {len(all_topic_checkpoints):,} topic segments")

    # Step 3b: Fixed checkpoints
    fixed_chunker = FixedChunker(summarizer)
    fixed_checkpoints = fixed_chunker.chunk_all(dataset.conversations)

    # Step 4: Save checkpoints to disk
    topic_out = CHECKPOINTS_DIR / "topic_checkpoints.json"
    fixed_out = CHECKPOINTS_DIR / "fixed_checkpoints.json"
    logger.info(f"Saving topic checkpoints → {topic_out}")
    with open(topic_out, "w", encoding="utf-8") as f:
        json.dump(
            [cp.model_dump(exclude={"messages", "embedding"}) for cp in all_topic_checkpoints],
            f, indent=2, ensure_ascii=False,
        )
    logger.info(f"Saving fixed checkpoints → {fixed_out}")
    with open(fixed_out, "w", encoding="utf-8") as f:
        json.dump(
            [cp.model_dump(exclude={"embedding"}) for cp in fixed_checkpoints],
            f, indent=2, ensure_ascii=False,
        )

    # Step 4: Build ChromaDB index
    logger.info("Step 4/4 — Building ChromaDB vector index...")
    embedder = Embedder(model=model)
    indexer = Indexer(embedder)
    indexer.build_from_checkpoints(
        topic_checkpoints=all_topic_checkpoints,
        fixed_checkpoints=fixed_checkpoints,
        conversations=dataset.conversations,
    )

    print("\n" + "=" * 50)
    print("  BUILD RESULTS")
    print("=" * 50)
    print(f"  Topic segments created : {len(all_topic_checkpoints):,}")
    print(f"  Fixed checkpoints      : {len(fixed_checkpoints):,}")
    avg_segs = len(all_topic_checkpoints) / dataset.total_conversations
    print(f"  Avg segments/convo     : {avg_segs:.1f}")
    print(f"  Checkpoints saved to   : {CHECKPOINTS_DIR}")
    print(f"  Index saved to         : {INDEX_DIR}")
    print("=" * 50)
    first_conv_cps = [cp for cp in all_topic_checkpoints if cp.conversation_id == 0]
    if first_conv_cps:
        print("\n  Sample topic segments (first conversation):")
        for cp in first_conv_cps:
            print(f"\n  [{cp.checkpoint_id}] — {cp.message_count} msgs")
            print(f"  Label  : {cp.topic_label}")
            print(f"  Summary: {cp.summary[:120]}{'...' if len(cp.summary) > 120 else ''}")
    print("\n" + "=" * 50)


def cmd_persona():
    """Phase 4: Extract user personas from all conversations."""
    from data.parser import parse_conversations
    from persona.extractor import PersonaExtractor

    logger.info("Phase 4 — Persona extraction starting...")
    dataset = parse_conversations()

    extractor = PersonaExtractor(run_facts=True, run_interests=True)
    personas, conv_personas = extractor.extract_all(dataset.conversations)
    personas_file, conv_file = extractor.save(personas, conv_personas)

    print("\n" + "=" * 58)
    print("  PERSONA EXTRACTION RESULTS")
    print("=" * 58)

    for speaker, persona in personas.items():
        print(f"\n  {speaker} (aggregate across {persona.total_conversations:,} convos):")
        print(f"    Messages analyzed : {persona.total_messages:,}")
        print(f"    Traits detected   : {', '.join(t.trait for t in persona.traits[:5])}")
        if persona.style:
            print(f"    Avg msg length    : {persona.style.avg_message_length:.0f} words")
            print(f"    Avg sentiment     : {persona.style.avg_sentiment:+.3f}")
            print(f"    Question rate     : {persona.style.question_rate:.0%}")
        if persona.values:
            v = persona.values
            scores = {
                "family": v.family_focus, "career": v.career_focus,
                "social": v.social_focus, "intellectual": v.intellectual_focus,
            }
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Top values        : {', '.join(f'{k} ({sv:.0%})' for k, sv in top)}")
        if persona.interests:
            print(f"    Top interests:")
            for interest in sorted(persona.interests, key=lambda x: x.weight, reverse=True)[:3]:
                print(f"      {interest.label} (weight={interest.weight:.3f})")

    print(f"\n  Per-conversation personas: {len(conv_personas):,} conversations have content")

    # Show a sample conversation persona
    if conv_personas:
        sample_id = next(iter(conv_personas))
        sample = conv_personas[sample_id]
        print(f"\n  Sample — Conversation {sample_id}:")
        if sample.user1_facts:
            print(f"    User 1 facts: {len(sample.user1_facts)}")
            for f in sample.user1_facts[:3]:
                print(f"      [{f.category}] {f.subject} {f.predicate} {f.obj}")
        if sample.user2_facts:
            print(f"    User 2 facts: {len(sample.user2_facts)}")
            for f in sample.user2_facts[:3]:
                print(f"      [{f.category}] {f.subject} {f.predicate} {f.obj}")

    print(f"\n  Saved to:")
    print(f"    {personas_file}")
    print(f"    {conv_file}")
    print("=" * 58)


def cmd_serve():
    """Phase 5: Launch the Gradio chatbot UI."""
    from config import INDEX_DIR, PERSONA_DIR

    # Pre-flight checks
    index_ok = (INDEX_DIR / "chroma.sqlite3").exists()
    personas_ok = (PERSONA_DIR / "personas.json").exists()
    conv_personas_ok = (PERSONA_DIR / "conv_personas.json").exists()

    if not index_ok:
        logger.error(
            "ChromaDB index not found. Run 'python main.py build' first."
        )
        sys.exit(1)

    if not personas_ok or not conv_personas_ok:
        logger.warning(
            "Persona files not found. Run 'python main.py persona' for persona-enriched responses. "
            "Continuing in RAG-only mode."
        )

    logger.info("Launching Gradio chatbot...")
    from serve.app import build_ui
    app, theme, css = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=theme,
        css=css,
        share=False,
        show_error=True,
        inbrowser=True,
    )


def cmd_drift():
    """Round 2 — Phase 1: Detect persona drift across topic segments."""
    import json
    from data.parser import parse_conversations
    from data.models import TopicCheckpoint
    from persona.drift import DriftDetector
    from config import CHECKPOINTS_DIR, DRIFT_TIMELINES_FILE

    topic_file = CHECKPOINTS_DIR / "topic_checkpoints.json"
    if not topic_file.exists():
        logger.error(
            f"topic_checkpoints.json not found at {topic_file}. "
            "Run 'python main.py build' first."
        )
        return

    logger.info("Loading topic checkpoints...")
    with open(topic_file, "r", encoding="utf-8") as f:
        raw_checkpoints = json.load(f)

    # Reconstruct TopicCheckpoint objects (messages excluded when saved, so we pass empty list)
    checkpoints = []
    for cp_dict in raw_checkpoints:
        try:
            cp_dict.setdefault("messages", [])
            checkpoints.append(TopicCheckpoint(**cp_dict))
        except Exception as e:
            logger.warning(f"Skipping malformed checkpoint: {e}")
            continue
    logger.info(f"Loaded {len(checkpoints):,} topic checkpoints.")

    logger.info("Parsing conversations (needed for message-level stats)...")
    dataset = parse_conversations()

    detector = DriftDetector()
    timelines = detector.detect_all(dataset.conversations, checkpoints)
    output_path = detector.save(timelines)

    # ── Summary ──────────────────────────────────────────────────────────────
    events_total = sum(t.drift_event_count for t in timelines.values())
    convs_with_drift = len({
        t.conversation_id for t in timelines.values() if t.drift_event_count > 0
    })

    print("\n" + "=" * 58)
    print("  DRIFT DETECTION RESULTS")
    print("=" * 58)
    print(f"  Timelines produced      : {len(timelines):,}")
    print(f"  Total drift events      : {events_total:,}")
    print(f"  Convos with drift       : {convs_with_drift:,}")
    print(f"  Output file             : {output_path}")

    # Sample output — show first conversation with drift
    sample = next(
        (t for t in timelines.values() if t.drift_event_count > 0), None
    )
    if sample:
        print(f"\n  Sample — Conv {sample.conversation_id} / {sample.speaker}:")
        print(sample.to_context_string())
    print("=" * 58)

def cmd_intent_train():
    """Round 2 — Phase C: Train the offline intent classifier (SVM on embeddings)."""
    from intent.train import main as train_main
    train_main()


def main():
    parser = argparse.ArgumentParser(
        description="RAG Chatbot — Conversation Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  parse              Parse conversations.csv and report stats
  build              Full build pipeline (phases 2-3): chunk → summarize → index
  build --index-only Re-index from saved checkpoints (skips Phase 2, ~5 min)
  persona            Phase 4: Extract user personas from all conversations
  drift              Round 2: Detect mood/tone drift across topic segments
  intent-train       Round 2: Train the offline intent classifier
  serve              Launch the Gradio chatbot (phase 5)
        """,
    )
    parser.add_argument("command", choices=["parse", "build", "persona", "drift", "intent-train", "serve"])
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="(build only) Skip chunking, re-index from saved checkpoint files.",
    )
    args = parser.parse_args()

    if args.command == "parse":
        cmd_parse()
    elif args.command == "build":
        cmd_build(index_only=args.index_only)
    elif args.command == "persona":
        cmd_persona()
    elif args.command == "drift":
        cmd_drift()
    elif args.command == "intent-train":
        cmd_intent_train()
    elif args.command == "serve":
        cmd_serve()


if __name__ == "__main__":
    main()
