"""
scripts/preview_persona.py — Detailed diagnostic of persona extraction.

Shows every section of the persona for a sample conversation, with the
source messages that contributed to each finding.

Run with:
    python scripts/preview_persona.py
    python scripts/preview_persona.py --n 3        # Use first 3 conversations
    python scripts/preview_persona.py --conv 42    # Use conversation ID 42
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.parser import parse_conversations
from persona.style import StyleAnalyzer
from persona.traits import TraitClassifier
from persona.facts import FactExtractor, aggregate_facts
from persona.interests import InterestExtractor
from persona.values import ValuesAnalyzer
from persona.events import EventDetector


def sep(title: str = "", width: int = 65):
    if title:
        print(f"\n{'─' * 4} {title} {'─' * max(0, width - len(title) - 6)}")
    else:
        print("─" * width)


def preview_persona(conversations, speaker: str = "User 1"):
    from data.models import Message

    msgs: list[Message] = []
    for conv in conversations:
        msgs.extend(m for m in conv.messages if m.speaker == speaker)

    if not msgs:
        print(f"  No messages found for {speaker}")
        return

    print(f"\n{'=' * 65}")
    print(f"  PERSONA PREVIEW: {speaker}")
    print(f"  Source: {len(conversations)} conversation(s), {len(msgs)} messages")
    print(f"{'=' * 65}")

    # ── 1. STYLE ANALYSIS ──────────────────────────────────────────────
    sep("1. STYLE ANALYSIS (StyleAnalyzer)")
    style_analyzer = StyleAnalyzer()
    style = style_analyzer.analyze(msgs)

    print(f"\n  Message Length:")
    print(f"    avg={style.avg_message_length:.1f} words | median={style.median_message_length:.1f} | max={style.max_message_length}")
    print(f"    short (<5w): {style.pct_short_messages:.0%} | long (>30w): {style.pct_long_messages:.0%}")

    print(f"\n  Punctuation / Expression:")
    print(f"    questions: {style.question_rate:.0%} | exclamations: {style.exclamation_rate:.0%}")
    print(f"    ellipsis:  {style.ellipsis_rate:.0%} | emoji: {style.emoji_rate:.0%} | caps: {style.caps_rate:.0%}")

    print(f"\n  Sentiment (VADER):")
    print(f"    avg compound: {style.avg_sentiment:+.3f} | variance: {style.sentiment_variance:.3f}")
    print(f"    positive: {style.positive_rate:.0%} | negative: {style.negative_rate:.0%} | neutral: {style.neutral_rate:.0%}")

    print(f"\n  Language Features:")
    print(f"    formality:        {style.formality_score:.0%} (0=casual, 1=formal)")
    print(f"    vocab richness:   {style.vocabulary_richness:.0%} (unique/total words)")
    print(f"    self-disclosure:  {style.self_disclosure_rate:.0%} (messages using 'I')")
    print(f"    hedging:          {style.hedging_rate:.0%} (maybe/perhaps/I think)")
    print(f"    certainty:        {style.certainty_rate:.0%} (definitely/absolutely)")
    print(f"    humor:            {style.humor_rate:.0%} (lol/haha/funny)")
    print(f"    agreement:        {style.agreement_rate:.0%} (same/agree/exactly)")
    print(f"    empathy:          {style.empathy_rate:.0%} (sorry/I understand)")

    # Show 3 sample messages
    print(f"\n  Sample messages used:")
    for m in msgs[:3]:
        print(f"    [{m.conversation_id}] {m.speaker}: {m.text[:90]}")

    # ── 2. TRAITS ──────────────────────────────────────────────────────
    sep("2. TRAIT CLASSIFICATION (TraitClassifier)")
    trait_classifier = TraitClassifier()
    traits = trait_classifier.classify(style, msgs)

    if traits:
        for t in traits:
            bar = "█" * int(t.confidence * 20)
            print(f"    {t.trait:<15} {bar:<20} {t.confidence:.0%}  ({t.evidence_count} evidence msgs)")
    else:
        print("  No traits detected (too few messages?)")

    # ── 3. VALUES ──────────────────────────────────────────────────────
    sep("3. VALUE FOCUS SCORES (ValuesAnalyzer)")
    values_analyzer = ValuesAnalyzer()
    values = values_analyzer.analyze(msgs)
    v = values.__dict__ if hasattr(values, '__dict__') else values.model_dump()
    for dim, score in sorted(v.items(), key=lambda x: x[1], reverse=True):
        if isinstance(score, float):
            bar = "█" * int(score * 20)
            print(f"    {dim.replace('_focus',''):<15} {bar:<20} {score:.0%}")

    # ── 4. LIFE EVENTS ─────────────────────────────────────────────────
    sep("4. LIFE EVENT DETECTION (EventDetector)")
    event_detector = EventDetector()
    events = event_detector.detect(msgs)

    if events:
        for ev in events[:10]:
            print(f"    [{ev.event_type}] conf={ev.confidence:.0%}")
            print(f"       \"{ev.description[:80]}\"")
    else:
        print("  No life events detected in this sample.")

    # ── 5. FACTS (SVO triples) ─────────────────────────────────────────
    sep("5. FACT EXTRACTION — SVO Triples (FactExtractor)")
    try:
        fact_extractor = FactExtractor()
        if fact_extractor._nlp is None:
            print("  spaCy not installed — run: python -m spacy download en_core_web_sm")
        else:
            all_facts = []
            for conv in conversations:
                conv_msgs = [m for m in conv.messages if m.speaker == speaker]
                facts = fact_extractor.extract(conv_msgs, speaker, conv.conversation_id)
                all_facts.extend(facts)

            if all_facts:
                aggregated = aggregate_facts(all_facts, min_frequency=1)
                print(f"\n  {len(all_facts)} raw facts → {len(aggregated)} after deduplication\n")
                for f in aggregated[:15]:
                    print(f"    [{f.category:<14}] conf={f.confidence:.0%}  \"{f.subject} {f.predicate} {f.obj}\"")
                    print(f"       evidence: \"{f.raw_text[:80]}\"")
            else:
                print("  No facts extracted from this sample.")
    except Exception as e:
        print(f"  FactExtractor error: {e}")

    # ── 6. LDA INTERESTS ───────────────────────────────────────────────
    sep("6. LDA INTEREST DISCOVERY (InterestExtractor)")
    interest_extractor = InterestExtractor(n_topics=8)
    try:
        interests = interest_extractor.extract(msgs)
        if interests:
            print(f"\n  {len(interests)} interest topics discovered:\n")
            for interest in interests:
                bar = "█" * int(interest.weight * 30)
                print(f"    Topic {interest.topic_id}: {interest.label}")
                print(f"      weight={interest.weight:.3f}  {bar}")
                print(f"      keywords: {', '.join(interest.keywords)}")
        else:
            print("  Too few messages for LDA (need ≥5 with enough vocabulary).")
    except Exception as e:
        print(f"  LDA error: {e}")

    print(f"\n{'=' * 65}\n")


def main():
    parser = argparse.ArgumentParser(description="Preview persona extraction in detail")
    parser.add_argument("--n", type=int, default=5, help="Number of conversations to use (default: 5)")
    parser.add_argument("--conv", type=int, default=None, help="Single conversation ID to preview")
    args = parser.parse_args()

    print("Parsing conversations...")
    dataset = parse_conversations()

    if args.conv is not None:
        convs = [c for c in dataset.conversations if c.conversation_id == args.conv]
        if not convs:
            print(f"Conversation {args.conv} not found.")
            return
    else:
        convs = dataset.conversations[:args.n]

    print(f"Using {len(convs)} conversation(s) ({sum(c.message_count for c in convs)} total messages)")

    for speaker in ["User 1", "User 2"]:
        preview_persona(convs, speaker)


if __name__ == "__main__":
    main()
