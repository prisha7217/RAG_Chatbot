"""
persona/extractor.py — Orchestrator for the full persona extraction pipeline.

Two-tier output:
    1. UserPersona (aggregate, per speaker label)
       - Style stats, traits, values, LDA interests
       - NO facts — these contradict across 11,003 different people
       - Saved to: outputs/persona/personas.json

    2. ConversationPersona (per conversation)
       - SVO facts, life events, named entities for that specific pair
       - Internally consistent — one conversation = one pair of people
       - Saved to: outputs/persona/conv_personas.json

Usage:
    from persona.extractor import PersonaExtractor
    extractor = PersonaExtractor()
    personas, conv_personas = extractor.extract_all(conversations)
    extractor.save(personas, conv_personas)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, Counter
from pathlib import Path

from data.models import Conversation, Message
from persona.schema import (
    UserPersona, ConversationPersona,
    PersonaNamedEntity, PersonaFact, PersonaLifeEvent,
)
from persona.facts import FactExtractor, aggregate_facts
from persona.style import StyleAnalyzer
from persona.traits import TraitClassifier
from persona.interests import InterestExtractor
from persona.values import ValuesAnalyzer
from persona.events import EventDetector
from config import PERSONA_DIR, MIN_FACT_FREQUENCY

logger = logging.getLogger(__name__)

SPEAKERS = ["User 1", "User 2"]


class PersonaExtractor:
    """
    Orchestrates persona extraction with a two-tier architecture:

    Global tier (UserPersona):
        Style, traits, values, and LDA interests aggregated across all
        conversations. Robust population-level signals.

    Conversation tier (ConversationPersona):
        Facts, life events, and named entities extracted per conversation.
        Internally consistent — no cross-person contradictions.
    """

    def __init__(
        self,
        run_facts: bool = True,
        run_interests: bool = True,
        run_ner: bool = True,
    ):
        self._fact_extractor = FactExtractor() if run_facts else None
        self._style_analyzer = StyleAnalyzer()
        self._trait_classifier = TraitClassifier()
        self._interest_extractor = InterestExtractor() if run_interests else None
        self._values_analyzer = ValuesAnalyzer()
        self._event_detector = EventDetector()
        self._run_facts = run_facts
        self._run_ner = run_ner and (
            self._fact_extractor is not None and
            self._fact_extractor._nlp is not None
        )

    def extract_all(
        self,
        conversations: list[Conversation],
    ) -> tuple[dict[str, UserPersona], dict[int, ConversationPersona]]:
        """
        Extract personas at both tiers in a single pass over conversations.

        Returns:
            (personas, conv_personas) where:
              - personas maps speaker → UserPersona (aggregate)
              - conv_personas maps conversation_id → ConversationPersona
        """
        logger.info(f"Starting persona extraction on {len(conversations):,} conversations...")

        # ── Global accumulation (for UserPersona) ─────────────────────────
        all_messages: dict[str, list[Message]] = defaultdict(list)
        conv_count: dict[str, int] = defaultdict(int)
        # No global facts — those live in conv_personas now

        # ── Per-conversation accumulation (for ConversationPersona) ───────
        conv_personas: dict[int, ConversationPersona] = {}

        for i, conv in enumerate(conversations):
            conv_user1_facts: list[PersonaFact] = []
            conv_user2_facts: list[PersonaFact] = []
            conv_user1_events: list[PersonaLifeEvent] = []
            conv_user2_events: list[PersonaLifeEvent] = []
            conv_entities: list[tuple[str, str]] = []

            for speaker in SPEAKERS:
                speaker_msgs = [m for m in conv.messages if m.speaker == speaker]
                if not speaker_msgs:
                    continue

                all_messages[speaker].extend(speaker_msgs)
                conv_count[speaker] += 1

                # Single spaCy pass: facts + NER together
                if self._fact_extractor and self._run_facts:
                    facts, ents = self._fact_extractor.extract_with_entities(
                        speaker_msgs, speaker, conv.conversation_id
                    )
                    if speaker == "User 1":
                        conv_user1_facts.extend(facts)
                    else:
                        conv_user2_facts.extend(facts)
                    conv_entities.extend(ents)

                # Life events per conversation
                speaker_events = self._event_detector.detect(speaker_msgs)
                if speaker == "User 1":
                    conv_user1_events.extend(speaker_events)
                else:
                    conv_user2_events.extend(speaker_events)

            # Build ConversationPersona for this conversation
            named_entities = []
            if self._run_ner:
                entity_counts: Counter = Counter()
                for etype, etext in conv_entities:
                    entity_counts[(etype, etext)] += 1
                for (etype, etext), count in entity_counts.most_common(10):
                    named_entities.append(PersonaNamedEntity(
                        entity_type=etype, text=etext, mention_count=count
                    ))

            # Only store conversation personas that have at least some facts/events
            has_content = (
                conv_user1_facts or conv_user2_facts or
                conv_user1_events or conv_user2_events
            )
            if has_content:
                conv_personas[conv.conversation_id] = ConversationPersona(
                    conversation_id=conv.conversation_id,
                    user1_facts=conv_user1_facts,
                    user2_facts=conv_user2_facts,
                    user1_events=conv_user1_events,
                    user2_events=conv_user2_events,
                    named_entities=named_entities,
                )

            if (i + 1) % 1000 == 0:
                logger.info(
                    f"  Processed {i+1:,}/{len(conversations):,} conversations... "
                    f"({len(conv_personas):,} with persona content)"
                )

        # ── Build aggregate UserPersonas (style, traits, values, interests) ─
        personas: dict[str, UserPersona] = {}

        for speaker in SPEAKERS:
            msgs = all_messages[speaker]
            if not msgs:
                continue

            logger.info(f"Building aggregate persona for {speaker} ({len(msgs):,} messages)...")

            style = self._style_analyzer.analyze(msgs)
            traits = self._trait_classifier.classify(style, msgs)

            interests = []
            if self._interest_extractor:
                sample = msgs[:15000]
                interests = self._interest_extractor.extract(sample)

            values = self._values_analyzer.analyze(msgs)

            personas[speaker] = UserPersona(
                speaker=speaker,
                total_conversations=conv_count[speaker],
                total_messages=len(msgs),
                interests=interests,
                traits=traits,
                style=style,
                values=values,
            )

            logger.info(
                f"  {speaker}: {len(interests)} interests | "
                f"{len(traits)} traits | style computed"
            )

        logger.info(
            f"Extraction complete. "
            f"{len(conv_personas):,}/{len(conversations):,} conversations have persona content."
        )

        return personas, conv_personas

    def save(
        self,
        personas: dict[str, UserPersona],
        conv_personas: dict[int, ConversationPersona],
        output_dir: Path = PERSONA_DIR,
    ) -> tuple[Path, Path]:
        """Save both persona outputs to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Global personas
        personas_file = output_dir / "personas.json"
        with open(personas_file, "w", encoding="utf-8") as f:
            json.dump(
                {speaker: p.model_dump() for speaker, p in personas.items()},
                f, indent=2, ensure_ascii=False,
            )
        logger.info(f"Aggregate personas saved to: {personas_file}")

        # Per-conversation personas
        conv_file = output_dir / "conv_personas.json"
        with open(conv_file, "w", encoding="utf-8") as f:
            json.dump(
                {str(cid): cp.model_dump() for cid, cp in conv_personas.items()},
                f, indent=2, ensure_ascii=False,
            )
        logger.info(f"Conversation personas saved to: {conv_file} ({len(conv_personas):,} entries)")

        return personas_file, conv_file
