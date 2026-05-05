"""
serve/context_builder.py — Persona-aware context assembler for the RAG pipeline.

Loads both persona outputs at startup:
    - personas.json        → UserPersona (aggregate style/traits/values/interests)
    - conv_personas.json   → ConversationPersona (per-conversation facts/events)

When the retriever returns results, this module:
    1. Extracts unique conversation_ids from the retrieved chunks
    2. Looks up the ConversationPersona for each retrieved conversation
    3. Composes a structured context string:
           [Global Persona] + [Conversation Persona Facts] + [Retrieved Text]

This context string is passed to the Generator as the LLM's input.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from retrieval.retriever import RetrievalContext, RetrievalResult
from config import PERSONA_DIR

logger = logging.getLogger(__name__)


class PersonaContextBuilder:
    """
    Loads persona data and enriches retrieval context with persona facts.

    Usage:
        builder = PersonaContextBuilder()
        full_context = builder.build(query, retrieval_context, mode="unified")
    """

    def __init__(
        self,
        personas_path: Path = PERSONA_DIR / "personas.json",
        conv_personas_path: Path = PERSONA_DIR / "conv_personas.json",
    ):
        self._global_personas: dict = {}
        self._conv_personas: dict[str, dict] = {}

        # Load global aggregate personas
        if personas_path.exists():
            with open(personas_path, encoding="utf-8") as f:
                self._global_personas = json.load(f)
            logger.info(f"Loaded global personas for: {list(self._global_personas.keys())}")
        else:
            logger.warning(f"personas.json not found at {personas_path}. Run 'python main.py persona' first.")

        # Load per-conversation personas
        if conv_personas_path.exists():
            with open(conv_personas_path, encoding="utf-8") as f:
                self._conv_personas = json.load(f)
            logger.info(f"Loaded {len(self._conv_personas):,} conversation personas.")
        else:
            logger.warning(f"conv_personas.json not found at {conv_personas_path}.")

    # ─── Public API ──────────────────────────────────────────────────────────

    def build(
        self,
        query: str,
        retrieval_context: RetrievalContext,
        mode: str = "unified",
    ) -> str:
        """
        Build the full context string for the LLM.

        Args:
            query: The user's question.
            retrieval_context: Results from the Retriever.
            mode: One of:
                "unified"      — global persona + conv facts + retrieved text (default)
                "rag_only"     — retrieved text only, no persona context
                "persona_only" — persona context only, no retrieved text

        Returns:
            A formatted string ready to be passed to the LLM.
        """
        sections = []

        # 1. Global persona context
        if mode in ("unified", "persona_only") and self._global_personas:
            sections.append(self._build_global_persona_section())

        # 2. Conversation-specific persona facts (from retrieved conv IDs)
        if mode in ("unified", "persona_only") and self._conv_personas:
            conv_ids = self._extract_conv_ids(retrieval_context)
            if conv_ids:
                sections.append(self._build_conv_persona_section(conv_ids))

        # 3. Retrieved text
        if mode in ("unified", "rag_only"):
            sections.append(retrieval_context.to_context_string())

        return "\n\n".join(sections)

    def has_personas(self) -> bool:
        """Returns True if persona data was loaded successfully."""
        return bool(self._global_personas or self._conv_personas)

    # ─── Private Helpers ─────────────────────────────────────────────────────

    def _extract_conv_ids(self, context: RetrievalContext) -> list[str]:
        """Get unique conversation IDs from all retrieval results."""
        seen = set()
        ids = []
        for result in context.all_results:
            cid = str(result.metadata.get("conversation_id", ""))
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
        return ids[:5]  # Cap at 5 conversations to avoid context bloat

    def _build_global_persona_section(self) -> str:
        """Format the aggregate UserPersona context block."""
        lines = ["=== AGGREGATE COMMUNICATION PROFILES ==="]
        lines.append(
            "These statistics are averaged across ALL conversations and ALL users in the dataset. "
            "They reflect general behavioural tendencies, NOT a single person.\n"
        )

        for speaker, persona_data in self._global_personas.items():
            lines.append(f"  {speaker}:")

            # Traits
            traits = persona_data.get("traits", [])
            if traits:
                top_traits = sorted(traits, key=lambda t: t.get("confidence", 0), reverse=True)[:4]
                trait_str = ", ".join(
                    f"{t['trait']} ({t['confidence']:.0%})" for t in top_traits
                )
                lines.append(f"    Personality: {trait_str}")

            # Style
            style = persona_data.get("style")
            if style:
                lines.append(
                    f"    Style: avg {style.get('avg_message_length', 0):.0f} words/msg | "
                    f"sentiment {style.get('avg_sentiment', 0):+.2f} | "
                    f"asks questions {style.get('question_rate', 0):.0%} of messages"
                )

            # Values
            values = persona_data.get("values")
            if values:
                sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)[:3]
                val_str = ", ".join(
                    f"{k.replace('_focus', '')} ({v:.0%})" for k, v in sorted_values if v > 0.08
                )
                if val_str:
                    lines.append(f"    Values: {val_str}")

            lines.append("")

        return "\n".join(lines)

    def _build_conv_persona_section(self, conv_ids: list[str]) -> str:
        """Format ConversationPersona facts for the retrieved conversations."""
        lines = ["=== CONVERSATION-SPECIFIC PERSONA FACTS ==="]
        lines.append(
            "IMPORTANT: Each conversation below is a DIFFERENT pair of people. "
            "User 1 and User 2 are role labels only — they are NOT the same individuals across conversations. "
            "Conflicting facts are expected and correct.\n"
        )

        found_any = False
        for cid in conv_ids:
            conv_data = self._conv_personas.get(cid)
            if not conv_data:
                continue

            u1_facts = conv_data.get("user1_facts", [])
            u2_facts = conv_data.get("user2_facts", [])
            u1_events = conv_data.get("user1_events", [])
            u2_events = conv_data.get("user2_events", [])

            if not (u1_facts or u2_facts or u1_events or u2_events):
                continue

            found_any = True
            lines.append(f"  Conversation {cid}:")

            for speaker, facts in [("User 1", u1_facts), ("User 2", u2_facts)]:
                if facts:
                    # Group by category
                    by_cat: dict[str, list[str]] = {}
                    for f in facts[:8]:
                        pred = f.get("predicate", "").strip()
                        obj = f.get("obj", "").strip()
                        item = f"{pred} {obj}".strip()
                        if item:
                            by_cat.setdefault(f.get("category", "other"), []).append(item)

                    if by_cat:
                        parts = []
                        for cat, items in by_cat.items():
                            parts.append(f"[{cat}] {' | '.join(items[:2])}")
                        lines.append(f"    {speaker}: {' • '.join(parts)}")

            for speaker, events in [("User 1", u1_events), ("User 2", u2_events)]:
                if events:
                    ev = events[0]
                    lines.append(
                        f"    {speaker} life event [{ev.get('event_type', '')}]: "
                        f"{ev.get('description', '')[:80]}"
                    )

            lines.append("")

        if not found_any:
            lines.append("  (No persona facts found for the retrieved conversations.)")

        return "\n".join(lines)
