"""
generation/generator.py — LLM-powered answer generator with template fallback.

Primary path: Groq API (llama-3.3-70b-versatile) — fast, free tier, sub-second.
Secondary:    OpenAI, then Gemini.
Fallback:     Deterministic template that extracts relevant facts from context.

Usage:
    from generation.generator import Generator
    gen = Generator()
    answer = gen.generate(query, retrieval_context, full_context_str)
"""

from __future__ import annotations

import logging
import os
from retrieval.retriever import RetrievalContext
from config import OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, GROQ_MODEL, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a large conversation dataset.

CRITICAL DATASET CONTEXT:
- Each retrieved conversation is a DIFFERENT pair of people.
- "User 1" and "User 2" are role labels, NOT the same individuals across conversations.
- Facts from different conversations WILL conflict — this is expected and correct.
- Do NOT merge facts across conversations into a single profile.

Your job:
1. Answer based ONLY on what the retrieved context contains.
2. When presenting facts, attribute them to specific conversations:
   e.g. "In one conversation, User 1 mentioned having a dog. In another, they worked as a nurse."
3. If multiple conversations give conflicting answers, say: "Across different conversations, users mentioned X, Y, and Z."
4. For personality/style questions, use the aggregate profile which is statistically valid.
5. Do NOT invent, infer, or assume facts not in the context.
6. Keep answers concise and well-structured.

Do not mention ChromaDB, embeddings, vector search, or technical retrieval details.
"""


class Generator:
    """
    Answer generator: tries Groq → OpenAI → Gemini → template fallback.
    Always returns a response.
    """

    def __init__(self):
        self._groq_available = bool(GROQ_API_KEY)
        self._openai_available = bool(OPENAI_API_KEY)
        self._gemini_available = bool(GEMINI_API_KEY)

        if self._groq_available:
            logger.info(f"Generator: Groq API configured (model: {GROQ_MODEL}).")
        elif self._openai_available:
            logger.info("Generator: OpenAI API configured.")
        elif self._gemini_available:
            logger.info("Generator: Gemini API configured.")
        else:
            logger.warning(
                "Generator: No API key found. Using template fallback. "
                "Set GROQ_API_KEY for fast free LLM responses."
            )

    def generate(
        self,
        query: str,
        context: RetrievalContext,
        full_context_str: str | None = None,
    ) -> str:
        """
        Generate an answer to the query using retrieved context.

        Args:
            query:            The user's question.
            context:          RetrievalContext from the Retriever.
            full_context_str: Pre-built persona-enriched context string.
                              If None, falls back to context.to_context_string().
        """
        context_str = full_context_str or context.to_context_string()

        if self._groq_available:
            answer = self._call_groq(query, context_str)
            if answer:
                return answer

        if self._openai_available:
            answer = self._call_openai(query, context_str)
            if answer:
                return answer

        if self._gemini_available:
            answer = self._call_gemini(query, context_str)
            if answer:
                return answer

        logger.info("Using template fallback for answer generation.")
        return self._template_fallback(query, context, full_context_str or "")

    # ─── LLM API Paths ───────────────────────────────────────────────────────

    def _call_groq(self, query: str, context_str: str) -> str | None:
        """Call Groq API (llama-3.3-70b-versatile). Returns None on any failure."""
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=768,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Groq call failed: {e}")
            return None

    def _call_openai(self, query: str, context_str: str) -> str | None:
        """Call OpenAI API. Returns None on any failure."""
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
                ],
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}")
            return None

    def _call_gemini(self, query: str, context_str: str) -> str | None:
        """Call Gemini API. Returns None on any failure."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_str}\n\nQuestion: {query}"
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini call failed: {e}")
            return None

    # ─── Template Fallback ────────────────────────────────────────────────────

    def _template_fallback(
        self,
        query: str,
        context: RetrievalContext,
        full_context_str: str = "",
    ) -> str:
        """
        Template-based answer. Produces clean, readable output without an LLM.

        Strategy:
          1. Detect the query topic (pets, location, hobby, personality, etc.)
          2. Show matching persona facts from retrieved conversations, grouped cleanly
          3. Show the most relevant conversation excerpts with speaker context
          4. Concise footer with retrieval stats
        """
        all_results = context.all_results

        if not all_results:
            return (
                f'No relevant information found for: **"{query}"**\n\n'
                "Try rephrasing your question or asking about a different topic."
            )

        query_lower = query.lower()
        STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
            "have", "has", "had", "what", "who", "where", "when", "how", "any",
            "some", "about", "tell", "me", "user", "1", "2", "there", "those",
            "these", "often", "most", "common", "talk", "like", "get", "know",
        }
        query_words = set(query_lower.split()) - STOPWORDS

        TOPIC_KEYWORDS = {
            "pet":         {"pet", "dog", "cat", "animal", "pets"},
            "location":    {"location", "live", "from", "city", "state", "place", "where"},
            "hobby":       {"hobby", "hobbies", "interest", "enjoy", "pastime", "leisure"},
            "occupation":  {"job", "work", "career", "occupation", "profession", "study"},
            "personality": {"personality", "style", "trait", "character", "behave", "communicate"},
            "family":      {"family", "kids", "children", "married", "spouse", "parents"},
            "food":        {"food", "eat", "cook", "restaurant", "cuisine", "diet"},
            "music":       {"music", "song", "band", "listen", "genre"},
            "education":   {"education", "school", "college", "degree", "study", "university"},
        }
        detected_topics = {
            topic for topic, kws in TOPIC_KEYWORDS.items()
            if kws & query_words
        }

        sections: list[str] = []

        # ── 1. Persona facts — shown per-conversation, not merged ─────────────
        if full_context_str and "CONVERSATION-SPECIFIC PERSONA FACTS" in full_context_str:
            conv_facts: dict[str, dict[str, dict[str, list[str]]]] = {}
            # conv_facts[conv_id][speaker][category] = [val1, val2, ...]

            in_facts, current_conv = False, "?"
            for line in full_context_str.split("\n"):
                if "CONVERSATION-SPECIFIC PERSONA FACTS" in line:
                    in_facts = True
                    continue
                if not in_facts:
                    continue
                if line.startswith("==="):
                    break
                stripped = line.strip()
                if stripped.startswith("Conversation") and stripped.endswith(":"):
                    current_conv = stripped[len("Conversation"):].rstrip(":").strip()
                    continue
                if "User 1:" in stripped or "User 2:" in stripped:
                    try:
                        speaker, rest = stripped.split(":", 1)
                        speaker = speaker.strip()
                        line_lower = rest.lower()
                        is_relevant = (
                            not detected_topics
                            or any(w in line_lower for w in query_words)
                            or any(t in line_lower for t in detected_topics)
                        )
                        if not is_relevant:
                            continue
                        conv_facts.setdefault(current_conv, {}).setdefault(speaker, {})
                        for chunk in rest.split("•"):
                            chunk = chunk.strip()
                            if chunk.startswith("[") and "]" in chunk:
                                cat_end = chunk.index("]")
                                cat = chunk[1:cat_end].strip()
                                val = chunk[cat_end + 1:].strip(" |").strip()
                                if val:
                                    conv_facts[current_conv][speaker].setdefault(cat, []).append(val)
                    except ValueError:
                        continue

            if conv_facts:
                fact_lines = [
                    "**What people said** *(each conversation is a different pair of people):*\n"
                ]
                for conv_id, speakers in list(conv_facts.items())[:4]:
                    fact_lines.append(f"*Conversation {conv_id}:*")
                    for speaker, cats in speakers.items():
                        for cat, vals in list(cats.items())[:3]:
                            cat_label = cat.replace("_", " ").title()
                            unique_vals = list(dict.fromkeys(vals))[:2]
                            fact_lines.append(
                                f"  - **{speaker}** ({cat_label}): {' · '.join(unique_vals)}"
                            )
                    fact_lines.append("")
                sections.append("\n".join(fact_lines))

        # ── 2. Aggregate persona for personality queries ──────────────────────
        if "personality" in detected_topics and full_context_str and "AGGREGATE" in full_context_str:
            agg_lines = ["**Communication profiles:**\n"]
            in_agg = False
            for line in full_context_str.split("\n"):
                if "AGGREGATE" in line:
                    in_agg = True
                    continue
                if in_agg and line.startswith("==="):
                    break
                if in_agg and line.strip() and not line.startswith(("These describe", "Analyzed")):
                    agg_lines.append(line.rstrip())
            if len(agg_lines) > 1:
                sections.append("\n".join(agg_lines))

        # ── 3. Conversation excerpts ──────────────────────────────────────────
        topic_hits = [r for r in all_results if r.source == "topic"][:4]
        chunk_hits = [r for r in all_results if r.source == "chunk"][:3]
        excerpt_source = topic_hits or chunk_hits

        if excerpt_source:
            excerpt_lines = ["**From the conversations:**\n"]
            shown = 0
            for r in excerpt_source:
                if shown >= 3:
                    break
                cid = r.metadata.get("conversation_id", "?")
                label = r.metadata.get("topic_label", "")

                raw_lines = r.text.split("\n") if "\n" in r.text else r.text.split(" | ")
                hits, ctx_lines = [], []
                for tl in raw_lines:
                    tl = tl.strip()
                    if not tl:
                        continue
                    if any(w in tl.lower() for w in query_words):
                        hits.append(tl)
                    elif len(ctx_lines) < 2:
                        ctx_lines.append(tl)

                display = (hits or ctx_lines)[:3]
                if not display:
                    continue

                header = f"*Conversation {cid}*" + (f" — {label}" if label else "")
                excerpt_lines.append(header)
                for tl in display:
                    excerpt_lines.append(f"> {tl}")
                excerpt_lines.append("")
                shown += 1

            if shown > 0:
                sections.append("\n".join(excerpt_lines))

        # ── 4. Last resort ────────────────────────────────────────────────────
        if not sections:
            best = all_results[0]
            sections.append(f"> {best.text[:300]}{'...' if len(best.text) > 300 else ''}")

        footer = (
            f"\n---\n*{len(all_results)} segments retrieved · "
            "Set `GROQ_API_KEY` for natural language answers*"
        )

        return "\n\n".join(sections) + footer
