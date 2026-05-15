"""
retrieval/conflict_resolver.py — Entity-aware re-ranking with contradiction detection.

Sits between the Retriever and the answer generator as a middleware layer.
Takes a RetrievalContext, re-ranks its results using a composite score, and
flags groups of results that likely contain contradictory facts.

Composite score (weights defined in config.py):
    0.40 × cosine similarity   (original retrieval score)
    0.30 × entity overlap      (shared named tokens between query and chunk)
    0.20 × emotional weight    (|VADER compound| — emotionally charged = more relevant)
    0.10 × recency             (conversation_id order — weak tiebreaker)

Contradiction detection:
    A group of results is flagged as contradictory when:
    1. Negation clash:  one chunk contains a negation of a key entity phrase
                        that appears positively in another chunk.
    2. Sentiment spread: VADER compound scores span > CONFLICT_SENTIMENT_SPREAD
                        (e.g. one chunk is very positive, another is very negative
                        about the same entity).

Usage (serve phase):
    from retrieval.conflict_resolver import ConflictResolver
    resolver = ConflictResolver()
    resolved_ctx = resolver.resolve(retrieval_context, query)
    # resolved_ctx.chunk_results are now re-ranked
    # resolved_ctx.contradiction_flags lists any detected conflicts
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from retrieval.retriever import RetrievalContext, RetrievalResult
from config import (
    CONFLICT_COSINE_WEIGHT,
    CONFLICT_ENTITY_WEIGHT,
    CONFLICT_EMOTION_WEIGHT,
    CONFLICT_RECENCY_WEIGHT,
    CONFLICT_SENTIMENT_SPREAD,
)

logger = logging.getLogger(__name__)

# Common English stopwords to exclude from entity extraction
_STOPWORDS = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "it", "its", "this", "that",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "a", "an", "the", "and", "or",
    "but", "in", "on", "at", "to", "for", "of", "with", "about",
    "by", "from", "up", "as", "if", "then", "than", "so", "no", "not",
    "user", "1", "2", "one", "two", "what", "did", "say", "tell",
    "remind", "mention", "talk", "spoke", "said", "asked",
    # High-frequency conversational filler — causes false-positive entity matching
    "ago", "all", "also", "always", "any", "around", "away",
    "back", "before", "both", "came", "come", "cool", "doing",
    "done", "each", "even", "ever", "few", "get", "gets", "good",
    "got", "great", "hear", "here", "how", "i'm", "i've", "i'll",
    "i'd", "just", "know", "let", "like", "long", "lot", "make",
    "many", "much", "now", "off", "oh", "okay", "old", "once",
    "only", "out", "over", "own", "really", "right", "same", "see",
    "since", "some", "sometimes", "soon", "sorry", "still", "such",
    "sure", "take", "that's", "there", "though", "through", "time",
    "too", "try", "very", "want", "well", "went", "who", "why",
    "wow", "yeah", "yet", "you're",
    # Contractions that regex keeps whole (apostrophe included)
    "what's", "it's", "he's", "she's", "that'll", "they've",
    "we've", "you've", "they're", "we're", "who's", "here's",
    "there's", "where's", "what're", "how's",
    # Common verbs/adjectives that produce noisy negation matches
    "sound", "sounds", "going", "gone", "nice", "glad", "agree",
    "hearing", "interesting", "basically", "actually", "probably",
    "anyway", "perhaps", "totally", "literally", "honestly",
})

# Maximum number of results to include in pairwise contradiction comparison.
# With N results, pairs = N*(N-1)/2.  Cap at 8 → max 28 pairs (vs 153 for 18).
_MAX_CONTRADICTION_RESULTS = 8

# Negation words that invert the sentiment/claim of nearby terms
_NEGATIONS = frozenset({
    "not", "no", "never", "none", "nothing", "nobody", "nowhere",
    "neither", "nor", "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't", "can't", "couldn't", "won't",
    "wouldn't", "shouldn't", "hasn't", "haven't", "hadn't",
    "without", "barely", "hardly", "scarcely",
})


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ContradictionFlag:
    """Records a detected contradiction between two retrieved results."""
    result_a_id: str
    result_b_id: str
    conflict_type: str           # "negation" | "sentiment_spread"
    entity: str                  # The entity phrase that is contradicted
    sentiment_a: float
    sentiment_b: float
    description: str

    def to_debug_string(self) -> str:
        return (
            f"⚠ Contradiction [{self.conflict_type}] on '{self.entity}': "
            f"{self.result_a_id} (sentiment={self.sentiment_a:+.2f}) vs "
            f"{self.result_b_id} (sentiment={self.sentiment_b:+.2f}) — "
            f"{self.description}"
        )


@dataclass
class ResolvedContext:
    """
    The conflict-resolved retrieval context.

    Drop-in replacement for RetrievalContext in the serve pipeline.
    Results are re-ranked by composite score. Contradiction flags are
    surfaced separately for the debug panel.
    """
    query: str
    topic_results: list[RetrievalResult]
    fixed_results: list[RetrievalResult]
    chunk_results: list[RetrievalResult]     # Re-ranked
    contradiction_flags: list[ContradictionFlag] = field(default_factory=list)

    @property
    def all_results(self) -> list[RetrievalResult]:
        combined = self.topic_results + self.fixed_results + self.chunk_results
        return sorted(combined, key=lambda r: r.score, reverse=True)

    @property
    def has_contradictions(self) -> bool:
        return len(self.contradiction_flags) > 0

    def contradiction_summary(self) -> str:
        """One-line summary for the LLM system prompt."""
        if not self.contradiction_flags:
            return ""
        entities = {f.entity for f in self.contradiction_flags}
        return (
            f"⚠ {len(self.contradiction_flags)} potential contradiction(s) detected "
            f"across retrieved chunks (entities: {', '.join(sorted(entities))}). "
            f"Treat conflicting claims with caution."
        )

    def to_context_string(self) -> str:
        """Format resolved context for the LLM, same shape as RetrievalContext."""
        lines = [f"Query: {self.query}\n"]

        if self.contradiction_flags:
            lines.append(f"[{self.contradiction_summary()}]\n")

        if self.topic_results:
            lines.append("=== TOPIC SUMMARIES (most relevant conversation segments) ===")
            for r in self.topic_results:
                label = r.metadata.get("topic_label", "")
                conv_id = r.metadata.get("conversation_id", "")
                lines.append(f"[Conv {conv_id} | {label} | score={r.score:.2f}]")
                lines.append(r.text)
                lines.append("")

        if self.chunk_results:
            lines.append("=== RAW MESSAGE CHUNKS (re-ranked by entity-aware score) ===")
            for r in self.chunk_results:
                conv_id = r.metadata.get("conversation_id", "")
                contradiction_mark = ""
                # Mark chunks involved in contradictions
                for flag in self.contradiction_flags:
                    if r.doc_id in (flag.result_a_id, flag.result_b_id):
                        contradiction_mark = " ⚠CONFLICT"
                        break
                lines.append(f"[Conv {conv_id} | score={r.score:.2f}{contradiction_mark}]")
                lines.append(r.text)
                lines.append("")

        if self.fixed_results:
            lines.append("=== POSITIONAL SUMMARIES (timeline context) ===")
            for r in self.fixed_results:
                start = r.metadata.get("start_global_index", "")
                end = r.metadata.get("end_global_index", "")
                lines.append(f"[Messages {start}–{end} | score={r.score:.2f}]")
                lines.append(r.text)
                lines.append("")

        return "\n".join(lines)


# ─── Main class ───────────────────────────────────────────────────────────────

class ConflictResolver:
    """
    Entity-aware re-ranker and contradiction detector.

    Stateless except for the VADER analyser and weights (loaded at init).
    Call resolve() once per query.
    """

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()
        self._weights = {
            "cosine":  CONFLICT_COSINE_WEIGHT,
            "entity":  CONFLICT_ENTITY_WEIGHT,
            "emotion": CONFLICT_EMOTION_WEIGHT,
            "recency": CONFLICT_RECENCY_WEIGHT,
        }
        logger.debug(
            f"ConflictResolver initialised — weights: "
            f"cosine={CONFLICT_COSINE_WEIGHT}, entity={CONFLICT_ENTITY_WEIGHT}, "
            f"emotion={CONFLICT_EMOTION_WEIGHT}, recency={CONFLICT_RECENCY_WEIGHT}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(
        self,
        ctx: RetrievalContext,
        query: Optional[str] = None,
    ) -> ResolvedContext:
        """
        Re-rank results and detect contradictions.

        Args:
            ctx:   The raw RetrievalContext from the Retriever.
            query: The original user query (uses ctx.query if omitted).

        Returns:
            A ResolvedContext with re-ranked chunk_results and contradiction_flags.
        """
        query = query or ctx.query
        query_entities = self._extract_entities(query)

        # Build a sorted list of all conversation_ids for recency normalisation
        all_conv_ids = sorted({
            r.metadata.get("conversation_id", 0)
            for r in ctx.all_results
            if isinstance(r.metadata.get("conversation_id"), int)
        })

        # Re-rank chunk results (most entity-sensitive)
        reranked_chunks = self._rerank(
            ctx.chunk_results, query_entities, all_conv_ids
        )
        # Also re-rank topic results
        reranked_topics = self._rerank(
            ctx.topic_results, query_entities, all_conv_ids
        )

        # Detect contradictions across re-ranked chunks + topics
        flags = self._detect_contradictions(
            reranked_chunks + reranked_topics, query_entities
        )

        if flags:
            logger.info(
                f"ConflictResolver: {len(flags)} contradiction(s) found for query: "
                f"'{query[:60]}'"
            )

        return ResolvedContext(
            query=query,
            topic_results=reranked_topics,
            fixed_results=ctx.fixed_results,   # Fixed chunks not re-ranked (positional)
            chunk_results=reranked_chunks,
            contradiction_flags=flags,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _rerank(
        self,
        results: list[RetrievalResult],
        query_entities: set[str],
        all_conv_ids: list[int],
    ) -> list[RetrievalResult]:
        """Re-rank a list of results by composite score, updating result.score."""
        if not results:
            return []

        max_conv_id = max(all_conv_ids) if all_conv_ids else 1

        scored = []
        for r in results:
            composite = self._composite_score(r, query_entities, max_conv_id)
            # Mutate score in place (keeps original cosine for reference in metadata)
            r.metadata["original_cosine"] = r.score
            r.score = composite
            scored.append(r)

        return sorted(scored, key=lambda r: r.score, reverse=True)

    def _composite_score(
        self,
        result: RetrievalResult,
        query_entities: set[str],
        max_conv_id: int,
    ) -> float:
        """
        Compute composite relevance score for a single result.

        Score components:
            cosine  : original retrieval similarity (0–1)
            entity  : fraction of query entities found in this chunk (0–1)
            emotion : |VADER compound| of chunk text — emotionally charged = salient (0–1)
            recency : normalised conversation_id (0–1, higher = later conversation)
        """
        cosine_score = result.metadata.get("original_cosine", result.score)

        # Entity overlap
        chunk_entities = self._extract_entities(result.text)
        if query_entities:
            overlap = len(query_entities & chunk_entities) / len(query_entities)
        else:
            overlap = 0.0

        # Emotional weight
        vader_scores = self._vader.polarity_scores(result.text)
        emotion_score = abs(vader_scores["compound"])  # 0–1

        # Recency (weak signal)
        conv_id = result.metadata.get("conversation_id", 0)
        if isinstance(conv_id, int) and max_conv_id > 0:
            recency_score = conv_id / max_conv_id
        else:
            recency_score = 0.0

        composite = (
            self._weights["cosine"]  * cosine_score +
            self._weights["entity"]  * overlap +
            self._weights["emotion"] * emotion_score +
            self._weights["recency"] * recency_score
        )

        logger.debug(
            f"  {result.doc_id}: cosine={cosine_score:.3f}, entity={overlap:.3f}, "
            f"emotion={emotion_score:.3f}, recency={recency_score:.3f} "
            f"→ composite={composite:.3f}"
        )
        return round(composite, 4)

    # ── Contradiction detection ───────────────────────────────────────────────

    def _detect_contradictions(
        self,
        results: list[RetrievalResult],
        query_entities: set[str],
    ) -> list[ContradictionFlag]:
        """
        Compare top-N result pairs and flag contradictions.

        Capped at _MAX_CONTRADICTION_RESULTS to avoid pairwise explosion:
        18 results → 153 pairs (too noisy); 8 results → 28 pairs (useful).

        Two types:
          1. Negation clash  — one chunk negates an entity phrase in another
          2. Sentiment spread — VADER compound scores are far apart for the
                                same entity (above CONFLICT_SENTIMENT_SPREAD)
        """
        flags: list[ContradictionFlag] = []
        # Only compare the top-ranked results to keep signal-to-noise high
        candidates = results[:_MAX_CONTRADICTION_RESULTS]
        n = len(candidates)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = candidates[i], candidates[j]
                pair_flags = self._compare_pair(a, b, query_entities)
                flags.extend(pair_flags)

        return flags

    def _compare_pair(
        self,
        a: RetrievalResult,
        b: RetrievalResult,
        query_entities: set[str],
    ) -> list[ContradictionFlag]:
        """Check a single pair of results for contradictions."""
        flags = []

        sentiment_a = self._vader.polarity_scores(a.text)["compound"]
        sentiment_b = self._vader.polarity_scores(b.text)["compound"]

        # Only check pairs that share at least one query entity — otherwise
        # they're probably talking about different things entirely
        entities_a = self._extract_entities(a.text)
        entities_b = self._extract_entities(b.text)
        shared = (entities_a & entities_b) | (query_entities & entities_a) | (query_entities & entities_b)

        if not shared:
            return []

        entity_label = ", ".join(sorted(shared)[:3])  # Top 3 for readability

        # ── Check 1: Negation clash ──────────────────────────────────────────
        negation_flag = self._check_negation_clash(a, b, shared)
        if negation_flag:
            flags.append(ContradictionFlag(
                result_a_id=a.doc_id,
                result_b_id=b.doc_id,
                conflict_type="negation",
                entity=entity_label,
                sentiment_a=sentiment_a,
                sentiment_b=sentiment_b,
                description=negation_flag,
            ))

        # ── Check 2: Sentiment spread ────────────────────────────────────────
        spread = abs(sentiment_a - sentiment_b)
        if spread >= CONFLICT_SENTIMENT_SPREAD:
            # Only flag if both have non-negligible sentiment (avoid neutral/neutral)
            if abs(sentiment_a) > 0.1 and abs(sentiment_b) > 0.1:
                flags.append(ContradictionFlag(
                    result_a_id=a.doc_id,
                    result_b_id=b.doc_id,
                    conflict_type="sentiment_spread",
                    entity=entity_label,
                    sentiment_a=sentiment_a,
                    sentiment_b=sentiment_b,
                    description=(
                        f"Opposing emotional valence about '{entity_label}' "
                        f"(spread={spread:.2f} ≥ threshold={CONFLICT_SENTIMENT_SPREAD})"
                    ),
                ))

        return flags

    def _check_negation_clash(
        self,
        a: RetrievalResult,
        b: RetrievalResult,
        shared_entities: set[str],
    ) -> Optional[str]:
        """
        Return a description string if one chunk negates an entity in the other,
        else return None.

        Heuristic: an entity in chunk A is "negated" if a negation word appears
        within 3 tokens of that entity in chunk B (or vice versa).
        """
        for entity in shared_entities:
            if len(entity) < 3:  # Skip very short tokens
                continue
            negated_in_a = self._is_negated(entity, a.text)
            negated_in_b = self._is_negated(entity, b.text)

            if negated_in_a != negated_in_b:
                negator = "A" if negated_in_a else "B"
                return (
                    f"Entity '{entity}' is negated in chunk {negator} "
                    f"but not in the other"
                )
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> set[str]:
        """
        Extract meaningful content tokens from text.

        Heuristic: lowercase alphabetic tokens ≥ 4 chars that are not stopwords.
        Min length 4 (not 3) cuts out short filler tokens like 'ago', 'all',
        'got', 'let' that survived the stopword list.
        No NER dependency — keeps the resolver CPU-only and fast.
        """
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        return {t for t in tokens if len(t) >= 4 and t not in _STOPWORDS}

    def _is_negated(self, entity: str, text: str) -> bool:
        """
        Check if `entity` appears within 3 tokens of a negation word in `text`.

        Returns True if a negation word is found within the window.
        """
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        entity_lower = entity.lower()

        for idx, token in enumerate(tokens):
            if token == entity_lower or entity_lower in token:
                # Check ±3 token window for negation words
                window_start = max(0, idx - 3)
                window_end = min(len(tokens), idx + 4)
                window = tokens[window_start:window_end]
                if any(w in _NEGATIONS for w in window):
                    return True
        return False
