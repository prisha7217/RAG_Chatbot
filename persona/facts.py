"""
persona/facts.py — Extract factual SVO triples about users via spaCy NLP.

Uses spaCy's dependency parser and NER to find statements like:
    "I study radiology"     → (I, study, radiology) → category: occupation
    "I have a dog"          → (I, have, dog)         → category: pet
    "I live in Portland"    → (I, live, Portland)    → category: location
    "My sister is a nurse"  → (sister, is, nurse)    → category: relationship

Only first-person statements (subject = "I" or "my") are extracted — these
are facts the speaker is asserting about themselves.

Usage:
    from persona.facts import FactExtractor
    extractor = FactExtractor()
    facts = extractor.extract(messages, speaker, conversation_id)
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import spacy

from data.models import Message
from persona.schema import PersonaFact

logger = logging.getLogger(__name__)

# ─── Category Keyword Mapping ─────────────────────────────────────────────────
# Maps verbs/predicates + objects to semantic categories.
# Order matters: first match wins.

CATEGORY_RULES: list[tuple[str, set[str], set[str]]] = [
    # (category, trigger_verbs, object_keywords)
    ("occupation", {"work", "study", "studying", "teach", "nurse", "doctor",
                    "engineer", "manage", "run", "own", "freelance", "major",
                    "train", "practice", "specialize", "employed", "hired"},
                   {"school", "college", "university", "hospital", "office",
                    "company", "radiology", "medicine", "law", "engineering",
                    "nursing", "education", "research", "music", "art", "design",
                    "tech", "software", "business", "marketing", "accounting",
                    "finance", "journalism", "pharmacy", "dentistry"}),

    ("education", {"study", "studying", "major", "graduate", "graduated",
                   "attend", "enrolled", "finished", "completed", "earned",
                   "pursuing", "getting"},
                  {"degree", "diploma", "bachelor", "master", "phd", "doctorate",
                   "certificate", "class", "course", "major", "minor", "thesis",
                   "dissertation", "college", "university", "school", "mba",
                   "stem", "humanities", "liberal arts"}),

    ("pet",        {"have", "own", "adopted", "rescue", "rescued", "got",
                    "taking care", "raising"},
                   {"dog", "cat", "pet", "puppy", "kitten", "fish", "bird",
                    "hamster", "rabbit", "turtle", "parrot", "guinea pig",
                    "goldfish", "lizard", "snake", "ferret"}),

    ("location",   {"live", "living", "moved", "move", "grew", "from", "born",
                    "raised", "visit", "visiting", "relocated", "relocating"},
                   {"city", "town", "state", "country", "portland", "new york",
                    "chicago", "california", "texas", "florida", "london",
                    "canada", "australia", "seattle", "boston", "atlanta",
                    "denver", "phoenix", "miami", "los angeles", "suburb"}),

    ("hobby",      {"love", "enjoy", "like", "play", "playing", "run", "running",
                    "cook", "cooking", "hike", "hiking", "paint", "painting",
                    "read", "reading", "write", "writing", "garden", "gardening",
                    "swim", "swimming", "cycle", "cycling", "knit", "knitting",
                    "craft", "dance", "dancing", "sing", "singing", "collect",
                    "collecting", "fish", "fishing", "hunt", "hunting",
                    "sew", "sewing", "bake", "baking"},
                   {"music", "cooking", "hiking", "reading", "yoga", "art",
                    "sports", "guitar", "piano", "drums", "violin", "chess",
                    "gaming", "climbing", "photography", "travel", "baking",
                    "fishing", "running", "cycling", "knitting", "painting",
                    "pottery", "woodworking", "sculpting", "embroidery"}),

    ("relationship", {"have", "married", "dating", "divorced", "single",
                      "engaged", "expecting", "raised", "adopted"},
                     {"wife", "husband", "girlfriend", "boyfriend", "partner",
                      "spouse", "son", "daughter", "kids", "children", "baby",
                      "sister", "brother", "mom", "dad", "parents", "grandma",
                      "grandpa", "family", "fiance", "fiancee", "newborn"}),

    ("achievement", {"won", "earned", "finished", "completed", "graduated",
                     "published", "launched", "built", "started", "received",
                     "awarded", "achieved", "accomplished", "ranked"},
                    {"award", "medal", "trophy", "prize", "scholarship",
                     "degree", "certification", "competition", "championship",
                     "record", "milestone", "promotion", "published", "book",
                     "paper", "patent", "business", "startup"}),

    ("health",     {"have", "dealing", "diagnosed", "suffer", "recovering",
                    "managing", "treating", "struggle"},
                   {"diabetes", "asthma", "anxiety", "depression", "adhd",
                    "chronic", "disability", "allergies", "cancer", "arthritis",
                    "condition", "illness", "disorder", "syndrome", "therapy",
                    "medication", "treatment", "surgery", "recovery"}),

    ("entertainment", {"love", "enjoy", "like", "watch", "watching", "listen",
                       "listening", "playing", "play", "follow", "read"},
                      {"movie", "movies", "film", "show", "series", "netflix",
                       "spotify", "podcast", "band", "album", "song", "game",
                       "video game", "book", "novel", "comic", "anime",
                       "documentary", "concert", "festival", "sports team"}),

    ("food",       {"love", "enjoy", "eat", "like", "hate", "prefer", "make",
                    "cook", "bake", "order", "crave"},
                   {"food", "pizza", "sushi", "burger", "pasta", "chicken",
                    "vegan", "vegetarian", "coffee", "tea", "beer", "wine",
                    "chocolate", "ice cream", "dessert", "spicy", "seafood",
                    "steak", "tacos", "ramen", "curry", "salad"}),

    ("vehicle",    {"have", "own", "drive", "ride", "bought", "restored",
                    "fixing", "building"},
                   {"car", "truck", "motorcycle", "bike", "bicycle", "van",
                    "suv", "impala", "tesla", "jeep", "vehicle", "camper",
                    "rv", "boat", "classic car", "muscle car"}),

    ("belief",     {"believe", "think", "feel", "support", "oppose", "agree",
                    "disagree", "value", "care"},
                   {"religion", "god", "church", "politics", "conservative",
                    "liberal", "environment", "climate", "veganism", "rights",
                    "equality", "justice", "freedom", "democracy", "abortion",
                    "gun", "immigration"}),
]

# First-person pronouns/subjects we care about
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our"}


class FactExtractor:
    """
    Extracts first-person factual SVO triples from messages using spaCy.
    Uses nlp.pipe() for batched processing (3-5x faster than one-at-a-time).
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self._nlp = spacy.load(model_name)
            logger.info(f"FactExtractor: loaded spaCy model '{model_name}'")
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            self._nlp = None

    def extract(
        self,
        messages: list[Message],
        speaker: str,
        conversation_id: int,
        batch_size: int = 64,
    ) -> list[PersonaFact]:
        """
        Extract first-person facts from messages using batched spaCy processing.
        Returns (facts, entities) but only facts are surfaced here.
        Use extract_with_entities() when you also need NER output.
        """
        facts, _ = self.extract_with_entities(messages, speaker, conversation_id, batch_size)
        return facts

    def extract_with_entities(
        self,
        messages: list[Message],
        speaker: str,
        conversation_id: int,
        batch_size: int = 64,
    ) -> tuple[list[PersonaFact], list[tuple[str, str]]]:
        """
        Extract facts AND named entities in a single spaCy pass (no duplicate work).

        Returns:
            (facts, entities) where entities is a list of (label, text) tuples.
        """
        if self._nlp is None:
            return [], []

        facts: list[PersonaFact] = []
        entities: list[tuple[str, str]] = []
        texts = [m.text for m in messages]

        # nlp.pipe() processes in batches — much faster than calling nlp() per message
        for doc, msg in zip(self._nlp.pipe(texts, batch_size=batch_size), messages):
            facts.extend(self._extract_facts_from_doc(doc, msg.text, speaker, conversation_id))
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC", "ORG", "PERSON", "WORK_OF_ART"):
                    entities.append((ent.label_, ent.text))

        return facts, entities

    def _extract_facts_from_doc(self, doc, text: str, speaker: str, conv_id: int) -> list[PersonaFact]:
        """Extract facts from an already-parsed spaCy Doc object."""
        facts: list[PersonaFact] = []

        for token in doc:
            # Look for verbs where the subject is first-person
            if token.pos_ != "VERB":
                continue

            subjects = [
                child for child in token.children
                if child.dep_ in ("nsubj", "nsubjpass")
            ]
            is_first_person = any(subj.text.lower() in FIRST_PERSON for subj in subjects)
            if not is_first_person:
                continue

            objects = [
                child for child in token.children
                if child.dep_ in ("dobj", "attr", "pobj", "prep", "acomp", "xcomp")
            ]
            if not objects:
                continue

            verb = token.lemma_.lower() or token.text.lower()  # fallback if lemma is empty
            for obj_token in objects:
                obj_text = self._get_span(obj_token)
                category = self._classify(verb, obj_text.lower())
                if category:
                    subj_text = subjects[0].text if subjects else "I"
                    facts.append(PersonaFact(
                        category=category,
                        subject=subj_text,
                        predicate=verb,
                        obj=obj_text,
                        raw_text=text,
                        speaker=speaker,
                        conversation_id=conv_id,
                        confidence=0.8,
                    ))

        # NER-based location facts
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                context = text.lower()
                if any(p in context for p in ["i live", "i'm from", "i moved", "i grew", "i was born"]):
                    facts.append(PersonaFact(
                        category="location",
                        subject="I",
                        predicate="live/from",
                        obj=ent.text,
                        raw_text=text,
                        speaker=speaker,
                        conversation_id=conv_id,
                        confidence=0.9,
                    ))


        return facts

    def _get_span(self, token) -> str:
        """Get the full noun phrase span for a token."""
        # Walk up to get the full subtree for compound nouns
        subtree = list(token.subtree)
        # Limit to 5 tokens to avoid huge phrases
        words = [t.text for t in subtree[:5] if not t.is_punct]
        return " ".join(words).strip() or token.text

    def _classify(self, verb: str, obj_lower: str) -> str | None:
        """
        Classify a (verb, object) pair into a category.
        Returns the category string or None if no match.
        """
        for category, trigger_verbs, object_keywords in CATEGORY_RULES:
            verb_match = verb in trigger_verbs
            obj_match = any(kw in obj_lower for kw in object_keywords)

            if verb_match and obj_match:
                return category
            # Loose match: object strongly implies a category regardless of verb
            if obj_match and category in ("pet", "location", "vehicle"):
                return category

        return None


def aggregate_facts(
    all_facts: list[PersonaFact],
    min_frequency: int = 1,
) -> list[PersonaFact]:
    """
    Deduplicate and score facts by frequency.

    Facts that appear multiple times (same category + object) get higher
    confidence scores. Facts below min_frequency are filtered out.

    Args:
        all_facts: All extracted facts for one speaker.
        min_frequency: Minimum occurrence count to keep a fact.

    Returns:
        Deduplicated facts sorted by confidence descending.
    """
    # Count occurrences of (category, normalized_obj) pairs
    counter: Counter = Counter()
    fact_map: dict[tuple, PersonaFact] = {}

    for fact in all_facts:
        key = (fact.category, fact.obj.lower().strip())
        counter[key] += 1
        if key not in fact_map:
            fact_map[key] = fact

    # Build deduplicated list with boosted confidence
    result = []
    max_count = max(counter.values()) if counter else 1

    for key, count in counter.items():
        if count < min_frequency:
            continue
        base_fact = fact_map[key]
        boosted_confidence = min(1.0, base_fact.confidence + (count / max_count) * 0.2)
        result.append(
            PersonaFact(
                category=base_fact.category,
                subject=base_fact.subject,
                predicate=base_fact.predicate,
                obj=base_fact.obj,
                raw_text=base_fact.raw_text,
                speaker=base_fact.speaker,
                conversation_id=base_fact.conversation_id,
                confidence=round(boosted_confidence, 3),
            )
        )

    return sorted(result, key=lambda f: f.confidence, reverse=True)
