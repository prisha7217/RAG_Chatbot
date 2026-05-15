"""
intent/generate_training_data.py — Synthetic training data generator.

Uses the Groq API to generate realistic labeled examples for the intent
classifier. The LLM generates training data; it never does the classification
itself (that runs fully offline via the trained SVM).

Usage:
    python intent/generate_training_data.py

Output:
    intent/training_data.json  — list of {"text": str, "label": str}

Requires GROQ_API_KEY in environment (or .env file).
After running, manually review the output and fix any bad examples before
running `python main.py intent-train`.
"""

from __future__ import annotations

import json
import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from config import INTENT_CLASSES, INTENT_TRAINING_DATA_FILE, INTENT_EXAMPLES_PER_CLASS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Prompt templates per intent class ────────────────────────────────────────

CLASS_PROMPTS = {
    "reminder": """
You are generating training data for a chatbot that analyses conversations between strangers.
The chatbot has access to thousands of PersonaChat-style dialogues on topics like:
pets, hobbies, jobs, food, music, sports, family, travel, books, movies.

Generate exactly {n} short, realistic user questions that ask the chatbot to RECALL or
RETRIEVE something specific from a past conversation. These should feel like natural
memory/lookup requests.

Examples of the style:
- "Remind me what User 1 said about their dog"
- "What did User 2 mention about their job?"
- "Did User 1 talk about college?"
- "Can you recall if anyone mentioned a cat?"
- "What was said about cooking in that conversation?"

Rules:
- One question per line, no numbering or bullets
- Vary the phrasing significantly — avoid repeating the same sentence structure
- Mix in references to both User 1 and User 2
- Include varied topics: pets, careers, hobbies, family, food, music, sports, travel
- Keep each question under 20 words
- Do NOT include emotional or distressing questions
- Do NOT include list/enumeration requests

Output exactly {n} lines, nothing else.
""",

    "emotional-support": """
You are generating training data for a chatbot that analyses conversations between strangers.
The chatbot has access to thousands of PersonaChat-style dialogues.

Generate exactly {n} short, realistic user questions that express CONCERN about a person's
emotional state, ask about sad/difficult moments, or seek to understand negative emotional
content in a conversation.

Examples of the style:
- "User 2 seems really upset, what happened?"
- "Why did User 1 sound sad when talking about their family?"
- "Was User 1 going through a difficult time?"
- "User 2 mentioned something painful, can you explain?"
- "What made User 1 feel lonely in this conversation?"

Rules:
- One question per line, no numbering or bullets
- Vary phrasing significantly
- Include emotional keywords: sad, upset, difficult, lonely, frustrated, anxious, worried, depressed
- Mix in both User 1 and User 2
- Keep each question under 20 words

Output exactly {n} lines, nothing else.
""",

    "action-item": """
You are generating training data for a chatbot that analyses conversations between strangers.
The chatbot has access to thousands of PersonaChat-style dialogues.

Generate exactly {n} short, realistic user questions that ask for a STRUCTURED LIST,
ENUMERATION, or COMPREHENSIVE SUMMARY across conversations. These are bulk-retrieval
requests, not single-conversation lookups.

Examples of the style:
- "List all mentions of pets across conversations"
- "Show me every time someone talked about their job"
- "Find all users who mentioned running"
- "Summarise all references to family in these dialogues"
- "Give me a breakdown of topics mentioned most often"
- "Enumerate every mention of music"

Rules:
- One question per line, no numbering or bullets
- Must include action verbs: list, show, find, enumerate, summarise, compile, gather
- Vary the topics: hobbies, pets, jobs, food, travel, sports, books, music
- Keep each question under 20 words
- Do NOT include emotional questions

Output exactly {n} lines, nothing else.
""",

    "small-talk": """
You are generating training data for a chatbot that analyses conversations between strangers.

Generate exactly {n} short phrases that are CASUAL GREETINGS or SMALL TALK directed at
a chatbot — not requests for information, not questions about the dataset.

Examples of the style:
- "Hi!"
- "Hello there"
- "How are you doing?"
- "Thanks for the help"
- "You're really helpful"
- "Nice one"
- "That's interesting"
- "Cool, got it"
- "Sounds good"
- "See you later"

Rules:
- One phrase per line, no numbering or bullets
- Very short — under 10 words each
- Natural, casual tone
- Mix greetings, thanks, affirmations, farewells, filler phrases
- Do NOT include any questions about the dataset or conversations

Output exactly {n} lines, nothing else.
""",

    "unknown": """
You are generating training data for a chatbot that analyses PersonaChat-style conversations.

Generate exactly {n} short user messages that DON'T clearly fit into:
- reminder (recall something specific)
- emotional-support (asking about sad/difficult feelings)
- action-item (list/enumerate/summarise)
- small-talk (greetings and filler)

These should be ambiguous, vague, or general queries that don't map cleanly to any of the above.

Examples of the style:
- "What is PersonaChat?"
- "How does this work?"
- "Tell me about conversation 5"
- "What can you do?"
- "Explain the data"
- "Who are User 1 and User 2?"
- "What topics are covered?"
- "Give me an overview"
- "Is this based on real conversations?"

Rules:
- One message per line, no numbering or bullets
- Vary the type: meta questions, vague requests, general curiosity
- Keep each under 15 words

Output exactly {n} lines, nothing else.
""",
}


# Classes where English has limited phrase variety → over-generate and trim
# small-talk: "Hi/Hello/Hey" variants collapse fast after dedup
# emotional-support: "User X seems sad, what happened?" template repeats
GENERATION_MULTIPLIER = {
    "reminder":           2,
    "action-item":        2,
    "unknown":            1,
    "emotional-support":  4,
    "small-talk":         3,   # highest multiplier — fewest unique phrases
}


def generate_examples(client: Groq, label: str, n: int = INTENT_EXAMPLES_PER_CLASS) -> list[dict]:
    """Call Groq to generate n examples for a given intent class.

    Retries up to 3 times with exponential backoff on rate-limit errors.
    """
    from groq import RateLimitError, APIError

    multiplier = GENERATION_MULTIPLIER.get(label, 1)
    request_n = min(n * multiplier, 300)
    prompt = CLASS_PROMPTS[label].format(n=request_n).strip()
    logger.info(f"  Generating ~{request_n} '{label}' examples (target: {n}, multiplier: {multiplier}×)...")

    # Exponential backoff: 30s → 60s → 120s
    wait_times = [30, 60, 120]
    last_error = None

    for attempt, wait in enumerate(wait_times, start=1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.90,
                max_tokens=request_n * 30,
            )
            break  # success
        except RateLimitError as e:
            last_error = e
            if attempt == len(wait_times):
                logger.error(f"Rate limit hit {attempt} times for '{label}'. Giving up.")
                raise
            logger.warning(
                f"  Rate limit hit (attempt {attempt}/{len(wait_times)}). "
                f"Waiting {wait}s before retry..."
            )
            for remaining in range(wait, 0, -5):
                logger.info(f"    Retrying in {remaining}s...")
                time.sleep(5)
        except APIError as e:
            logger.error(f"  Groq API error for '{label}': {e}")
            raise

    raw_text = response.choices[0].message.content.strip()
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # Deduplicate (exact match after stripping list markers)
    seen = set()
    examples = []
    for line in lines:
        cleaned = line.lstrip("0123456789.-) ").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            examples.append({"text": cleaned, "label": label})

    # Trim to exactly n after dedup
    examples = examples[:n]

    actual = len(examples)
    if actual < n:
        logger.warning(
            f"  Only got {actual}/{n} unique examples for '{label}' after dedup. "
            f"You can manually add more to training_data.json."
        )
    else:
        logger.info(f"  ✓ {actual} '{label}' examples (deduplicated from {len(lines)} generated)")

    return examples


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    output_path = INTENT_TRAINING_DATA_FILE

    # ── Load existing data (top-up mode) ──────────────────────────────────────
    existing: list[dict] = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        logger.info(f"Loaded {len(existing)} existing examples from {output_path}")

    from collections import Counter
    current_counts = Counter(e["label"] for e in existing)

    # Find which classes are short
    short_classes = {
        label: INTENT_EXAMPLES_PER_CLASS - current_counts.get(label, 0)
        for label in INTENT_CLASSES
        if current_counts.get(label, 0) < INTENT_EXAMPLES_PER_CLASS
    }

    if not short_classes:
        logger.info("All classes already have enough examples. Nothing to generate.")
        logger.info("Current distribution:")
        for label, count in current_counts.most_common():
            logger.info(f"  {label:20s}: {count}")
        return

    logger.info(f"Classes needing top-up: {short_classes}")
    logger.info("─" * 56)

    # Build existing text set per label for cross-run dedup
    existing_texts = {
        label: {e["text"] for e in existing if e["label"] == label}
        for label in INTENT_CLASSES
    }

    all_examples = list(existing)  # start with what we have

    for label, needed in short_classes.items():
        logger.info(f"  Generating {needed} more '{label}' examples (have {current_counts.get(label,0)}/{INTENT_EXAMPLES_PER_CLASS})...")

        multiplier = GENERATION_MULTIPLIER.get(label, 1)
        # Ask for 3× needed to account for dedup against existing
        request_n = min(needed * max(multiplier, 3), 300)
        prompt = CLASS_PROMPTS[label].format(n=request_n).strip()

        from groq import RateLimitError, APIError
        wait_times = [30, 60, 120]

        for attempt, wait in enumerate(wait_times, start=1):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.90,
                    max_tokens=request_n * 30,
                )
                break
            except RateLimitError:
                if attempt == len(wait_times):
                    logger.error(f"Rate limit hit {attempt} times. Giving up on '{label}'.")
                    raise
                logger.warning(f"  Rate limit. Waiting {wait}s (attempt {attempt}/{len(wait_times)})...")
                for remaining in range(wait, 0, -5):
                    logger.info(f"    Retrying in {remaining}s...")
                    time.sleep(5)
            except APIError as e:
                logger.error(f"  API error: {e}")
                raise

        raw_text = response.choices[0].message.content.strip()
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # Dedup against BOTH existing and newly generated examples
        seen = set(existing_texts.get(label, set()))
        new_examples = []
        for line in lines:
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                new_examples.append({"text": cleaned, "label": label})
                if len(new_examples) == needed:
                    break  # got exactly what we need

        got = len(new_examples)
        if got < needed:
            logger.warning(f"  Only got {got}/{needed} new unique examples for '{label}'.")
        else:
            logger.info(f"  ✓ {got} new '{label}' examples added")

        all_examples.extend(new_examples)

        if label != list(short_classes.keys())[-1]:
            logger.info("  Waiting 10s before next class...")
            time.sleep(10)

    # Save merged result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    final_counts = Counter(e["label"] for e in all_examples)
    logger.info("─" * 56)
    logger.info(f"Saved {len(all_examples)} total examples to: {output_path}")
    logger.info("Final label distribution:")
    for label, count in final_counts.most_common():
        status = "✓" if count >= INTENT_EXAMPLES_PER_CLASS else f"⚠ short by {INTENT_EXAMPLES_PER_CLASS - count}"
        logger.info(f"  {label:20s}: {count:3d}  {status}")

    logger.info("")
    logger.info("Next: python main.py intent-train")


if __name__ == "__main__":
    main()
