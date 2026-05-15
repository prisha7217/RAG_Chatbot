"""
config.py — Central configuration and constants for the RAG Chatbot system.
All tunable parameters, file paths, and model names live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ─── Project Root ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = ROOT_DIR / "logs"

# ─── Output Subdirectories ───────────────────────────────────────────────────
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PERSONA_DIR = OUTPUTS_DIR / "persona"
INDEX_DIR = OUTPUTS_DIR / "index"

# ─── Data ────────────────────────────────────────────────────────────────────
CONVERSATIONS_CSV = ROOT_DIR / "conversations.csv"

# ─── Embedding Model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 80MB, fast, CPU-friendly

# ─── Topic Chunking ──────────────────────────────────────────────────────────
WINDOW_SIZE = 5                  # Sliding window size (messages)
COSINE_WEIGHT = 0.7              # Weight for cosine similarity signal
KEYWORD_WEIGHT = 0.3             # Weight for keyword overlap signal
BASE_TOPIC_THRESHOLD = 0.35      # Default combined-score threshold for topic break
ADAPTIVE_THRESHOLD_FACTOR = 0.5  # mean - factor * std  →  adaptive threshold
MIN_TOPIC_MESSAGES = 3           # Minimum messages per topic segment
MAX_TOPIC_MESSAGES = 15          # Maximum messages before forcing a split
TOP_K_KEYWORDS = 10              # TF-IDF keywords extracted per window

# ─── Summarization ───────────────────────────────────────────────────────────
SUMMARY_TOP_K = 3                # Number of messages to pick for extractive summary

# ─── Fixed Checkpoints ───────────────────────────────────────────────────────
FIXED_CHECKPOINT_SIZE = 100      # Messages per fixed checkpoint

# ─── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K_TOPICS = 5
RETRIEVAL_TOP_K_CHUNKS = 10
RETRIEVAL_TOP_K_FIXED = 3
MESSAGE_CHUNK_SIZE = 7           # Messages per raw chunk stored in vector DB

# ─── ChromaDB ────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_TOPICS = "topic_summaries"
CHROMA_COLLECTION_FIXED = "fixed_summaries"
CHROMA_COLLECTION_CHUNKS = "message_chunks"

# ─── Generation (Serve Phase) ────────────────────────────────────────────────
# Set via environment variable: export GROQ_API_KEY=...
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.3-70b-versatile"   # Fast, large context, free tier
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"        # Fallback model if using OpenAI
LLM_TEMPERATURE = 0.3

# ─── Persona Extraction ──────────────────────────────────────────────────────
MIN_FACT_FREQUENCY = 3           # Min conversations a fact must appear in → high confidence
LDA_N_TOPICS = 15                # Number of LDA topics to discover
LDA_MAX_FEATURES = 1000          # Max vocabulary size for LDA
TRAIT_AUTO_LABEL_SAMPLE = 500    # Messages to auto-label for trait classifier bootstrap

# ─── Ensure output directories exist ─────────────────────────────────────────
for _dir in [CHECKPOINTS_DIR, PERSONA_DIR, INDEX_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─── Round 2: Drift Detection ─────────────────────────────────────────────────
DRIFT_SENTIMENT_THRESHOLD = 0.15   # Minimum sentiment delta to flag a drift event
DRIFT_STYLE_THRESHOLD     = 0.20   # Minimum change in question/exclamation rate for style drift
DRIFT_TIMELINES_FILE      = PERSONA_DIR / "drift_timelines.json"


# ─── Round 2: Intent Classification ──────────────────────────────────────────
INTENT_CLASSES = ["reminder", "emotional-support", "action-item", "small-talk", "unknown"]
INTENT_TRAINING_DATA_FILE = ROOT_DIR / "intent" / "training_data.json"
INTENT_MODEL_DIR          = OUTPUTS_DIR / "intent"
INTENT_MODEL_FILE         = INTENT_MODEL_DIR / "intent_svm.pkl"
INTENT_EXAMPLES_PER_CLASS = 100    # Examples to generate per class via Groq
INTENT_CONFIDENCE_FLOOR   = 0.55   # Below this confidence → fall back to "unknown"

# Ensure intent model directory exists
INTENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Round 2: Conflict Resolution ─────────────────────────────────────────────
CONFLICT_COSINE_WEIGHT    = 0.40   # Original vector similarity
CONFLICT_ENTITY_WEIGHT    = 0.30   # Entity overlap with query
CONFLICT_EMOTION_WEIGHT   = 0.20   # Emotional significance (VADER |compound|)
CONFLICT_RECENCY_WEIGHT   = 0.10   # Conversation_id order (weak tiebreaker)
CONFLICT_SENTIMENT_SPREAD = 0.40   # VADER polarity spread threshold → contradiction flag
