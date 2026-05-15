"""
intent/quick_compare.py — Compare SVM vs TF-IDF on custom queries.

Trains TF-IDF+LogReg on the fly (since compare.py doesn't save it),
then runs both classifiers on a set of test queries side by side.

Usage:
    python intent/quick_compare.py
"""

import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from config import INTENT_MODEL_FILE, INTENT_TRAINING_DATA_FILE

# ─── Load training data ───────────────────────────────────────────────────────
with open(INTENT_TRAINING_DATA_FILE) as f:
    data = json.load(f)
texts = [d["text"] for d in data]
labels = [d["label"] for d in data]

# ─── Train TF-IDF + LogReg (on-the-fly, not saved) ───────────────────────────
print("Training TF-IDF + LogReg...")
vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf = vec.fit_transform(texts)
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_tfidf, labels)

# ─── Load SVM ─────────────────────────────────────────────────────────────────
print("Loading SVM + sentence-transformer embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
with open(INTENT_MODEL_FILE, "rb") as f:
    svm = pickle.load(f)

# ─── Queries to test ──────────────────────────────────────────────────────────
# Edit this list to test your own queries
TEST_QUERIES = [
    "what did they talk about",
    "can you tell me more",
    "anything about their dog?",
    "that conversation was interesting",
    "why did user 1 get upset",
    "remind me what user 1 said",
    "list all mentions of pets",
    "hi there",
    "how does this work?",
    "did they mention travel?",
]

# ─── Run comparison ───────────────────────────────────────────────────────────
print()
print(f"{'Query':<38} {'SVM':<28} {'TF-IDF'}")
print("-" * 84)

for q in TEST_QUERIES:
    # SVM prediction
    emb = embedder.encode([q], normalize_embeddings=True)
    svm_proba = svm.predict_proba(emb)[0]
    svm_idx = svm_proba.argmax()
    svm_label = svm.classes_[svm_idx]
    svm_conf = svm_proba[svm_idx]

    # TF-IDF prediction
    tfidf_proba = lr.predict_proba(vec.transform([q]))[0]
    tfidf_idx = tfidf_proba.argmax()
    tfidf_label = lr.classes_[tfidf_idx]
    tfidf_conf = tfidf_proba[tfidf_idx]

    # Format: label (conf%) — flag if they disagree
    svm_str   = f"{svm_label} ({svm_conf:.0%})"
    tfidf_str = f"{tfidf_label} ({tfidf_conf:.0%})"
    flag = "  ← DISAGREE" if svm_label != tfidf_label else ""

    print(f"{q:<38} {svm_str:<28} {tfidf_str}{flag}")

print()
print("Tip: edit TEST_QUERIES in this script to test your own queries.")
