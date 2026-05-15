"""
intent/__init__.py — Offline intent classification module.

Classifies user queries into 5 intent categories using an SVM trained on
sentence-transformer embeddings (all-MiniLM-L6-v2).

Intents:
    reminder          — "Remind me what User 1 said about dogs"
    emotional-support — "User 2 seems sad, what happened?"
    action-item       — "List all mentions of careers"
    small-talk        — "Hi, how are you?"
    unknown           — Everything else (default unified pipeline)

Usage:
    from intent.classifier import IntentClassifier
    classifier = IntentClassifier()
    label, confidence = classifier.predict(query, embedder)
"""
