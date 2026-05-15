# Persona-Aware RAG — Round 2: Intelligence Layer

This document covers the second phase of the project, which adds three new intelligence modules on top of the base RAG system. These modules make the chatbot significantly smarter about **how** to answer, not just **what** to retrieve.

> **This document covers Round 2 only.** See [README.md](README.md) for the base system (topic detection, summarization, retrieval, persona extraction, Gradio UI).

---

## Table of Contents

- [What Changed](#what-changed)
- [How Drift Detection Works](#how-drift-detection-works)
- [How Intent Classification Works](#how-intent-classification-works)
- [How Conflict Resolution Works](#how-conflict-resolution-works)
- [How the Serve Phase Changed](#how-the-serve-phase-changed)
- [Updated Project Structure](#updated-project-structure)
- [How to Run the New Steps](#how-to-run-the-new-steps)
- [New Configuration Settings](#new-configuration-settings)
- [Design Decisions](#design-decisions)

---

## What Changed

Round 1 built a system that could **find and present** relevant conversation data. Round 2 adds a layer that understands the **intent behind the query** and **resolves conflicts** in retrieved results before they reach the LLM.

```
Round 1 pipeline:
  query → embed → ChromaDB → context string → Groq → answer

Round 2 pipeline:
  query → IntentClassifier → [small-talk fast-path]
                            ↓
                       embed → ChromaDB
                            ↓
                     ConflictResolver (re-rank + contradiction detection)
                            ↓
               context string (+ tone hint + drift timeline + contradiction warning)
                            ↓
                          Groq → answer
```

Three new modules were added, one new build step, and the serve pipeline was wired to use all of them.

---

## How Drift Detection Works

**Module:** `persona/drift.py` — run once, offline, as a build step

The drift detector analyses **how each person's mood and communication style shifts throughout a conversation**. It reads the topic checkpoints that were already generated in Round 1 and computes a timeline of emotional changes for each speaker in each conversation.

### What it measures per segment

For each topic segment, it computes:

| Signal | How | What it captures |
|---|---|---|
| **VADER sentiment** | Average compound score of all messages | Overall emotional tone (+1 = very positive, -1 = very negative) |
| **Question rate** | Questions / total messages | Curiosity level |
| **Exclamation rate** | Exclamations / total messages | Emotional expressiveness |
| **Message length** | Average words per message | Engagement / talkativeness |

Each segment gets a **mood label** that combines sentiment and style:

```
avg_sentiment > 0.3  AND  question_rate > 0.3  →  "very positive & curious"
avg_sentiment < 0.0  AND  exclamation_rate < 0.1  →  "negative & formal"
avg_sentiment ≈ 0    AND  message_length > 20  →  "neutral & talkative"
```

### What counts as a drift event

After scoring all segments, adjacent pairs are compared. A drift event is flagged when:

- **Sentiment drift**: the compound score changes by more than 0.15 between segments
- **Style drift**: the question or exclamation rate changes by more than 0.20

Each event records what the trigger topic was (the topic label of the new segment) and which keywords the topic introduced.

### Example

```
Conversation 2 — User 2:

Segment 0: "Enjoy & Today & Parmesan & Chicken"
  → very positive & curious (sentiment: +0.67)

Segment 1: "Pets & Dog & Best & Cat"
  → neutral & curious (sentiment: +0.14)
  ⚑ DRIFT EVENT: sentiment dropped 0.53 (topic: pets)

Segment 2: "Misses & Miss & Does"
  → negative & formal (sentiment: -0.08)
  ⚑ DRIFT EVENT: continued drop 0.21 (topic: someone being missed)

Segment 3: "Chatting & Talking & Hope"
  → very positive & expressive (sentiment: +0.49)
  ⚑ DRIFT EVENT: recovered 0.57 (ended on a cheerful note)
```

This timeline gets saved and surfaced in two ways: injected into the LLM context so it can explain emotional shifts, and shown in the debug panel so you can see exactly when and why a conversation's mood changed.

### Output

```
outputs/persona/drift_timelines.json
  Format: {"0_User 1": {conversation_id, speaker, segments, drift_events}, ...}
  Size:   ~62MB
  Stats:  21,843 timelines across 11,000 conversations, 48,488 total drift events
```

---

## How Intent Classification Works

**Module:** `intent/classifier.py` — loaded at serve startup, ~5ms per query

The intent classifier figures out **what kind of question the user is asking** before retrieval even starts. This lets the system handle different query types differently instead of sending everything through the same generic RAG pipeline.

### The five intents

| Intent | What it means | Example queries |
|---|---|---|
| `reminder` | Looking up a specific fact | *"What did user 1 say about their dog?"*, *"Remind me what they mentioned about travel"* |
| `emotional-support` | Asking about emotional/sad content | *"Why did user 2 seem upset?"*, *"Was anyone going through something difficult?"* |
| `action-item` | Wants a list or enumeration | *"List all the hobbies mentioned"*, *"What topics come up most?"* |
| `small-talk` | Greeting or filler | *"Hi there"*, *"How does this work?"*, *"Thanks!"* |
| `unknown` | Everything else | Ambiguous or out-of-scope queries — routed through standard RAG |

### Why SVM, not the LLM?

The LLM could classify intent too, but that would cost an API call (and its latency) before we even start retrieval. Instead, we train a small **SVM classifier on sentence-transformer embeddings**:

- Embeddings come from `all-MiniLM-L6-v2` — the same model already loaded for retrieval, so no extra model to load
- Inference is ~5ms on CPU
- No API calls, fully offline
- Confidence threshold of 55% — below this, the query falls back to `unknown` rather than misclassifying

### How the model was trained

```
1. Generate 100 synthetic examples per class using Groq
   (uses real dataset topic labels as context for realistic phrasing)

2. Embed all 500 examples using all-MiniLM-L6-v2

3. 80/20 train/test split + 5-fold cross-validation

4. Train SVC(kernel='rbf', probability=True)
   → Test accuracy: 100% on held-out set

5. Save to outputs/intent/intent_svm.pkl
```

The training data was manually curated after generation to cover **implicit** phrasings — things like *"anything about their dog?"* (a reminder, even though it doesn't say "remind me") or *"can you tell me more"* (unknown, because it's too vague to classify).

### How it affects the pipeline

- **`small-talk`** → immediate friendly redirect response, zero retrieval cost, sub-200ms latency
- **`reminder`** → LLM receives `[TONE GUIDANCE]`: be concise, cite specific conversations
- **`emotional-support`** → LLM receives `[TONE GUIDANCE]`: be empathetic, acknowledge feelings first
- **`action-item`** → LLM receives `[TONE GUIDANCE]`: format as a structured list
- **`unknown`** → standard pipeline, no tone hint

---

## How Conflict Resolution Works

**Module:** `retrieval/conflict_resolver.py` — runs after retrieval, before the LLM

The conflict resolver sits between ChromaDB and the answer generator. It does two things: **re-rank** the retrieved results using a smarter scoring formula, and **detect contradictions** between results that might confuse the LLM.

### Re-ranking: composite score

Raw ChromaDB results are ranked purely by cosine similarity. The resolver replaces that with a composite score that considers four factors:

```
composite = 0.40 × cosine_similarity
          + 0.30 × entity_overlap       ← how many query keywords appear in this chunk
          + 0.20 × emotional_weight     ← |VADER compound| of the chunk text
          + 0.10 × recency              ← conversation_id normalised (later = higher)
```

**Why entity overlap matters:** a chunk about *"my golden retriever Buddy"* might have slightly lower cosine similarity than a generic pet chunk, but it contains the specific entity the user asked about. The 30% entity weight promotes it above the generic one.

**Why emotional weight matters:** emotionally charged chunks (high sentiment magnitude, either positive or negative) tend to be more content-rich than neutral filler. A chunk with a VADER score of +0.95 is probably someone talking about something they care about, which is usually more relevant than a neutral confirmation.

The original cosine score is preserved in `metadata["original_cosine"]` so you can see the delta in the debug panel.

### Contradiction detection

After re-ranking, the top 8 results are compared in pairs (28 comparisons max). Two types of contradictions are flagged:

**Type 1 — Negation clash:**
One chunk mentions an entity, and another chunk negates that same entity within a 3-token window.

```
Chunk A: "User 1 has a dog — a golden retriever named Buddy."  (positive)
Chunk B: "User 1 doesn't have any pets."                       (negated)

→ ⚠ Contradiction [negation] on 'golden, retriever': 
     "Entity 'dog' is negated in chunk B but not in the other"
```

**Type 2 — Sentiment spread:**
Two chunks share meaningful entities but have VADER compound scores that are far apart (threshold: 0.60 apart), and both are non-neutral (|score| > 0.1).

```
Chunk A (sentiment: +0.89): positive story about a pet
Chunk B (sentiment: -0.75): someone expressing grief about losing a pet

→ ⚠ Contradiction [sentiment_spread] on 'dog, pet':
     "Opposing emotional valence (spread=1.64 ≥ threshold=0.60)"
```

### What counts as an entity (and what doesn't)

The entity extractor uses a filtered token set — lowercase alphabetic tokens, minimum 4 characters, excluding an extended stopword list. The stopwords include not just standard English stopwords but also:

- Contractions: `it's`, `what's`, `he's`, `they're`, etc.
- High-frequency conversational filler: `sorry`, `interesting`, `basically`, `sounds`, `going`, `anyway`, `honestly`
- Common verbs: `make`, `take`, `want`, `know`, `hear`, `done`

This prevents the detector from flagging `"what's"` as a contradicted entity just because a negation word appeared nearby, which was the source of most false positives in early testing.

### What the LLM sees

Detected contradictions are surfaced in two ways:

1. **In the context string** — a `⚠` warning header before the retrieved text tells the LLM conflicting claims exist and to flag them in its answer
2. **In the debug panel** — each contradiction flag shows which chunks are involved, what entity triggered it, and the sentiment polarity of both sides

---

## How the Serve Phase Changed

The `serve/app.py` and `serve/context_builder.py` were both updated to use the new modules. Here's the new request flow:

```
User sends message
      │
      ▼
IntentClassifier.predict(message, embedder)
      │
      ├─ small-talk? → immediate reply (skip everything below)
      │
      ▼
Retriever.retrieve(query)         (unchanged from Round 1)
      │
      ▼
ConflictResolver.resolve(ctx)
      │  - re-ranks chunk_results by composite score
      │  - detects contradictions across top-8 results
      │  - returns ResolvedContext (drop-in for RetrievalContext)
      │
      ▼
PersonaContextBuilder.build(query, resolved_ctx, mode, intent)
      │  - [TONE GUIDANCE] hint (based on intent)
      │  - ⚠ contradiction warning (if flags exist)
      │  - aggregate persona section
      │  - per-conversation persona facts
      │  - PERSONA DRIFT TIMELINE (mood shifts for retrieved conversations)
      │  - retrieved text (from resolved_ctx.to_context_string())
      │
      ▼
Generator.generate(query, resolved_ctx, full_context_str)
      │  - Groq sees: tone hint + contradiction warning + drift timeline + retrieved text
      │
      ▼
Debug panel:
      - Intent badge (icon + label + confidence %)
      - Contradiction flags (type, entity, both chunk IDs, sentiment polarity)
      - Top segments: composite score + original cosine side-by-side
      - ⚠ marker on chunks involved in contradictions
      - 📊 Drift Timeline (seg→seg | type | Δ)
```

### Small-talk fast-path

Queries classified as `small-talk` return immediately with a canned response that redirects the user to dataset queries. The entire retrieval + generation pipeline is skipped, which makes these responses ~10× faster and prevents the LLM wasting tokens on off-topic input.

---

## Updated Project Structure

```
rag_task/
├── intent/                        # NEW — Intent classification module
│   ├── __init__.py
│   ├── classifier.py              # IntentClassifier: load SVM, predict()
│   ├── train.py                   # SVM training script (run once)
│   ├── generate_training_data.py  # Groq-based synthetic data generation
│   ├── training_data.json         # 500 curated examples (100 per class)
│   └── quick_compare.py           # SVM vs TF-IDF side-by-side comparison
│
├── persona/
│   ├── drift.py                   # NEW — DriftDetector: mood timeline per speaker
│   └── schema.py                  # UPDATED — added DriftEvent, ConversationTimeline
│
├── retrieval/
│   ├── conflict_resolver.py       # NEW — entity-aware re-ranker + contradiction detector
│   └── retriever.py               # Unchanged
│
├── serve/
│   ├── app.py                     # UPDATED — intent + conflict resolver wired in
│   └── context_builder.py         # UPDATED — drift timelines, tone hints, contradiction warning
│
├── generation/
│   └── generator.py               # UPDATED — system prompt aware of new context signals
│
├── outputs/
│   ├── intent/
│   │   └── intent_svm.pkl         # NEW — trained SVM model (~few KB)
│   └── persona/
│       └── drift_timelines.json   # NEW — 21,843 timelines, 48,488 events (~62MB)
│
├── tests/
│   ├── test_drift.py              # NEW — drift detection unit tests
│   └── test_conflict_resolver.py  # NEW — 15 tests for conflict resolver
│
└── config.py                      # UPDATED — drift + intent + conflict constants
```

---

## How to Run the New Steps

Round 2 adds two new build steps. Run them **after** the Round 1 build and persona steps:

### Step 3 — Detect Drift

```bash
python main.py drift
```

**What this does:**
1. Loads all topic checkpoints from `outputs/checkpoints/`
2. Groups segments by conversation_id and speaker
3. Computes VADER sentiment + style stats per segment
4. Compares adjacent segments → generates drift events
5. Saves `outputs/persona/drift_timelines.json`

**Time:** ~5 minutes on CPU  
**Output:** `outputs/persona/drift_timelines.json` (62MB)

### Step 4 — Train the Intent Classifier

```bash
# First, generate training data (needs GROQ_API_KEY):
python intent/generate_training_data.py

# Then train the SVM:
python main.py intent-train
```

**What this does:**
1. Loads `intent/training_data.json` (500 examples)
2. Embeds all examples using `all-MiniLM-L6-v2`
3. Trains SVM with 5-fold cross-validation
4. Prints accuracy + per-class report
5. Saves model to `outputs/intent/intent_svm.pkl`

**Time:** ~2 minutes  
**Output:** `outputs/intent/intent_svm.pkl`

> If you skip this step, the classifier will log a warning and return `("unknown", 0.0)` for all queries — the chatbot will still work, just without intent routing.

### Step 5 — Launch with Full Round 2 Pipeline

```bash
python main.py serve
```

Same command as before — the new modules load automatically at startup. You'll see these log lines confirming everything loaded:

```
INFO | serve.app | Loading intent classifier...
INFO | intent.classifier | Intent classifier loaded: 5 classes — action-item, emotional-support, reminder, small-talk, unknown
INFO | serve.app | Loading conflict resolver...
INFO | serve.context_builder | Loaded drift timelines for 10,922 conversations.
```

---

## New Configuration Settings

All new constants are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `DRIFT_SENTIMENT_THRESHOLD` | `0.15` | Minimum sentiment delta to flag a drift event |
| `DRIFT_STYLE_THRESHOLD` | `0.20` | Minimum change in question/exclamation rate for style drift |
| `INTENT_CLASSES` | 5 classes | The 5 intent labels the SVM knows |
| `INTENT_CONFIDENCE_FLOOR` | `0.55` | Below this confidence → fall back to `unknown` |
| `INTENT_EXAMPLES_PER_CLASS` | `100` | Examples generated per class during training data creation |
| `CONFLICT_COSINE_WEIGHT` | `0.40` | Weight for original vector similarity in composite score |
| `CONFLICT_ENTITY_WEIGHT` | `0.30` | Weight for entity overlap with query |
| `CONFLICT_EMOTION_WEIGHT` | `0.20` | Weight for emotional significance (VADER \|compound\|) |
| `CONFLICT_RECENCY_WEIGHT` | `0.10` | Weight for conversation_id recency (weak tiebreaker) |
| `CONFLICT_SENTIMENT_SPREAD` | `0.60` | VADER spread threshold to flag sentiment contradiction |

---

## Design Decisions

### Why not use spaCy NER for entity extraction in the conflict resolver?

The conflict resolver runs on every query at serve time. Loading spaCy's NER pipeline would add ~200ms per query and a significant memory footprint. The simple stopword-filtered token approach (4+ char tokens, extended stopword list) achieves the same practical result for conversational text — the entities that matter for contradiction detection (pet names, locations, occupations) are typically 4+ characters and not in any stopword list. The false-positive rate after tuning is low enough that spaCy NER wouldn't add meaningful value.

### Why train SVM instead of using a zero-shot LLM classifier?

Three reasons: latency (5ms vs 300ms+), cost (zero API calls per query), and reliability (doesn't depend on prompt engineering or API availability). The 100-example-per-class training set was sufficient for 100% test accuracy because the five intents have genuinely distinct semantic signatures in embedding space — `small-talk` clusters far from `reminder`, `emotional-support` clusters far from `action-item`, etc.

### Why cap contradiction comparison at top 8 results?

With 18 retrieved results, pairwise comparison produces 153 pairs. Even with a tight entity filter, this generates too much noise — casual conversation chunks inevitably share some common words that look like contradictions. Capping at 8 results gives 28 pairs and focuses the detector on the results most likely to be used by the LLM, which is where contradictions actually matter.

### Why merge both speakers' drift events per conversation?

The drift timelines file stores separate timelines for User 1 and User 2 within the same conversation. For the serve-phase lookup, we merge them into a single list per `conversation_id`. This means the context builder can look up by conversation ID (which is what we have from retrieval metadata) rather than having to know which speaker to look up. The slight loss of speaker attribution in the debug panel is worth the simpler lookup.
