"""
serve/app.py — Gradio 6.x chatbot UI for the RAG + Persona system.

Three response modes:
    Unified     — Persona context + RAG retrieval + template answer
    RAG Only    — Retrieval + template, no persona context
    Persona Only — Persona context only, no retrieval

Run with:
    python main.py serve
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever
from generation.generator import Generator
from serve.context_builder import PersonaContextBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Startup — load all heavy components once ────────────────────────────────

logger.info("Loading embedding model...")
_model = SentenceTransformer(EMBEDDING_MODEL)
_embedder = Embedder(model=_model)

logger.info("Connecting to ChromaDB index...")
_retriever = Retriever(_embedder)

logger.info("Loading generator...")
_generator = Generator()

logger.info("Loading persona context builder...")
_persona_builder = PersonaContextBuilder()

logger.info("All components loaded. Starting Gradio UI...")

# ─── Core chat function ───────────────────────────────────────────────────────

def chat(
    message: str,
    history: list[dict],
    mode: str,
    top_k: int,
) -> tuple[list[dict], str]:
    if not message.strip():
        return history, ""

    t0 = time.time()
    import re

    mode_key = {
        "🔗 Unified (Recommended)": "unified",
        "📚 RAG Only": "rag_only",
        "👤 Persona Only": "persona_only",
    }.get(mode, "unified")

    # ── Detect direct conversation ID reference ───────────────────────────────
    # Matches: "conversation 42", "conv 42", "conv. 42", "#42"
    conv_id_match = re.search(
        r'\b(?:conversation|conv\.?)\s*#?(\d+)\b|#(\d+)\b',
        message, re.IGNORECASE
    )
    direct_conv_id: int | None = None
    if conv_id_match:
        raw_id = conv_id_match.group(1) or conv_id_match.group(2)
        direct_conv_id = int(raw_id)

    # ── Retrieve ──────────────────────────────────────────────────────────────
    if direct_conv_id is not None:
        retrieval_ctx = _retriever.retrieve_conversation(direct_conv_id)
    else:
        retrieval_ctx = _retriever.retrieve(query=message, top_k_topics=int(top_k))

    # Build persona-enriched context
    full_context = _persona_builder.build(
        query=message,
        retrieval_context=retrieval_ctx,
        mode=mode_key,
    )

    # Generate
    answer = _generator.generate(
        query=message,
        context=retrieval_ctx,
        full_context_str=full_context,
    )

    elapsed = time.time() - t0

    # Debug panel
    top_results = retrieval_ctx.all_results[:5]
    debug_lines = [
        f"**Mode:** {mode} | **Latency:** {elapsed:.2f}s | **Results:** {len(retrieval_ctx.all_results)}",
        f"**Lookup:** {'Direct (conv ' + str(direct_conv_id) + ')' if direct_conv_id is not None else 'Semantic search'}",
        "",
        "**Top retrieved segments:**",
    ]
    for r in top_results:
        cid = r.metadata.get("conversation_id", "?")
        label = r.metadata.get("topic_label", "")
        debug_lines.append(
            f"- `[{r.source}]` Conv {cid} | score={r.score:.3f}"
            + (f" | {label}" if label else "")
        )

    conv_ids = list({
        str(r.metadata.get("conversation_id", ""))
        for r in top_results
        if r.metadata.get("conversation_id") is not None
    })[:4]
    if conv_ids and _persona_builder.has_personas():
        debug_lines.append(f"\n**Persona lookups:** conv IDs {conv_ids}")

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return new_history, "\n".join(debug_lines)


# ─── Gradio UI (Gradio 6.x compatible) ───────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

CSS = """
.chat-wrap { height: 500px !important; overflow-y: auto; }
.debug-md { font-size: 0.82em; }
"""

EXAMPLE_QUESTIONS = [
    "Does User 1 have any pets?",
    "What are User 2's hobbies?",
    "What do the users talk about most often?",
    "Is there anyone who works in healthcare?",
    "Tell me about User 1's personality and communication style.",
    "What kinds of food or music do users like?",
    "Tell me about conversation 0",
    "What did the people in conversation 42 talk about?",
    "Show me the persona for conversation 100",
]


def build_ui():
    with gr.Blocks(title="RAG Persona Chatbot") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:20px 0 10px 0;">
            <h1 style="font-size:1.9em; font-weight:700; margin:0; color:#a78bfa;">
                💬 Conversation Intelligence Chatbot
            </h1>
            <p style="color:#94a3b8; margin:6px 0 0 0; font-size:0.92em;">
                RAG + Persona-Aware · 191,592 messages · 11,003 conversations · 9,303 persona profiles
            </p>
        </div>
        """)

        with gr.Row():
            # ── Sidebar ───────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=230):
                gr.Markdown("### ⚙️ Settings")

                mode = gr.Radio(
                    choices=[
                        "🔗 Unified (Recommended)",
                        "📚 RAG Only",
                        "👤 Persona Only",
                    ],
                    value="🔗 Unified (Recommended)",
                    label="Response Mode",
                )

                top_k = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Retrieved Segments",
                )

                gr.Markdown("""---
**🔗 Unified** — Persona facts + retrieved conversations. Best for most queries.

**📚 RAG Only** — Pure retrieval. Best for "what did they say about X?"

**👤 Persona Only** — Best for "what is User 1 like?"

---
**⚠️ About this dataset**

Each conversation is a **different pair of people**. "User 1" and "User 2" are role labels — not the same individuals across conversations.

Conflicting facts (e.g. different locations or pets) are **expected** — they come from different people. The system shows facts per-conversation for clarity.
""")

            # ── Chat panel ────────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=480,
                    show_label=False,
                    elem_classes=["chat-wrap"],
                )

                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask anything about the conversations or users...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send ▶", variant="primary", scale=1, min_width=80)

                with gr.Accordion("🔍 Debug — Retrieved Segments", open=False):
                    debug_box = gr.Markdown(
                        value="*Submit a query to see retrieval details.*",
                        elem_classes=["debug-md"],
                    )

                gr.Markdown("**💡 Try these:**")
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=msg_box,
                    label="",
                )

        # ── Event wiring ──────────────────────────────────────────────────────
        def submit(message, history, mode, top_k):
            if not message.strip():
                return history, "", message
            new_history, debug = chat(message, history or [], mode, top_k)
            return new_history, debug, ""

        send_btn.click(
            fn=submit,
            inputs=[msg_box, chatbot, mode, top_k],
            outputs=[chatbot, debug_box, msg_box],
        )

        msg_box.submit(
            fn=submit,
            inputs=[msg_box, chatbot, mode, top_k],
            outputs=[chatbot, debug_box, msg_box],
        )

    return demo, THEME, CSS


if __name__ == "__main__":
    app, theme, css = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=theme,
        css=css,
        share=False,
        show_error=True,
        inbrowser=True,
    )
