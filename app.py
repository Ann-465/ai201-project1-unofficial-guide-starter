"""
app.py — Gradio web interface for the RowdyHacks Unofficial Guide RAG system
-----------------------------------------------------------------------------
Run with:   python app.py
Then open:  http://localhost:7860

The interface has:
  • Question input (text box + Enter key support)
  • Answer output (grounded in retrieved documents)
  • Sources panel (programmatically extracted — never fabricated)
  • Retrieval details (expandable — shows distance scores for each chunk)

All generation is handled by query.py. This file only handles UI layout.
"""

import sys

# ── Check Gradio is installed ─────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    print("❌  Gradio is not installed.")
    print("    Run:  pip install gradio>=4.0.0")
    sys.exit(1)

# ── Import generation layer ───────────────────────────────────────────────────
try:
    from query import ask
    from embed_and_retrieve import load_store
except ImportError as e:
    print(f"❌  Could not import project modules: {e}")
    print("    Make sure query.py and embed_and_retrieve.py are in the same directory.")
    sys.exit(1)

# ── Load vector store once at startup ────────────────────────────────────────
print("Loading vector store...")
load_store()
print("Ready.")


# ══════════════════════════════════════════════════════════════════════════════
#  HANDLER — called by Gradio on every query
# ══════════════════════════════════════════════════════════════════════════════

def handle_query(question: str):
    """
    Takes the user's question, calls ask(), formats outputs for Gradio.

    Returns three strings:
      answer_text   — the LLM's grounded response
      sources_text  — bullet list of source documents
      debug_text    — retrieval details (distances, source IDs)
    """
    question = question.strip()
    if not question:
        return (
            "Please enter a question.",
            "",
            ""
        )

    result = ask(question)

    # ── Answer ────────────────────────────────────────────────────────────────
    answer_text = result["answer"]

    # ── Sources (programmatic — from chunk metadata, not the LLM) ────────────
    if result["sources"]:
        sources_text = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources_text = "No sources — the system declined to answer this question."

    # ── Retrieval debug panel ─────────────────────────────────────────────────
    debug_lines = ["Top retrieved chunks (before generation):\n"]
    for rank, chunk in enumerate(result["chunks"], 1):
        dist  = chunk["distance"]
        src   = chunk["source_id"]
        words = chunk["chunk"].split()
        preview = " ".join(words[:20]) + ("…" if len(words) > 20 else "")
        quality = "strong" if dist < 0.35 else ("good" if dist < 0.55 else "weak")
        debug_lines.append(
            f"Rank {rank}  dist={dist:.3f} [{quality}]  [{src}]\n"
            f"  {preview}\n"
        )

    debug_text = "\n".join(debug_lines)

    return answer_text, sources_text, debug_text


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLE_QUESTIONS = [
    "How should beginners prepare before attending RowdyHacks?",
    "What makes a project stand out to judges?",
    "How do participants recommend finding teammates?",
    "What advice do experienced hackers give about sleep?",
    "What projects won awards at RowdyHacks XI?",
    "What is the prize money for RowdyHacks XII?",   # out-of-scope — should decline
]

with gr.Blocks(title="RowdyHacks Unofficial Guide") as demo:

    gr.Markdown(
        """
        # 🤠 RowdyHacks Unofficial Guide
        Ask anything about preparing for, competing in, or winning at RowdyHacks —
        the annual hackathon at the University of Texas at San Antonio.

        Answers are grounded in real participant experiences, project write-ups,
        and survival guides. Sources are shown for every response.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. How should I find teammates at RowdyHacks?",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("**Example questions**")
            for ex in EXAMPLE_QUESTIONS:
                gr.Button(ex, size="sm").click(
                    fn=lambda q=ex: q,
                    outputs=question_box,
                )

    answer_box = gr.Textbox(
        label="Answer",
        lines=8,
        interactive=False,
        placeholder="Answer will appear here…",
    )

    sources_box = gr.Textbox(
        label="Retrieved from",
        lines=4,
        interactive=False,
        placeholder="Source documents will appear here…",
    )

    with gr.Accordion("Retrieval details (for debugging)", open=False):
        debug_box = gr.Textbox(
            label="Top retrieved chunks",
            lines=12,
            interactive=False,
        )

    # Wire up the button and Enter key
    ask_btn.click(
        fn=handle_query,
        inputs=question_box,
        outputs=[answer_box, sources_box, debug_box],
    )
    question_box.submit(
        fn=handle_query,
        inputs=question_box,
        outputs=[answer_box, sources_box, debug_box],
    )

    gr.Markdown(
        """
        ---
        *Answers are generated only from collected documents.
        If the knowledge base doesn't cover your question, the system will say so
        rather than guessing.*
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LAUNCH
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
