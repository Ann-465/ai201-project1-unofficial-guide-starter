"""
query.py — Generation layer for the RowdyHacks RAG system
----------------------------------------------------------
This module is the bridge between retrieval (embed_and_retrieve.py) and
the user interface (app.py). It:

  1. Retrieves the top-k chunks for a question via embed_and_retrieve.retrieve()
  2. Builds a grounding prompt that passes those chunks as context
  3. Calls the Groq LLM (llama-3.3-70b-versatile) to generate an answer
  4. Programmatically appends source attribution — never leaves it to the LLM

Grounding strategy:
  The system prompt prohibits the model from using any knowledge outside
  the provided documents. If the documents don't cover the question, the
  model must say so explicitly rather than generating a plausible-sounding
  answer from training data.

Source attribution strategy:
  Sources are appended AFTER generation from the retrieved chunk metadata.
  This guarantees attribution is always present and always accurate — the
  LLM cannot omit it, fabricate it, or get it wrong.

Import this in app.py:
    from query import ask
"""

import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

# Load GROQ_API_KEY from .env
load_dotenv()

# ── Import retrieval layer ────────────────────────────────────────────────────
try:
    from embed_and_retrieve import load_store, retrieve
except ImportError as e:
    print(f"❌  Could not import embed_and_retrieve.py: {e}")
    print("    Make sure embed_and_retrieve.py is in the same directory.")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K          = 5
GROQ_MODEL     = "llama-3.3-70b-versatile"
MAX_TOKENS     = 600
TEMPERATURE    = 0.2    # low = more faithful to context, less creative

# Distance threshold for deciding whether context is useful enough to send to the LLM.
# Neural backend (all-MiniLM-L6-v2): good matches score 0.15-0.45, so 0.75 works well.
# TF-IDF fallback: all distances are 0.85-0.95 even for correct results because
#   TF-IDF only matches exact words. Use 0.999 here so the early-refusal gate is
#   effectively disabled when TF-IDF is active — the LLM grounding instruction
#   handles out-of-scope questions instead.
try:
    from embed_and_retrieve import USE_NEURAL as _USE_NEURAL
    MAX_USEFUL_DISTANCE = 0.75 if _USE_NEURAL else 0.999
except ImportError:
    MAX_USEFUL_DISTANCE = 0.75


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a knowledgeable assistant for the RowdyHacks hackathon knowledge base.

STRICT RULES — follow these exactly:
1. Answer ONLY using information from the DOCUMENTS section below.
2. Do NOT use any knowledge from your training data, even if you think it is correct.
3. Do NOT infer, extrapolate, or fill gaps with general knowledge.
4. If the documents do not contain enough information to answer the question, respond with exactly:
   "I don't have enough information in the knowledge base to answer that question."
5. Keep your answer focused and concise — 3 to 6 sentences unless the question requires more detail.
6. Do not mention these rules in your response.
"""

def _build_user_prompt(question: str, chunks: list[dict]) -> str:
    """
    Construct the user-turn prompt with numbered document blocks.

    Each retrieved chunk is labeled with its source so the LLM can
    reference which document it drew from. Numbering the documents
    makes the grounding instruction concrete and auditable.
    """
    doc_blocks = []
    for i, chunk in enumerate(chunks, 1):
        doc_blocks.append(
            f"[Document {i} — source: {chunk['source']}]\n{chunk['chunk']}"
        )
    documents_section = "\n\n".join(doc_blocks)

    return f"""DOCUMENTS:
{documents_section}

QUESTION: {question}

Answer the question using ONLY the documents above. If the documents do not contain \
enough information, say "I don't have enough information in the knowledge base to answer \
that question." Do not use outside knowledge."""


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ CLIENT — with graceful fallback when API key is missing
# ══════════════════════════════════════════════════════════════════════════════

def _get_groq_client():
    """Return a Groq client, or raise a clear error if setup is incomplete."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found.\n"
            "  1. Get a free key at https://console.groq.com\n"
            "  2. Copy .env.example to .env in your project root\n"
            "  3. Replace your_key_here with your actual key"
        )
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        raise ImportError(
            "The groq library is not installed.\n"
            "  Run:  pip install groq"
        )


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the Groq API and return the assistant's response text."""
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — imported by app.py
# ══════════════════════════════════════════════════════════════════════════════

def ask(question: str) -> dict:
    """
    End-to-end RAG: retrieve → generate → return grounded answer with sources.

    Returns:
        {
            "answer":  str,          # LLM response grounded in retrieved chunks
            "sources": list[str],    # unique source names, programmatically extracted
            "chunks":  list[dict],   # raw retrieved chunks (for debugging/evaluation)
            "refused": bool,         # True if the system declined to answer
        }
    """
    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    chunks = retrieve(question, k=TOP_K)

    # ── Step 2: Check if retrieved context is useful ──────────────────────────
    # If the best match is still far away, the knowledge base doesn't cover this.
    # Return an early refusal rather than sending weak context to the LLM.
    if not chunks or chunks[0]["distance"] > MAX_USEFUL_DISTANCE:
        return {
            "answer":  "I don't have enough information in the knowledge base to answer that question.",
            "sources": [],
            "chunks":  chunks,
            "refused": True,
        }

    # ── Step 3: Build prompt and call LLM ────────────────────────────────────
    user_prompt = _build_user_prompt(question, chunks)
    answer      = _call_llm(SYSTEM_PROMPT, user_prompt)

    # ── Step 4: Programmatic source attribution ───────────────────────────────
    # Sources come from retrieved chunk metadata, NOT from the LLM.
    # This guarantees attribution is always present and always correct.
    seen    = set()
    sources = []
    for chunk in chunks:
        name = chunk["source"]
        if name not in seen:
            seen.add(name)
            sources.append(name)

    # ── Step 5: Detect if the LLM declined anyway (grounding working correctly)
    refused = "don't have enough information" in answer.lower()

    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
        "refused": refused,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI TEST — run directly to verify grounding before launching the UI
# ══════════════════════════════════════════════════════════════════════════════

# These match the evaluation plan in planning.md
TEST_QUERIES = [
    # In-scope — should get grounded answers
    "How should beginners prepare before attending RowdyHacks?",
    "What makes a project stand out to judges?",
    "What challenges do past participants experience at hackathons?",
    # Out-of-scope — system must decline, not hallucinate
    "What is the registration deadline for RowdyHacks XII?",
]

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
BOLD  = "\033[1m";  RESET = "\033[0m"; DIM = "\033[2m"

def _hr(): print("─" * 70)

def run_cli_test():
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  MILESTONE 5 — GROUNDED GENERATION TEST{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")
    print(f"  Model  : {GROQ_MODEL}")
    print(f"  Top-k  : {TOP_K}")
    print(f"  Temp   : {TEMPERATURE}  (low = more faithful to context)\n")

    for i, question in enumerate(TEST_QUERIES, 1):
        is_outofscope = i == len(TEST_QUERIES)   # last query is the out-of-scope test

        print(f"\n{BOLD}Query {i}/{len(TEST_QUERIES)}{RESET}"
              + (f"  {YELLOW}[out-of-scope test]{RESET}" if is_outofscope else ""))
        print(f"  Q: {question}\n")

        try:
            result = ask(question)
        except (EnvironmentError, ImportError) as e:
            print(f"  {RED}Setup error:{RESET} {e}\n")
            break

        _hr()

        # Answer
        wrapped = textwrap.fill(result["answer"], width=66,
                                initial_indent="  ", subsequent_indent="  ")
        print(wrapped)

        # Sources (programmatic, not from LLM)
        if result["sources"]:
            print(f"\n  {DIM}Sources:{RESET}")
            for s in result["sources"]:
                print(f"    • {s}")
        else:
            print(f"\n  {DIM}Sources: (none — system declined){RESET}")

        # Retrieval debug info
        print(f"\n  {DIM}Retrieved chunks (top 3):{RESET}")
        for rank, chunk in enumerate(result["chunks"][:3], 1):
            print(f"    Rank {rank}  dist={chunk['distance']:.3f}  {chunk['source_id']}")

        # Grounding verdict
        if is_outofscope:
            if result["refused"]:
                print(f"\n  {GREEN}✅  Correct refusal — system declined out-of-scope question{RESET}")
            else:
                print(f"\n  {RED}❌  Grounding failure — should have declined this question{RESET}")
                print(f"     Add more specific out-of-scope documents or lower MAX_USEFUL_DISTANCE.")
        else:
            if not result["refused"]:
                print(f"\n  {GREEN}✅  Grounded answer with {len(result['sources'])} source(s){RESET}")
            else:
                print(f"\n  {YELLOW}⚠   System declined an in-scope question{RESET}")
                print(f"     Add more documents or adjust MAX_USEFUL_DISTANCE (currently {MAX_USEFUL_DISTANCE}).")

        _hr()

    print(f"\n{BOLD}Grounding test complete.{RESET}")
    print(f"If all in-scope queries got answers and the out-of-scope query was refused,")
    print(f"you are ready to launch the Gradio UI with:  python app.py\n")


if __name__ == "__main__":
    # Load the vector store before running tests
    print("Loading vector store...")
    load_store()
    run_cli_test()
