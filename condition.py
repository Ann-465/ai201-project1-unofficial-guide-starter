
"""
Milestone 3 Checkpoint — Chunk Quality Inspector
-------------------------------------------------
Runs every check from the Milestone 3 checklist and prints 5 random
chunks from chunks.json. All checks must pass before moving to Milestone 4.
 
Usage:
    python check_chunks.py                    # use default data/chunks.json
    python check_chunks.py --file my/path.json
"""
 
import json
import random
import re
import sys
import textwrap
import argparse
from pathlib import Path
from collections import Counter
 
# ── Thresholds (match the spec) ───────────────────────────────────────────────
MIN_TOTAL_CHUNKS   = 50      # fewer → chunks probably too large
MAX_TOTAL_CHUNKS   = 2000    # more  → chunks probably too small
MIN_CHUNK_WORDS    = 40      # below this is a fragment
MAX_CHUNK_WORDS    = 600     # above this is too diluted
IDEAL_MIN_WORDS    = 150     # healthy lower bound for this corpus
IDEAL_MAX_WORDS    = 350     # healthy upper bound for this corpus
 
# ── Patterns that signal bad output ──────────────────────────────────────────
HTML_TAG_RE      = re.compile(r"<[a-z][a-z0-9]*[\s/>]", re.IGNORECASE)
HTML_ENTITY_RE   = re.compile(r"&(amp|nbsp|lt|gt|quot|#\d+);", re.IGNORECASE)
NAV_NOISE_RE     = re.compile(
    r"\b(cookie\s*policy|accept\s*cookies|sign\s*in|log\s*in|"
    r"subscribe|newsletter|privacy\s*policy|terms\s*of\s*service|"
    r"all\s*rights\s*reserved|copyright\s*\d{4}|"
    r"read\s*more|share\s*this|follow\s*us)\b",
    re.IGNORECASE,
)
 
# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
 
def ok(msg):   print(f"  {GREEN}✅  {msg}{RESET}")
def fail(msg): print(f"  {RED}❌  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠   {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")
 
 
# ── Individual checks ─────────────────────────────────────────────────────────
 
def check_total_count(chunks: list) -> bool:
    n = len(chunks)
    if n < MIN_TOTAL_CHUNKS:
        fail(f"Only {n} chunks total — below the minimum of {MIN_TOTAL_CHUNKS}. "
             f"Chunks may be too large, or some documents failed to load.")
        return False
    if n > MAX_TOTAL_CHUNKS:
        fail(f"{n} chunks total — above the maximum of {MAX_TOTAL_CHUNKS}. "
             f"Chunks may be too small to carry enough semantic signal.")
        return False
    ok(f"Total chunk count: {n}  (target: {MIN_TOTAL_CHUNKS}–{MAX_TOTAL_CHUNKS})")
    return True
 
 
def check_empty_chunks(chunks: list) -> bool:
    empties = [i for i, c in enumerate(chunks) if not c["chunk"].strip()]
    if empties:
        fail(f"{len(empties)} empty chunk(s) found at indices: {empties[:10]}"
             f"{'...' if len(empties) > 10 else ''}. "
             f"Add a len(chunk) > 0 filter in chunk_text().")
        return False
    ok("No empty chunks found.")
    return True
 
 
def check_html_artifacts(chunks: list) -> bool:
    bad = [(i, c["source_id"]) for i, c in enumerate(chunks)
           if HTML_TAG_RE.search(c["chunk"])]
    if bad:
        fail(f"{len(bad)} chunk(s) still contain HTML tags. "
             f"Example sources: {list({s for _,s in bad[:3]})}. "
             f"Cleaning did not fully strip HTML — re-run with a stricter BeautifulSoup pass.")
        return False
    ok("No HTML tags found in any chunk.")
    return True
 
 
def check_html_entities(chunks: list) -> bool:
    bad = [(i, c["source_id"]) for i, c in enumerate(chunks)
           if HTML_ENTITY_RE.search(c["chunk"])]
    if bad:
        fail(f"{len(bad)} chunk(s) contain HTML entities (&amp;, &nbsp;, etc.). "
             f"Example sources: {list({s for _,s in bad[:3]})}. "
             f"Check the HTML_ENTITIES replacement dict in ingest.py.")
        return False
    ok("No HTML entities found in any chunk.")
    return True
 
 
def check_nav_noise(chunks: list) -> bool:
    bad = [(i, c["source_id"]) for i, c in enumerate(chunks)
           if NAV_NOISE_RE.search(c["chunk"])]
    if bad:
        warn(f"{len(bad)} chunk(s) may contain nav/boilerplate text "
             f"(cookie banners, sign-in prompts, copyright lines). "
             f"Example sources: {list({s for _,s in bad[:3]})}. "
             f"Consider tightening BOILERPLATE_PATTERNS in ingest.py.")
        return False   # warn only — not a hard failure
    ok("No nav/boilerplate noise detected.")
    return True
 
 
def check_chunk_sizes(chunks: list) -> bool:
    word_counts = [len(c["chunk"].split()) for c in chunks]
 
    fragments  = sum(1 for w in word_counts if w < MIN_CHUNK_WORDS)
    oversized  = sum(1 for w in word_counts if w > MAX_CHUNK_WORDS)
    in_range   = sum(1 for w in word_counts if IDEAL_MIN_WORDS <= w <= IDEAL_MAX_WORDS)
    pct        = round(100 * in_range / len(chunks))
 
    passed = True
    if fragments:
        fail(f"{fragments} chunk(s) are under {MIN_CHUNK_WORDS} words — fragments with no "
             f"standalone meaning. Raise MIN_CHUNK_WORDS or fix sentence splitting.")
        passed = False
    if oversized:
        fail(f"{oversized} chunk(s) exceed {MAX_CHUNK_WORDS} words — too diluted for "
             f"precise retrieval. Reduce CHUNK_SIZE in ingest.py.")
        passed = False
    if passed:
        ok(f"Chunk sizes look healthy: "
           f"min={min(word_counts)}w  max={max(word_counts)}w  avg={sum(word_counts)//len(word_counts)}w  "
           f"{pct}% in ideal range ({IDEAL_MIN_WORDS}–{IDEAL_MAX_WORDS}w).")
    return passed
 
 
def check_source_diversity(chunks: list) -> bool:
    sources = Counter(c["source_id"] for c in chunks)
    n_sources = len(sources)
    if n_sources < 3:
        fail(f"Chunks come from only {n_sources} distinct source(s). "
             f"Make sure all documents loaded correctly.")
        return False
    ok(f"Chunks span {n_sources} distinct sources: {', '.join(sorted(sources.keys()))}.")
    return True
 
 
def check_metadata(chunks: list) -> bool:
    missing_source    = [i for i, c in enumerate(chunks) if not c.get("source")]
    missing_source_id = [i for i, c in enumerate(chunks) if not c.get("source_id")]
    missing_index     = [i for i, c in enumerate(chunks) if c.get("chunk_index") is None]
 
    if missing_source or missing_source_id or missing_index:
        if missing_source:
            fail(f"{len(missing_source)} chunk(s) missing 'source' field — attribution will break.")
        if missing_source_id:
            fail(f"{len(missing_source_id)} chunk(s) missing 'source_id' field.")
        if missing_index:
            fail(f"{len(missing_index)} chunk(s) missing 'chunk_index' field.")
        return False
    ok("All chunks have 'source', 'source_id', and 'chunk_index' metadata.")
    return True
 
 
def check_uniform_length(chunks: list) -> bool:
    """Warn if all chunks are suspiciously identical in length (mechanical splitting)."""
    word_counts = [len(c["chunk"].split()) for c in chunks]
    unique = len(set(word_counts))
    if unique == 1:
        warn("All chunks are exactly the same length — this suggests purely mechanical "
             "splitting with no sentence or paragraph awareness. Consider splitting on "
             "paragraph boundaries first.")
        return False
    ok(f"Chunk lengths vary naturally ({unique} distinct word-count values — not mechanical).")
    return True
 
 
# ── Random sample display ─────────────────────────────────────────────────────
 
def print_random_chunks(chunks: list, n: int = 5):
    sample = random.sample(chunks, min(n, len(chunks)))
    print(f"\n{'═'*70}")
    print(f"  5 RANDOM CHUNKS FOR MANUAL INSPECTION")
    print(f"{'═'*70}")
    print(
        "  For each chunk, ask yourself:\n"
        "  • Does this make sense on its own?\n"
        "  • Could someone answer a question from this chunk alone?\n"
        "  • Is the source attribution correct for the text shown?\n"
    )
 
    for idx, chunk in enumerate(sample, 1):
        words = chunk["chunk"].split()
        word_count = len(words)
        preview = " ".join(words[:12]) + ("..." if len(words) > 12 else "")
        print(f"{'─'*70}")
        print(f"  Chunk {idx}/5")
        print(f"  Source ID : {chunk['source_id']}")
        print(f"  Source    : {chunk['source']}")
        print(f"  Position  : chunk #{chunk['chunk_index']} in document")
        print(f"  Length    : {word_count} words")
        print(f"  Starts    : \"{preview}\"")
        print()
        wrapped = textwrap.fill(
            chunk["chunk"], width=68,
            initial_indent="  ", subsequent_indent="  "
        )
        print(wrapped)
        print()
 
    print(f"{'═'*70}")
 
 
# ── Orchestrator ──────────────────────────────────────────────────────────────
 
def run_checkpoint(chunks_path: Path):
    if not chunks_path.is_absolute():
        chunks_path = Path(__file__).resolve().parent / chunks_path

    if not chunks_path.exists():
        print(f"{RED}❌  File not found: {chunks_path}{RESET}")
        print("    Run  python ingest.py --from-disk  first to generate chunks.json.")
        sys.exit(1)
 
    raw_text = chunks_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        print(f"{RED}❌  {chunks_path} is empty.{RESET}")
        print("    Re-run your ingest step to generate a valid JSON list of chunks.")
        sys.exit(1)

    try:
        chunks = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"{RED}❌  {chunks_path} is not valid JSON: {exc}{RESET}")
        print("    Make sure the file contains a JSON array of chunk objects.")
        sys.exit(1)

    if not chunks:
        print(f"{RED}❌  chunks.json is empty.{RESET}")
        sys.exit(1)
 
    header("═" * 70)
    header(f"  MILESTONE 3 CHECKPOINT — {chunks_path}")
    header("═" * 70)
 
    checks = [
        ("Total chunk count",        check_total_count),
        ("No empty chunks",          check_empty_chunks),
        ("No HTML tags",             check_html_artifacts),
        ("No HTML entities",         check_html_entities),
        ("No nav/boilerplate noise", check_nav_noise),
        ("Chunk size range",         check_chunk_sizes),
        ("Source diversity",         check_source_diversity),
        ("Metadata completeness",    check_metadata),
        ("Non-uniform lengths",      check_uniform_length),
    ]
 
    results = {}
    header("\nRunning checks...\n")
    for label, fn in checks:
        results[label] = fn(chunks)
 
    # Summary
    passed  = [l for l, r in results.items() if r]
    failed  = [l for l, r in results.items() if not r]
 
    header(f"\n{'═'*70}")
    header(f"  SUMMARY")
    print(f"{'═'*70}")
    print(f"  {GREEN}{len(passed)}/{len(checks)} checks passed{RESET}   "
          f"{RED}{len(failed)} failed/warned{RESET}")
 
    if failed:
        print(f"\n  {RED}Failed / warned:{RESET}")
        for label in failed:
            print(f"    • {label}")
        print(f"\n  {YELLOW}Fix the issues above before moving to Milestone 4.{RESET}")
        print(f"  Bad chunks cannot be fixed by tuning retrieval later.\n")
    else:
        print(f"\n  {GREEN}{BOLD}All checks passed — ready for Milestone 4 (embedding).{RESET}\n")
 
    # Always print the 5 random chunks regardless of check results
    print_random_chunks(chunks, n=5)
 
    if failed:
        sys.exit(1)   # non-zero exit so CI / scripts can detect failure
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", default="script.json",
        help="Path to chunks.json (default: script.json)"
    )
    args = parser.parse_args()
    run_checkpoint(Path(args.file))