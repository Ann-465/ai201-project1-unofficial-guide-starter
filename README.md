# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

RowdyHacks participant knowledge — practical advice, lessons learned, and event-specific information compiled from past participants, survival guides, project write-ups, and community discussions.

This knowledge is valuable because the official RowdyHacks website covers registration and logistics, not the real student experience. Information about how to scope a project for 24 hours, which workshops are worth attending, how teams actually form on the ground, and what judges respond to is scattered across Reddit threads, Notion guides, Devpost pages, and Medium articles. A first-time participant has no single place to find it. This system consolidates that knowledge into one searchable, answerable interface.
---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | RowdyHakcs website | the lastest hackathon at the University of Texas at San Antonio| https://rowdyhacks.org|
| 2 | Survival Guide| Information that help the student prepare and serve for the hackathone| https://acmutsa.notion.site/RHXI-Survival-Guide-186c7f3b374281c2a72edb1df6e4daa6|
| 3 | RowdyHacks Instagram account| Information about the past couple of RowdHacks| https://www.instagram.com/rowdyhacks/|
| 4 | Reddit| Tips for hackathons| https://www.reddit.com/r/csMajors/comments/17irlcq/any_tips_for_hackathon/|
| 5 | Quora| Tips for preparing for hackathons| https://www.quora.com/How-do-you-prepare-for-and-win-hackathons|
| 6 | Medium article| Tips and Strategy| https://medium.com/@allankong/ive-won-thousands-in-hackathons-here-are-my-tips-and-strategies-72267f9f3974|
| 7 | Devpost| Information about the past projects in Hackthons and the upcoming Hackathons| https://devpost.com|
| 8 | Reddit| Hackathone ideas| https://www.reddit.com/r/AI_Agents/comments/1kh559v/megathread_post_your_hackathon_ideas_here/|
| 9 | OpenAI Developer Community| Hackathon Ideas| https://community.openai.com/t/unofficial-weekend-project-hackathon-ideas/1150059|
| 10 | Devpost| Last RowdyHacks information page| https://rowdyhacks-xi.devpost.com/?ref_feature=challenge&ref_medium=discover|


---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:** ~300 words (word count, not characters or tokens)

**Overlap:** 50 words

**Why these choices fit your documents:**
The corpus is a mix of short Reddit and Quora comments (50–200 words) and longer guides and articles (500–1000 words). A 300-word chunk is large enough to capture a complete idea — a full piece of advice, a project description, or a preparation checklist — while staying small enough for a specific query to match precisely. Shorter chunks (under 100 words) would embed too little meaning for the semantic search to distinguish signal from noise. Larger chunks (over 500 words) would merge multiple topics (e.g., team formation AND time management AND presentations) into one vector, making it harder for any single query to match cleanly.

The 50-word overlap prevents information from being lost at chunk boundaries. When a key point spans two consecutive chunks — for example, an intro sentence followed by its explanation — the overlapping words carry enough context for either chunk to be retrieved on its own. This became relevant in the documented failure case below, where a chunk boundary mid-paragraph cut a section about judging criteria in two.

**Preprocessing before chunking:** HTML tags, navigation elements, cookie banners, footers, and repeated site headers were stripped using BeautifulSoup before any text reached the chunker. HTML entities (`&amp;`, `&nbsp;`) were replaced with their plain-text equivalents. Lines shorter than 3 characters were dropped. All of this ran in `ingest.py` via `clean_html()` before `chunk_text()` was called.

**Final chunk count:** 14 chunks across 7 source documents (2 chunks per document). This is below the 50-chunk minimum recommended in the instructions. The sample documents are compact text files manually copied from sources that blocked automated scraping. Adding the full text of each source would bring the corpus to approximately 60–100 chunks.



---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:** `all-MiniLM-L6-v2` from the `sentence-transformers` library, stored in a local ChromaDB collection with cosine distance.

This model was chosen because it runs entirely locally with no API key or rate limits, and it is specifically optimized for semantic similarity — it can match "How should I prepare?" to a chunk that says "Preparation starts weeks before the event" even with no shared words. It produces 384-dimensional vectors and embeds quickly enough that re-embedding the full corpus takes under a second.

**Production tradeoff reflection:**
In a production system I would weigh four tradeoffs. First, **context length**: `all-MiniLM-L6-v2` has a 256-token input limit, which is fine for 300-word chunks but would truncate longer documents without re-chunking. OpenAI's `text-embedding-3-large` supports 8,191 tokens, enabling much larger chunks or whole-document embedding. Second, **multilingual support**: UTSA has a large Spanish-speaking student population. A participant might query in Spanish even if the documents are in English. `multilingual-e5-large` handles cross-lingual retrieval; MiniLM does not. Third, **domain accuracy**: MiniLM was trained on general web text, not hackathon-specific language. A model fine-tuned on developer community text (like `e5-mistral-7b-instruct`) would likely produce lower distances on queries about APIs, sprint cycles, and demo presentations. Fourth, **cost and latency**: API-hosted models like `text-embedding-3-large` add ~50–100ms per query and accrue per-token charges at scale; local models add zero marginal cost but require a machine with enough RAM to load them.


---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**
The word "ONLY" and the explicit "Do NOT" instructions are intentional — softer phrasings like "try to use the documents" do not reliably prevent the model from drawing on training knowledge. The exact refusal phrase in rule 4 makes it possible to detect refusals programmatically by checking whether `"don't have enough information"` appears in the response.

**How source attribution is surfaced in the response:**
Sources are extracted **after generation** from retrieved chunk metadata — not from the LLM's response text. In `query.py`:

```python
seen, sources = set(), []
for chunk in chunks:
    name = chunk["source"]   # from ChromaDB metadata, not from LLM
    if name not in seen:
        seen.add(name)
        sources.append(name)
```

This guarantees attribution is always present and always accurate. The LLM cannot omit a source, fabricate a source name, or get the document title wrong. The sources list in the Gradio UI is populated entirely from this programmatic extraction.
---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | How should beginners prepare before attending RowdyHacks? | What to bring, advance preparation steps, day-of tips | "Confirm your team two weeks out, set up shared GitHub one week out, arrive at the opening ceremony for sponsor credits, bring a power strip." | Relevant — survival_guide at dist=0.356 | Accurate |
| 2 | What challenges did past participants experience at hackathons? | Scope creep, sleep deprivation, team formation issues, technical blockers | "Scope creep, sleep deprivation around 3am, chaotic on-site team formation, API rate limits catching first-timers off guard." | Relevant — reddit_hackathon_tips top result | Accurate |
| 3 | What advice do experienced participants give first-time hackers? | Scope aggressively, read prize tracks, sleep | "Scope aggressively, read all prize tracks before committing to an idea, take a 3–4 hour nap instead of pushing through 36 hours." | Relevant — reddit and quora sources retrieved | Accurate |
| 4 | What makes a project stand out to judges? | Clear problem statement, polished demo, judging criteria alignment | "I don't have enough information in the knowledge base to answer that question." | Partially relevant — best chunk at dist=0.547, key content split across chunk boundary | Inaccurate |
| 5 | How do participants recommend finding teammates? | Attend team formation event, complementary skills, form team before event | "Attend the team formation event, aim for 3–4 people with complementary skills, form your team before the event if possible." | Relevant — survival_guide and reddit sources retrieved | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:** "What makes a project stand out to judges?"

**What the system returned:** `"I don't have enough information in the knowledge base to answer that question."`

**Root cause (tied to a specific pipeline stage):**

The failure is at the **chunking stage**, with a secondary effect on **retrieval**. The Medium article by Allan Kong contains two sections that directly answer this question: "Start With the Judging Criteria" and "Building a Winning Demo." Both are in `medium_hackathon_tips`. However, the word-count chunker split the document at word 300, which fell mid-paragraph inside "Technical Stack Choices" — the section *between* the two relevant sections. This produced:

- Chunk #0 (words 1–300): covers judging criteria and idea selection, but trails off mid-sentence into tech stack content
- Chunk #1 (words 251–506): **begins with an orphaned sentence fragment** — `"one person to learn the new tool while others build the rest of the system in familiar territory. Building a Winning Demo…"`

When `medium_hackathon_tips` chunk #1 was retrieved (dist=0.733, rank 4), it opened with that incomplete sentence. The model received this as part of its context but the fragment signaled incomplete information, and with no other chunk scoring strongly enough, it triggered the refusal.

The chunker never saw the paragraph boundary — it simply counted 300 words and cut. A word-count splitter is unaware of document structure.

**What you would change to fix it:**

Switch to **paragraph-aware chunking** — split on double newlines first, then merge adjacent paragraphs until reaching the target word count. This would keep "Building a Winning Demo" as an intact section rather than splitting it across two chunks. Alternatively, reducing `CHUNK_SIZE` to 150 words would decrease the chance of important sections being bisected, though it would also increase total chunk count and require tuning `top_k` upward to compensate.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->
**One way the spec helped you during implementation:**

The requirement to write the evaluation plan in `planning.md` before any code forced me to commit to specific, testable questions with explicit expected answers before I had seen any system output. When I reached Milestone 6, I had clear success criteria already written down, which made the accuracy judgments honest rather than post-hoc rationalization. Without that pre-commitment, I would have been tempted to phrase expected answers vaguely enough to match whatever the system returned.

**One way your implementation diverged from the spec, and why:**

The spec assumed all 10 source documents would be fetched programmatically — the pipeline was designed around `requests` + BeautifulSoup fetching live URLs. In practice, every source returned 403 Forbidden: Reddit, Quora, Medium, Devpost, and the official RowdyHacks site all block non-browser traffic. I pivoted to manually copying text from each source into `.txt` files and running `python ingest.py --from-disk`. The `--from-disk` flag was not in the original spec; I added it specifically because of this failure. The result was actually cleaner text (no JavaScript rendering artifacts, no partial page loads) but it meant the automated fetch path in `ingest.py` is largely unused. If I were building this again, I would treat `--from-disk` as the primary path and automated fetching as an optional enhancement.



---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

- *What I gave the AI:* My `planning.md` Documents section (10 source URLs and file types), the Chunking Strategy section (300-word chunks, 50-word overlap), and the pipeline diagram showing the five stages.
- *What it produced:* A working `load_documents()` function that read `.txt` files from a `/data` directory and a `chunk_text()` function that split by word count with the specified overlap. The HTML cleaning used BeautifulSoup and stripped common tag types.
- *What I changed or overrode:* The initial `is_boilerplate_element()` function only checked CSS `class` attributes for nav/ad keywords. It missed boilerplate in `id` attributes and `aria-label` attributes, which several sources used. I extended the function to check all three attributes. I also found that the cleaning removed Reddit Markdown link syntax `[text](url)` before the link text was extracted, losing useful context. I added a separate substitution pass that replaced `[text](url)` with just `text` before stripping Markdown entirely.

**Instance 2**

- *What I gave the AI:* My grounding requirement ("answers from retrieved context only, with source attribution"), the desired output format (`{"answer": str, "sources": list[str]}`), and a request to write the system prompt for `llama-3.3-70b-versatile`.
- *What it produced:* A system prompt that read: `"You are a helpful assistant. Please answer the question using only the provided documents. If you cannot find the answer, say so."`
- *What I changed or overrode:* I tested this draft and the model answered out-of-scope questions with confident training-data responses roughly half the time — the word "please" and "using only" left too much latitude. I rewrote the system prompt with numbered imperative rules, replaced "using only" with "Answer ONLY using," added an explicit "Do NOT use any knowledge from your training data," and specified the exact refusal phrase the model must use. After this rewrite, out-of-scope refusals became consistent across all test cases.Sonnet 4.6 LowClaude is AI and can make mistakes. Please double-check responses.
