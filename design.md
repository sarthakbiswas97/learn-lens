# LearnLens -- System Design Document

## 1. Problem Statement

Every day, ML engineers and researchers consume dozens of articles, papers, tweets, blog posts, and videos. By the next morning, 90% is forgotten. Bookmarks pile up unread. Coding mistakes repeat weeks later. Hours disappear into tangential topics while genuinely important content sits untouched.

Existing tools fall into two categories, and neither solves this:

**Frontier AI (ChatGPT, Claude, Gemini):**
- Cannot access your browsing history or bookmarks
- Cannot track what you actually read vs. skimmed
- Cannot remember your coding mistakes across sessions
- Cannot run continuously on every page you visit
- Sending your full daily activity to an API is expensive and a privacy nightmare

**Second Brain Tools (Obsidian, Notion, Readwise, Mem.ai):**
- They are libraries -- passive storage systems
- They organize everything equally, never judge your choices
- They never tell you what is a waste of your time
- They store, but they do not prioritize, nag, or warn

LearnLens is neither. It is a **coach** -- an opinionated mentor that has opinions about your learning, judges your choices, and tells you when you are wasting time.

---

## 2. First-Principles: Why 3 Models, Not 1

Effective learning requires three distinct cognitive functions. Each maps cleanly to a specialized model:

### The Three Cognitive Functions

| Function | Question It Answers | Compute Profile | Model Type Needed |
|----------|-------------------|-----------------|-------------------|
| **Connection** | "What relates to what?" | Dense vector math, pairwise similarity | Embedding model (small, fast) |
| **Prioritization** | "What matters given my goals?" | Structured reasoning over (content, goals) pairs | Fine-tuned classifier/scorer |
| **Synthesis** | "What should I do about all this?" | Long-form text generation from structured context | Instruction-tuned generator |

### Why not one large model?

A single 7B+ model could theoretically do all three, but:

1. **Budget**: Hackathon track requires <=4B total parameters (Tiny Titan)
2. **Specialization**: A 137M embedding model produces better vectors than a 3B generalist trying to do embeddings. A fine-tuned 1B scorer outperforms a 3B generalist at structured priority scoring. Each model excels at its narrow task.
3. **Pipeline clarity**: Three models with well-defined inputs and outputs are easier to debug, test, and iterate on than one monolithic model trying to do everything
4. **Cost**: Embedding 500 articles through a 3B model is 20x slower than through a 137M model. The embedding model runs thousands of times per session; it must be fast.

### Why not two models?

You could merge Prioritizer + Mentor into one model, but:
- The Prioritizer needs to output **structured JSON** (score, rationale, action) -- fine-tuned behavior
- The Mentor needs to output **flowing markdown** (daily briefing) -- general instruction-following
- These are fundamentally different output formats requiring different training objectives
- Keeping them separate means you can fine-tune Model 2 without touching Model 3

---

## 3. System Architecture

```
                                    USER INPUT
                                        |
                    +-------------------+-------------------+
                    |                   |                   |
              Paste URLs        Upload Bookmarks    Upload Chrome
              (manual)          (HTML export)       History (SQLite)
                    |                   |                   |
                    +-------------------+-------------------+
                                        |
                                   INGESTION
                              (trafilatura / BS4)
                                        |
                                        v
                              +------------------+
                              |    SQLite DB     |
                              |  (content table) |
                              +------------------+
                                        |
                                        v
                              MODEL 1: CONNECTOR
                            nomic-embed-text-v1.5
                                    (137M)
                                        |
                              embed each document
                                        |
                                        v
                              +------------------+
                              |    SQLite DB     |
                              | (embeddings tbl) |
                              +------------------+
                                        |
                                        v
                              MODEL 2: PRIORITIZER
                            MiniCPM5-1B + LoRA-scorer
                                        |
                           score each item vs goals
                                        |
                                        v
                              +------------------+
                              |    SQLite DB     |
                              |  (scores table)  |
                              +------------------+
                                        |
                     +------------------+------------------+
                     |                                     |
               Top-K scored items                  Forgotten items
               + Mistake patterns                  (high relevance,
               + User goals                        no interaction 7+ days)
                     |                                     |
                     +------------------+------------------+
                                        |
                                        v
                              MODEL 3: MENTOR
                           MiniCPM5-1B + LoRA-mentor
                              (shared base weights)
                                        |
                             generate daily briefing
                                        |
                                        v
                              +------------------+
                              |   GRADIO UI      |
                              |   (5 tabs)       |
                              +------------------+
```

**Key design decision:** The pipeline is **sequential, not parallel**. Each model's output feeds the next. This is simpler to debug, and the models are small enough that total latency is acceptable (~5-10 seconds for a full pipeline run).

**SQLite is the single source of truth** between pipeline stages. Every intermediate result (embeddings, scores) is persisted. This means:
- You can re-run any stage without re-running earlier ones
- The UI can read directly from SQLite without waiting for the full pipeline
- Debugging is trivial -- just inspect the database

---

## 4. Model Selection Rationale

### Model 1 -- Connector: nomic-ai/nomic-embed-text-v1.5

| Property | Value |
|----------|-------|
| Parameters | 137M |
| Embedding dimensions | 768 (default), supports Matryoshka: 512, 256, 128, 64 |
| Max sequence length | 8192 tokens |
| MTEB score | ~62.4 |
| License | Apache 2.0 |

**Why this model:**
- **8192 token context**: Most articles are 2000-5000 tokens. This model embeds the full article without chunking. Smaller models (all-MiniLM-L6-v2) truncate at 256 tokens, losing 90%+ of content.
- **Matryoshka support**: Can store 256-dim vectors (1 KB each) instead of 768-dim (3 KB each). At 10,000 documents, this is 10 MB vs 30 MB. Quality drops only ~1.2 MTEB points.
- **MTEB competitive**: 62.4 MTEB average, competitive with models 3x its size
- **Negligible budget impact**: 137M is ~3% of the 4B parameter budget

**Input/Output contract:**
- Input: text with mandatory prefix (`search_document: <text>` for indexing, `search_query: <text>` for querying)
- Output: float32 numpy array of shape (1, 768) or truncated to (1, 256)

**Role in pipeline:**
1. Embed all ingested content into vectors
2. Compute pairwise cosine similarity for knowledge map (UMAP visualization)
3. Find "forgotten" items: content with high cosine similarity to recent items but no user interaction in 7+ days
4. Power the "Connections You Missed" section of the daily briefing

---

### Model 2 -- Prioritizer: openbmb/MiniCPM5-1B

| Property | Value |
|----------|-------|
| Parameters | ~1.08B |
| Architecture | LlamaForCausalLM, GQA (16 Q-heads, 2 KV-heads) |
| Context | 131,072 tokens |
| License | Apache 2.0 |
| Variants | Base, SFT, GGUF |

**Why this model:**
- **OpenBMB sponsor target**: Using MiniCPM qualifies for the $10K OpenBMB Special Award ($5K per track)
- **1B sweet spot**: Large enough for structured reasoning, small enough for fast LoRA fine-tuning (~30 min on A10G)
- **131K context**: Can process long articles + goals in a single pass
- **Think/no-think modes**: `enable_thinking=False` for fast structured output in production

**Input/Output contract:**
- Input: chat-formatted prompt containing content summary (title, first 500 chars, URL) + user goals list
- Output: structured JSON:
  ```json
  {
    "score": 8,
    "rationale": "Directly relevant to your goal of learning post-training techniques",
    "action": "Read today -- high alignment with GRPO learning goal"
  }
  ```

**Fine-tuning approach:**
- Knowledge distillation from Nemotron-Super-49B-v1 (teacher) via NVIDIA NIM API
- Generate ~2000 training examples (content + goals -> priority JSON)
- LoRA SFT with TRL SFTTrainer (r=16, alpha=32)
- Train on Modal A10G (~$5-10 credits, ~30 min)
- Push merged adapter to HuggingFace Hub

---

### Model 3 -- Mentor: MiniCPM5-1B + LoRA-mentor

| Property | Value |
|----------|-------|
| Base Parameters | ~1.08B (shared with Model 2) |
| LoRA Parameters | ~16M (r=32, alpha=64) |
| Architecture | LlamaForCausalLM, GQA (16 Q-heads, 2 KV-heads) |
| Context | 131,072 tokens |
| License | Apache 2.0 |

**Why this model:**
- **Shared base with Model 2**: Load MiniCPM5-1B once (~2GB), attach two LoRA adapters. Total VRAM: ~2GB + a few MB per adapter
- **Tiny Titan eligible**: Total ~2.2B params (137M + 1B + 1B) instead of ~4.3B with SmolLM3
- **Same chat format as Model 2**: No tokenizer mismatch, same `enable_thinking=False` API
- **Adapter swapping at runtime**: `model.set_adapter("mentor")` vs `model.set_adapter("scorer")`
- **OpenBMB sponsor relevance**: Both adapters built on their flagship model

**Input/Output contract:**
- Input: structured prompt containing:
  - Top-K scored items (from Model 2) with scores and rationales
  - Forgotten items (from Model 1 similarity + interaction tracking)
  - Mistake patterns (from mistake tracker)
  - User goals (from goals table)
- Output: markdown daily briefing with sections:
  - **Today's Focus**: Top 3-5 items to read/study, with reasons
  - **You're Forgetting**: Items bookmarked 7+ days ago that are highly relevant but never opened
  - **Watch Out**: Mistake patterns that relate to today's focus areas
  - **Connections You Missed**: Surprising links between recent content

**Personality directive**: The mentor is opinionated, not neutral. It uses direct language:
- "Stop reading about Redis internals" not "You might consider deprioritizing Redis content"
- "This is 10x more relevant to your goals" not "This content has high alignment"
- "You forgot about this -- again" not "This item has low interaction count"

**Fine-tuning approach:**
- Knowledge distillation from Nemotron-Ultra-550B (teacher) via NVIDIA NIM API
- Generate ~2000 training examples (synthetic briefing contexts -> markdown briefings)
- LoRA SFT with TRL SFTTrainer (r=32, alpha=64, max_seq_length=4096)
- Train on Modal A10G (~$5-10 credits, ~30 min)
- Push adapter only to HuggingFace Hub (sarthakbiswas/learnlens-mentor-lora)

---

## 5. Data Pipeline

### Stage 1: Ingestion

Three input methods, each producing `ContentItem` records:

| Method | Parser | What It Extracts |
|--------|--------|-----------------|
| Paste URL | trafilatura | title, body text, URL, word count |
| Upload bookmarks HTML | BeautifulSoup | URLs + titles from `<DT><A>` tags, then trafilatura for each URL |
| Upload Chrome history SQLite | Direct SQL query | URLs + titles + visit counts + last visit from `urls` table |

**Deduplication**: By URL. If a URL already exists in the content table, skip it (upsert pattern).

**Content limits**: Body text truncated to 4096 characters for embedding. Full text stored separately for display.

### Stage 2: Processing

- Strip HTML, normalize whitespace, remove boilerplate (trafilatura handles most of this)
- Extract metadata: word count, estimated read time, domain, publication date if available
- Classify source type: article, paper, documentation, social media post, video transcript

### Stage 3: Embedding (Model 1)

- Batch embed all unembedded content (query: `SELECT * FROM content WHERE id NOT IN (SELECT content_id FROM embeddings)`)
- Store embeddings as numpy float32 BLOBs in SQLite
- Build similarity matrix for knowledge map (UMAP)

### Stage 4: Scoring (Model 2)

- For each unscored content-goal pair, run Model 2
- Cache scores in the `scores` table (invalidated when goals change)
- Score range: 1-10, with rationale text

### Stage 5: Generation (Model 3)

- Assemble briefing context:
  - Top-K scored items (default K=10)
  - Forgotten items: embedded 7+ days ago, cosine similarity > 0.6 to any active goal embedding, zero interactions
  - Mistake patterns from mistake tracker
  - Active user goals
- Generate markdown briefing
- Store briefing in `briefings` table with timestamp and referenced content IDs

---

## 6. Storage Schema

All data in a single SQLite file. WAL mode enabled for concurrent reads.

### Tables

```
content
-------
id              INTEGER PRIMARY KEY AUTOINCREMENT
url             TEXT UNIQUE NOT NULL
title           TEXT NOT NULL
body_text       TEXT NOT NULL
source_type     TEXT NOT NULL  -- 'url' | 'bookmark' | 'history'
word_count      INTEGER
domain          TEXT
ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP

embeddings
----------
id              INTEGER PRIMARY KEY AUTOINCREMENT
content_id      INTEGER NOT NULL REFERENCES content(id)
vector          BLOB NOT NULL  -- numpy float32 tobytes()
dim_size        INTEGER NOT NULL  -- 768 or 256
model_version   TEXT NOT NULL
created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP

goals
-----
id              INTEGER PRIMARY KEY AUTOINCREMENT
goal_text       TEXT NOT NULL
priority        INTEGER NOT NULL DEFAULT 3  -- 1 (low) to 5 (critical)
is_active       BOOLEAN NOT NULL DEFAULT 1
created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP

scores
------
id              INTEGER PRIMARY KEY AUTOINCREMENT
content_id      INTEGER NOT NULL REFERENCES content(id)
goal_id         INTEGER NOT NULL REFERENCES goals(id)
score           REAL NOT NULL  -- 1.0 to 10.0
rationale       TEXT NOT NULL
suggested_action TEXT
scored_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
model_version   TEXT NOT NULL

mistakes
--------
id              INTEGER PRIMARY KEY AUTOINCREMENT
pattern         TEXT NOT NULL  -- short label, e.g. "forgot to freeze embeddings"
description     TEXT NOT NULL  -- full description of the mistake
examples        TEXT           -- JSON array of specific instances
severity        INTEGER NOT NULL DEFAULT 3  -- 1 (minor) to 5 (critical)
created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP

briefings
---------
id              INTEGER PRIMARY KEY AUTOINCREMENT
content         TEXT NOT NULL  -- full markdown briefing
items_referenced TEXT NOT NULL  -- JSON array of content IDs used
generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP

interactions
------------
id              INTEGER PRIMARY KEY AUTOINCREMENT
content_id      INTEGER NOT NULL REFERENCES content(id)
action          TEXT NOT NULL  -- 'viewed' | 'dismissed' | 'saved'
acted_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

### Indexes

```sql
CREATE UNIQUE INDEX idx_embeddings_content ON embeddings(content_id);
CREATE INDEX idx_scores_content_goal ON scores(content_id, goal_id);
CREATE INDEX idx_scores_score ON scores(score DESC);
CREATE INDEX idx_content_ingested ON content(ingested_at);
CREATE INDEX idx_interactions_content ON interactions(content_id);
CREATE INDEX idx_goals_active ON goals(is_active);
```

---

## 7. Gradio UI Layout

### Tab 1: Daily Briefing (Default View)

The first thing you see when you open LearnLens. Not a dashboard, not a graph -- an opinionated briefing.

- **Generated briefing display**: Rendered markdown with sections (Today's Focus, You're Forgetting, Watch Out, Connections You Missed)
- **Regenerate button**: Re-runs Model 3 with current data
- **Date selector**: Browse past briefings
- **"This is wrong" feedback**: Mark specific recommendations as irrelevant (feeds back into scoring)

### Tab 2: Priority Queue

- **Sortable table**: All scored content, columns: Title, Score, Goal, Rationale, Action, Date Added
- **Filter by goal**: Dropdown to filter by specific goal
- **Bulk dismiss**: Select multiple items to mark as "not relevant"
- **Score threshold slider**: Show only items above N/10

### Tab 3: Forgotten Items

- **Card layout**: Each card shows title, relevance score, days since ingested, similar recent items
- **"Open" button**: Marks as interacted, removes from forgotten list
- **"Not relevant" button**: Dismisses permanently

### Tab 4: Knowledge Map

- **2D scatter plot**: UMAP projection of all content embeddings
- **Color coding**: By topic cluster (k-means on embeddings)
- **Hover tooltip**: Title, score, date
- **Click to expand**: Show full content summary
- **Goal markers**: User goals embedded and shown as distinct markers

### Tab 5: Mistake Tracker

- **Add mistake form**: Pattern name, description, example instances
- **Mistake list**: All tracked patterns with severity, edit/delete
- **Related content**: For each mistake, show recent content that relates to it (cosine similarity between mistake description embedding and content embeddings)

---

## 8. Training Pipeline (Dual LoRA Knowledge Distillation)

### Why Distillation?

MiniCPM5-1B out of the box is a general-purpose model. It does not know how to score learning content against goals, nor how to generate opinionated briefings. We need to teach it both behaviors via two specialized LoRA adapters.

Options:
1. **Manual annotation**: Label 2000+ examples by hand -- too slow for hackathon
2. **Prompt engineering**: Use MiniCPM5-1B with detailed system prompts -- output quality is inconsistent at 1B for both tasks
3. **Knowledge distillation**: Use Nemotron-Ultra-550B (teacher) to generate gold labels, then fine-tune MiniCPM5-1B (student) with dual LoRA adapters -- best quality, automated, targets NVIDIA sponsor prize

### Adapter 1: Scorer (Prioritizer)

```
Step 1: Generate synthetic content-goal pairs (40+ templates, 20 goals)
                    |
                    v
Step 2: Send to Nemotron-Ultra-550B via NVIDIA NIM API
        (system prompt: "You are a learning prioritization expert")
        (output: JSON with score, rationale, action)
                    |
                    v
Step 3: Validate and filter responses
                    |
                    v
Step 4: LoRA fine-tune MiniCPM5-1B
        (TRL SFTTrainer, r=16, alpha=32, max_seq_length=2048)
        (Modal A10G, ~30 min, ~$5-10)
                    |
                    v
Step 5: Push adapter to HuggingFace Hub
        (sarthakbiswas/learnlens-scorer-lora)
```

### Adapter 2: Mentor (Briefing Generator)

```
Step 1: Generate synthetic briefing contexts
        (goals + scored items + forgotten items + mistakes)
                    |
                    v
Step 2: Send to Nemotron-Ultra-550B via NVIDIA NIM API
        (system prompt: "You are an opinionated learning mentor")
        (output: markdown briefing with 4 sections)
                    |
                    v
Step 3: Validate all 4 sections present
                    |
                    v
Step 4: LoRA fine-tune MiniCPM5-1B
        (TRL SFTTrainer, r=32, alpha=64, max_seq_length=4096)
        (Modal A10G, ~30 min, ~$5-10)
                    |
                    v
Step 5: Push adapter to HuggingFace Hub
        (sarthakbiswas/learnlens-mentor-lora)
```

### Data Generation Strategy

Generate diverse examples across:
- **Content types**: ML papers, blog posts, documentation, tutorials, news articles, social media threads
- **Goal types**: "learning post-training", "preparing for interviews", "building a RAG system", "understanding transformers"
- **Score distribution**: Ensure roughly uniform distribution across 1-10 (avoid all-high-score bias)
- **Edge cases**: Content that seems relevant but is not (clickbait), content that seems irrelevant but is (foundational knowledge)

Target: 2000 examples minimum, 5000 if API budget allows.

---

## 9. Privacy and Local-First Architecture

### Design Principles

1. **No external API calls at inference time**: All 3 models run locally. The only network call is initial model download from HuggingFace Hub.
2. **Data stays in SQLite**: Single file, easy to backup, delete, or move. No cloud sync.
3. **Chrome history is read-only**: The app reads a copy of the Chrome history SQLite file. It never modifies the original.
4. **No telemetry**: No analytics, no usage tracking, no crash reporting.
5. **Export everything**: User can download their full SQLite database at any time.

### HF Spaces Exception

On HuggingFace Spaces (demo deployment), data lives in the Space's ephemeral storage. It is:
- Per-session (not shared between users)
- Deleted when the Space restarts
- Never sent to any external service

For persistent use, users run LearnLens locally.

---

## 10. Sponsor Targeting Strategy

| Sponsor/Track | How LearnLens Qualifies | Prize |
|---------------|------------------------|-------|
| **OpenBMB Special Award** | MiniCPM5-1B as shared base for both prioritizer and mentor | $5K per track ($10K total) |
| **Tiny Titan** | Total ~2.2B params (137M + 1B + 1B) | Track prize |
| **Backyard AI** | Local-first, privacy-preserving, runs on consumer hardware | Track prize |
| **Llama Champion badge** | MiniCPM5-1B served via GGUF/llama.cpp | Merit badge |
| **NVIDIA** | 550B -> 1B knowledge distillation from Nemotron Ultra via NIM API | Sponsor prize |
| **HuggingFace ecosystem** | Gradio app, HF Spaces, sentence-transformers, PEFT, TRL | Community recognition |

### Maximizing Visibility

- README prominently lists all models and sponsors
- Demo video shows each model's contribution
- Blog post explains the 3-model architecture and why small models are the only option
- Tag all relevant organizations in social media announcement
