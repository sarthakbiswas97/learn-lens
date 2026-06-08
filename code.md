# LearnLens -- Implementation Guide

## 1. Tech Stack

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| python | >=3.11,<3.13 | Runtime |
| gradio | >=5.0 | UI framework (hackathon requirement) |
| transformers | >=5.6 | Model loading (MiniCPM5 requires >=5.6) |
| sentence-transformers | >=4.1 | Embedding model (nomic-embed-text-v1.5) |
| torch | >=2.7 | Backend for all models |
| pydantic | >=2.12 | Config and data validation |
| trafilatura | >=2.0 | Web content extraction |
| beautifulsoup4 | >=4.13 | Bookmarks HTML parsing |
| numpy | >=2.0 | Embedding storage and similarity |
| umap-learn | >=0.5 | Knowledge map dimensionality reduction |
| plotly | >=6.0 | Knowledge map interactive visualization |

### Training Dependencies (optional group)

| Library | Version | Purpose |
|---------|---------|---------|
| trl | >=0.21 | SFTTrainer for LoRA fine-tuning |
| peft | >=0.15 | LoRA adapter management |
| modal | >=0.73 | Cloud GPU for training |
| openai | >=1.0 | NVIDIA NIM API client (OpenAI-compatible) |
| datasets | >=3.0 | Training data loading |

### Dev Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| pytest | >=9.0 | Testing |
| pytest-cov | >=6.0 | Coverage reporting |
| ruff | >=0.15 | Linting + formatting |

### HF Spaces Runtime

| Library | Version | Purpose |
|---------|---------|---------|
| spaces | (auto-installed) | @spaces.GPU decorator for ZeroGPU |

---

## 2. Build Sequence

### Phase 1: Storage + Ingestion (Day 1-2)

**What to build first and why:** The database is the foundation. Every other component reads from or writes to it. Build this first so you can test everything else in isolation.

1. `learnlens/storage/database.py` -- SQLite connection, schema creation, WAL mode
2. `learnlens/storage/queries.py` -- CRUD operations for all 7 tables
3. `learnlens/pipeline/ingestion.py` -- URL fetching (trafilatura), bookmarks HTML parsing, Chrome history reading
4. `tests/unit/test_database.py` -- Schema creation, insert/query/delete
5. `tests/unit/test_ingestion.py` -- URL parsing, bookmarks parsing, history parsing

**Milestone:** Can ingest a URL and see it in the database.

### Phase 2: Embedding Pipeline (Day 2-3)

1. `learnlens/models/connector.py` -- Load nomic-embed-text-v1.5, embed documents, embed queries
2. `learnlens/pipeline/embedding.py` -- Batch embed unembedded content, store vectors, similarity search
3. `tests/unit/test_connector.py` -- Embedding shape, prefix handling
4. `tests/unit/test_embedding.py` -- Similarity search correctness

**Milestone:** Can embed content and find similar items.

### Phase 3: Training Pipeline (Day 3-4)

1. `training/generate_data.py` -- Nemotron data generation for scorer via NIM API
2. `training/generate_mentor_data.py` -- Nemotron data generation for mentor via NIM API
3. `training/train_prioritizer.py` -- LoRA fine-tune scorer adapter on Modal
4. `training/train_mentor.py` -- LoRA fine-tune mentor adapter on Modal
5. Push both adapters to HuggingFace Hub

**Milestone:** Have both LoRA adapters (scorer + mentor) on HF Hub.

### Phase 4: Scoring + Briefing (Day 4-5)

1. `learnlens/models/prioritizer.py` -- Load MiniCPM5-1B + LoRA, score content
2. `learnlens/models/mentor.py` -- Load MiniCPM5-1B + LoRA-mentor, generate briefing
3. `learnlens/pipeline/scoring.py` -- Score all unscored content-goal pairs
4. `learnlens/pipeline/briefing.py` -- Assemble context, generate briefing, store
5. `tests/unit/test_prioritizer.py` -- Output parsing, score range
6. `tests/unit/test_mentor.py` -- Briefing structure validation

**Milestone:** Full pipeline from URL to generated briefing.

### Phase 5: Gradio UI (Day 5-6)

1. `learnlens/app.py` -- Main Gradio app with tab layout
2. `learnlens/ui/briefing_tab.py` -- Daily briefing display
3. `learnlens/ui/priority_tab.py` -- Priority queue table
4. `learnlens/ui/forgotten_tab.py` -- Forgotten items cards
5. `learnlens/ui/knowledge_tab.py` -- UMAP scatter plot
6. `learnlens/ui/mistakes_tab.py` -- Mistake tracker form + list

**Milestone:** Working Gradio app with all 5 tabs.

### Phase 6: Polish + Deploy (Day 6-7)

1. README.md with architecture diagram, screenshots, sponsor credits
2. Demo video with real data
3. HF Spaces deployment (push to Space repo)
4. Error handling edge cases, loading states, empty states
5. Final testing pass

**Milestone:** Live on HuggingFace Spaces, demo video recorded.

---

## 3. Model Loading Patterns

### Model 1: nomic-embed-text-v1.5

```python
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Load model (137M params)
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
)

# CRITICAL: All inputs MUST have task prefixes
# For indexing documents:
doc_embeddings = model.encode(
    ["search_document: " + text for text in documents],
    show_progress_bar=True,
    batch_size=32,
)

# For querying:
query_embedding = model.encode(
    ["search_query: " + query_text]
)

# Matryoshka truncation (768 -> 256 dims) for storage optimization
def truncate_embedding(embedding, target_dim=256):
    """Truncate and re-normalize for Matryoshka dimensionality reduction."""
    tensor = torch.tensor(embedding)
    normalized = F.layer_norm(tensor, normalized_shape=(tensor.shape[-1],))
    truncated = normalized[..., :target_dim]
    result = F.normalize(truncated, p=2, dim=-1)
    return result.numpy()

# Cosine similarity (model.similarity uses cosine by default)
similarities = model.similarity(query_embedding, doc_embeddings)
```

**Key details:**
- `trust_remote_code=True` required for older library versions
- Prefixes are mandatory: `search_document:` for indexing, `search_query:` for querying
- Other valid prefixes: `clustering:`, `classification:`
- Matryoshka dims: 768 (default), 512, 256, 128, 64
- At 256 dims: MTEB drops only ~1.2 points (62.4 -> 61.0), storage drops 3x

---

### Model 2: MiniCPM5-1B (with LoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM5-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load LoRA adapter (after fine-tuning)
model = PeftModel.from_pretrained(
    base_model,
    "sarthakbiswas/learnlens-prioritizer-lora",
)

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")

# Inference
messages = [
    {
        "role": "system",
        "content": (
            "You are a learning prioritization expert. Given a piece of content "
            "and a list of learning goals, output a JSON object with: "
            "score (1-10), rationale (one sentence), action (what to do)."
        ),
    },
    {
        "role": "user",
        "content": f"Content: {title}\n{summary}\n\nGoals:\n{goals_text}",
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,  # Fast structured output, no reasoning chain
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
# Parse JSON from response
```

**Key details:**
- Requires `transformers>=5.6`
- `enable_thinking=False` for structured output (faster, no thinking tokens)
- `enable_thinking=True` uses temp=0.9, top_p=0.95 (for reasoning tasks)
- GQA architecture: 16 query heads, 2 KV heads
- GGUF variant available: `openbmb/MiniCPM5-1B-GGUF`

---

### Model 3: MiniCPM5-1B + LoRA-mentor

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load shared base model (same as Model 2)
base_model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM5-1B",
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")

# Attach mentor LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "sarthakbiswas/learnlens-mentor-lora",
    adapter_name="mentor",
)

# Before generation, activate the mentor adapter
model.set_adapter("mentor")

messages = [
    {
        "role": "system",
        "content": (
            "You are an opinionated learning mentor. Be direct and specific. "
            "Tell the user what to focus on, what they are forgetting, "
            "and what mistakes to watch for. Do not be neutral -- have opinions."
        ),
    },
    {
        "role": "user",
        "content": briefing_context,  # assembled from scored items, forgotten items, mistakes
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Direct output, no reasoning
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
)

briefing = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
```

**Key details:**
- Shared base with Model 2: load once, attach two LoRA adapters
- `model.set_adapter("mentor")` before briefing generation
- `model.set_adapter("scorer")` before priority scoring
- Sampling: temperature=0.6, top_p=0.95
- `enable_thinking=False` for direct output, no reasoning
- 131K context -- easily fits all briefing context
- GGUF available for llama.cpp serving (Llama Champion badge)

---

## 4. SQLite Schema DDL

```sql
-- Enable WAL mode for concurrent reads
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    body_text TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN ('url', 'bookmark', 'history')),
    word_count INTEGER,
    domain TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id INTEGER NOT NULL REFERENCES content(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    dim_size INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_text TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 3 CHECK(priority BETWEEN 1 AND 5),
    is_active BOOLEAN NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id INTEGER NOT NULL REFERENCES content(id) ON DELETE CASCADE,
    goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
    score REAL NOT NULL CHECK(score BETWEEN 1.0 AND 10.0),
    rationale TEXT NOT NULL,
    suggested_action TEXT,
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mistakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL,
    description TEXT NOT NULL,
    examples TEXT,
    severity INTEGER NOT NULL DEFAULT 3 CHECK(severity BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS briefings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    items_referenced TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id INTEGER NOT NULL REFERENCES content(id) ON DELETE CASCADE,
    action TEXT NOT NULL CHECK(action IN ('viewed', 'dismissed', 'saved')),
    acted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_content ON embeddings(content_id);
CREATE INDEX IF NOT EXISTS idx_scores_content_goal ON scores(content_id, goal_id);
CREATE INDEX IF NOT EXISTS idx_scores_score ON scores(score DESC);
CREATE INDEX IF NOT EXISTS idx_content_ingested ON content(ingested_at);
CREATE INDEX IF NOT EXISTS idx_interactions_content ON interactions(content_id);
CREATE INDEX IF NOT EXISTS idx_goals_active ON goals(is_active);
```

### Numpy Embedding Storage

```python
import numpy as np
import sqlite3

def store_embedding(
    conn: sqlite3.Connection,
    content_id: int,
    vector: np.ndarray,
    model_version: str,
) -> int:
    """Store a numpy embedding as a BLOB in SQLite."""
    dim_size = vector.shape[-1]
    blob = vector.astype(np.float32).tobytes()
    cursor = conn.execute(
        "INSERT INTO embeddings (content_id, vector, dim_size, model_version) VALUES (?, ?, ?, ?)",
        (content_id, blob, dim_size, model_version),
    )
    return cursor.lastrowid


def load_embedding(blob: bytes, dim_size: int = 768) -> np.ndarray:
    """Load a numpy embedding from a SQLite BLOB."""
    return np.frombuffer(blob, dtype=np.float32).reshape(1, dim_size)


def load_all_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """Load all embeddings into a single matrix for batch similarity."""
    rows = conn.execute(
        "SELECT content_id, vector, dim_size FROM embeddings ORDER BY content_id"
    ).fetchall()
    if not rows:
        return [], np.array([])

    content_ids = [row[0] for row in rows]
    dim_size = rows[0][2]
    matrix = np.stack([
        np.frombuffer(row[1], dtype=np.float32).reshape(dim_size)
        for row in rows
    ])
    return content_ids, matrix
```

---

## 5. Cosine Similarity Search

For <100k documents, brute-force numpy is fast and simple. No FAISS/Annoy needed.

```python
import numpy as np

def cosine_search(
    query_vec: np.ndarray,
    all_vecs: np.ndarray,
    content_ids: list[int],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Brute-force cosine similarity search.

    Args:
        query_vec: shape (1, dim) or (dim,), normalized
        all_vecs: shape (N, dim), normalized
        content_ids: list of content IDs matching rows in all_vecs
        top_k: number of results to return

    Returns:
        list of (content_id, similarity_score) sorted descending
    """
    query_normalized = query_vec.flatten() / np.linalg.norm(query_vec)
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    vecs_normalized = all_vecs / np.where(norms > 0, norms, 1)

    similarities = vecs_normalized @ query_normalized  # shape (N,)

    # Use argpartition for efficient top-K (O(N) instead of O(N log N))
    if top_k < len(similarities):
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    else:
        top_indices = np.argsort(similarities)[::-1]

    return [(content_ids[i], float(similarities[i])) for i in top_indices]
```

**Performance estimate:**
- 10,000 documents x 256 dims: ~2ms per query
- 100,000 documents x 256 dims: ~15ms per query
- More than sufficient for interactive use

---

## 6. Knowledge Distillation Pipeline

### Step 1: Generate Training Data from Nemotron

```python
# training/generate_data.py
import json
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_NIM_API_KEY"],
)

SYSTEM_PROMPT = """You are a learning prioritization expert. Given a piece of learning content \
and a set of learning goals, assess how relevant and important this content is for the learner.

Output a JSON object with exactly these fields:
- "score": integer 1-10 (1=irrelevant, 10=critical to read immediately)
- "rationale": one sentence explaining why this score
- "action": one sentence suggesting what the learner should do

Be honest and opinionated. Score low if content is tangential. Score high only if it directly \
advances a stated goal. Consider recency, depth, and practical applicability."""

def generate_training_example(
    content_title: str,
    content_summary: str,
    goals: list[str],
) -> dict | None:
    """Generate a single training example using Nemotron as teacher."""
    goals_text = "\n".join(f"- {g}" for g in goals)
    user_prompt = f"Content: {content_title}\n{content_summary}\n\nLearning Goals:\n{goals_text}"

    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-super-49b-v1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )

    assistant_content = response.choices[0].message.content

    # Validate JSON structure
    try:
        parsed = json.loads(assistant_content)
        assert 1 <= parsed["score"] <= 10
        assert "rationale" in parsed
        assert "action" in parsed
    except (json.JSONDecodeError, KeyError, AssertionError):
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def generate_dataset(output_path: str, num_examples: int = 2000) -> None:
    """Generate full training dataset."""
    # ... iterate over synthetic content-goal pairs
    # ... call generate_training_example for each
    # ... write valid examples to JSONL
    with open(output_path, "w") as f:
        for example in examples:
            result = generate_training_example(
                example["title"],
                example["summary"],
                example["goals"],
            )
            if result is not None:
                f.write(json.dumps(result) + "\n")
```

### Step 2: LoRA Fine-Tuning on Modal

```python
# training/train_prioritizer.py
import modal

app = modal.App("learnlens-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7",
        "transformers>=5.6",
        "trl>=0.21",
        "peft>=0.15",
        "datasets>=3.0",
        "accelerate>=1.0",
        "bitsandbytes>=0.45",
    )
)

@app.function(
    gpu="A10G",
    timeout=3600,
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
)
def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "openbmb/MiniCPM5-1B",
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # Load distillation data
    dataset = load_dataset("json", data_files="distillation_data.jsonl", split="train")

    # Training config
    training_config = SFTConfig(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=2048,
        dataset_text_field=None,  # using messages format
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # Merge and push
    merged_model = trainer.model.merge_and_unload()
    merged_model.push_to_hub("sarthakbiswas/learnlens-prioritizer")
    tokenizer.push_to_hub("sarthakbiswas/learnlens-prioritizer")
```

### Cost Estimates

| Resource | Cost |
|----------|------|
| NVIDIA NIM API (2000 examples) | ~$0-2 (free tier: 1000 credits) |
| Modal A10G (30 min training) | ~$5-10 of $250 credits |
| HuggingFace Hub storage | Free |
| **Total** | **~$5-12** |

---

## 7. Gradio App Structure

### Entry Point

```python
# learnlens/app.py
import gradio as gr
import spaces
from learnlens.config import load_config
from learnlens.storage.database import Database
from learnlens.models.connector import ConnectorModel
from learnlens.models.prioritizer import PrioritizerModel
from learnlens.models.mentor import MentorModel
from learnlens.ui import (
    briefing_tab,
    priority_tab,
    forgotten_tab,
    knowledge_tab,
    mistakes_tab,
)

# Load models at module level (ZeroGPU loads before @spaces.GPU activation)
config = load_config()
connector = ConnectorModel(config)
prioritizer = PrioritizerModel(config)
mentor = MentorModel(config)

def main():
    db = Database(config.db_path)

    with gr.Blocks(title="LearnLens", theme=gr.themes.Soft()) as app:
        gr.Markdown("# LearnLens -- Your AI Learning Mentor")
        gr.Markdown("*An opinionated mentor that tells you what to focus on and what to stop wasting time on.*")

        with gr.Tabs():
            briefing_tab.create(db, connector, prioritizer, mentor, config)
            priority_tab.create(db, config)
            forgotten_tab.create(db, connector, config)
            knowledge_tab.create(db, connector, config)
            mistakes_tab.create(db, connector, config)

    app.launch()


if __name__ == "__main__":
    main()
```

### Tab Pattern

Each tab module exports a single `create()` function:

```python
# learnlens/ui/briefing_tab.py
import gradio as gr
import spaces

@spaces.GPU(duration=120)
def generate_briefing_gpu(db, connector, prioritizer, mentor, config):
    """GPU-accelerated briefing generation."""
    # ... run full pipeline: embed -> score -> generate
    pass

def create(db, connector, prioritizer, mentor, config):
    with gr.Tab("Daily Briefing"):
        briefing_output = gr.Markdown(value="Click 'Generate' to create your daily briefing.")
        with gr.Row():
            generate_btn = gr.Button("Generate Briefing", variant="primary")
            date_picker = gr.DateTime(label="Browse Past Briefings")
        # ... wire up event handlers
```

### ZeroGPU Pattern

```python
import spaces

# Models loaded at module level (CUDA emulation at import time)
# Real GPU allocated ONLY inside @spaces.GPU functions

@spaces.GPU(duration=120)  # 120 second timeout
def inference_function(input_text):
    # GPU is active here
    embedding = connector.embed_documents([input_text])
    score = prioritizer.score(content, goals)
    briefing = mentor.generate_briefing(context)
    return briefing
```

**Important ZeroGPU details:**
- Default timeout: 60s per call (configurable with `duration=N`)
- Free tier: 5 min/day (free accounts), 40 min/day (PRO)
- Models load to CUDA at import time via emulation, real GPU only during decorated functions
- All 3 models (~4.1B params in bf16) fit in a single ZeroGPU allocation (48 GB VRAM)

---

## 8. HF Spaces Deployment

### README YAML Header

```yaml
---
title: LearnLens
emoji: "\U0001F50D"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0"
app_file: learnlens/app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
short_description: AI learning mentor that tells you what to focus on
tags:
  - learnlens
  - learning-mentor
  - small-models
  - minicpm
  - smollm
---
```

### Hardware Options

| Option | VRAM | Cost | Best For |
|--------|------|------|----------|
| **zero-a10g** (recommended) | 48 GB | Free (PRO) | Demo, judging |
| t4-small | 16 GB | $0.40/hr | Tight on VRAM for 3 models |
| l4 | 24 GB | $0.80/hr | Safe option if ZeroGPU quota runs out |

### Deployment Steps

1. Create HF Space: `huggingface-cli repo create learnlens --type space --space-sdk gradio`
2. Clone and push code
3. Set secrets in Space settings: `NVIDIA_NIM_API_KEY` (only needed for training, not inference)
4. Space auto-builds and deploys

---

## 9. Input Methods

### Method A: Paste URL

```python
import trafilatura

def fetch_and_extract(url: str) -> tuple[str, str] | None:
    """Fetch URL and extract title + body text."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None

    result = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        output_format="txt",
    )
    if result is None:
        return None

    metadata = trafilatura.extract(
        downloaded,
        output_format="xmltei",
        include_comments=False,
    )
    # Extract title from metadata or fallback to first line
    title = _extract_title(metadata, result)
    return title, result
```

### Method B: Upload Bookmarks HTML

```python
from bs4 import BeautifulSoup
from pathlib import Path

def parse_bookmarks_html(file_path: Path) -> list[dict[str, str]]:
    """Parse Chrome/Firefox bookmarks HTML export."""
    html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    bookmarks = []
    for link in soup.find_all("a"):
        url = link.get("href", "")
        title = link.get_text(strip=True)
        if url.startswith("http"):
            bookmarks.append({"url": url, "title": title})
    return bookmarks
```

### Method C: Upload Chrome History SQLite

```python
import sqlite3
from pathlib import Path

def parse_chrome_history(
    file_path: Path,
    limit: int = 500,
) -> list[dict[str, str]]:
    """Read Chrome history SQLite file (user uploads a copy)."""
    conn = sqlite3.connect(str(file_path))
    rows = conn.execute(
        """
        SELECT url, title, visit_count, last_visit_time
        FROM urls
        WHERE url LIKE 'http%'
        ORDER BY last_visit_time DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    return [
        {
            "url": row[0],
            "title": row[1] or "",
            "visit_count": row[2],
        }
        for row in rows
    ]
```

---

## 10. Error Handling and Validation

### Pydantic Models for Pipeline Data

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class ContentItem:
    id: int
    url: str
    title: str
    body_text: str
    source_type: str  # 'url' | 'bookmark' | 'history'
    word_count: int
    ingested_at: datetime

@dataclass(frozen=True)
class Goal:
    id: int
    goal_text: str
    priority: int  # 1-5
    is_active: bool

@dataclass(frozen=True)
class ScoredItem:
    content: ContentItem
    score: float  # 1.0-10.0
    rationale: str
    suggested_action: str
    goal: Goal

@dataclass(frozen=True)
class BriefingRequest:
    scored_items: tuple[ScoredItem, ...]
    forgotten_items: tuple[ContentItem, ...]
    mistakes: tuple[str, ...]
    goals: tuple[Goal, ...]

@dataclass(frozen=True)
class MistakePattern:
    id: int
    pattern: str
    description: str
    examples: tuple[str, ...]
    severity: int  # 1-5
```

### Logging Pattern

```python
import logging

logger = logging.getLogger(__name__)

# In every module:
logger.info("Embedding %d documents", len(documents))
logger.warning("Failed to fetch URL: %s", url)
logger.error("Model inference failed: %s", str(e))
```

### Graceful Degradation

If a model fails, the pipeline should degrade gracefully:
- Model 1 fails: Show content without embeddings, disable knowledge map
- Model 2 fails: Show content in chronological order (no scoring), skip priority queue
- Model 3 fails: Show scored items as a raw list instead of generated briefing

### Input Validation

- URL format: must start with `http://` or `https://`
- File type: bookmarks must be `.html`, history must be `.sqlite` or `.db`
- Content length: body_text capped at 50,000 characters
- Goal text: 1-500 characters, at least one goal required for scoring
- Mistake pattern: 1-200 characters for pattern name

---

## 11. Testing Strategy

### Unit Tests

```
tests/
├── unit/
│   ├── test_database.py          # Schema creation, CRUD, WAL mode
│   ├── test_queries.py           # All query functions, edge cases
│   ├── test_ingestion.py         # URL parsing, bookmarks, history
│   ├── test_embedding.py         # Similarity search, Matryoshka truncation
│   ├── test_connector.py         # Embedding shape, prefix handling
│   ├── test_prioritizer.py       # Output JSON parsing, score validation
│   ├── test_mentor.py            # Briefing structure, markdown output
│   ├── test_config.py            # Config loading, defaults, overrides
│   └── test_types.py             # Frozen dataclass immutability
├── integration/
│   ├── test_pipeline.py          # Full pipeline URL -> briefing (mocked models)
│   └── test_gradio_app.py        # App launches, tabs render
└── conftest.py                   # Shared fixtures (tmp database, sample data)
```

### Key Fixtures

```python
# tests/conftest.py
import pytest
from learnlens.storage.database import Database

@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database for testing."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    yield db
    db.close()

@pytest.fixture
def sample_content():
    """Sample content items for testing."""
    return [
        ContentItem(
            id=1,
            url="https://example.com/article1",
            title="Understanding LoRA Fine-Tuning",
            body_text="LoRA (Low-Rank Adaptation) is a technique...",
            source_type="url",
            word_count=1500,
            ingested_at=datetime.now(),
        ),
        # ... more samples
    ]

@pytest.fixture
def sample_embeddings():
    """Pre-computed embeddings for testing similarity."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 256)).astype(np.float32)
```

### Running Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit/

# With coverage
uv run pytest --cov=learnlens --cov-report=term-missing

# Specific test
uv run pytest tests/unit/test_database.py -v
```
