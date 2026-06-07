# LearnLens 🔍

**AI learning mentor powered by 3 small models.**

LearnLens is not a bookmark manager. It is an **opinionated coach** that tells you what to focus on, what you're forgetting, and what mistakes to watch for.

---

## Why LearnLens?

Every day, ML engineers and researchers consume dozens of articles, papers, tweets, and videos. By the next morning, 90% is forgotten.

- **ChatGPT/Claude** can't access your browsing history or remember your mistakes
- **Obsidian/Notion** are passive libraries -- they store, but never judge or prioritize

LearnLens runs **locally**, reads your content, and actively tells you when you're wasting time.

---

## Architecture

```
User Input (URLs / Bookmarks / Chrome History)
         |
    Ingestion (trafilatura / BS4)
         |
    +------------------+
    |   SQLite DB      |
    +------------------+
         |
    Model 1: Connector (nomic-embed-text-v1.5, 137M)
         |  -> embeddings, similarity, forgotten items
         |
    Model 2: Prioritizer (MiniCPM5-1B + LoRA, ~1B)
         |  -> structured JSON scoring 1-10 vs goals
         |
    Model 3: Mentor (SmolLM3-3B, 3B)
         |  -> opinionated markdown daily briefing
         |
    Gradio UI (5 tabs)
```

**Total parameters: ~4.1B** (Tiny Titan track eligible)

---

## Features

- **Daily Briefing** — opinionated markdown report on what to read today
- **Priority Queue** — all content scored 1-10 against your goals
- **Forgotten Items** — high-relevance content you haven't touched in 7+ days
- **Knowledge Map** — UMAP visualization of your content embeddings
- **Mistake Tracker** — track patterns you want to avoid repeating

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run locally
uv run learnlens

# Seed demo data
uv run python scripts/seed_demo_data.py
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Gradio 5.0+ |
| Embedding | nomic-embed-text-v1.5 (137M) |
| Prioritizer | MiniCPM5-1B + LoRA (~1B) |
| Mentor | SmolLM3-3B (3B) |
| Storage | SQLite (WAL mode) |
| Training | Modal + TRL + NVIDIA NIM |

---

## Training

```bash
# Generate distillation data (requires NVIDIA_NIM_API_KEY)
uv run python training/generate_data.py

# Fine-tune on Modal
modal run training/train_prioritizer.py
```

---

## Sponsor Alignment

| Sponsor | How We Qualify |
|---------|---------------|
| **OpenBMB** | MiniCPM5-1B as core Model 2 |
| **Tiny Titan** | ~4.1B total parameters |
| **Backyard AI** | Local-first, privacy-preserving |
| **Llama Champion** | GGUF export for llama.cpp |
| **NVIDIA** | Knowledge distillation from Nemotron |
| **HuggingFace** | Gradio, HF Spaces, SmolLM3 |

---

## License

Apache 2.0
