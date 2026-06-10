---
title: LearnLens
emoji: "🔍"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0"
app_file: learnlens/app.py
pinned: false
license: apache-2.0
hardware: a10g
short_description: AI learning mentor that tells you what to focus on
tags:
  - learnlens
  - learning-mentor
  - small-models
  - minicpm
  - nemotron
  - knowledge-distillation
---

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
    Model 2: Prioritizer (MiniCPM5-1B + LoRA-scorer, ~1B)
         |  -> structured JSON scoring 1-10 vs goals
         |
    Model 3: Mentor (MiniCPM5-1B + LoRA-mentor, ~1B)
         |  -> opinionated markdown daily briefing
         |
    Gradio UI (5 tabs)
```

**Key innovation:** Models 2 and 3 share the same MiniCPM5-1B base weights (~2GB) and swap LoRA adapters at runtime. Total parameters: **~2.2B** (Tiny Titan track eligible).

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
| Prioritizer | MiniCPM5-1B + LoRA-scorer (~1B) |
| Mentor | MiniCPM5-1B + LoRA-mentor (~1B) |
| Shared Base | MiniCPM5-1B loaded once, 2 adapters swapped |
| Storage | SQLite (WAL mode) |
| Training | Modal + TRL + NVIDIA NIM |

---

## Training

```bash
# Generate distillation data (requires NVIDIA_NIM_API_KEY)
uv run python training/generate_data.py
uv run python training/generate_mentor_data.py

# Fine-tune both adapters on Modal
modal run training/train_prioritizer.py
modal run training/train_mentor.py
```

---

## Sponsor Alignment

| Sponsor | How We Qualify |
|---------|---------------|
| **OpenBMB** | MiniCPM5-1B as shared base for both prioritizer and mentor |
| **Tiny Titan** | ~2.2B total parameters (137M + 1B + 1B) |
| **Backyard AI** | Local-first, privacy-preserving |
| **Llama Champion** | GGUF export for llama.cpp |
| **NVIDIA** | 49B -> 1B knowledge distillation via Nemotron Super |
| **HuggingFace** | Gradio, HF Spaces, PEFT, TRL |

---

## License

Apache 2.0
