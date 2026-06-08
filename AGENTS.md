# LearnLens Development Guide

## Quick Start

# Install dependencies
uv sync

# Run locally
uv run learnlens

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

## Training Pipeline

# Generate distillation data (requires NVIDIA_NIM_API_KEY)
uv run python training/generate_data.py
uv run python training/generate_mentor_data.py

# Fine-tune on Modal (requires modal setup)
modal run training/train_prioritizer.py
modal run training/train_mentor.py

## Project Structure

- learnlens/models/ -- Model wrappers (connector, prioritizer, mentor)
- learnlens/pipeline/ -- Data processing (ingestion, embedding, scoring, briefing)
- learnlens/storage/ -- SQLite database operations
- learnlens/ui/ -- Gradio tab components
- training/ -- Model training scripts (not part of main package)

## Conventions

- Type annotations on all functions
- Frozen dataclasses for data types
- logging, never print()
- pathlib.Path, never os.path
- Parameterized SQL queries only
- 200-400 lines per file, 800 max
