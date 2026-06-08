# LearnLens -- Project Structure Guide

This guide defines the project layout, conventions, and module contracts for LearnLens. It is modeled after the [ml-intern](../ml-intern/) project structure, adapted for a Gradio-based application.

---

## 1. Project Structure

```
learnlens/
├── learnlens/                      # Main Python package
│   ├── __init__.py                 # Package init, version
│   ├── app.py                      # Gradio app entry point (main)
│   ├── config.py                   # Pydantic settings, config loading
│   │
│   ├── models/                     # Model loading and inference
│   │   ├── __init__.py
│   │   ├── types.py                # Shared data types (frozen dataclasses)
│   │   ├── connector.py            # nomic-embed-text-v1.5 (embedding)
│   │   ├── prioritizer.py          # MiniCPM5-1B + LoRA (scoring)
│   │   └── mentor.py               # MiniCPM5-1B + LoRA (briefing generation)
│   │
│   ├── pipeline/                   # Data processing pipeline
│   │   ├── __init__.py
│   │   ├── ingestion.py            # URL fetch, bookmarks parse, history parse
│   │   ├── embedding.py            # Batch embed, similarity search
│   │   ├── scoring.py              # Priority scoring against goals
│   │   └── briefing.py             # Daily briefing assembly and generation
│   │
│   ├── storage/                    # SQLite operations
│   │   ├── __init__.py
│   │   ├── database.py             # Connection, schema, migrations, WAL
│   │   └── queries.py              # CRUD for all 7 tables
│   │
│   ├── ui/                         # Gradio tab components
│   │   ├── __init__.py
│   │   ├── briefing_tab.py         # Daily briefing display + generate
│   │   ├── priority_tab.py         # Scored content table + filters
│   │   ├── forgotten_tab.py        # Forgotten items cards
│   │   ├── knowledge_tab.py        # UMAP scatter plot
│   │   └── mistakes_tab.py         # Mistake tracker form + list
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── logging.py              # Logging configuration
│       └── text.py                 # Text cleaning, truncation, extraction
│
├── training/                       # Model training (not part of main package)
│   ├── generate_data.py            # Nemotron distillation data generation (scorer)
│   ├── generate_mentor_data.py     # Nemotron distillation data generation (mentor)
│   ├── train_prioritizer.py        # LoRA fine-tuning scorer adapter on Modal
│   ├── train_mentor.py             # LoRA fine-tuning mentor adapter on Modal
│   └── export_gguf.py              # GGUF conversion for llama.cpp
│
├── configs/
│   └── default.json                # Default configuration values
│
├── tests/
│   ├── conftest.py                 # Shared fixtures (tmp_db, sample data)
│   ├── unit/
│   │   ├── test_database.py
│   │   ├── test_queries.py
│   │   ├── test_ingestion.py
│   │   ├── test_embedding.py
│   │   ├── test_connector.py
│   │   ├── test_prioritizer.py
│   │   ├── test_mentor.py
│   │   ├── test_config.py
│   │   └── test_types.py
│   └── integration/
│       ├── test_pipeline.py
│       └── test_gradio_app.py
│
├── scripts/
│   └── seed_demo_data.py           # Populate DB with demo content for judging
│
├── data/                           # Local data directory (gitignored)
│   └── learnlens.db                # SQLite database (created at runtime)
│
├── pyproject.toml                  # Project metadata, dependencies, tools
├── uv.lock                         # Locked dependency versions
├── .env.example                    # Template for environment variables
├── .gitignore                      # Git ignore rules
├── README.md                       # User-facing documentation
├── AGENTS.md                       # Development guide (for AI agents + devs)
├── LICENSE                         # Apache 2.0
└── design.md                       # Architecture document (this companion doc)
```

### Mapping from ml-intern

| ml-intern | LearnLens | Rationale |
|-----------|-----------|-----------|
| `agent/main.py` (CLI entry) | `learnlens/app.py` (Gradio entry) | Gradio replaces CLI as the interface |
| `agent/config.py` | `learnlens/config.py` | Same Pydantic config pattern |
| `agent/core/` | `learnlens/pipeline/` | Core processing logic |
| `agent/tools/` | `learnlens/models/` | External capability wrappers |
| `agent/context_manager/` | `learnlens/storage/` | State persistence layer |
| `agent/utils/` | `learnlens/utils/` | Shared utilities |
| `backend/` (FastAPI) | Not needed | Gradio handles HTTP serving |
| `frontend/` (React) | `learnlens/ui/` | Gradio tabs replace React components |
| `configs/` | `configs/` | Same JSON config pattern |
| `tests/` | `tests/` | Same pytest structure |
| `scripts/` | `scripts/` + `training/` | Training scripts separated out |

---

## 2. Conventions

### Package Management

- **uv** for dependency management and virtual environment
- `pyproject.toml` as the single source of truth for metadata and deps
- `uv.lock` committed to git for reproducible builds
- Dependency groups: core (default), `training` (optional), `dev` (optional)

### Configuration

- **Pydantic BaseModel** for typed, validated config
- **JSON config file** (`configs/default.json`) for defaults
- **Environment variables** for secrets (API keys, tokens)
- **Config loading hierarchy**: JSON defaults -> env var overrides -> runtime overrides

```python
# learnlens/config.py
from pydantic import BaseModel, Field
from pathlib import Path

class LearnLensConfig(BaseModel):
    """Application configuration."""

    # Storage
    db_path: Path = Field(default=Path("data/learnlens.db"))

    # Model IDs
    embedding_model_id: str = "nomic-ai/nomic-embed-text-v1.5"
    prioritizer_adapter_id: str = "sarthakbiswas/learnlens-scorer-lora"
    prioritizer_base_model_id: str = "openbmb/MiniCPM5-1B"
    mentor_adapter_id: str = "sarthakbiswas/learnlens-mentor-lora"

    # Embedding
    embedding_dim: int = Field(default=256, description="Matryoshka dim (768, 512, 256, 128, 64)")

    # Pipeline
    max_content_length: int = Field(default=4096, description="Max chars for embedding input")
    top_k_briefing: int = Field(default=10, description="Top scored items for briefing")
    forgotten_days_threshold: int = Field(default=7, description="Days before item is 'forgotten'")
    forgotten_similarity_threshold: float = Field(default=0.6, description="Min cosine sim to goals")

    # Generation
    mentor_temperature: float = 0.6
    mentor_top_p: float = 0.95
    mentor_max_tokens: int = 2048
    prioritizer_temperature: float = 0.7
    prioritizer_max_tokens: int = 256


def load_config(config_path: Path | None = None) -> LearnLensConfig:
    """Load config from JSON file, with env var overrides."""
    ...
```

### Type Annotations

- **All function signatures** must have type annotations
- **Return types** always specified
- Use `from __future__ import annotations` for forward references
- Prefer `list[str]` over `List[str]` (Python 3.11+)
- Use `X | None` over `Optional[X]`

### Immutability

- **Frozen dataclasses** for all data transfer objects
- **NamedTuples** for simple return types
- **Never mutate** function arguments or shared state
- Create new objects instead of modifying existing ones

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ContentItem:
    """Immutable content record."""
    id: int
    url: str
    title: str
    body_text: str
    source_type: str
    word_count: int
    ingested_at: datetime
```

### Logging

- `logging.getLogger(__name__)` in every module
- **Never use `print()`** for operational output
- Log levels: DEBUG (verbose pipeline steps), INFO (operations), WARNING (degraded), ERROR (failures)
- Configure once in `learnlens/utils/logging.py`

```python
# learnlens/utils/logging.py
import logging

def configure_logging(level: str = "INFO") -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
```

### File Paths

- **pathlib.Path** exclusively, never `os.path`
- Relative paths resolved from project root
- Database path configurable via config

### File Size

- **200-400 lines typical**, 800 absolute maximum per file
- If a file exceeds 400 lines, consider splitting by responsibility
- Each file should have a single clear purpose

### Naming

| Thing | Convention | Example |
|-------|-----------|---------|
| Files | `snake_case.py` | `briefing_tab.py` |
| Classes | `PascalCase` | `ConnectorModel` |
| Functions | `snake_case` | `embed_documents()` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_EMBEDDING_DIM` |
| Booleans | `is_`, `has_`, `should_` prefix | `is_active`, `has_embedding` |
| Private | leading underscore | `_parse_json_response()` |

### Error Handling

- Handle errors explicitly, never swallow silently
- Use specific exception types, not bare `except`
- Log errors with context before re-raising or handling
- Graceful degradation: if one model fails, the rest should still work

---

## 3. Module Contracts

### learnlens/models/types.py -- Shared Data Types

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class ContentItem:
    id: int
    url: str
    title: str
    body_text: str
    source_type: str      # 'url' | 'bookmark' | 'history'
    word_count: int
    ingested_at: datetime

@dataclass(frozen=True)
class Goal:
    id: int
    goal_text: str
    priority: int          # 1-5
    is_active: bool

@dataclass(frozen=True)
class ScoredItem:
    content: ContentItem
    score: float           # 1.0-10.0
    rationale: str
    suggested_action: str
    goal: Goal

@dataclass(frozen=True)
class MistakePattern:
    id: int
    pattern: str           # short label
    description: str       # full description
    examples: tuple[str, ...]
    severity: int          # 1-5

@dataclass(frozen=True)
class BriefingRequest:
    scored_items: tuple[ScoredItem, ...]
    forgotten_items: tuple[ContentItem, ...]
    mistakes: tuple[MistakePattern, ...]
    goals: tuple[Goal, ...]
```

---

### learnlens/models/connector.py

```python
class ConnectorModel:
    """Wrapper for nomic-embed-text-v1.5 embedding model."""

    def __init__(self, config: LearnLensConfig) -> None: ...
    def load(self) -> None: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents with 'search_document:' prefix.

        Args:
            texts: raw document texts (prefix added internally)

        Returns:
            numpy array of shape (len(texts), config.embedding_dim)
        """
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with 'search_query:' prefix.

        Returns:
            numpy array of shape (1, config.embedding_dim)
        """
        ...
```

**Dependencies:** sentence-transformers, torch, numpy
**Side effects:** loads model to GPU on `load()`

---

### learnlens/models/prioritizer.py

```python
class PrioritizerModel:
    """Wrapper for MiniCPM5-1B + LoRA-scorer priority scorer."""

    def __init__(
        self,
        config: LearnLensConfig,
        base_model: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
    ) -> None: ...
    def load(self) -> None: ...

    def score(
        self,
        content: ContentItem,
        goals: list[Goal],
    ) -> ScoredItem:
        """Score a content item against user goals.

        Returns:
            ScoredItem with score (1-10), rationale, and suggested action
        """
        ...
```

**Dependencies:** transformers, peft, torch
**Side effects:** attaches LoRA-scorer adapter to shared base model on `load()`

---

### learnlens/models/mentor.py

```python
class MentorModel:
    """Wrapper for MiniCPM5-1B + LoRA-mentor briefing generator."""

    def __init__(
        self,
        config: LearnLensConfig,
        base_model: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
    ) -> None: ...
    def load(self) -> None: ...

    def generate_briefing(self, request: BriefingRequest) -> str:
        """Generate a daily briefing from scored items, forgotten items, and mistakes.

        Returns:
            Markdown-formatted daily briefing string
        """
        ...
```

**Dependencies:** transformers, peft, torch
**Side effects:** attaches LoRA-mentor adapter to shared base model on `load()`

---

### learnlens/pipeline/ingestion.py

```python
def ingest_url(url: str, db: Database) -> ContentItem | None:
    """Fetch URL content and store in database.

    Returns None if URL is unreachable or content extraction fails.
    Deduplicates by URL (skips if already exists).
    """
    ...

def ingest_bookmarks(file_path: Path, db: Database) -> list[ContentItem]:
    """Parse bookmarks HTML export and ingest all URLs.

    Returns list of successfully ingested items.
    Fetches content for each bookmark URL via trafilatura.
    """
    ...

def ingest_chrome_history(
    file_path: Path,
    db: Database,
    limit: int = 500,
) -> list[ContentItem]:
    """Read Chrome history SQLite and ingest top URLs by visit count.

    User uploads a COPY of their Chrome history file.
    Only processes http/https URLs.
    """
    ...
```

**Dependencies:** trafilatura, beautifulsoup4, sqlite3
**Side effects:** writes to database, makes HTTP requests (trafilatura)

---

### learnlens/pipeline/embedding.py

```python
def embed_new_content(db: Database, model: ConnectorModel) -> int:
    """Embed all content that doesn't have embeddings yet.

    Returns count of newly embedded items.
    """
    ...

def find_similar(
    query_vec: np.ndarray,
    db: Database,
    top_k: int = 10,
) -> list[tuple[ContentItem, float]]:
    """Find top-K most similar content items to query vector.

    Returns list of (ContentItem, cosine_similarity) pairs, sorted descending.
    Uses brute-force cosine similarity (fast for <100K docs).
    """
    ...

def find_forgotten_items(
    db: Database,
    model: ConnectorModel,
    goals: list[Goal],
    days_threshold: int = 7,
    similarity_threshold: float = 0.6,
) -> list[ContentItem]:
    """Find items ingested 7+ days ago with high goal relevance but no interactions.

    Embeds active goals, computes similarity to all old content,
    filters by interaction count = 0.
    """
    ...
```

**Dependencies:** numpy, learnlens.storage, learnlens.models.connector
**Side effects:** reads/writes database

---

### learnlens/pipeline/scoring.py

```python
def score_content(
    db: Database,
    model: PrioritizerModel,
    goals: list[Goal],
) -> int:
    """Score all unscored content against active goals.

    Returns count of new scores generated.
    Only scores content-goal pairs that don't already have a score.
    """
    ...

def invalidate_scores(db: Database, goal_id: int) -> int:
    """Delete scores for a specific goal (when goal text changes).

    Returns count of deleted scores.
    """
    ...
```

**Dependencies:** learnlens.storage, learnlens.models.prioritizer
**Side effects:** reads/writes database, runs model inference

---

### learnlens/pipeline/briefing.py

```python
def generate_daily_briefing(
    db: Database,
    connector: ConnectorModel,
    prioritizer: PrioritizerModel,
    mentor: MentorModel,
    config: LearnLensConfig,
) -> str:
    """Run full pipeline and generate daily briefing.

    Pipeline steps:
    1. Embed any new content (Model 1)
    2. Score unscored content (Model 2)
    3. Find forgotten items (Model 1 + interaction data)
    4. Load mistake patterns
    5. Assemble BriefingRequest
    6. Generate briefing markdown (Model 3)
    7. Store briefing in database

    Returns:
        Markdown-formatted daily briefing string
    """
    ...
```

**Dependencies:** all pipeline modules, all model modules
**Side effects:** full pipeline execution (DB reads/writes, model inference)

---

### learnlens/storage/database.py

```python
class Database:
    """SQLite database connection and schema management."""

    def __init__(self, path: Path) -> None:
        """Open connection, enable WAL mode, create schema if needed."""
        ...

    def close(self) -> None: ...

    def __enter__(self) -> Database: ...
    def __exit__(self, *args) -> None: ...

    @property
    def connection(self) -> sqlite3.Connection: ...
```

**Dependencies:** sqlite3, pathlib
**Side effects:** creates SQLite file, enables WAL, creates tables

---

### learnlens/storage/queries.py

```python
# Content CRUD
def insert_content(conn, url, title, body_text, source_type, word_count, domain) -> int: ...
def get_content_by_id(conn, content_id) -> ContentItem | None: ...
def get_content_by_url(conn, url) -> ContentItem | None: ...
def get_unembedded_content(conn) -> list[ContentItem]: ...
def get_all_content(conn, limit=100, offset=0) -> list[ContentItem]: ...

# Embedding CRUD
def insert_embedding(conn, content_id, vector, dim_size, model_version) -> int: ...
def get_embedding(conn, content_id) -> np.ndarray | None: ...
def get_all_embeddings(conn) -> tuple[list[int], np.ndarray]: ...

# Goal CRUD
def insert_goal(conn, goal_text, priority) -> int: ...
def get_active_goals(conn) -> list[Goal]: ...
def update_goal(conn, goal_id, goal_text, priority, is_active) -> None: ...
def delete_goal(conn, goal_id) -> None: ...

# Score CRUD
def insert_score(conn, content_id, goal_id, score, rationale, action, model_version) -> int: ...
def get_unscored_content(conn, goal_id) -> list[ContentItem]: ...
def get_top_scored(conn, top_k=10) -> list[ScoredItem]: ...
def delete_scores_for_goal(conn, goal_id) -> int: ...

# Mistake CRUD
def insert_mistake(conn, pattern, description, examples, severity) -> int: ...
def get_all_mistakes(conn) -> list[MistakePattern]: ...
def update_mistake(conn, mistake_id, pattern, description, examples, severity) -> None: ...
def delete_mistake(conn, mistake_id) -> None: ...

# Briefing CRUD
def insert_briefing(conn, content, items_referenced) -> int: ...
def get_latest_briefing(conn) -> tuple[str, datetime] | None: ...
def get_briefing_by_date(conn, date) -> str | None: ...

# Interaction CRUD
def insert_interaction(conn, content_id, action) -> None: ...
def get_items_without_interactions(conn, days_threshold=7) -> list[int]: ...
```

**All queries use parameterized statements. Never string concatenation.**

---

### learnlens/ui/*.py -- Tab Components

Each tab module exports a single `create()` function:

```python
def create(db: Database, *models, config: LearnLensConfig) -> None:
    """Build Gradio tab components within a gr.Tab context."""
    with gr.Tab("Tab Name"):
        # ... Gradio components and event handlers
```

Tab modules should:
- Define their own event handler functions
- Use `@spaces.GPU` for any function that runs model inference
- Accept only the models they actually need (not all models)
- Handle empty states gracefully (no content yet, no goals yet, etc.)

---

## 4. pyproject.toml Template

```toml
[project]
name = "learnlens"
version = "0.1.0"
description = "AI learning mentor powered by 3 small models"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.11,<3.13"
authors = [
    { name = "Sarthak Biswas" },
]

dependencies = [
    "gradio>=5.0",
    "transformers>=5.6",
    "sentence-transformers>=4.1",
    "torch>=2.7",
    "peft>=0.15",
    "pydantic>=2.12",
    "trafilatura>=2.0",
    "beautifulsoup4>=4.13",
    "numpy>=2.0",
    "umap-learn>=0.5",
    "plotly>=6.0",
]

[project.optional-dependencies]
training = [
    "trl>=0.21",
    "modal>=0.73",
    "openai>=1.0",
    "datasets>=3.0",
    "accelerate>=1.0",
    "bitsandbytes>=0.45",
]
dev = [
    "pytest>=9.0",
    "pytest-cov>=6.0",
    "ruff>=0.15",
]

[project.scripts]
learnlens = "learnlens.app:main"

[tool.uv]
dev-dependencies = [
    "pytest>=9.0",
    "pytest-cov>=6.0",
    "ruff>=0.15",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests (model loading)",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["learnlens"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 5. Environment and Secrets

### .env.example

```bash
# Required for training only (not needed at inference)
NVIDIA_NIM_API_KEY=nvapi-xxxx

# Required for pushing models to HF Hub
HF_TOKEN=hf_xxxx

# Optional: custom database path
LEARNLENS_DB_PATH=./data/learnlens.db
```

### .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.ruff_cache/

# Virtual environment
.venv/

# Data (local only)
data/
*.db
*.sqlite

# Training artifacts
training/data/
training/output/
*.gguf

# Secrets
.env

# OS
.DS_Store

# IDE
.vscode/
.idea/

# uv
.python-version
```

---

## 6. AGENTS.md Template

```markdown
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
```

---

## 7. Development Workflow

### Daily Development Cycle

```bash
# 1. Start dev
uv sync

# 2. Make changes
# ... edit files ...

# 3. Lint
uv run ruff check . --fix
uv run ruff format .

# 4. Test
uv run pytest

# 5. Run locally to verify
uv run learnlens
```

### Adding Dependencies

```bash
# Add a core dependency
uv add <package>

# Add a training dependency
uv add --optional training <package>

# Add a dev dependency
uv add --group dev <package>
```

### Pre-commit Checks

Before every commit:
1. `uv run ruff check .` -- no lint errors
2. `uv run ruff format --check .` -- formatting matches
3. `uv run pytest` -- all tests pass
4. No `.env` or secrets in staged files
