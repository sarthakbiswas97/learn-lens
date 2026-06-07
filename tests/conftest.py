"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from learnlens.models.types import ContentItem, Goal
from learnlens.storage.database import Database


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    """Temporary SQLite database for testing."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    yield db
    db.close()


@pytest.fixture
def sample_content() -> list[ContentItem]:
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
        ContentItem(
            id=2,
            url="https://example.com/article2",
            title="Redis Internals Explained",
            body_text="Redis uses an event loop...",
            source_type="url",
            word_count=800,
            ingested_at=datetime.now(),
        ),
    ]


@pytest.fixture
def sample_goal() -> Goal:
    """Sample learning goal for testing."""
    return Goal(
        id=1,
        goal_text="learning post-training techniques",
        priority=5,
        is_active=True,
    )


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Pre-computed random embeddings for testing similarity."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 256)).astype(np.float32)
