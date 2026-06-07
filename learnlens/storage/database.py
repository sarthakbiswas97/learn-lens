"""SQLite database connection and schema management."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
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

CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_content ON embeddings(content_id);
CREATE INDEX IF NOT EXISTS idx_scores_content_goal ON scores(content_id, goal_id);
CREATE INDEX IF NOT EXISTS idx_scores_score ON scores(score DESC);
CREATE INDEX IF NOT EXISTS idx_content_ingested ON content(ingested_at);
CREATE INDEX IF NOT EXISTS idx_interactions_content ON interactions(content_id);
CREATE INDEX IF NOT EXISTS idx_goals_active ON goals(is_active);
"""


class Database:
    """SQLite database connection and schema management."""

    def __init__(self, path: Path) -> None:
        """Open connection, enable WAL mode, create schema if needed."""
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        logger.info("Database initialized at %s", path)

    def _create_schema(self) -> None:
        """Execute schema DDL."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("Database connection closed")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the underlying SQLite connection."""
        return self._conn
