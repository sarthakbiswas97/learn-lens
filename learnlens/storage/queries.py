"""CRUD operations for all database tables."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime

import numpy as np

from learnlens.models.types import ContentItem, Goal, MistakePattern, ScoredItem

logger = logging.getLogger(__name__)


def _row_to_content(row: sqlite3.Row) -> ContentItem:
    return ContentItem(
        id=row["id"],
        url=row["url"],
        title=row["title"],
        body_text=row["body_text"],
        source_type=row["source_type"],
        word_count=row["word_count"] or 0,
        ingested_at=datetime.fromisoformat(row["ingested_at"]),
    )


def _row_to_goal(row: sqlite3.Row) -> Goal:
    return Goal(
        id=row["id"],
        goal_text=row["goal_text"],
        priority=row["priority"],
        is_active=bool(row["is_active"]),
    )


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def insert_content(
    conn: sqlite3.Connection,
    url: str,
    title: str,
    body_text: str,
    source_type: str,
    word_count: int | None,
    domain: str | None,
) -> int:
    """Insert a content item. Returns the row id (new or existing)."""
    conn.execute(
        """
        INSERT INTO content (url, title, body_text, source_type, word_count, domain)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            title=excluded.title,
            body_text=excluded.body_text,
            word_count=excluded.word_count,
            domain=excluded.domain
        """,
        (url, title, body_text, source_type, word_count, domain),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM content WHERE url = ?", (url,)).fetchone()
    return row["id"] if row else 0


def get_content_by_id(conn: sqlite3.Connection, content_id: int) -> ContentItem | None:
    row = conn.execute(
        "SELECT * FROM content WHERE id = ?", (content_id,)
    ).fetchone()
    return _row_to_content(row) if row else None


def get_content_by_url(conn: sqlite3.Connection, url: str) -> ContentItem | None:
    row = conn.execute(
        "SELECT * FROM content WHERE url = ?", (url,)
    ).fetchone()
    return _row_to_content(row) if row else None


def get_unembedded_content(conn: sqlite3.Connection) -> list[ContentItem]:
    rows = conn.execute(
        """
        SELECT c.* FROM content c
        WHERE c.id NOT IN (SELECT content_id FROM embeddings)
        """
    ).fetchall()
    return [_row_to_content(row) for row in rows]


def get_all_content(
    conn: sqlite3.Connection, limit: int = 100, offset: int = 0
) -> list[ContentItem]:
    rows = conn.execute(
        "SELECT * FROM content ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [_row_to_content(row) for row in rows]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def insert_embedding(
    conn: sqlite3.Connection,
    content_id: int,
    vector: np.ndarray,
    dim_size: int,
    model_version: str,
) -> int:
    """Store a numpy embedding as a BLOB."""
    blob = vector.astype(np.float32).tobytes()
    cursor = conn.execute(
        "INSERT INTO embeddings (content_id, vector, dim_size, model_version) VALUES (?, ?, ?, ?)",
        (content_id, blob, dim_size, model_version),
    )
    conn.commit()
    return cursor.lastrowid


def get_embedding(conn: sqlite3.Connection, content_id: int) -> np.ndarray | None:
    row = conn.execute(
        "SELECT vector, dim_size FROM embeddings WHERE content_id = ?", (content_id,)
    ).fetchone()
    if row is None:
        return None
    return np.frombuffer(row["vector"], dtype=np.float32).reshape(row["dim_size"])


def get_all_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    rows = conn.execute(
        "SELECT content_id, vector, dim_size FROM embeddings ORDER BY content_id"
    ).fetchall()
    if not rows:
        return [], np.array([])
    content_ids = [row["content_id"] for row in rows]
    dim_size = rows[0]["dim_size"]
    matrix = np.stack([
        np.frombuffer(row["vector"], dtype=np.float32).reshape(dim_size)
        for row in rows
    ])
    return content_ids, matrix


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------

def insert_goal(conn: sqlite3.Connection, goal_text: str, priority: int = 3) -> int:
    cursor = conn.execute(
        "INSERT INTO goals (goal_text, priority) VALUES (?, ?)",
        (goal_text, priority),
    )
    conn.commit()
    return cursor.lastrowid


def get_active_goals(conn: sqlite3.Connection) -> list[Goal]:
    rows = conn.execute(
        "SELECT * FROM goals WHERE is_active = 1 ORDER BY priority DESC, created_at"
    ).fetchall()
    return [_row_to_goal(row) for row in rows]


def update_goal(
    conn: sqlite3.Connection,
    goal_id: int,
    goal_text: str,
    priority: int,
    is_active: bool,
) -> None:
    conn.execute(
        "UPDATE goals SET goal_text = ?, priority = ?, is_active = ? WHERE id = ?",
        (goal_text, priority, int(is_active), goal_id),
    )
    conn.commit()


def delete_goal(conn: sqlite3.Connection, goal_id: int) -> None:
    conn.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def insert_score(
    conn: sqlite3.Connection,
    content_id: int,
    goal_id: int,
    score: float,
    rationale: str,
    suggested_action: str | None,
    model_version: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO scores (content_id, goal_id, score, rationale, suggested_action, model_version)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (content_id, goal_id, score, rationale, suggested_action, model_version),
    )
    conn.commit()
    return cursor.lastrowid


def get_unscored_content(conn: sqlite3.Connection, goal_id: int) -> list[ContentItem]:
    rows = conn.execute(
        """
        SELECT c.* FROM content c
        WHERE c.id NOT IN (
            SELECT content_id FROM scores WHERE goal_id = ?
        )
        """,
        (goal_id,),
    ).fetchall()
    return [_row_to_content(row) for row in rows]


def get_top_scored(conn: sqlite3.Connection, top_k: int = 10) -> list[ScoredItem]:
    rows = conn.execute(
        """
        SELECT
            s.score AS s_score,
            s.rationale AS s_rationale,
            s.suggested_action AS s_suggested_action,
            c.id AS c_id,
            c.url AS c_url,
            c.title AS c_title,
            c.body_text AS c_body_text,
            c.source_type AS c_source_type,
            c.word_count AS c_word_count,
            c.ingested_at AS c_ingested_at,
            g.id AS g_id,
            g.goal_text AS g_goal_text,
            g.priority AS g_priority,
            g.is_active AS g_is_active
        FROM scores s
        JOIN content c ON s.content_id = c.id
        JOIN goals g ON s.goal_id = g.id
        ORDER BY s.score DESC
        LIMIT ?
        """,
        (top_k,),
    ).fetchall()
    scored_items = []
    for row in rows:
        content = ContentItem(
            id=row["c_id"],
            url=row["c_url"],
            title=row["c_title"],
            body_text=row["c_body_text"],
            source_type=row["c_source_type"],
            word_count=row["c_word_count"] or 0,
            ingested_at=datetime.fromisoformat(row["c_ingested_at"]),
        )
        goal = Goal(
            id=row["g_id"],
            goal_text=row["g_goal_text"],
            priority=row["g_priority"],
            is_active=bool(row["g_is_active"]),
        )
        scored_items.append(
            ScoredItem(
                content=content,
                score=row["s_score"],
                rationale=row["s_rationale"],
                suggested_action=row["s_suggested_action"] or "",
                goal=goal,
            )
        )
    return scored_items


def delete_scores_for_goal(conn: sqlite3.Connection, goal_id: int) -> int:
    cursor = conn.execute("DELETE FROM scores WHERE goal_id = ?", (goal_id,))
    conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Mistakes
# ---------------------------------------------------------------------------

def insert_mistake(
    conn: sqlite3.Connection,
    pattern: str,
    description: str,
    examples: tuple[str, ...] | None,
    severity: int = 3,
) -> int:
    examples_json = json.dumps(list(examples)) if examples else None
    cursor = conn.execute(
        "INSERT INTO mistakes (pattern, description, examples, severity) VALUES (?, ?, ?, ?)",
        (pattern, description, examples_json, severity),
    )
    conn.commit()
    return cursor.lastrowid


def get_all_mistakes(conn: sqlite3.Connection) -> list[MistakePattern]:
    rows = conn.execute("SELECT * FROM mistakes ORDER BY severity DESC, created_at").fetchall()
    mistakes = []
    for row in rows:
        examples = tuple(json.loads(row["examples"])) if row["examples"] else ()
        mistakes.append(
            MistakePattern(
                id=row["id"],
                pattern=row["pattern"],
                description=row["description"],
                examples=examples,
                severity=row["severity"],
            )
        )
    return mistakes


def update_mistake(
    conn: sqlite3.Connection,
    mistake_id: int,
    pattern: str,
    description: str,
    examples: tuple[str, ...] | None,
    severity: int,
) -> None:
    examples_json = json.dumps(list(examples)) if examples else None
    conn.execute(
        "UPDATE mistakes SET pattern = ?, description = ?, examples = ?, severity = ? WHERE id = ?",
        (pattern, description, examples_json, severity, mistake_id),
    )
    conn.commit()


def delete_mistake(conn: sqlite3.Connection, mistake_id: int) -> None:
    conn.execute("DELETE FROM mistakes WHERE id = ?", (mistake_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Briefings
# ---------------------------------------------------------------------------

def insert_briefing(
    conn: sqlite3.Connection, content: str, items_referenced: list[int]
) -> int:
    cursor = conn.execute(
        "INSERT INTO briefings (content, items_referenced) VALUES (?, ?)",
        (content, json.dumps(items_referenced)),
    )
    conn.commit()
    return cursor.lastrowid


def get_latest_briefing(conn: sqlite3.Connection) -> tuple[str, datetime] | None:
    row = conn.execute(
        "SELECT content, generated_at FROM briefings ORDER BY generated_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return row["content"], datetime.fromisoformat(row["generated_at"])


def get_briefing_by_date(conn: sqlite3.Connection, date: datetime) -> str | None:
    row = conn.execute(
        "SELECT content FROM briefings WHERE date(generated_at) = date(?) LIMIT 1",
        (date.isoformat(),),
    ).fetchone()
    return row["content"] if row else None


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

def insert_interaction(
    conn: sqlite3.Connection, content_id: int, action: str
) -> None:
    conn.execute(
        "INSERT INTO interactions (content_id, action) VALUES (?, ?)",
        (content_id, action),
    )
    conn.commit()


def get_items_without_interactions(
    conn: sqlite3.Connection, days_threshold: int = 7
) -> list[int]:
    rows = conn.execute(
        """
        SELECT c.id FROM content c
        WHERE c.ingested_at <= datetime('now', '-' || ? || ' days')
        AND c.id NOT IN (SELECT content_id FROM interactions)
        """,
        (days_threshold,),
    ).fetchall()
    return [row["id"] for row in rows]
