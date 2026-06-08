"""Tests for CRUD queries."""

from __future__ import annotations

import numpy as np

from learnlens.storage.database import Database
from learnlens.storage.queries import (
    delete_goal,
    get_active_goals,
    get_content_by_id,
    get_content_by_url,
    get_top_scored,
    insert_content,
    insert_embedding,
    insert_goal,
    insert_score,
)


def test_insert_and_get_content(tmp_db: Database) -> None:
    row_id = insert_content(
        tmp_db.connection,
        url="https://example.com/test",
        title="Test Article",
        body_text="This is a test.",
        source_type="url",
        word_count=10,
        domain="example.com",
    )
    assert row_id > 0

    item = get_content_by_id(tmp_db.connection, row_id)
    assert item is not None
    assert item.title == "Test Article"

    by_url = get_content_by_url(tmp_db.connection, "https://example.com/test")
    assert by_url is not None
    assert by_url.id == row_id


def test_insert_goal_and_get_active(tmp_db: Database) -> None:
    gid = insert_goal(tmp_db.connection, "learn transformers", priority=5)
    goals = get_active_goals(tmp_db.connection)
    assert len(goals) == 1
    assert goals[0].goal_text == "learn transformers"

    delete_goal(tmp_db.connection, gid)
    assert len(get_active_goals(tmp_db.connection)) == 0


def test_insert_embedding_and_retrieve(tmp_db: Database) -> None:
    cid = insert_content(
        tmp_db.connection,
        url="https://example.com/embed",
        title="Embed Test",
        body_text="content",
        source_type="url",
        word_count=2,
        domain=None,
    )
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    eid = insert_embedding(tmp_db.connection, cid, vector, 3, "test-model")
    assert eid > 0

    from learnlens.storage.queries import get_embedding

    retrieved = get_embedding(tmp_db.connection, cid)
    assert retrieved is not None
    np.testing.assert_allclose(retrieved, vector, atol=1e-6)


def test_insert_score_and_get_top(tmp_db: Database) -> None:
    cid = insert_content(
        tmp_db.connection,
        url="https://example.com/score",
        title="Score Test",
        body_text="content",
        source_type="url",
        word_count=2,
        domain=None,
    )
    gid = insert_goal(tmp_db.connection, "test goal", priority=3)
    insert_score(tmp_db.connection, cid, gid, 8.5, "very relevant", "read now", "test")

    top = get_top_scored(tmp_db.connection, top_k=10)
    assert len(top) == 1
    assert top[0].score == 8.5
