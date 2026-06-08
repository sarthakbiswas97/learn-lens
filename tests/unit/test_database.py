"""Tests for database schema and connection."""

from __future__ import annotations

from pathlib import Path

from learnlens.storage.database import Database


def test_database_creates_file(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    assert db_path.exists()
    db.close()


def test_wal_mode_enabled(tmp_db: Database) -> None:
    row = tmp_db.connection.execute("PRAGMA journal_mode").fetchone()
    assert row[0] == "wal"


def test_foreign_keys_enabled(tmp_db: Database) -> None:
    row = tmp_db.connection.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1


def test_all_tables_exist(tmp_db: Database) -> None:
    tables = tmp_db.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {t[0] for t in tables}
    expected = {"content", "embeddings", "goals", "scores", "mistakes", "briefings", "interactions"}
    assert expected.issubset(names)
