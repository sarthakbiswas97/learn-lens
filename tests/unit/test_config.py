"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from learnlens.config import LearnLensConfig, load_config


def test_default_config() -> None:
    config = LearnLensConfig()
    assert config.embedding_dim == 256
    assert config.db_path == Path("data/learnlens.db")


def test_load_from_file(tmp_path: Path) -> None:
    config_path = tmp_path / "test.json"
    config_path.write_text('{"embedding_dim": 128, "top_k_briefing": 5}')
    config = load_config(config_path)
    assert config.embedding_dim == 128
    assert config.top_k_briefing == 5
