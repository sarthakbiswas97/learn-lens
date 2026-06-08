"""Integration test of full pipeline with mocked LLMs."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from learnlens.config import LearnLensConfig
from learnlens.models.connector import ConnectorModel
from learnlens.models.mentor import MentorModel
from learnlens.models.prioritizer import PrioritizerModel
from learnlens.models.types import BriefingRequest, ContentItem, Goal, ScoredItem
from learnlens.pipeline.briefing import generate_daily_briefing
from learnlens.storage.database import Database
from learnlens.storage.queries import (
    get_active_goals,
    get_top_scored,
    insert_content,
    insert_goal,
    insert_mistake,
)


class MockPrioritizer(PrioritizerModel):
    """Prioritizer that returns deterministic scores without loading a model."""

    def __init__(self, config: LearnLensConfig, **_kwargs: object) -> None:
        self._config = config

    def load(self) -> None:
        pass

    def score(self, content: ContentItem, goals: list[Goal]) -> ScoredItem:
        primary = max(goals, key=lambda g: g.priority) if goals else Goal(
            id=0, goal_text="general", priority=3, is_active=True
        )
        return ScoredItem(
            content=content,
            score=7.5,
            rationale="Mock rationale: relevant to goal",
            suggested_action="Mock action: read this",
            goal=primary,
        )


class MockMentor(MentorModel):
    """Mentor that returns a simple markdown briefing without loading a model."""

    def __init__(self, config: LearnLensConfig, **_kwargs: object) -> None:
        self._config = config

    def load(self) -> None:
        pass

    def generate_briefing(self, request: BriefingRequest) -> str:
        lines = ["# Daily Briefing (Mock)", ""]
        lines.append("## Today's Focus")
        for item in request.scored_items:
            lines.append(f"- {item.content.title}: {item.score}")
        return "\n".join(lines)


@pytest.fixture
def mock_config(tmp_path) -> LearnLensConfig:
    return LearnLensConfig(db_path=tmp_path / "test_pipeline.db")


@pytest.fixture
def seeded_db(mock_config: LearnLensConfig) -> Database:
    db = Database(mock_config.db_path)
    # Seed content
    insert_content(
        db.connection,
        "https://example.com/lora",
        "LoRA Fine-Tuning Guide",
        "LoRA reduces trainable parameters by injecting low-rank matrices...",
        "url",
        1500,
        "example.com",
    )
    insert_content(
        db.connection,
        "https://example.com/transformers",
        "Understanding Transformers",
        "The transformer architecture uses self-attention mechanisms...",
        "url",
        2000,
        "example.com",
    )
    # Seed goals
    insert_goal(db.connection, "learning post-training techniques", 5)
    insert_goal(db.connection, "understanding transformers", 4)
    # Seed mistakes
    insert_mistake(
        db.connection,
        "forgot to freeze embeddings",
        "Did not freeze pretrained embeddings during fine-tuning.",
        None,
        4,
    )
    yield db
    db.close()


@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline_mocked(seeded_db: Database, mock_config: LearnLensConfig) -> None:
    """Run full pipeline: embed -> score -> brief with mocked LLMs."""
    connector = ConnectorModel(mock_config)
    connector.load()

    prioritizer = MockPrioritizer(mock_config)
    mentor = MockMentor(mock_config)

    briefing = generate_daily_briefing(
        seeded_db, connector, prioritizer, mentor, mock_config
    )

    # Verify briefing was generated and stored
    assert "Daily Briefing" in briefing
    assert "LoRA" in briefing or "Transformers" in briefing

    # Verify scores were created (2 items x 2 goals = 4 scores)
    scores = get_top_scored(seeded_db.connection, top_k=10)
    assert len(scores) == 4
    assert scores[0].score == 7.5
    assert scores[0].rationale == "Mock rationale: relevant to goal"

    # Verify goals exist
    goals = get_active_goals(seeded_db.connection)
    assert len(goals) == 2


@pytest.mark.integration
def test_find_similar_with_real_embeddings(seeded_db: Database, mock_config: LearnLensConfig) -> None:
    """Test similarity search with real embeddings."""
    from learnlens.pipeline.embedding import embed_new_content, find_similar

    connector = ConnectorModel(mock_config)
    connector.load()

    count = embed_new_content(seeded_db, connector)
    assert count == 2

    # Query for transformer-related content
    query_vec = connector.embed_query("transformer architecture")
    results = find_similar(query_vec, seeded_db, top_k=2)

    assert len(results) == 2
    # The transformer article should be most similar
    titles = [r[0].title for r in results]
    assert "Understanding Transformers" in titles
