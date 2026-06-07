"""Tests for data type immutability."""

from __future__ import annotations

from datetime import datetime

import pytest

from learnlens.models.types import ContentItem, Goal


def test_content_item_is_frozen() -> None:
    item = ContentItem(
        id=1,
        url="https://example.com",
        title="Test",
        body_text="text",
        source_type="url",
        word_count=10,
        ingested_at=datetime.now(),
    )
    with pytest.raises(AttributeError):
        item.title = "New Title"


def test_goal_is_frozen() -> None:
    goal = Goal(id=1, goal_text="learn", priority=3, is_active=True)
    with pytest.raises(AttributeError):
        goal.priority = 5
