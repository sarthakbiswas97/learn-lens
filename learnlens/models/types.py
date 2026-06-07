"""Shared data types for LearnLens."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ContentItem:
    """Immutable content record."""

    id: int
    url: str
    title: str
    body_text: str
    source_type: str  # 'url' | 'bookmark' | 'history'
    word_count: int
    ingested_at: datetime


@dataclass(frozen=True)
class Goal:
    """Immutable learning goal record."""

    id: int
    goal_text: str
    priority: int  # 1-5
    is_active: bool


@dataclass(frozen=True)
class ScoredItem:
    """Immutable scored content record."""

    content: ContentItem
    score: float  # 1.0-10.0
    rationale: str
    suggested_action: str
    goal: Goal


@dataclass(frozen=True)
class MistakePattern:
    """Immutable mistake pattern record."""

    id: int
    pattern: str  # short label
    description: str  # full description
    examples: tuple[str, ...]
    severity: int  # 1-5


@dataclass(frozen=True)
class BriefingRequest:
    """Input to the Mentor model for daily briefing generation."""

    scored_items: tuple[ScoredItem, ...]
    forgotten_items: tuple[ContentItem, ...]
    mistakes: tuple[MistakePattern, ...]
    goals: tuple[Goal, ...]
