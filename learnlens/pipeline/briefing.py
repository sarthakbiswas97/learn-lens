"""Daily briefing assembly and generation."""

from __future__ import annotations

import logging

from learnlens.config import LearnLensConfig
from learnlens.models.connector import ConnectorModel
from learnlens.models.mentor import MentorModel
from learnlens.models.prioritizer import PrioritizerModel
from learnlens.models.types import BriefingRequest
from learnlens.pipeline.embedding import embed_new_content, find_forgotten_items
from learnlens.pipeline.scoring import score_content
from learnlens.storage.database import Database
from learnlens.storage.queries import (
    get_active_goals,
    get_all_mistakes,
    get_top_scored,
    insert_briefing,
)

logger = logging.getLogger(__name__)


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
        Markdown-formatted daily briefing string.
    """
    logger.info("Starting daily briefing generation")

    # Step 1: embed new content
    try:
        embed_new_content(db, connector)
    except Exception:
        logger.exception("Embedding step failed; continuing without new embeddings")

    # Step 2: score unscored content
    try:
        goals = get_active_goals(db.connection)
        score_content(db, prioritizer, goals)
    except Exception:
        logger.exception("Scoring step failed; continuing with existing scores")
        goals = get_active_goals(db.connection)

    # Step 3: find forgotten items
    forgotten_items = []
    try:
        forgotten_items = find_forgotten_items(db, connector, goals, config)
    except Exception:
        logger.exception("Forgotten items step failed; continuing without")

    # Step 4: load mistakes
    mistakes = []
    try:
        mistakes = get_all_mistakes(db.connection)
    except Exception:
        logger.exception("Mistake loading failed; continuing without")

    # Step 5: assemble top scored items
    scored_items = []
    try:
        scored_items = get_top_scored(db.connection, top_k=config.top_k_briefing)
    except Exception:
        logger.exception("Top scored loading failed; continuing without")

    # Step 6: generate briefing
    request = BriefingRequest(
        scored_items=tuple(scored_items),
        forgotten_items=tuple(forgotten_items),
        mistakes=tuple(mistakes),
        goals=tuple(goals),
    )

    briefing = ""
    try:
        briefing = mentor.generate_briefing(request)
    except Exception:
        logger.exception("Briefing generation failed")
        # Graceful degradation: render raw scored items
        briefing = _fallback_briefing(request)

    # Step 7: store briefing
    try:
        referenced = [s.content.id for s in scored_items]
        insert_briefing(db.connection, briefing, referenced)
    except Exception:
        logger.exception("Failed to store briefing")

    logger.info("Daily briefing generation complete (%d chars)", len(briefing))
    return briefing


def _fallback_briefing(request: BriefingRequest) -> str:
    """Render a simple markdown list when the mentor model fails."""
    lines = ["# Daily Briefing (Fallback)", ""]

    lines.append("## Today's Focus")
    for item in request.scored_items:
        lines.append(f"- **{item.content.title}** ({item.score}/10)")
        lines.append(f"  - {item.rationale}")
    lines.append("")

    if request.forgotten_items:
        lines.append("## You're Forgetting")
        for item in request.forgotten_items:
            lines.append(f"- {item.title}")
        lines.append("")

    if request.mistakes:
        lines.append("## Watch Out")
        for m in request.mistakes:
            lines.append(f"- {m.pattern}: {m.description}")
        lines.append("")

    return "\n".join(lines)
