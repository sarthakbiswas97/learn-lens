"""Priority scoring against user goals."""

from __future__ import annotations

import logging

from learnlens.models.prioritizer import PrioritizerModel
from learnlens.models.types import Goal
from learnlens.storage.database import Database
from learnlens.storage.queries import (
    delete_scores_for_goal,
    get_active_goals,
    get_unscored_content,
    insert_score,
)

logger = logging.getLogger(__name__)


def score_content(
    db: Database,
    model: PrioritizerModel,
    goals: list[Goal] | None = None,
) -> int:
    """Score all unscored content against active goals.

    Returns count of new scores generated.
    Only scores content-goal pairs that don't already have a score.
    """
    if goals is None:
        goals = get_active_goals(db.connection)

    if not goals:
        logger.warning("No active goals; skipping scoring")
        return 0

    total_new = 0
    for goal in goals:
        items = get_unscored_content(db.connection, goal.id)
        if not items:
            continue

        logger.info("Scoring %d items against goal '%s'", len(items), goal.goal_text)
        for item in items:
            try:
                scored = model.score(item, [goal])
                insert_score(
                    conn=db.connection,
                    content_id=item.id,
                    goal_id=goal.id,
                    score=scored.score,
                    rationale=scored.rationale,
                    suggested_action=scored.suggested_action,
                    model_version="minicpm5-1b-lora",
                )
                total_new += 1
            except Exception:
                logger.exception("Failed to score content %d against goal %d", item.id, goal.id)

    logger.info("Generated %d new scores", total_new)
    return total_new


def invalidate_scores(db: Database, goal_id: int) -> int:
    """Delete scores for a specific goal (when goal text changes).

    Returns count of deleted scores.
    """
    count = delete_scores_for_goal(db.connection, goal_id)
    logger.info("Invalidated %d scores for goal %d", count, goal_id)
    return count
