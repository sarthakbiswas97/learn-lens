"""Forgotten items cards."""

from __future__ import annotations

import gradio as gr

from learnlens.config import LearnLensConfig
from learnlens.models.connector import ConnectorModel
from learnlens.storage.database import Database


def create(db: Database, connector: ConnectorModel, config: LearnLensConfig) -> None:
    """Build the Forgotten Items tab."""
    with gr.Tab("Forgotten Items"):
        gr.Markdown("## Items You Are Forgetting")
        gr.Markdown("High-relevance content you haven't interacted with in 7+ days.")

        list_box = gr.JSON(label="Forgotten Items")
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(fn=lambda: _load_forgotten(db, connector, config), outputs=list_box)


def _load_forgotten(
    db: Database, connector: ConnectorModel, config: LearnLensConfig
) -> list[dict[str, str]]:
    from learnlens.models.types import Goal
    from learnlens.pipeline.embedding import find_forgotten_items
    from learnlens.storage.queries import get_active_goals

    goals = get_active_goals(db.connection)
    items = find_forgotten_items(db, connector, goals, config)
    return [{"title": item.title, "url": item.url} for item in items]
