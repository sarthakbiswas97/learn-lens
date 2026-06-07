"""Priority queue table + filters."""

from __future__ import annotations

import gradio as gr

from learnlens.config import LearnLensConfig
from learnlens.storage.database import Database


def create(db: Database, config: LearnLensConfig) -> None:
    """Build the Priority Queue tab."""
    with gr.Tab("Priority Queue"):
        gr.Markdown("## Priority Queue")
        gr.Markdown("Scored content sorted by relevance to your goals.")

        table = gr.Dataframe(
            headers=["Title", "Score", "Goal", "Rationale", "Action", "Added"],
            label="Scored Items",
        )
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(fn=lambda: _load_scores(db), outputs=table)


def _load_scores(db: Database) -> list[list[str]]:
    from learnlens.storage.queries import get_top_scored

    items = get_top_scored(db.connection, top_k=50)
    return [
        [
            s.content.title,
            str(s.score),
            s.goal.goal_text,
            s.rationale,
            s.suggested_action,
            str(s.content.ingested_at),
        ]
        for s in items
    ]
