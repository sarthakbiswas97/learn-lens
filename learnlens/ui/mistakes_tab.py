"""Mistake tracker form + list."""

from __future__ import annotations

import gradio as gr

from learnlens.config import LearnLensConfig
from learnlens.storage.database import Database


def create(db: Database, config: LearnLensConfig) -> None:
    """Build the Mistake Tracker tab."""
    with gr.Tab("Mistake Tracker"):
        gr.Markdown("## Mistake Tracker")
        gr.Markdown("Track patterns you want to avoid repeating.")

        with gr.Row():
            pattern_input = gr.Textbox(
                label="Pattern", placeholder="e.g., forgot to freeze embeddings"
            )
            desc_input = gr.Textbox(label="Description", placeholder="Detailed description...")
            severity_input = gr.Slider(1, 5, value=3, step=1, label="Severity")

        add_btn = gr.Button("Add Mistake")
        list_box = gr.JSON(label="Mistakes")

        add_btn.click(
            fn=lambda p, d, s: _add_mistake(db, p, d, s),
            inputs=[pattern_input, desc_input, severity_input],
            outputs=list_box,
        )

        refresh_btn = gr.Button("Refresh List")
        refresh_btn.click(fn=lambda: _load_mistakes(db), outputs=list_box)


def _add_mistake(
    db: Database, pattern: str, description: str, severity: int
) -> list[dict[str, str]]:
    from learnlens.storage.queries import insert_mistake

    insert_mistake(db.connection, pattern, description, None, severity)
    return _load_mistakes(db)


def _load_mistakes(db: Database) -> list[dict[str, str]]:
    from learnlens.storage.queries import get_all_mistakes

    mistakes = get_all_mistakes(db.connection)
    return [
        {
            "pattern": m.pattern,
            "description": m.description,
            "severity": m.severity,
        }
        for m in mistakes
    ]
