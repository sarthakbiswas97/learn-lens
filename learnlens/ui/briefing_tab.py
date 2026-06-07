"""Daily briefing display + generate."""

from __future__ import annotations

import gradio as gr

from learnlens.config import LearnLensConfig
from learnlens.models.connector import ConnectorModel
from learnlens.models.mentor import MentorModel
from learnlens.models.prioritizer import PrioritizerModel
from learnlens.storage.database import Database


def create(
    db: Database,
    connector: ConnectorModel,
    prioritizer: PrioritizerModel,
    mentor: MentorModel,
    config: LearnLensConfig,
) -> None:
    """Build the Daily Briefing tab."""
    with gr.Tab("Daily Briefing"):
        gr.Markdown("## Your Daily Briefing")
        briefing_output = gr.Markdown(value="Click 'Generate' to create your daily briefing.")

        with gr.Row():
            generate_btn = gr.Button("Generate Briefing", variant="primary")

        generate_btn.click(
            fn=lambda: _generate(db, connector, prioritizer, mentor, config),
            outputs=briefing_output,
        )


def _generate(
    db: Database,
    connector: ConnectorModel,
    prioritizer: PrioritizerModel,
    mentor: MentorModel,
    config: LearnLensConfig,
) -> str:
    from learnlens.pipeline.briefing import generate_daily_briefing

    return generate_daily_briefing(db, connector, prioritizer, mentor, config)
