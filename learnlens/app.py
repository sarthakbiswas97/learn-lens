"""Gradio app entry point."""

from __future__ import annotations

import logging

import gradio as gr

from learnlens.config import load_config
from learnlens.models.connector import ConnectorModel
from learnlens.models.mentor import MentorModel
from learnlens.models.prioritizer import PrioritizerModel
from learnlens.storage.database import Database
from learnlens.ui import (
    briefing_tab,
    forgotten_tab,
    knowledge_tab,
    mistakes_tab,
    priority_tab,
)
from learnlens.utils.logging import configure_logging

logger = logging.getLogger(__name__)

# Load models at module level (ZeroGPU loads before @spaces.GPU activation)
config = load_config()
connector = ConnectorModel(config)
prioritizer = PrioritizerModel(config)
mentor = MentorModel(config)


def main() -> None:
    configure_logging()
    db = Database(config.db_path)

    with gr.Blocks(title="LearnLens", theme=gr.themes.Soft()) as app:
        gr.Markdown("# LearnLens -- Your AI Learning Mentor")
        gr.Markdown(
            "*An opinionated mentor that tells you what to focus on "
            "and what to stop wasting time on.*"
        )

        with gr.Tabs():
            briefing_tab.create(db, connector, prioritizer, mentor, config)
            priority_tab.create(db, config)
            forgotten_tab.create(db, connector, config)
            knowledge_tab.create(db, connector, config)
            mistakes_tab.create(db, config)

    app.launch()


if __name__ == "__main__":
    main()
