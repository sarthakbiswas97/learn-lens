"""Gradio app entry point."""

from __future__ import annotations

import logging

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Conditionally import spaces for HF Spaces ZeroGPU
try:
    import spaces
except ImportError:
    spaces = None  # type: ignore[assignment]

    class _DummySpaces:  # noqa: D401
        """No-op decorator when spaces is not installed (local dev)."""

        def GPU(self, **kwargs):  # noqa: N802
            def decorator(func):
                return func

            return decorator

    spaces = _DummySpaces()

# Load config
config = load_config()

# Lazy model loading state
_connector: ConnectorModel | None = None
_prioritizer: PrioritizerModel | None = None
_mentor: MentorModel | None = None


def _get_connector() -> ConnectorModel:
    """Lazy-load embedding model."""
    global _connector
    if _connector is None:
        _connector = ConnectorModel(config)
        _connector.load()
    return _connector


def _get_llm_models() -> tuple[PrioritizerModel, MentorModel]:
    """Lazy-load shared MiniCPM5-1B base + dual LoRA adapters."""
    global _prioritizer, _mentor
    if _prioritizer is None or _mentor is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.bfloat16 if device == "cuda" else torch.float16
        logger.info("Loading MiniCPM5-1B base to %s (%s)", device, dtype)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.prioritizer_base_model_id,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            config.prioritizer_base_model_id,
            trust_remote_code=True,
        )
        _prioritizer = PrioritizerModel(config, base_model=base_model, tokenizer=tokenizer)
        _mentor = MentorModel(config, base_model=base_model, tokenizer=tokenizer)
        _prioritizer.load()
        _mentor.load()
    return _prioritizer, _mentor


def main() -> None:
    configure_logging()
    db = Database(config.db_path)

    # Eager load models locally (HF Spaces loads at module level via ZeroGPU)
    connector = _get_connector()
    prioritizer, mentor = _get_llm_models()

    with gr.Blocks(title="LearnLens") as app:
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
