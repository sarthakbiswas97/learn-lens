"""Pydantic settings and configuration loading."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/default.json")


class LearnLensConfig(BaseModel):
    """Application configuration."""

    # Storage
    db_path: Path = Field(default=Path("data/learnlens.db"))

    # Model IDs
    embedding_model_id: str = "nomic-ai/nomic-embed-text-v1.5"
    prioritizer_adapter_id: str = "sarthakbiswas/learnlens-scorer-sft-v1"
    prioritizer_base_model_id: str = "openbmb/MiniCPM5-1B"
    mentor_adapter_id: str = "sarthakbiswas/learnlens-mentor-sft-v1"

    # Embedding
    embedding_dim: int = Field(default=256, description="Matryoshka dim (768, 512, 256, 128, 64)")

    # Pipeline
    max_content_length: int = Field(default=4096, description="Max chars for embedding input")
    top_k_briefing: int = Field(default=10, description="Top scored items for briefing")
    forgotten_days_threshold: int = Field(default=7, description="Days before item is 'forgotten'")
    forgotten_similarity_threshold: float = Field(
        default=0.6, description="Min cosine sim to goals"
    )

    # Generation
    mentor_temperature: float = 0.6
    mentor_top_p: float = 0.95
    mentor_max_tokens: int = 2048
    prioritizer_temperature: float = 0.7
    prioritizer_max_tokens: int = 256


def load_config(config_path: Path | None = None) -> LearnLensConfig:
    """Load config from JSON file, with environment variable overrides."""
    path = config_path or DEFAULT_CONFIG_PATH
    data: dict[str, object] = {}

    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Loaded config from %s", path)

    # Environment overrides
    if db_path_env := os.getenv("LEARNLENS_DB_PATH"):
        data["db_path"] = Path(db_path_env)

    return LearnLensConfig(**data)
