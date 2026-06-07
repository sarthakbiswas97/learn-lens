"""Populate DB with demo content for judging."""

from __future__ import annotations

import logging

from learnlens.config import load_config
from learnlens.storage.database import Database
from learnlens.storage.queries import insert_content, insert_goal, insert_mistake
from learnlens.utils.logging import configure_logging

logger = logging.getLogger(__name__)

DEMO_CONTENT = [
    {
        "url": "https://arxiv.org/abs/1706.03762",
        "title": "Attention Is All You Need",
        "body_text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        "source_type": "url",
        "word_count": 4500,
        "domain": "arxiv.org",
    },
    {
        "url": "https://huggingface.co/blog/rlhf",
        "title": "RLHF: Reinforcement Learning from Human Feedback",
        "body_text": "RLHF is a technique that trains a reward model to capture human preferences...",
        "source_type": "url",
        "word_count": 3200,
        "domain": "huggingface.co",
    },
    {
        "url": "https://blog.google/technology/ai/google-gemini-ai/",
        "title": "Introducing Gemini",
        "body_text": "Gemini is our most capable and general model, built from the ground up to be multimodal...",
        "source_type": "url",
        "word_count": 2100,
        "domain": "blog.google",
    },
]

DEMO_GOALS = [
    ("learning post-training techniques", 5),
    ("building a RAG system", 4),
    ("understanding transformers", 5),
]

DEMO_MISTAKES = [
    ("forgot to freeze embeddings", "Did not freeze pretrained embeddings during fine-tuning, causing catastrophic forgetting.", 4),
    ("wrong learning rate", "Used lr=1e-3 for a 7B model fine-tune, causing divergence.", 5),
]


def seed() -> None:
    configure_logging()
    config = load_config()
    db = Database(config.db_path)

    logger.info("Seeding demo data")

    for item in DEMO_CONTENT:
        insert_content(db.connection, **item)

    for goal_text, priority in DEMO_GOALS:
        insert_goal(db.connection, goal_text, priority)

    for pattern, description, severity in DEMO_MISTAKES:
        insert_mistake(db.connection, pattern, description, None, severity)

    logger.info("Demo data seeded successfully")
    db.close()


if __name__ == "__main__":
    seed()
