"""Generate distillation training data from Nemotron via NVIDIA NIM API."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a learning prioritization expert. Given a piece of learning content \
and a set of learning goals, assess how relevant and important this content is for the learner.

Output a JSON object with exactly these fields:
- "score": integer 1-10 (1=irrelevant, 10=critical to read immediately)
- "rationale": one sentence explaining why this score
- "action": one sentence suggesting what the learner should do

Be honest and opinionated. Score low if content is tangential. Score high only if it directly \
advances a stated goal. Consider recency, depth, and practical applicability."""

# Synthetic content-goal pairs for distillation
_CONTENT_TEMPLATES = [
    {
        "title": "Understanding Transformer Architecture",
        "summary": "A deep dive into the original 'Attention Is All You Need' paper, covering self-attention, multi-head attention, positional encoding, and the encoder-decoder structure.",
        "goals": ["learning post-training techniques", "building a RAG system"],
    },
    {
        "title": "Redis Internals: How It Handles 1M Concurrent Connections",
        "summary": "An exploration of Redis's event loop, memory management, and networking stack optimizations for high concurrency.",
        "goals": ["preparing for system design interviews", "building a RAG system"],
    },
    {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "summary": "Explains how LoRA reduces trainable parameters by injecting low-rank matrices into transformer layers, making fine-tuning feasible on consumer GPUs.",
        "goals": ["learning post-training techniques", "understanding transformers"],
    },
    {
        "title": "Twitter Thread: 10 VS Code Extensions You Need",
        "summary": "A curated list of VS Code extensions for productivity, including themes, linters, and AI assistants.",
        "goals": ["building a RAG system"],
    },
    {
        "title": "GRPO: Group Relative Policy Optimization Explained",
        "summary": "Deep dive into DeepSeek's GRPO algorithm, a reinforcement learning method for LLM reasoning that eliminates the need for a separate critic model.",
        "goals": ["learning post-training techniques"],
    },
]

_GOAL_POOL = [
    "learning post-training techniques",
    "preparing for system design interviews",
    "building a RAG system",
    "understanding transformers",
    "mastering PyTorch",
    "learning MLOps",
    "preparing for ML interviews",
    "understanding diffusion models",
]


def _generate_synthetic_pair() -> dict[str, object]:
    """Generate a random synthetic content-goal pair."""
    template = random.choice(_CONTENT_TEMPLATES)
    # Vary goals
    num_goals = random.randint(1, 3)
    goals = random.sample(_GOAL_POOL, num_goals)
    return {
        "title": template["title"],
        "summary": template["summary"],
        "goals": goals,
    }


def generate_training_example(
    client: OpenAI,
    content_title: str,
    content_summary: str,
    goals: list[str],
) -> dict | None:
    """Generate a single training example using Nemotron as teacher."""
    goals_text = "\n".join(f"- {g}" for g in goals)
    user_prompt = f"Content: {content_title}\n{content_summary}\n\nLearning Goals:\n{goals_text}"

    try:
        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-super-49b-v1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=512,
        )
    except Exception as e:
        logger.warning("NIM API call failed: %s", e)
        return None

    assistant_content = response.choices[0].message.content

    # Validate JSON structure
    try:
        parsed = json.loads(assistant_content)
        assert 1 <= parsed["score"] <= 10
        assert "rationale" in parsed
        assert "action" in parsed
    except (json.JSONDecodeError, KeyError, AssertionError):
        logger.warning("Malformed response from Nemotron: %s", assistant_content[:200])
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def generate_dataset(output_path: str, num_examples: int = 2000) -> None:
    """Generate full training dataset and write to JSONL."""
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_NIM_API_KEY"],
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    attempt = 0

    with open(output_file, "w", encoding="utf-8") as f:
        while valid_count < num_examples and attempt < num_examples * 3:
            attempt += 1
            pair = _generate_synthetic_pair()
            result = generate_training_example(
                client,
                pair["title"],
                pair["summary"],
                pair["goals"],
            )
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                valid_count += 1
                if valid_count % 100 == 0:
                    logger.info("Generated %d/%d valid examples", valid_count, num_examples)

    logger.info("Dataset complete: %d examples written to %s", valid_count, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_dataset("training/data/distillation_data.jsonl", num_examples=500)
