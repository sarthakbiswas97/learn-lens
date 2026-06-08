"""Generate distillation training data for mentor (briefing generation)."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

from openai import OpenAI

from training.generate_data import _CONTENT_TEMPLATES, _GOAL_POOL, _mutate_text

logger = logging.getLogger(__name__)

_MENTOR_SYSTEM_PROMPT = """You are an opinionated learning mentor. Given a learner's context (their goals, top-priority content, forgotten items, and mistake patterns), generate a daily briefing in markdown with these exact sections:

## Today's Focus
## You're Forgetting
## Watch Out
## Connections You Missed

Be direct and blunt. Say "Stop reading about X" not "Consider deprioritizing X". Say "This is 10x more relevant" not "This has high alignment"."""

_MISTAKE_POOL = [
    ("forgot to freeze embeddings", "Did not freeze pretrained embeddings during fine-tuning, causing catastrophic forgetting.", 4),
    ("wrong learning rate", "Used lr=1e-3 for a 7B model fine-tune, causing divergence.", 5),
    ("no gradient clipping", "Training exploded because max_grad_norm was not set during multi-GPU training.", 4),
    ("evaluated on training set", "Reported metrics from the training split instead of a held-out validation set.", 5),
    ("forgot to set random seed", "Results were not reproducible because no seed was set for torch, numpy, or random.", 3),
    ("leaked test data", "Used the test set for hyperparameter tuning, invalidating final evaluation.", 5),
    ("ignored class imbalance", "Trained on imbalanced data without weighted loss or resampling, producing a biased model.", 4),
    ("no input validation", "Production API crashed on malformed JSON because input was not validated before inference.", 4),
]


def _build_synthetic_context() -> str:
    """Build a simulated briefing context from templates."""
    goals = random.sample(_GOAL_POOL, random.randint(1, 4))
    num_scored = random.randint(3, 10)
    scored_templates = random.sample(_CONTENT_TEMPLATES, num_scored)
    scored_items = []
    for tpl in scored_templates:
        score = random.randint(1, 10)
        scored_items.append({
            "title": _mutate_text(tpl["title"]),
            "score": score,
            "rationale": f"Relevance score {score} based on alignment with learner goals.",
            "action": "Read immediately" if score >= 8 else "Review when convenient" if score >= 5 else "Skip unless bored",
        })
    # Sort by score descending
    scored_items.sort(key=lambda x: x["score"], reverse=True)

    # Forgotten items: 0-3 random low-score items from a while ago
    num_forgotten = random.randint(0, 3)
    forgotten = []
    if num_forgotten > 0 and len(scored_items) > 3:
        forgotten = random.sample(scored_items[-5:], num_forgotten)

    # Mistakes: 0-2 random patterns
    num_mistakes = random.randint(0, 2)
    mistakes = random.sample(_MISTAKE_POOL, num_mistakes)

    lines: list[str] = ["# Your Learning Context", ""]

    lines.append("## Your Goals")
    for goal in goals:
        lines.append(f"- {goal}")
    lines.append("")

    lines.append("## Top Priority Content")
    for item in scored_items[:10]:
        lines.append(f"- **{item['title']}** (score: {item['score']})")
        lines.append(f"  - Rationale: {item['rationale']}")
        lines.append(f"  - Action: {item['action']}")
    lines.append("")

    if forgotten:
        lines.append("## Items You Are Forgetting")
        for item in forgotten:
            lines.append(f"- {item['title']}")
        lines.append("")

    if mistakes:
        lines.append("## Mistakes to Watch For")
        for pattern, description, severity in mistakes:
            lines.append(f"- **{pattern}** (severity {severity})")
            lines.append(f"  - {description}")
        lines.append("")

    lines.append("Generate the daily briefing based on the above context.")
    return "\n".join(lines)


def generate_mentor_example(client: OpenAI) -> dict | None:
    """Generate a single mentor training example using Nemotron Ultra as teacher."""
    context = _build_synthetic_context()

    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[
                {"role": "system", "content": _MENTOR_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.6,
            max_tokens=2048,
        )
    except Exception as e:
        logger.warning("NIM API call failed: %s", e)
        return None

    assistant_content = response.choices[0].message.content

    # Validate all 4 sections are present
    required_sections = [
        "## Today's Focus",
        "## You're Forgetting",
        "## Watch Out",
        "## Connections You Missed",
    ]
    missing = [s for s in required_sections if s not in assistant_content]
    if missing:
        logger.warning("Missing sections in mentor response: %s", missing)
        return None

    return {
        "messages": [
            {"role": "system", "content": _MENTOR_SYSTEM_PROMPT},
            {"role": "user", "content": context},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def generate_mentor_dataset(output_path: str, num_examples: int = 2000) -> None:
    """Generate full mentor training dataset and write to JSONL."""
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
            result = generate_mentor_example(client)
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                valid_count += 1
                if valid_count % 100 == 0:
                    logger.info("Generated %d/%d valid mentor examples", valid_count, num_examples)

    logger.info("Mentor dataset complete: %d examples written to %s", valid_count, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_mentor_dataset("training/data/mentor_data.jsonl", num_examples=500)
