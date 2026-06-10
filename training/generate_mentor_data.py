"""Generate distillation training data for mentor (briefing generation)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

from openai import OpenAI

from training.generate_data import _CONTENT_TEMPLATES, _GOAL_POOL, _mutate_text
from training.utils import (
    compute_sha256,
    estimate_nim_cost,
    save_data_manifest,
    set_all_seeds,
)

logger = logging.getLogger(__name__)

_MENTOR_SYSTEM_PROMPT = """You are an opinionated learning mentor. Given a learner's context, generate a daily briefing in markdown.

You MUST include ALL of these sections, in this order:

## Today's Focus
(List 3-5 top priority items with scores and specific actions)

## You're Forgetting
(List relevant items the user bookmarked but never opened)

## Watch Out
(List mistake patterns to avoid, tied to today's focus)

## Connections You Missed
(Point out surprising links between recent content items)

CRITICAL: Include all 4 section headers exactly as shown above. Do not skip any section. Even if a section has limited content, include the header and write something brief.

Be direct and blunt. Say "Stop reading about X" not "Consider deprioritizing X". Say "This is 10x more relevant" not "This has high alignment"."""

_MISTAKE_POOL = [
    (
        "forgot to freeze embeddings",
        "Did not freeze pretrained embeddings during fine-tuning, causing catastrophic forgetting.",
        4,
    ),
    ("wrong learning rate", "Used lr=1e-3 for a 7B model fine-tune, causing divergence.", 5),
    (
        "no gradient clipping",
        "Training exploded because max_grad_norm was not set during multi-GPU training.",
        4,
    ),
    (
        "evaluated on training set",
        "Reported metrics from the training split instead of a held-out validation set.",
        5,
    ),
    (
        "forgot to set random seed",
        "Results were not reproducible because no seed was set for torch, numpy, or random.",
        3,
    ),
    (
        "leaked test data",
        "Used the test set for hyperparameter tuning, invalidating final evaluation.",
        5,
    ),
    (
        "ignored class imbalance",
        "Trained on imbalanced data without weighted loss or resampling, producing a biased model.",
        4,
    ),
    (
        "no input validation",
        "Production API crashed on malformed JSON because input was not validated before inference.",
        4,
    ),
]


def _build_synthetic_context(rng: random.Random) -> str:
    """Build a simulated briefing context from templates."""
    goals = rng.sample(_GOAL_POOL, rng.randint(1, 4))
    num_scored = rng.randint(3, 10)
    scored_templates = rng.sample(_CONTENT_TEMPLATES, num_scored)
    scored_items = []
    for tpl in scored_templates:
        score = rng.randint(1, 10)
        scored_items.append(
            {
                "title": _mutate_text(tpl["title"], rng),
                "score": score,
                "rationale": f"Relevance score {score} based on alignment with learner goals.",
                "action": "Read immediately"
                if score >= 8
                else "Review when convenient"
                if score >= 5
                else "Skip unless bored",
            }
        )
    # Sort by score descending
    scored_items.sort(key=lambda x: x["score"], reverse=True)

    # Forgotten items: 0-3 random low-score items from a while ago
    num_forgotten = rng.randint(0, 3)
    forgotten = []
    if num_forgotten > 0 and len(scored_items) > 3:
        forgotten = rng.sample(scored_items[-5:], num_forgotten)

    # Mistakes: 0-2 random patterns
    num_mistakes = rng.randint(0, 2)
    mistakes = rng.sample(_MISTAKE_POOL, num_mistakes)

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


def generate_mentor_example(client: OpenAI, rng: random.Random) -> dict | None:
    """Generate a single mentor training example using Nemotron Ultra as teacher."""
    context = _build_synthetic_context(rng)

    try:
        response = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
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

    # Validate all 4 sections are present (case-insensitive, flexible matching)
    required_sections = [
        "today's focus",
        "you're forgetting",
        "watch out",
        "connections you missed",
    ]
    content_lower = assistant_content.lower()
    missing = [s for s in required_sections if s not in content_lower]
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


def generate_mentor_dataset(
    output_path: Path,
    num_examples: int = 500,
    eval_holdout: int = 50,
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    """Generate full mentor training dataset and write to JSONL."""
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_NIM_API_KEY"],
        timeout=60,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    eval_file = output_file.parent / "eval_holdout_mentor.jsonl"

    rng = random.Random(seed)

    target_total = 3 if dry_run else num_examples
    target_train = max(0, target_total - eval_holdout)

    if dry_run:
        print(f"\n[DRY RUN] Will generate {target_total} example(s) (no files written).")
    else:
        print(
            f"\nGenerating {target_total} examples ({target_train} train + {eval_holdout} eval holdout)"
        )

    if not dry_run and not estimate_nim_cost(target_total):
        print("Aborted by user.")
        return

    valid_count = 0
    attempt = 0
    max_attempts = target_total * 3

    train_f = open(output_file, "w", encoding="utf-8") if not dry_run else None  # noqa: SIM115
    eval_f = open(eval_file, "w", encoding="utf-8") if not dry_run else None  # noqa: SIM115

    try:
        while valid_count < target_total and attempt < max_attempts:
            attempt += 1
            result = generate_mentor_example(client, rng)
            if result is not None:
                if valid_count < eval_holdout and not dry_run:
                    eval_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    eval_f.flush()
                elif not dry_run:
                    train_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    train_f.flush()
                valid_count += 1
                if valid_count % 50 == 0 or dry_run:
                    logger.info("Generated %d/%d valid mentor examples", valid_count, target_total)
    finally:
        if train_f:
            train_f.close()
        if eval_f:
            eval_f.close()

    if dry_run:
        print(f"\nDry-run complete: {valid_count} valid examples generated.")
        print(f"Cost would be: ~${valid_count * 0.005:.2f}")
        print("Exiting without writing files.")
        return

    logger.info(
        "Mentor dataset complete: %d train + %d eval written to %s",
        target_train,
        eval_holdout,
        output_file.parent,
    )

    data_hash = compute_sha256(output_file)
    save_data_manifest(
        output_path=output_file,
        script="generate_mentor_data.py",
        teacher_model="nvidia/llama-3.3-nemotron-super-49b-v1",
        total_attempts=attempt,
        valid_examples=valid_count,
        output_file=output_file,
        data_hash=data_hash,
        estimated_cost=f"${valid_count * 0.005:.2f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mentor distillation data via Nemotron Ultra"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=500,
        help="Total examples to generate (including eval holdout)",
    )
    parser.add_argument(
        "--eval-holdout", type=int, default=50, help="Number of examples to reserve for eval"
    )
    parser.add_argument(
        "--output", type=str, default="training/data/mentor_data.jsonl", help="Output JSONL path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run", action="store_true", help="Process 3 examples, print cost estimate, exit"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    set_all_seeds(args.seed)

    generate_mentor_dataset(
        output_path=Path(args.output),
        num_examples=args.num_examples,
        eval_holdout=args.eval_holdout,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
