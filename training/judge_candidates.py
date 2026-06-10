"""Judge Best-of-N candidates using Nemotron 49B as judge."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from openai import OpenAI


def _set_all_seeds(seed: int = 42) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    import json

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _save_jsonl(path: Path, records: list[dict]) -> None:
    import json

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _estimate_nim_cost(num_calls: int) -> bool:
    cost = num_calls * 0.005
    print(f"\nEstimated NIM API cost: ${cost:.2f} for {num_calls} judge calls")
    return True


logger = logging.getLogger(__name__)

JUDGE_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

SCORER_JUDGE_PROMPT = """You are an expert judge evaluating AI-generated learning prioritization responses.

Given a prompt and 4 candidate responses, pick the BEST one (respond with just the number 0, 1, 2, or 3).

Evaluate based on:
1. JSON validity and correct structure (score, rationale, action)
2. Score reasonableness (1-10, appropriate for the content and goals)
3. Rationale quality (specific, one sentence, explains the score)
4. Action quality (actionable, one sentence, suggests what to do)

Prompt:
{prompt}

Candidate 0:
{candidate_0}

Candidate 1:
{candidate_1}

Candidate 2:
{candidate_2}

Candidate 3:
{candidate_3}

Respond with ONLY the number (0, 1, 2, or 3) of the best candidate."""

MENTOR_JUDGE_PROMPT = """You are an expert judge evaluating AI-generated daily briefing responses.

Given a prompt and 4 candidate responses, pick the BEST one (respond with just the number 0, 1, 2, or 3).

Evaluate based on:
1. All 4 sections present (Today's Focus, You're Forgetting, Watch Out, Connections You Missed)
2. Directness and opinionated tone (not neutral, gives specific advice)
3. Specificity (mentions actual items from the context, not generic advice)
4. Actionability (tells the user exactly what to do)

Prompt:
{prompt}

Candidate 0:
{candidate_0}

Candidate 1:
{candidate_1}

Candidate 2:
{candidate_2}

Candidate 3:
{candidate_3}

Respond with ONLY the number (0, 1, 2, or 3) of the best candidate."""


def judge_candidates(
    client: OpenAI,
    prompt_messages: list[dict],
    candidates: list[str],
    judge_prompt_template: str,
) -> int | None:
    """Ask Nemotron to pick the best candidate. Returns index or None on failure."""
    prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages)

    judge_prompt = judge_prompt_template.format(
        prompt=prompt_text,
        candidate_0=candidates[0],
        candidate_1=candidates[1],
        candidate_2=candidates[2],
        candidate_3=candidates[3],
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.3,
            max_tokens=10,
            timeout=30,
        )
    except Exception as e:
        logger.warning("Judge API call failed: %s", e)
        return None

    content = response.choices[0].message.content.strip()
    # Extract first digit
    for char in content:
        if char in "0123":
            return int(char)

    logger.warning("Judge returned unexpected response: %s", content)
    return None


def run_judging(
    candidates_path: Path,
    output_path: Path,
    judge_prompt_template: str,
    dry_run: bool = False,
) -> None:
    """Run the full judging pipeline."""
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_NIM_API_KEY"],
        timeout=30,
    )

    records = _load_jsonl(candidates_path)
    target = 3 if dry_run else len(records)

    if not dry_run and not _estimate_nim_cost(target):
        print("Aborted by user.")
        return

    best_records: list[dict] = []
    attempt = 0
    valid = 0

    for i, record in enumerate(records[:target]):
        best_idx = judge_candidates(
            client,
            record["messages"],
            record["candidates"],
            judge_prompt_template,
        )
        attempt += 1
        if best_idx is not None:
            best_records.append(
                {
                    "messages": record["messages"]
                    + [{"role": "assistant", "content": record["candidates"][best_idx]}]
                }
            )
            valid += 1
        else:
            # Fallback: pick first candidate
            best_records.append(
                {
                    "messages": record["messages"]
                    + [{"role": "assistant", "content": record["candidates"][0]}]
                }
            )

        if (i + 1) % 50 == 0 or dry_run:
            logger.info("Judged %d/%d prompts", i + 1, target)

    if dry_run:
        print(f"\nDry-run complete: {valid}/{attempt} judged successfully.")
        return

    _save_jsonl(output_path, best_records)

    data_hash = _compute_sha256(output_path)
    manifest = {
        "script": "judge_candidates.py",
        "teacher_model": JUDGE_MODEL,
        "total_attempts": attempt,
        "valid_examples": valid,
        "output_file": str(output_path),
        "data_hash": f"sha256:{data_hash}",
        "estimated_cost": f"${attempt * 0.005:.2f}",
    }
    import json

    with open(output_path.parent / f"{output_path.stem}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge Best-of-N candidates")
    parser.add_argument("--candidates", type=str, required=True, help="Candidates JSONL path")
    parser.add_argument("--output", type=str, required=True, help="Output curated JSONL path")
    parser.add_argument("--task", type=str, choices=["scorer", "mentor"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    _set_all_seeds(args.seed)

    judge_prompt = SCORER_JUDGE_PROMPT if args.task == "scorer" else MENTOR_JUDGE_PROMPT
    run_judging(
        candidates_path=Path(args.candidates),
        output_path=Path(args.output),
        judge_prompt_template=judge_prompt,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
