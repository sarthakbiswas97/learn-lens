"""Evaluate SFT vs Best-of-N vs Combined adapters on held-out prompts."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

app = modal.App("learnlens-evaluate")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.7",
    "transformers>=5.6",
    "peft>=0.15",
    "datasets>=3.0",
    "accelerate>=1.0",
    "openai>=1.0",
)

BASE_MODEL_ID = "openbmb/MiniCPM5-1B"
JUDGE_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

ADAPTER_VARIANTS = {
    "sft": {
        "scorer": "sarthakbiswas/learnlens-scorer-sft-v1",
        "mentor": "sarthakbiswas/learnlens-mentor-sft-v1",
    },
}


def _run_scorer(
    model,
    tokenizer,
    prompt_messages: list[dict],
) -> str:
    """Run scorer model on a prompt, return raw response."""
    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.5,
        top_p=0.95,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def _run_mentor(
    model,
    tokenizer,
    prompt_messages: list[dict],
) -> str:
    """Run mentor model on a prompt, return raw response."""
    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def _judge_scorer(
    client,
    prompt_messages: list[dict],
    response: str,
) -> dict:
    """Ask Nemotron to score a scorer response."""
    prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages)
    judge_prompt = f"""Rate the following AI-generated learning prioritization response on a scale of 1-10 for overall quality.

Prompt:
{prompt_text}

Response:
{response}

Output JSON with exactly these fields:
- "score": integer 1-10 (overall quality)
- "json_valid": boolean (is the response valid JSON with score, rationale, action?)
- "rationale_quality": integer 1-10
- "action_quality": integer 1-10"""

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.3,
            max_tokens=256,
            timeout=30,
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        return {
            "overall_score": int(parsed.get("score", 5)),
            "json_valid": bool(parsed.get("json_valid", False)),
            "rationale_quality": int(parsed.get("rationale_quality", 5)),
            "action_quality": int(parsed.get("action_quality", 5)),
        }
    except Exception as e:
        logger.warning("Judge scoring failed: %s", e)
        return {
            "overall_score": 5,
            "json_valid": False,
            "rationale_quality": 5,
            "action_quality": 5,
        }


def _judge_mentor(
    client,
    prompt_messages: list[dict],
    response: str,
) -> dict:
    """Ask Nemotron to score a mentor response."""
    prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages)
    judge_prompt = f"""Rate the following AI-generated daily briefing on a scale of 1-10 for overall quality.

Prompt:
{prompt_text}

Response:
{response}

Output JSON with exactly these fields:
- "score": integer 1-10 (overall quality)
- "sections_complete": boolean (all 4 sections present?)
- "directness": integer 1-10 (opinionated, not neutral?)
- "specificity": integer 1-10 (mentions actual items from context?)"""

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.3,
            max_tokens=256,
            timeout=30,
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        return {
            "overall_score": int(parsed.get("score", 5)),
            "sections_complete": bool(parsed.get("sections_complete", False)),
            "directness": int(parsed.get("directness", 5)),
            "specificity": int(parsed.get("specificity", 5)),
        }
    except Exception as e:
        logger.warning("Judge scoring failed: %s", e)
        return {"overall_score": 5, "sections_complete": False, "directness": 5, "specificity": 5}


def _set_all_seeds(seed: int = 42) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_jsonl(path):
    import json

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@app.function(
    gpu="A10G",
    timeout=3600,
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("NVIDIA_NIM"),
    ],
)
def evaluate(
    eval_records: list[dict],
    task: str,
    seed: int = 42,
) -> dict:
    """Evaluate all adapter variants on held-out prompts."""
    from openai import OpenAI
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _set_all_seeds(seed)

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_NIM_API_KEY"],
        timeout=30,
    )

    # Load base model once
    logger.info("Loading base model: %s", BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )

    records = eval_records
    logger.info("Evaluating %d held-out prompts", len(records))

    results: dict[str, list[dict]] = {}

    for variant_name, adapters in ADAPTER_VARIANTS.items():
        adapter_id = adapters.get(task)
        if adapter_id is None:
            continue

        logger.info("Evaluating variant: %s (%s)", variant_name, adapter_id)

        # Load adapter
        model = PeftModel.from_pretrained(base_model, adapter_id)

        variant_results: list[dict] = []
        for i, record in enumerate(records):
            prompt_messages = [m for m in record["messages"] if m["role"] != "assistant"]

            if task == "scorer":
                response = _run_scorer(model, tokenizer, prompt_messages)
                judge_scores = _judge_scorer(client, prompt_messages, response)
            else:
                response = _run_mentor(model, tokenizer, prompt_messages)
                judge_scores = _judge_mentor(client, prompt_messages, response)

            variant_results.append(
                {
                    "prompt_index": i,
                    "response": response,
                    **judge_scores,
                }
            )

            if (i + 1) % 10 == 0:
                logger.info("%s: %d/%d done", variant_name, i + 1, len(records))

        results[variant_name] = variant_results

        # Unload adapter before next variant
        del model

    # Aggregate and save
    summary: dict[str, dict] = {}
    for variant_name, variant_results in results.items():
        overall_scores = [r["overall_score"] for r in variant_results]
        summary[variant_name] = {
            "mean_overall": sum(overall_scores) / len(overall_scores),
            "json_valid_rate": sum(1 for r in variant_results if r.get("json_valid"))
            / len(variant_results)
            if task == "scorer"
            else None,
            "sections_complete_rate": sum(1 for r in variant_results if r.get("sections_complete"))
            / len(variant_results)
            if task == "mentor"
            else None,
        }

    output = {
        "task": task,
        "num_prompts": len(records),
        "summary": summary,
        "details": results,
    }

    logger.info("Evaluation complete. Summary: %s", summary)
    return output


@app.local_entrypoint()
def main(
    eval_data: str = "training/data/eval_holdout_scorer.jsonl",
    output_dir: str = "training/output",
    task: str = "scorer",
    seed: int = 42,
) -> None:
    """Local entrypoint to launch evaluation."""
    import json

    logging.basicConfig(level=logging.INFO)

    records = _load_jsonl(eval_data)
    print(f"Loaded {len(records)} eval prompts from {eval_data}")

    result = evaluate.remote(
        eval_records=records,
        task=task,
        seed=seed,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{task}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nEvaluation result saved to {out_file}")
    print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate adapter variants")
    parser.add_argument("--eval-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="training/output")
    parser.add_argument("--task", type=str, choices=["scorer", "mentor"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        task=args.task,
        seed=args.seed,
    )
