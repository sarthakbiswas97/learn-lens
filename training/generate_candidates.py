"""Generate N=4 candidate responses per prompt for Best-of-N refinement."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

app = modal.App("learnlens-generate-candidates")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.7",
    "transformers>=5.6",
    "peft>=0.15",
    "datasets>=3.0",
    "accelerate>=1.0",
)

BASE_MODEL_ID = "openbmb/MiniCPM5-1B"
ADAPTER_ID = "sarthakbiswas/learnlens-scorer-sft-v1"
MENTOR_ADAPTER_ID = "sarthakbiswas/learnlens-mentor-sft-v1"


def _set_all_seeds(seed: int = 42) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_jsonl(path: str) -> list[dict]:
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
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_candidates(
    prompt_records: list[dict],
    adapter_id: str,
    num_candidates: int = 4,
    temperature: float = 0.7,
    seed: int = 42,
) -> list[dict]:
    """Generate candidate responses using SFT'd student model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _set_all_seeds(seed)

    logger.info("Loading base model: %s", BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )

    logger.info("Loading adapter: %s", adapter_id)
    model = PeftModel.from_pretrained(model, adapter_id)

    logger.info("Generating %d candidates for %d prompts", num_candidates, len(prompt_records))

    results: list[dict] = []
    for i, record in enumerate(prompt_records):
        messages = record["messages"]
        # Extract system + user messages (exclude assistant)
        prompt_messages = [m for m in messages if m["role"] != "assistant"]

        inputs = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        candidates: list[str] = []
        for _ in range(num_candidates):
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
            )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            candidates.append(response)

        results.append(
            {
                "messages": prompt_messages,
                "candidates": candidates,
            }
        )

        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d prompts", i + 1, len(prompt_records))

    logger.info("Generated %d candidate sets", len(results))
    return results


@app.local_entrypoint()
def main(
    data: str = "training/data/distillation_data.jsonl",
    output: str = "training/data/scorer_candidates.jsonl",
    adapter: str = ADAPTER_ID,
    num_candidates: int = 4,
    temperature: float = 0.7,
    seed: int = 42,
    batch_size: int = 50,
) -> None:
    """Local entrypoint to launch candidate generation in batches."""
    logging.basicConfig(level=logging.INFO)

    records = _load_jsonl(data)
    total = len(records)
    print(f"Loaded {total} prompts from {data}")

    out_file = Path(output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume: count already completed records
    completed = 0
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            completed = sum(1 for line in f if line.strip())
        print(f"Resuming: {completed}/{total} already done")

    # Process in batches to avoid long-lived gRPC connections
    for start in range(completed, total, batch_size):
        end = min(start + batch_size, total)
        batch = records[start:end]
        print(f"\nBatch {start // batch_size + 1}: prompts {start + 1}-{end} (size={len(batch)})")

        results = generate_candidates.remote(
            prompt_records=batch,
            adapter_id=adapter,
            num_candidates=num_candidates,
            temperature=temperature,
            seed=seed,
        )

        # Append incrementally
        with open(out_file, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved batch. Total: {end}/{total}")

    print(f"\nAll done! Saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Best-of-N candidates")
    parser.add_argument("--data", type=str, default="training/data/distillation_data.jsonl")
    parser.add_argument("--output", type=str, default="training/data/scorer_candidates.jsonl")
    parser.add_argument("--adapter", type=str, default=ADAPTER_ID)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(
        data=args.data,
        output=args.output,
        adapter=args.adapter,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        seed=args.seed,
        batch_size=args.batch_size,
    )
