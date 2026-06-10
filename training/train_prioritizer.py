"""LoRA fine-tune MiniCPM5-1B on Modal for priority scoring."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

app = modal.App("learnlens-training")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.7",
    "transformers>=5.6",
    "trl>=0.21",
    "peft>=0.15",
    "datasets>=3.0",
    "accelerate>=1.0",
    "bitsandbytes>=0.45",
    "tensorboard>=2.18",
)

BASE_MODEL_ID = "openbmb/MiniCPM5-1B"
HF_REPO = "sarthakbiswas/learnlens-scorer-sft-v1"


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


def _validate_jsonl(path: Path, required_fields: list[str] | None = None) -> list[dict]:
    import json

    required = required_fields or []
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: invalid JSON — {e}") from e
            missing = [fld for fld in required if fld not in record]
            if missing:
                raise ValueError(f"Line {line_num}: missing required fields {missing}")
            records.append(record)
    return records


def _save_run_metadata(
    output_dir: Path,
    run_name: str,
    experiment: str,
    task: str,
    base_model: str,
    data_file: Path,
    data_hash: str,
    data_size: int,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    max_seq_length: int,
    seed: int,
    modal_gpu: str,
    training_time_sec: float,
    final_train_loss: float,
) -> None:
    import json
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    metadata = {
        "run_name": run_name,
        "experiment": experiment,
        "task": task,
        "base_model": base_model,
        "data_file": str(data_file),
        "data_hash": f"sha256:{data_hash}",
        "data_size": data_size,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "max_seq_length": max_seq_length,
        "seed": seed,
        "modal_gpu": modal_gpu,
        "training_time_sec": round(training_time_sec, 1),
        "final_train_loss": round(final_train_loss, 4),
        "git_commit": commit,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _validate_data(records: list[dict], label: str = "data") -> list[dict]:
    """Validate records before any GPU work."""
    if not records:
        raise ValueError("No valid records found")
    # Check message structure
    for i, rec in enumerate(records[:10]):  # sample first 10
        msgs = rec.get("messages", [])
        roles = {m.get("role") for m in msgs}
        if not {"system", "user", "assistant"}.issubset(roles):
            raise ValueError(f"Record {i + 1} missing required roles (system/user/assistant)")
    logger.info("Data validation passed: %d records (%s)", len(records), label)
    return records


@app.function(
    gpu="A10G",
    timeout=3600,
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    train_records: list[dict],
    output_dir: str,
    dry_run: bool = False,
    resume_from: str | None = None,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    seed: int = 42,
) -> dict:
    """Train MiniCPM5-1B with LoRA on distillation data."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    _set_all_seeds(seed)
    start_time = time.time()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Validate data BEFORE loading model (fail fast, no GPU waste)
    # ------------------------------------------------------------------
    if dry_run:
        # Tiny synthetic dataset for dry-run
        dummy_records = [
            {
                "messages": [
                    {"role": "system", "content": "You are a learning prioritization expert."},
                    {
                        "role": "user",
                        "content": "Content: Test Article\nTest summary.\n\nLearning Goals:\n- test goal",
                    },
                    {
                        "role": "assistant",
                        "content": '{"score": 5, "rationale": "test", "action": "review"}',
                    },
                ]
            }
            for _ in range(10)
        ]
        dataset = Dataset.from_list(dummy_records)
        data_hash = "dry-run"
        data_size = 10
    else:
        _validate_data(train_records, label="train")
        dataset = Dataset.from_list(train_records)
        data_size = len(dataset)
        # Compute hash from JSON representation
        import hashlib
        import json

        data_hash = hashlib.sha256(json.dumps(train_records, sort_keys=True).encode()).hexdigest()

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
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

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # ------------------------------------------------------------------
    # 3. Training config
    # ------------------------------------------------------------------
    training_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=1 if dry_run else epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field=None,
        report_to="none" if dry_run else "tensorboard",
    )

    if resume_from:
        training_config.resume_from_checkpoint = resume_from

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    logger.info(
        "Starting training (dry_run=%s, epochs=%d, lr=%.0e)", dry_run, epochs, learning_rate
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    final_loss = train_result.training_loss if train_result else 0.0
    training_time = time.time() - start_time

    logger.info("Training complete in %.1f seconds. Final loss: %.4f", training_time, final_loss)

    # ------------------------------------------------------------------
    # 5. Save and push
    # ------------------------------------------------------------------
    adapter_dir = out_dir / "lora-scorer"
    trainer.model.save_pretrained(str(adapter_dir))

    if not dry_run:
        logger.info("Pushing adapter to %s", HF_REPO)
        trainer.model.push_to_hub(HF_REPO)
        tokenizer.push_to_hub(HF_REPO)

        # Save metadata
        _save_run_metadata(
            output_dir=out_dir,
            run_name="scorer-sft-v1",
            experiment="sft" if not resume_from else "combined",
            task="scorer",
            base_model=BASE_MODEL_ID,
            data_file=Path("remote_data"),
            data_hash=data_hash,
            data_size=data_size,
            lora_r=lora_r,
            lora_alpha=lora_r * 2,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=4,
            grad_accum=4,
            max_seq_length=1024,
            seed=seed,
            modal_gpu="A10G",
            training_time_sec=training_time,
            final_train_loss=final_loss,
        )

    return {
        "final_loss": final_loss,
        "training_time_sec": training_time,
        "dry_run": dry_run,
    }


def _load_jsonl(path: str) -> list[dict]:
    import json

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@app.local_entrypoint()
def main(
    data: str = "training/data/distillation_data.jsonl",
    output_dir: str = "training/output/scorer-sft-v1",
    dry_run: bool = False,
    resume: str | None = None,
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 16,
    seed: int = 42,
) -> None:
    """Local entrypoint to launch training."""
    logging.basicConfig(level=logging.INFO)

    if dry_run:
        print("\n[DRY RUN] Will validate data, load model, run 3 steps, and exit.\n")
        train_records = []
    else:
        data_path = Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data}")
        train_records = _load_jsonl(data)
        # Validate locally before sending to remote
        _validate_data(train_records, label="train")
        print(f"Loaded {len(train_records)} training records from {data}")

    result = train.remote(
        train_records=train_records,
        output_dir=output_dir,
        dry_run=dry_run,
        resume_from=resume,
        epochs=epochs,
        learning_rate=lr,
        lora_r=lora_r,
        seed=seed,
    )
    print(f"\nTraining result: {result}")


if __name__ == "__main__":
    # Also support direct invocation with argparse (for local testing without Modal)
    parser = argparse.ArgumentParser(description="Train scorer LoRA adapter")
    parser.add_argument("--data", type=str, default="training/data/distillation_data.jsonl")
    parser.add_argument("--output-dir", type=str, default="training/output/scorer-sft-v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(
        data=args.data,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=args.lora_r,
        seed=args.seed,
    )
