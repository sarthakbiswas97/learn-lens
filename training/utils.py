"""Shared utilities for training scripts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_NIM_COST_PER_CALL = 0.005


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_jsonl(path: Path, required_fields: list[str] | None = None) -> list[dict[str, Any]]:
    """Parse every line of a JSONL file and validate required fields.

    Args:
        path: Path to JSONL file.
        required_fields: Top-level keys that must be present in every record.

    Returns:
        List of parsed records.

    Raises:
        ValueError: If any line fails to parse or misses required fields.
    """
    required = required_fields or []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: invalid JSON — {e}") from e

            missing = [f for f in required if f not in record]
            if missing:
                raise ValueError(f"Line {line_num}: missing required fields {missing}")

            records.append(record)

    logger.info("Validated %d records in %s", len(records), path)
    return records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of records."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Save a list of records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def estimate_nim_cost(num_calls: int, cost_per_call: float = DEFAULT_NIM_COST_PER_CALL) -> bool:
    """Print cost estimate and ask user for confirmation.

    Returns:
        True if user confirms, False otherwise.
    """
    estimated = num_calls * cost_per_call
    print(
        f"\nEstimated NIM API cost: {num_calls} calls × ~${cost_per_call:.3f}/call = ~${estimated:.2f}"
    )
    if os.environ.get("LEARNLENS_YES"):
        return True
    try:
        response = input("Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return response in ("y", "yes")


def save_data_manifest(
    output_path: Path,
    script: str,
    teacher_model: str,
    total_attempts: int,
    valid_examples: int,
    output_file: Path,
    data_hash: str,
    estimated_cost: str,
) -> None:
    """Write a data generation manifest JSON."""
    import datetime

    reject_rate = (total_attempts - valid_examples) / total_attempts if total_attempts else 0.0
    manifest = {
        "script": script,
        "teacher_model": teacher_model,
        "total_attempts": total_attempts,
        "valid_examples": valid_examples,
        "reject_rate": round(reject_rate, 4),
        "output_file": str(output_file),
        "data_hash": f"sha256:{data_hash}",
        "estimated_cost": estimated_cost,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    manifest_path = output_path.parent / "data_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Data manifest saved to %s", manifest_path)


def save_run_metadata(
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
    change_from_prev: str = "",
) -> None:
    """Write training run metadata JSON."""
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
        "git_commit": get_git_commit(),
    }
    if change_from_prev:
        metadata["change_from_prev"] = change_from_prev

    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Run metadata saved to %s", metadata_path)


def get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seeds set to %d", seed)
