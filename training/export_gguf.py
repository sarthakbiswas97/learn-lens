"""GGUF conversion for llama.cpp serving."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "sarthakbiswas/learnlens-prioritizer"
DEFAULT_OUTPUT = "training/output/learnlens-prioritizer.gguf"


def export_gguf(
    model_id: str = DEFAULT_MODEL_ID,
    output_path: str = DEFAULT_OUTPUT,
    quantization: str = "Q4_K_M",
) -> None:
    """Export a HuggingFace model to GGUF format for llama.cpp.

    Requires llama.cpp to be installed and convert_hf_to_gguf.py available.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Save to temporary directory for conversion
    temp_dir = Path("training/output/temp_hf")
    temp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    # Find llama.cpp convert script
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
    if not convert_script.exists():
        logger.error(
            "convert_hf_to_gguf.py not found at %s. "
            "Please clone llama.cpp and run from repo root.",
            convert_script,
        )
        sys.exit(1)

    cmd = [
        sys.executable,
        str(convert_script),
        str(temp_dir),
        "--outfile",
        str(output_file),
        "--outtype",
        quantization,
    ]

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("GGUF exported to %s", output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_gguf()
