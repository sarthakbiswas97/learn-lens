"""GGUF conversion for llama.cpp serving."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "openbmb/MiniCPM5-1B"


def export_gguf(
    adapter_id: str,
    output_path: str,
    quantization: str = "Q4_K_M",
) -> None:
    """Export a merged base+LoRA adapter to GGUF format for llama.cpp.

    Requires llama.cpp to be installed and convert_hf_to_gguf.py available.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    logger.info("Loading LoRA adapter: %s", adapter_id)
    model = PeftModel.from_pretrained(base_model, adapter_id)

    logger.info("Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    # Save to temporary directory for conversion
    temp_dir = Path("training/output/temp_hf")
    temp_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    # Find llama.cpp convert script
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
    if not convert_script.exists():
        logger.error(
            "convert_hf_to_gguf.py not found at %s. Please clone llama.cpp and run from repo root.",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LoRA adapter to GGUF")
    parser.add_argument("--adapter-id", type=str, required=True, help="HuggingFace adapter ID")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF path")
    parser.add_argument("--quantization", type=str, default="Q4_K_M", help="Quantization type")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    export_gguf(args.adapter_id, args.output, args.quantization)


if __name__ == "__main__":
    main()
