"""LoRA fine-tune MiniCPM5-1B on Modal for priority scoring."""

from __future__ import annotations

import logging

import modal

logger = logging.getLogger(__name__)

app = modal.App("learnlens-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7",
        "transformers>=5.6",
        "trl>=0.21",
        "peft>=0.15",
        "datasets>=3.0",
        "accelerate>=1.0",
        "bitsandbytes>=0.45",
    )
)


@app.function(
    gpu="A10G",
    timeout=3600,
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
)
def train() -> None:
    """Train MiniCPM5-1B with LoRA on distillation data."""
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "openbmb/MiniCPM5-1B",
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "openbmb/MiniCPM5-1B",
        trust_remote_code=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
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

    # Load distillation data
    dataset = load_dataset(
        "json",
        data_files="/root/distillation_data.jsonl",
        split="train",
    )

    # Training config
    training_config = SFTConfig(
        output_dir="/root/output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=2048,
        dataset_text_field=None,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save and push adapter only (base model loaded separately at runtime)
    trainer.model.save_pretrained("/root/output/lora-scorer")
    trainer.model.push_to_hub("sarthakbiswas/learnlens-scorer-lora")
    tokenizer.push_to_hub("sarthakbiswas/learnlens-scorer-lora")
    logger.info("Adapter pushed to HuggingFace Hub")


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint to launch training."""
    train.remote()
