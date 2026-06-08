"""Generate distillation training data from Nemotron via NVIDIA NIM API."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a learning prioritization expert. Given a piece of learning content \
and a set of learning goals, assess how relevant and important this content is for the learner.

Output a JSON object with exactly these fields:
- "score": integer 1-10 (1=irrelevant, 10=critical to read immediately)
- "rationale": one sentence explaining why this score
- "action": one sentence suggesting what the learner should do

Be honest and opinionated. Score low if content is tangential. Score high only if it directly \
advances a stated goal. Consider recency, depth, and practical applicability."""

# Synthetic content-goal pairs for distillation
_CONTENT_TEMPLATES = [
    # ML Papers
    {
        "title": "Understanding Transformer Architecture",
        "summary": "A deep dive into the original 'Attention Is All You Need' paper, covering self-attention, multi-head attention, positional encoding, and the encoder-decoder structure.",
    },
    {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "summary": "Explains how LoRA reduces trainable parameters by injecting low-rank matrices into transformer layers, making fine-tuning feasible on consumer GPUs.",
    },
    {
        "title": "GRPO: Group Relative Policy Optimization Explained",
        "summary": "Deep dive into DeepSeek's GRPO algorithm, a reinforcement learning method for LLM reasoning that eliminates the need for a separate critic model.",
    },
    {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "summary": "DPO shows that RLHF can be simplified by treating the language model itself as the reward model, eliminating the need for a separate reward model and PPO.",
    },
    {
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "summary": "QLoRA enables finetuning of 65B parameter models on a single 48GB GPU by backpropagating gradients through a frozen 4-bit quantized model into LoRA adapters.",
    },
    {
        "title": "The Illustrated Stable Diffusion",
        "summary": "A visual guide to how latent diffusion models work, covering the variational autoencoder, U-Net, CLIP text encoder, and the diffusion timestep schedule.",
    },
    {
        "title": "Mixture of Experts Explained",
        "summary": "How Mixture of Experts (MoE) architectures scale model capacity without proportional compute cost, with analysis of Switch Transformers and recent MoE LLMs.",
    },
    {
        "title": "RAG vs Fine-tuning: Which Should You Use?",
        "summary": "A systematic comparison of retrieval-augmented generation and fine-tuning for domain adaptation, with decision frameworks and hybrid approaches.",
    },
    {
        "title": "Scaling Laws for Neural Language Models",
        "summary": "Empirical analysis showing that model performance scales predictably with compute, parameters, and dataset size, enabling extrapolation to optimal training budgets.",
    },
    {
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "summary": "Anthropic's approach to training harmless AI assistants using a constitution of principles and self-critique, reducing reliance on human harm labels.",
    },
    # Engineering / Blog posts
    {
        "title": "Redis Internals: How It Handles 1M Concurrent Connections",
        "summary": "An exploration of Redis's event loop, memory management, and networking stack optimizations for high concurrency.",
    },
    {
        "title": "Building a Real-time Feature Store at Netflix",
        "summary": "How Netflix designed and scaled their feature store to serve sub-10ms feature lookups for thousands of models across recommendation, search, and content delivery.",
    },
    {
        "title": "Kubernetes Operators: Patterns and Anti-patterns",
        "summary": "Best practices for building custom resource definitions and controllers, with examples from production ML serving infrastructure.",
    },
    {
        "title": "Kafka vs Pulsar: A Deep Dive for ML Pipelines",
        "summary": "Comparing distributed messaging systems for real-time model inference, feature logging, and training data pipelines at scale.",
    },
    {
        "title": "10x Faster Data Loading with PyTorch DataLoader",
        "summary": "Techniques for optimizing PyTorch data loading including multi-processing, pinned memory, prefetching, and custom collate functions.",
    },
    {
        "title": "Monitoring LLMs in Production: A Practical Guide",
        "summary": "How to set up observability for LLM-powered applications including latency tracking, token usage, output quality metrics, and drift detection.",
    },
    {
        "title": "CUDA Kernel Optimization for Deep Learning",
        "summary": "Introduction to writing custom CUDA kernels for PyTorch, covering memory coalescing, shared memory, occupancy, and profiling with Nsight.",
    },
    {
        "title": "GitOps for Machine Learning: A Practical Tutorial",
        "summary": "Applying GitOps principles to ML pipelines using ArgoCD, Kubeflow, and automated model promotion with canary deployments.",
    },
    # Documentation / Tutorials
    {
        "title": "PyTorch 2.0 Compile Mode: A Complete Guide",
        "summary": "How torch.compile works under the hood with TorchDynamo, TorchInductor, and Triton kernels, with benchmarks and common pitfalls.",
    },
    {
        "title": "HuggingFace PEFT Library: From Quickstart to Advanced",
        "summary": "Comprehensive guide to LoRA, IA3, prompt tuning, and adapter fusion with the PEFT library, including multi-adapter composition.",
    },
    {
        "title": "vLLM Production Deployment Handbook",
        "summary": "Setting up vLLM for high-throughput LLM serving including PagedAttention, continuous batching, tensor parallelism, and quantization.",
    },
    {
        "title": "OpenAI API Best Practices for Enterprise",
        "summary": "Rate limiting strategies, request batching, error handling, retry logic, and cost optimization for production OpenAI API integrations.",
    },
    {
        "title": "DeepSpeed ZeRO-3: Training 100B+ Models",
        "summary": "How DeepSpeed's ZeRO stage 3 partitions optimizer states, gradients, and parameters across GPUs to enable training models with trillions of parameters.",
    },
    {
        "title": "Weights & Biases for Experiment Tracking",
        "summary": "Complete W&B tutorial covering runs, sweeps, artifacts, model registry, and integrations with PyTorch Lightning and HuggingFace.",
    },
    # News / Announcements
    {
        "title": "Google Announces Gemini 2.5 Pro with 1M Context Window",
        "summary": "Google's latest multimodal model features a 1 million token context window, native audio and video understanding, and improved reasoning benchmarks.",
    },
    {
        "title": "OpenAI Releases GPT-5 with Multimodal Reasoning",
        "summary": "The newest GPT model introduces unified multimodal reasoning across text, images, audio, and video with significantly reduced hallucination rates.",
    },
    {
        "title": "Meta Releases Llama 4: 400B MoE Architecture",
        "summary": "Meta's latest open weights model uses a 400B parameter Mixture of Experts with 16 experts, achieving GPT-4 level performance on most benchmarks.",
    },
    {
        "title": "Mistral AI Launches Codestral: Code-First LLM",
        "summary": "Mistral's specialized code model supports 80+ programming languages with fill-in-the-middle capabilities and outperforms GPT-4 on HumanEval.",
    },
    {
        "title": "NVIDIA Blackwell Architecture: 4x Training Speedup",
        "summary": "NVIDIA's new B200 GPU introduces second-generation transformer engine and FP4 precision, enabling 4x faster training for trillion-parameter models.",
    },
    # Social media / Hot takes
    {
        "title": "Why I Stopped Using LangChain (Thread)",
        "summary": "A viral Twitter thread arguing that LangChain adds unnecessary abstraction, increases debugging difficulty, and pure Python with OpenAI client is superior for most projects.",
    },
    {
        "title": "5 Things Junior ML Engineers Get Wrong About Production",
        "summary": "Common mistakes including over-engineering pipelines, neglecting data validation, ignoring inference latency, insufficient monitoring, and poor error handling.",
    },
    {
        "title": "The Hype Cycle of Vector Databases: Where Are We Now?",
        "summary": "An opinionated analysis of when vector databases are necessary versus when PostgreSQL with pgvector or in-memory FAISS is sufficient.",
    },
    {
        "title": "Why Your RAG System is Probably Broken",
        "summary": "Common RAG failure modes including poor chunking, missing context, wrong embedding models, no reranking, and insufficient evaluation metrics.",
    },
    {
        "title": "AI Engineering vs ML Engineering: The Real Difference",
        "summary": "A nuanced take on how AI engineering (prompt engineering, RAG, evaluation) differs from traditional ML engineering (feature engineering, model training).",
    },
    # Irrelevant / Low relevance content
    {
        "title": "The Perfect Sourdough Bread Recipe",
        "summary": "A detailed guide to making sourdough at home, covering starter maintenance, autolyse, stretch and fold techniques, and baking in a Dutch oven.",
    },
    {
        "title": "2026 NBA Playoffs: Predictions and Analysis",
        "summary": "Breaking down the playoff matchups, key player injuries, statistical trends, and expert predictions for the NBA championship.",
    },
    {
        "title": "Top 10 Mediterranean Diet Meal Prep Ideas",
        "summary": "Healthy meal prep recipes focusing on olive oil, fish, whole grains, and fresh vegetables for busy professionals.",
    },
    {
        "title": "The History of Jazz in New Orleans",
        "summary": "From Buddy Bolden to Louis Armstrong, tracing the evolution of jazz music in New Orleans and its influence on modern music.",
    },
    {
        "title": "Minimalist Wardrobe: Building a Capsule Collection",
        "summary": "How to build a versatile 30-piece wardrobe that works for any occasion, with color palette strategies and quality fabric recommendations.",
    },
]

_GOAL_POOL = [
    "learning post-training techniques",
    "preparing for system design interviews",
    "building a RAG system",
    "understanding transformers",
    "mastering PyTorch",
    "learning MLOps",
    "preparing for ML interviews",
    "understanding diffusion models",
    "optimizing inference latency",
    "learning distributed training",
    "building production ML pipelines",
    "understanding quantization and compression",
    "learning CUDA and GPU programming",
    "mastering Kubernetes for ML",
    "understanding RLHF and alignment",
    "building multimodal AI systems",
    "learning vector databases and embeddings",
    "understanding Mixture of Experts",
    "mastering prompt engineering",
    "learning model evaluation and benchmarking",
]


_MUTATION_SWAP_TERMS = [
    ("transformer", "attention mechanism"),
    ("LoRA", "adapter tuning"),
    ("RAG", "retrieval augmentation"),
    ("fine-tuning", "post-training"),
    ("GPU", "accelerator"),
    ("model", "network"),
    ("inference", "serving"),
    ("training", "learning"),
    ("PyTorch", "JAX"),
    ("Kubernetes", "container orchestration"),
]


def _mutate_text(text: str) -> str:
    """Apply random term swaps to diversify text."""
    swaps = random.sample(_MUTATION_SWAP_TERMS, k=min(2, len(_MUTATION_SWAP_TERMS)))
    for old, new in swaps:
        if old in text and random.random() < 0.3:
            text = text.replace(old, new, 1)
    return text


def _generate_synthetic_pair() -> dict[str, object]:
    """Generate a random synthetic content-goal pair with mutations."""
    template = random.choice(_CONTENT_TEMPLATES)
    num_goals = random.randint(1, 4)
    goals = random.sample(_GOAL_POOL, num_goals)
    return {
        "title": _mutate_text(template["title"]),
        "summary": _mutate_text(template["summary"]),
        "goals": goals,
    }


def generate_training_example(
    client: OpenAI,
    content_title: str,
    content_summary: str,
    goals: list[str],
) -> dict | None:
    """Generate a single training example using Nemotron as teacher."""
    goals_text = "\n".join(f"- {g}" for g in goals)
    user_prompt = f"Content: {content_title}\n{content_summary}\n\nLearning Goals:\n{goals_text}"

    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=512,
        )
    except Exception as e:
        logger.warning("NIM API call failed: %s", e)
        return None

    assistant_content = response.choices[0].message.content

    # Validate JSON structure
    try:
        parsed = json.loads(assistant_content)
        assert 1 <= parsed["score"] <= 10
        assert "rationale" in parsed
        assert "action" in parsed
    except (json.JSONDecodeError, KeyError, AssertionError):
        logger.warning("Malformed response from Nemotron: %s", assistant_content[:200])
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def generate_dataset(output_path: str, num_examples: int = 2000) -> None:
    """Generate full training dataset and write to JSONL."""
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
            pair = _generate_synthetic_pair()
            result = generate_training_example(
                client,
                pair["title"],
                pair["summary"],
                pair["goals"],
            )
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                valid_count += 1
                if valid_count % 100 == 0:
                    logger.info("Generated %d/%d valid examples", valid_count, num_examples)

    logger.info("Dataset complete: %d examples written to %s", valid_count, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_dataset("training/data/distillation_data.jsonl", num_examples=500)
