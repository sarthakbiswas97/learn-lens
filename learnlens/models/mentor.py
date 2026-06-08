"""Wrapper for MiniCPM5-1B + LoRA mentor briefing generator."""

from __future__ import annotations

import logging

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from learnlens.config import LearnLensConfig
from learnlens.models.types import BriefingRequest

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an opinionated learning mentor. Be direct and specific. "
    "Tell the user what to focus on, what they are forgetting, "
    "and what mistakes to watch for. Do not be neutral -- have opinions. "
    "Use direct language: 'Stop reading about X' not 'You might consider deprioritizing X'. "
    "Structure your response in markdown with these sections:\n"
    "## Today's Focus\n"
    "## You're Forgetting\n"
    "## Watch Out\n"
    "## Connections You Missed"
)


class MentorModel:
    """Wrapper for MiniCPM5-1B + LoRA mentor briefing generator."""

    def __init__(
        self,
        config: LearnLensConfig,
        base_model: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
    ) -> None:
        self._config = config
        self._base_model = base_model
        self._tokenizer = tokenizer
        self._model: AutoModelForCausalLM | None = None
        self._adapter_loaded = False

    def load(self) -> None:
        """Attach mentor LoRA adapter to shared base model."""
        if self._base_model is None:
            raise RuntimeError("MentorModel requires a shared base_model to be provided")

        logger.info("Loading mentor adapter: %s", self._config.mentor_adapter_id)
        self._model = PeftModel.from_pretrained(
            self._base_model,
            self._config.mentor_adapter_id,
            adapter_name="mentor",
        )
        self._adapter_loaded = True
        logger.info("Mentor adapter loaded")

    def _ensure_loaded(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MentorModel not loaded. Call load() first.")
        return self._model, self._tokenizer

    def generate_briefing(self, request: BriefingRequest) -> str:
        """Generate a daily briefing from scored items, forgotten items, and mistakes.

        Returns:
            Markdown-formatted daily briefing string.
        """
        model, tokenizer = self._ensure_loaded()
        model.set_adapter("mentor")

        context = self._assemble_context(request)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=self._config.mentor_max_tokens,
            temperature=self._config.mentor_temperature,
            top_p=self._config.mentor_top_p,
            do_sample=True,
        )

        briefing = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        logger.info("Generated briefing (%d chars)", len(briefing))
        return briefing.strip()

    def _assemble_context(self, request: BriefingRequest) -> str:
        """Build structured context string from request data."""
        lines: list[str] = []

        lines.append("# Your Learning Context")
        lines.append("")

        # Goals
        lines.append("## Your Goals")
        for goal in request.goals:
            lines.append(f"- {goal.goal_text} (priority {goal.priority})")
        lines.append("")

        # Top scored items
        lines.append("## Top Priority Content")
        for item in request.scored_items[: self._config.top_k_briefing]:
            lines.append(f"- **{item.content.title}** (score: {item.score})")
            lines.append(f"  - Rationale: {item.rationale}")
            lines.append(f"  - Action: {item.suggested_action}")
            lines.append(f"  - URL: {item.content.url}")
        lines.append("")

        # Forgotten items
        if request.forgotten_items:
            lines.append("## Items You Are Forgetting")
            for item in request.forgotten_items:
                lines.append(f"- {item.title} ({item.url})")
            lines.append("")

        # Mistakes
        if request.mistakes:
            lines.append("## Mistakes to Watch For")
            for mistake in request.mistakes:
                lines.append(f"- **{mistake.pattern}** (severity {mistake.severity})")
                lines.append(f"  - {mistake.description}")
            lines.append("")

        lines.append("Generate the daily briefing based on the above context.")
        return "\n".join(lines)
