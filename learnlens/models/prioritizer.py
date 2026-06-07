"""Wrapper for MiniCPM5-1B + LoRA priority scorer."""

from __future__ import annotations

import json
import logging
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from learnlens.config import LearnLensConfig
from learnlens.models.types import ContentItem, Goal, ScoredItem

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a learning prioritization expert. Given a piece of content "
    "and a list of learning goals, output a JSON object with exactly these fields: "
    "score (integer 1-10), rationale (one sentence), action (one sentence suggesting what to do). "
    "Be honest and opinionated. Score low if content is tangential. "
    "Score high only if it directly advances a stated goal."
)


class PrioritizerModel:
    """Wrapper for MiniCPM5-1B + LoRA priority scorer."""

    def __init__(self, config: LearnLensConfig) -> None:
        self._config = config
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> None:
        """Load base model and optional LoRA adapter."""
        logger.info("Loading prioritizer base model: %s", self._config.prioritizer_base_model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.prioritizer_base_model_id,
            trust_remote_code=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self._config.prioritizer_base_model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Attempt to load LoRA adapter; fall back to base if unavailable
        try:
            self._model = PeftModel.from_pretrained(
                base_model,
                self._config.prioritizer_model_id,
            )
            logger.info("Loaded LoRA adapter: %s", self._config.prioritizer_model_id)
        except Exception:
            self._model = base_model
            logger.warning(
                "LoRA adapter not found (%s), using base model",
                self._config.prioritizer_model_id,
            )

    def _ensure_loaded(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self._model is None or self._tokenizer is None:
            self.load()
        return self._model, self._tokenizer  # type: ignore[return-value]

    def score(
        self,
        content: ContentItem,
        goals: list[Goal],
    ) -> ScoredItem:
        """Score a content item against user goals.

        Returns:
            ScoredItem with score (1-10), rationale, and suggested action.
        """
        model, tokenizer = self._ensure_loaded()

        goals_text = "\n".join(f"- {g.goal_text} (priority {g.priority})" for g in goals)
        summary = content.body_text[:500]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Title: {content.title}\n"
                    f"URL: {content.url}\n"
                    f"Summary: {summary}\n\n"
                    f"Learning Goals:\n{goals_text}"
                ),
            },
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=self._config.prioritizer_max_tokens,
            temperature=self._config.prioritizer_temperature,
            top_p=0.95,
            do_sample=True,
        )

        raw = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

        parsed = self._parse_response(raw)

        # Attach to highest-priority goal for grouping
        primary_goal = max(goals, key=lambda g: g.priority) if goals else Goal(
            id=0, goal_text="general", priority=3, is_active=True
        )

        return ScoredItem(
            content=content,
            score=parsed["score"],
            rationale=parsed["rationale"],
            suggested_action=parsed["action"],
            goal=primary_goal,
        )

    def _parse_response(self, raw: str) -> dict[str, object]:
        """Extract JSON from model response with fallbacks."""
        # Try to find JSON block
        if "```json" in raw:
            block = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            block = raw.split("```")[1].split("```")[0]
        else:
            block = raw

        block = block.strip()

        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            # Fallback regex extraction
            parsed = self._regex_extract(block)

        score = float(parsed.get("score", 5))
        score = max(1.0, min(10.0, score))

        return {
            "score": score,
            "rationale": str(parsed.get("rationale", "No rationale provided.")),
            "action": str(parsed.get("action", "Review at your discretion.")),
        }

    def _regex_extract(self, text: str) -> dict[str, str]:
        """Emergency regex extraction when JSON is malformed."""
        score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', text)
        action_match = re.search(r'"action"\s*:\s*"([^"]*)"', text)

        return {
            "score": score_match.group(1) if score_match else "5",
            "rationale": rationale_match.group(1) if rationale_match else "No rationale.",
            "action": action_match.group(1) if action_match else "Review content.",
        }
