"""Text cleaning, truncation, and extraction utilities."""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_length: int = 4096) -> str:
    """Truncate text to max_length characters, preserving word boundaries."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    return truncated.strip()


def extract_domain(url: str) -> str | None:
    """Extract domain from a URL."""
    match = re.match(r"https?://([^/]+)", url)
    return match.group(1) if match else None
