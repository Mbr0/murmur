#!/usr/bin/env python3
"""Pure transcription filter helpers used by the app and tests."""

from __future__ import annotations


DEFAULT_MIN_DURATION_SECONDS = 1.0
DEFAULT_MIN_MAX_LEVEL = 0.01
MIN_VALID_TEXT_LENGTH = 3


HALLUCINATION_TEXTS = {
    "",
    ".",
    "thank",
    "thank you",
    "thanks for listening",
    "thanks for watching",
    "bye",
    "goodbye",
    "see you",
    "subscribe",
    "you",
}


def should_skip_audio(duration_seconds: float, max_level: float) -> bool:
    """Return True when audio is too short or too quiet to transcribe reliably."""
    assert duration_seconds >= 0.0, "duration_seconds must be non-negative"
    assert max_level >= 0.0, "max_level must be non-negative"
    return duration_seconds < DEFAULT_MIN_DURATION_SECONDS or max_level < DEFAULT_MIN_MAX_LEVEL


def is_likely_hallucination(text: str) -> bool:
    """Return True when text matches low-signal hallucination patterns."""
    normalized = text.lower().strip()
    return normalized in HALLUCINATION_TEXTS or len(normalized) < MIN_VALID_TEXT_LENGTH
