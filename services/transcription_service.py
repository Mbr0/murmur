#!/usr/bin/env python3
"""Local transcription helpers for Whisper model calls."""

from __future__ import annotations

from typing import Any


def transcribe_audio(
    model: Any,
    audio_path: str,
    *,
    condition_on_previous_text: bool = False,
    no_speech_threshold: float = 0.6,
) -> dict[str, Any]:
    """Run a local Whisper transcription pass on an audio file path."""
    assert audio_path, "audio_path is required"
    return model.transcribe(
        audio_path,
        fp16=False,
        language=None,
        condition_on_previous_text=condition_on_previous_text,
        no_speech_threshold=no_speech_threshold,
    )


def transcribe_audio_file(model: Any, audio_path: str) -> dict[str, Any]:
    """Transcribe an uploaded file with default Whisper settings."""
    return model.transcribe(audio_path, fp16=False, language=None)


def extract_text(result: dict[str, Any]) -> str:
    """Return normalized transcription text from a Whisper result payload."""
    return result.get("text", "").strip()
