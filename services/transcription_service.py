#!/usr/bin/env python3
"""Local transcription helpers for Whisper model calls."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def resolve_whisper_device(torch_module: Any | None = None) -> str:
    """Return a safe Whisper device: MPS when available, otherwise CPU."""
    if torch_module is None:
        import torch as torch_module

    assert torch_module is not None, "torch module is required"
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Whisper device selected: %s", device)
    return device


def resolve_fp16(device: str) -> bool:
    """Enable fp16 only on CUDA. Whisper+MPS fp16 is unsafe."""
    assert device, "device is required"
    return device == "cuda"


def load_whisper_with_fallback(
    load_fn: Callable[[str], Any],
    device: str,
) -> tuple[Any, str, bool]:
    """Load Whisper; if MPS load fails, retry once on CPU with fp16=False."""
    assert device, "device is required"
    assert load_fn is not None, "load_fn is required"
    fp16 = resolve_fp16(device)
    try:
        model = load_fn(device)
        return model, device, fp16
    except Exception:
        if device != "mps":
            raise
        logger.warning(
            "MPS Whisper load failed; falling back to CPU (fp16=False)",
            exc_info=True,
        )
        model = load_fn("cpu")
        return model, "cpu", False


def _resolve_fp16_arg(device: str | None, fp16: bool | None) -> bool:
    if fp16 is not None:
        return fp16
    if device is None:
        return False
    return resolve_fp16(device)


def _prompt_kwargs(initial_prompt: str | None) -> dict[str, Any]:
    """Only forward ``initial_prompt`` when the caller actually supplied one."""
    if initial_prompt is None:
        return {}
    return {"initial_prompt": initial_prompt}


def transcribe_audio(
    model: Any,
    audio_path: str,
    *,
    condition_on_previous_text: bool = False,
    no_speech_threshold: float = 0.6,
    device: str | None = None,
    fp16: bool | None = None,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    """Run a local Whisper transcription pass on an audio file path."""
    assert audio_path, "audio_path is required"
    use_fp16 = _resolve_fp16_arg(device, fp16)
    return model.transcribe(
        audio_path,
        fp16=use_fp16,
        language=language,
        condition_on_previous_text=condition_on_previous_text,
        no_speech_threshold=no_speech_threshold,
        **_prompt_kwargs(initial_prompt),
    )


def transcribe_audio_file(
    model: Any,
    audio_path: str,
    *,
    device: str | None = None,
    fp16: bool | None = None,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    """Transcribe an uploaded file with default Whisper settings.

    Unused in the app: the production file-import path is the engine adapter, which
    calls :func:`transcribe_audio` with ``condition_on_previous_text=long_form``.
    """
    assert audio_path, "audio_path is required"
    use_fp16 = _resolve_fp16_arg(device, fp16)
    return model.transcribe(
        audio_path,
        fp16=use_fp16,
        language=language,
        **_prompt_kwargs(initial_prompt),
    )


def extract_text(result: dict[str, Any]) -> str:
    """Return normalized transcription text from a Whisper result payload."""
    return result.get("text", "").strip()
