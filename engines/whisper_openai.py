#!/usr/bin/env python3
"""Adapter that keeps the existing ``openai-whisper`` path behind :class:`Engine`.

This engine exists so Wave 0 can route the app through the engine contract
without changing what users get. It is replaced once the bake-off confirms D1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from engines.base import (
    LANGUAGE_AUTO,
    Engine,
    EngineInfo,
    EngineUnavailableError,
    Hints,
    Segment,
    Transcript,
)
from services.transcription_service import (
    extract_text,
    load_whisper_with_fallback,
    resolve_whisper_device,
    transcribe_audio,
)

logger = logging.getLogger(__name__)

ENGINE_ID = "whisper_openai"
ENGINE_NAME = "OpenAI Whisper (PyTorch)"


def _resolve_language(language: str | None) -> str | None:
    """Whisper detects the language itself when it is given ``None``."""
    if language is None or language == LANGUAGE_AUTO:
        return None
    return language


def _build_segments(result: dict[str, Any]) -> tuple[Segment, ...]:
    """Convert Whisper's segment dicts into contract segments."""
    raw = result.get("segments") or ()
    return tuple(
        Segment(
            start=float(item.get("start", 0.0)),
            end=float(item.get("end", 0.0)),
            text=str(item.get("text", "")).strip(),
        )
        for item in raw
    )


def _build_transcript(result: dict[str, Any], hints_applied: bool | None) -> Transcript:
    """Map a Whisper result payload onto :class:`Transcript`."""
    segments = _build_segments(result)
    duration_s = max((segment.end for segment in segments), default=None)
    return Transcript(
        text=extract_text(result),
        language=result.get("language"),
        duration_s=duration_s,
        segments=segments,
        engine_id=ENGINE_ID,
        hints_applied=hints_applied,
    )


class WhisperOpenAIEngine(Engine):
    """Wraps the in-process ``openai-whisper`` model the app shipped with."""

    supports_streaming = False
    supports_hints = True

    def __init__(
        self,
        model_name: str,
        whisper_module: Any | None = None,
        load_fn: Any | None = None,
    ) -> None:
        assert model_name, "model_name is required"
        self._model_name = model_name
        self._whisper_module = whisper_module
        self._load_fn = load_fn
        self._model: Any | None = None
        #: Device and precision the model actually loaded on; read by the app.
        self.device = "cpu"
        self.fp16 = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        if self._model is not None:
            return
        load_fn = self._load_fn or self._default_load_fn()
        device = resolve_whisper_device()
        logger.info("Loading Whisper model %s on device=%s", self._model_name, device)
        self._model, self.device, self.fp16 = load_whisper_with_fallback(load_fn, device)
        logger.info("Whisper model loaded on device=%s fp16=%s", self.device, self.fp16)

    def unload(self) -> None:
        self._model = None
        _empty_mps_cache()

    def info(self) -> EngineInfo:
        return EngineInfo(
            id=ENGINE_ID,
            name=ENGINE_NAME,
            model_id=self._model_name,
            size_bytes=0,
            languages=(LANGUAGE_AUTO,),
            supports_streaming=self.supports_streaming,
            supports_hints=self.supports_hints,
            requires_apple_silicon=False,
        )

    def runtime_summary(self) -> str:
        """What the model actually loaded onto; meaningful only after :meth:`load`."""
        return f"device={self.device} fp16={self.fp16}"

    def _transcribe(
        self,
        wav_path: Path,
        language: str | None,
        hints: Hints | None,
        long_form: bool,
    ) -> Transcript:
        initial_prompt = hints.as_prompt_text() if hints is not None else None
        result = transcribe_audio(
            self._model,
            str(wav_path),
            # A whole-file import may condition on the text it already produced;
            # dictation must not, so one bad window cannot poison the next.
            condition_on_previous_text=long_form,
            device=self.device,
            fp16=self.fp16,
            language=_resolve_language(language),
            initial_prompt=initial_prompt,
        )
        return _build_transcript(result, hints_applied=True if initial_prompt else None)

    def _default_load_fn(self):
        module = self._whisper_module or _import_whisper()
        return lambda device: module.load_model(self._model_name, device=device)


def _import_whisper():
    """Import ``whisper`` lazily; the runtime may be absent on this machine."""
    try:
        import whisper
    except ImportError as exc:
        raise EngineUnavailableError(
            f"openai-whisper is not available on this machine: {exc}"
        ) from exc
    return whisper


def _empty_mps_cache() -> None:
    """Best-effort MPS cache release. Only a missing torch is tolerated."""
    try:
        import torch
    except ImportError:
        return
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        return
    empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
    if empty_cache is None:
        return
    empty_cache()


ENGINE_CLASS = WhisperOpenAIEngine
