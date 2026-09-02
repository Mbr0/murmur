#!/usr/bin/env python3
"""Voxtral Mini 4B Realtime (MLX, 4-bit) transcription engine.

Runtime contract (mlx-audio @ 0.5.1, commit 6b54ec6ecd99d0ad77dfa33dd129707e31bf051c,
read from github.com/Blaizzy/mlx-audio on 2026-09-02)
=====================================================================

Loading
    ``from mlx_audio.stt.utils import load``
    ``load(model_path: str | Path, lazy: bool = False, strict: bool = False,
    **kwargs) -> mlx.nn.Module``
    A ``Path`` (or a ``str`` naming an existing path) is loaded from disk; a ``str``
    that is not a path is treated as a HuggingFace repo id and downloaded. Murmur
    always passes a ``Path`` so loading never touches the network -- ``model_store``
    owns downloads. ``mlx_audio.stt.utils.MODEL_REMAPPING`` maps the ``config.json``
    ``model_type`` ``"voxtral_realtime"`` onto
    ``mlx_audio.stt.models.voxtral_realtime``.

Batch transcription
    ``Model.generate(audio, *, max_tokens=4096, temperature=0.0, verbose=False,
    stream=False, transcription_delay_ms=None, **kwargs)``
    ``audio`` may be ``str | Path | mx.array | list[mx.array] | np.ndarray``. A path
    is read and resampled to 16 kHz mono float32 by ``Model._load_audio``; arrays are
    flattened and assumed to be 16 kHz float32 already. ``stream=True`` turns the
    same call into a generator of ``str`` deltas over a *complete* buffer -- that is
    not live input, so this engine does not use it.

Result object
    ``mlx_audio.stt.models.base.STTOutput(text, segments=None, language=None,
    prompt_tokens, generation_tokens, total_tokens, prompt_tps, generation_tps,
    total_time)``. Voxtral Realtime fills only ``text`` and the token/timing
    counters: ``segments`` and ``language`` stay ``None``, and ``total_time`` is
    wall-clock inference time, NOT audio duration. ``duration_s`` is therefore taken
    from the WAV header here.

Live streaming
    ``Model.create_streaming_session(*, max_tokens=4096, temperature=0.0,
    transcription_delay_ms=None) -> VoxtralStreamingSession`` exposing
      ``feed(samples: np.ndarray)``                  16 kHz mono float32; cheap,
                                                     thread-safe, no MLX work
      ``close()``                                    end-of-audio signal
      ``step(*, max_decode_tokens=4) -> list[str]``  one bounded unit of MLX work,
                                                     returns the deltas it emitted
      ``done`` (property)                            True once the utterance ended
      ``input_sample_rate``                          int, 16000
    ``Model.generate_streaming(source: StreamingAudioSource, ...)`` is a thin wrapper
    that owns a blocking read loop; this engine uses the session API directly so the
    caller keeps control of its own audio thread.
    NOTE: the ``create_streaming_session`` docstring mentions ``finalize_step()``;
    no such method exists at this commit. Do not rely on it.

Audio format
    ``SAMPLE_RATE = 16000``; ``RAW_AUDIO_LENGTH_PER_TOK = 1280`` samples per audio
    token (80 ms). Default ``transcription_delay_ms`` is 480, tunable 240..2400 as a
    latency/accuracy dial.

Language and biasing -- THE GAP
    Voxtral Realtime exposes NO language parameter and NO prompt/context/hotword
    parameter. Its decoder prompt is fixed: ``[BOS] + [STREAMING_PAD] * (n_left_pad +
    n_delay)``. ``generate()`` swallows unknown keyword arguments through ``**kwargs``,
    so passing ``initial_prompt=...`` would be a silent no-op that merely *looks* like
    biasing. This engine refuses to fake it and refuses to guess: ``supports_hints``
    is False, nothing is sent, and a pass that was given hints reports
    ``Transcript.hints_applied=False`` so the UI can say they were ignored rather than
    pretend. Language is likewise accepted and echoed, never sent to the model.

Re-check when upgrading mlx-audio
    1. a real ``language`` argument on ``Model.generate``
    2. a real biasing argument (``initial_prompt`` / ``prompt`` / ``system_prompt`` /
       ``hotwords`` / ``context``) on ``Model.generate``
    3. when upstream adds a biasing/language parameter, wire it explicitly and flip
       ``supports_hints`` -- never re-introduce detection by signature inspection,
       which can mis-wire an unrelated future parameter
    4. ``STTOutput.segments`` becoming populated (word/segment timestamps)
    5. the ``VoxtralStreamingSession.step()`` signature and ``done`` semantics
    6. ``mlx_audio.stt.utils.load`` still accepting a local ``Path``

Nothing from ``mlx`` or ``mlx_audio`` is imported at module import time: Murmur must
run on machines where MLX is absent (decision D7 -- Intel Macs use whisper.cpp).
"""

from __future__ import annotations

import platform
import wave
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

from engines.base import (
    LANGUAGE_AUTO,
    Engine,
    EngineError,
    EngineInfo,
    EngineUnavailableError,
    Hints,
    Partial,
    Segment,
    Transcript,
)
from services.model_profile_service import CHIP_APPLE_SILICON, detect_chip

#: Registry id; must match the module name so ``create_engine`` finds it.
ENGINE_ID = "voxtral_mlx"

ENGINE_NAME = "Voxtral Mini 4B Realtime (MLX, 4-bit)"

#: The only WAV shape this engine accepts. Resampling is out of scope: a mismatch is
#: a capture-settings bug in the caller, not something to paper over here.
SAMPLE_RATE_HZ = 16000
CHANNEL_COUNT = 1
SAMPLE_WIDTH_BYTES = 2

#: The 13 languages of the Voxtral Mini 4B Realtime 2602 model card, plus the auto
#: sentinel. The model has no language argument, so ``auto`` is the only mode it can
#: actually be run in; the rest describe what it understands, for the language picker.
LANGUAGES: tuple[str, ...] = (
    LANGUAGE_AUTO,
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "ko",
    "nl",
    "pt",
    "ru",
    "zh",
)

#: Safety net so a runtime that never reports ``done`` fails loudly instead of hanging.
MAX_DRAIN_STEPS = 100_000

#: Deltas per streaming step; mirrors ``Model.generate_streaming`` in mlx-audio.
STREAM_DECODE_TOKENS = 16


class VoxtralRuntime(Protocol):
    """The seam between this engine and mlx-audio.

    The engine never imports MLX itself; every call that needs it goes through an
    object shaped like this. Tests inject a fake; production gets
    :class:`_MlxAudioRuntime`.
    """

    def import_backend(self) -> None:
        """Import mlx/mlx-audio. Raises :class:`ImportError` when unavailable."""

    def load_model(self, model_path: Path) -> Any:
        """Load the model from a local directory and return the opaque handle."""

    def transcribe(self, model: Any, wav_path: Path, *, language: str | None, hints: Hints | None) -> Any:
        """Transcribe a validated 16 kHz mono 16-bit WAV. Returns an ``STTOutput``."""

    def create_session(self, model: Any) -> "VoxtralStreamSession":
        """Open a live streaming session over ``model``."""

    def clear_cache(self) -> None:
        """Release runtime caches after unload. Must be safe when never loaded."""


class VoxtralStreamSession(Protocol):
    """A live streaming session; PCM in, text deltas out."""

    @property
    def done(self) -> bool:
        """True once the utterance has been fully decoded."""

    def feed_pcm(self, chunk: bytes) -> None:
        """Queue one chunk of 16 kHz mono signed 16-bit little-endian PCM."""

    def close(self) -> None:
        """Signal end of audio."""

    def step(self) -> list[str]:
        """Run one bounded unit of work; return the text deltas it produced."""


class _MlxAudioSession:
    """Adapts :class:`VoxtralStreamSession` onto mlx-audio's streaming session."""

    def __init__(self, session: Any, numpy_module: Any) -> None:
        assert session is not None, "session is required"
        assert numpy_module is not None, "numpy module is required"
        self._session = session
        self._np = numpy_module

    @property
    def done(self) -> bool:
        return bool(self._session.done)

    def feed_pcm(self, chunk: bytes) -> None:
        samples = self._np.frombuffer(chunk, dtype="<i2").astype(self._np.float32) / 32768.0
        self._session.feed(samples)

    def close(self) -> None:
        self._session.close()

    def step(self) -> list[str]:
        return list(self._session.step(max_decode_tokens=STREAM_DECODE_TOKENS))


class _MlxAudioRuntime:
    """Default runtime: a thin wrapper over ``mlx_audio.stt``.

    Everything MLX-shaped is imported inside :meth:`import_backend`, never at module
    import time.
    """

    def __init__(self) -> None:
        self._load_stt_model: Any = None
        self._mx: Any = None
        self._np: Any = None

    def import_backend(self) -> None:
        import mlx.core as mx
        import numpy as np
        from mlx_audio.stt.utils import load as load_stt_model

        self._mx = mx
        self._np = np
        self._load_stt_model = load_stt_model

    def load_model(self, model_path: Path) -> Any:
        assert self._load_stt_model is not None, "import_backend() must run before load_model()"
        return self._load_stt_model(Path(model_path))

    def transcribe(
        self,
        model: Any,
        wav_path: Path,
        *,
        language: str | None,
        hints: Hints | None,
    ) -> Any:
        """Run ``Model.generate``.

        ``language`` and ``hints`` are accepted and ignored: Voxtral Realtime detects
        the language itself and has no biasing argument at all (see the module
        docstring). Forwarding either into ``**kwargs`` would be a silent no-op.
        """
        return model.generate(str(wav_path), temperature=0.0)

    def create_session(self, model: Any) -> _MlxAudioSession:
        assert self._np is not None, "import_backend() must run before create_session()"
        return _MlxAudioSession(model.create_streaming_session(temperature=0.0), self._np)

    def clear_cache(self) -> None:
        if self._mx is not None:
            self._mx.clear_cache()


class VoxtralMlxEngine(Engine):
    """Voxtral Mini 4B Realtime through mlx-audio, on Apple Silicon only.

    ``model_path`` is a directory the caller already resolved (``engines.model_store``
    owns downloading and verifying it); this engine never fetches anything.
    """

    supports_streaming = True
    #: Voxtral Realtime has no biasing parameter at all (see the module docstring).
    supports_hints = False

    def __init__(self, model_path: Path, runtime: VoxtralRuntime | None = None) -> None:
        assert model_path is not None, "model_path is required"
        self._model_path = Path(model_path)
        self._runtime: VoxtralRuntime = runtime if runtime is not None else _MlxAudioRuntime()
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> Path:
        return self._model_path

    def load(self) -> None:
        if self._model is not None:
            return
        if detect_chip() != CHIP_APPLE_SILICON:
            raise EngineUnavailableError(
                f"{ENGINE_NAME} needs Apple Silicon; this machine reports "
                f"platform.machine() == {platform.machine()!r}. "
                "Use the whisper.cpp engine instead."
            )
        if not self._model_path.exists():
            raise EngineUnavailableError(
                f"{ENGINE_NAME} model directory not found at {self._model_path}. "
                "Download the model before loading the engine."
            )
        try:
            self._runtime.import_backend()
        except ImportError as exc:
            raise EngineUnavailableError(
                f"{ENGINE_NAME} needs the MLX runtime, which is not installed: {exc}. "
                "Install it with: pip install 'mlx-audio[stt]' mlx"
            ) from exc
        model = self._runtime.load_model(self._model_path)
        assert model is not None, "runtime.load_model() returned None"
        self._model = model

    def unload(self) -> None:
        self._model = None
        self._runtime.clear_cache()

    def info(self) -> EngineInfo:
        return EngineInfo(
            id=ENGINE_ID,
            name=ENGINE_NAME,
            model_id=self._model_path.name,
            size_bytes=_path_size_bytes(self._model_path),
            languages=LANGUAGES,
            supports_streaming=self.supports_streaming,
            supports_hints=self.supports_hints,
            requires_apple_silicon=True,
        )

    def _transcribe(
        self,
        wav_path: Path,
        language: str | None,
        hints: Hints | None,
        long_form: bool,
    ) -> Transcript:
        """``long_form`` is accepted and ignored: the decoder has no such dial."""
        duration_s = _read_wav_duration(Path(wav_path))
        # Hints reach no biasing parameter, so say so instead of implying they did.
        hints_applied = False if (hints is not None and hints.as_prompt_text()) else None
        result = self._runtime.transcribe(self._model, Path(wav_path), language=language, hints=hints)
        return Transcript(
            text=(getattr(result, "text", "") or "").strip(),
            language=_resolve_language(result, language),
            duration_s=duration_s,
            segments=_read_segments(result),
            engine_id=ENGINE_ID,
            hints_applied=hints_applied,
        )

    def _stream(
        self,
        chunks: Iterable[bytes],
        language: str | None = None,
        hints: Hints | None = None,
    ) -> Iterator[Partial]:
        """``language`` and ``hints`` are accepted and ignored.

        The realtime session takes neither: there is no language flag and no
        initial-prompt equivalent to bias it with. They are in the signature so
        the live and batch paths describe an utterance identically, and because
        the app's rule for when a live decode may stand in for the batch one
        (:meth:`Engine.stream`) depends on the caller having passed the
        language, not on this engine having used it.
        """
        session = self._runtime.create_session(self._model)
        assert session is not None, "runtime.create_session() returned None"
        text = ""
        # close() is the end-of-audio signal AND the session's release. A consumer
        # that stops early (break) or a chunk that raises must still reach it, so it
        # lives in a finally; the contract says close() is idempotent.
        try:
            for chunk in chunks:
                if not isinstance(chunk, (bytes, bytearray)):
                    raise EngineError(
                        f"{ENGINE_NAME} streams 16-bit PCM bytes; got {type(chunk).__name__}"
                    )
                if not chunk:
                    continue
                if len(chunk) % SAMPLE_WIDTH_BYTES:
                    raise EngineError(
                        f"{ENGINE_NAME} streams 16-bit PCM; chunk of {len(chunk)} bytes is not "
                        "a whole number of samples"
                    )
                session.feed_pcm(bytes(chunk))
                for delta in session.step():
                    text += delta
                    yield Partial(text=text, is_final=False)
            session.close()
            drains = 0
            while not session.done:
                drains += 1
                if drains > MAX_DRAIN_STEPS:
                    raise EngineError(
                        f"{ENGINE_NAME} streaming session never finished after "
                        f"{MAX_DRAIN_STEPS} steps past end-of-audio"
                    )
                for delta in session.step():
                    text += delta
                    yield Partial(text=text, is_final=False)
            yield Partial(text=text.strip(), is_final=True)
        finally:
            session.close()


def _resolve_language(result: Any, requested: str | None) -> str | None:
    """Language the runtime reported, else the requested one, else unknown.

    Voxtral Realtime never reports one, so in practice this echoes the request.
    """
    detected = getattr(result, "language", None)
    if detected:
        return str(detected)
    if requested and requested != LANGUAGE_AUTO:
        return requested
    return None


def _read_segments(result: Any) -> tuple[Segment, ...]:
    """Map runtime segments onto :class:`Segment`; empty when it reports none."""
    raw = getattr(result, "segments", None)
    if not raw:
        return ()
    segments: list[Segment] = []
    for item in raw:
        if isinstance(item, Segment):
            segments.append(item)
            continue
        if isinstance(item, dict):
            segments.append(
                Segment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=str(item.get("text", "")),
                )
            )
            continue
        raise EngineError(f"{ENGINE_NAME} runtime returned an unreadable segment: {type(item).__name__}")
    return tuple(segments)


def _read_wav_duration(wav_path: Path) -> float:
    """Validate the WAV is 16 kHz mono 16-bit and return its duration in seconds.

    Resampling is deliberately out of scope: a mismatch means the capture settings
    are wrong, and the message says exactly what was found so the caller can fix it.
    """
    try:
        with wave.open(str(wav_path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
    except FileNotFoundError as exc:
        raise EngineError(f"{ENGINE_NAME} cannot read {wav_path}: file not found") from exc
    except (wave.Error, OSError, EOFError) as exc:
        raise EngineError(f"{ENGINE_NAME} cannot read {wav_path} as a WAV file: {exc}") from exc
    if channels != CHANNEL_COUNT or width != SAMPLE_WIDTH_BYTES or rate != SAMPLE_RATE_HZ:
        raise EngineError(
            f"{ENGINE_NAME} needs 16000 Hz mono 16-bit WAV; {wav_path} is "
            f"{rate} Hz, {channels} channel(s), {width * 8}-bit. Fix the capture settings."
        )
    if rate <= 0:
        raise EngineError(f"{ENGINE_NAME} cannot read {wav_path}: frame rate is {rate}")
    return frames / float(rate)


def _path_size_bytes(path: Path) -> int:
    """Total size of the model on disk; 0 when it has not been downloaded."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


#: Discovered by ``engines.create_engine("voxtral_mlx")``.
ENGINE_CLASS = VoxtralMlxEngine

__all__ = [
    "ENGINE_CLASS",
    "ENGINE_ID",
    "ENGINE_NAME",
    "LANGUAGES",
    "MAX_DRAIN_STEPS",
    "SAMPLE_RATE_HZ",
    "VoxtralMlxEngine",
    "VoxtralRuntime",
    "VoxtralStreamSession",
]
