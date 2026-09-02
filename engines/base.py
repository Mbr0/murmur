#!/usr/bin/env python3
"""Engine contract every Murmur transcription backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

#: Sentinel used in :attr:`EngineInfo.languages` and as a ``language`` value
#: when the engine should detect the language itself.
LANGUAGE_AUTO = "auto"

#: Languages a multilingual Whisper model transcribes well enough to offer in the
#: picker, ``auto`` first. Whisper itself claims ninety-nine, but a picker that
#: long is unusable and most of its tail is unusable too; this is the subset
#: Murmur stands behind. Shared so :mod:`engines.whispercpp` and anything else
#: describing a Whisper model report the same list.
WHISPER_LANGUAGES: tuple[str, ...] = (
    LANGUAGE_AUTO,
    "en",
    "fr",
    "nl",
    "de",
    "es",
    "it",
    "pt",
    "pl",
    "ru",
    "uk",
    "tr",
    "sv",
    "da",
    "no",
    "fi",
    "cs",
    "ro",
    "hu",
    "el",
    "ja",
    "zh",
    "ko",
    "ar",
    "hi",
    "id",
    "vi",
)


#: whisper.cpp's ``verbose_json`` reports ``language`` as one of whisper's own
#: English language names (e.g. ``"english"``, ``"french"``), not the ISO
#: 639-1 codes the rest of Murmur (and the Voxtral engine) use. Maps the name
#: for every code in :data:`WHISPER_LANGUAGES` to that code so
#: :func:`normalize_language_code` can undo it. Verified against upstream
#: whisper's ``tokenizer.py`` ``LANGUAGES`` table.
_LANGUAGE_NAME_TO_CODE: dict[str, str] = {
    "english": "en",
    "french": "fr",
    "dutch": "nl",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "polish": "pl",
    "russian": "ru",
    "ukrainian": "uk",
    "turkish": "tr",
    "swedish": "sv",
    "danish": "da",
    "norwegian": "no",
    "finnish": "fi",
    "czech": "cs",
    "romanian": "ro",
    "hungarian": "hu",
    "greek": "el",
    "japanese": "ja",
    "chinese": "zh",
    "korean": "ko",
    "arabic": "ar",
    "hindi": "hi",
    "indonesian": "id",
    "vietnamese": "vi",
}


def normalize_language_code(value: str | None) -> str | None:
    """Fold an engine's raw ``language`` into the ISO 639-1 code the app uses.

    ``None``, ``""`` and :data:`LANGUAGE_AUTO` all mean "no language" and
    return None. Everything else is lower-cased; a whisper language *name*
    (as whisper.cpp's ``verbose_json`` reports it, e.g. ``"english"``) is
    mapped to its code via :data:`_LANGUAGE_NAME_TO_CODE`. An ISO code, or
    anything else not in that table, passes through lower-cased and
    unchanged: this never raises.
    """
    if value is None:
        return None
    text = value.strip().lower()
    if not text or text == LANGUAGE_AUTO:
        return None
    return _LANGUAGE_NAME_TO_CODE.get(text, text)


class EngineError(Exception):
    """Base class for every engine failure."""


class EngineNotLoadedError(EngineError):
    """Raised when work is requested from an engine that has not been loaded."""


class EngineCapabilityError(EngineError):
    """Raised when an engine is asked for a capability it does not advertise."""


class EngineUnavailableError(EngineError):
    """Raised when the runtime an engine needs is missing on this machine."""


@dataclass(frozen=True)
class Segment:
    """One timed slice of a transcript, in seconds from the start of the clip."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """Result of a completed transcription pass."""

    text: str
    language: str | None
    duration_s: float | None
    segments: tuple[Segment, ...]
    engine_id: str
    #: True when the engine really biased the decode with the caller's
    #: :class:`Hints`, False when it was given hints it could not use, and None
    #: when there was nothing to apply. The UI can then say so rather than guess.
    hints_applied: bool | None = None


@dataclass(frozen=True)
class Partial:
    """Intermediate result emitted while streaming.

    ``text`` is the full text so far, not a delta: every partial carries the
    whole utterance as the engine understands it at that moment, so a consumer
    **replaces** what it holds rather than appending. A later partial may even
    be shorter than the one before it, because the decoder is free to revise a
    word it has already emitted.
    """

    text: str
    is_final: bool
    start_s: float | None = None
    end_s: float | None = None


@dataclass(frozen=True)
class Hints:
    """Caller-supplied bias for a transcription pass."""

    vocabulary: tuple[str, ...] = ()
    initial_prompt: str | None = None

    def as_prompt_text(self) -> str | None:
        """Fold these hints into the one prompt string every engine wants.

        The prompt comes first, then the vocabulary joined by ``", "``, the two
        halves separated by a single space. Blank terms are dropped; None when
        nothing at all remains.
        """
        pieces: list[str] = []
        prompt = (self.initial_prompt or "").strip()
        if prompt:
            pieces.append(prompt)
        terms = ", ".join(term.strip() for term in self.vocabulary if term and term.strip())
        if terms:
            pieces.append(terms)
        return " ".join(pieces) or None


@dataclass(frozen=True)
class EngineInfo:
    """Static description of an engine and the model it is configured with."""

    id: str
    name: str
    model_id: str
    size_bytes: int
    languages: tuple[str, ...]
    supports_streaming: bool
    supports_hints: bool
    requires_apple_silicon: bool


class Engine(ABC):
    """Abstract transcription engine.

    Subclasses implement :meth:`load`, :meth:`unload`, :meth:`info`,
    :attr:`is_loaded` and :meth:`_transcribe`; streaming engines set
    ``supports_streaming = True`` and implement :meth:`_stream`. The public
    :meth:`transcribe` and :meth:`stream` are template methods that enforce the
    load state and the advertised capabilities before delegating.
    """

    #: Overridden as a class attribute (or property) by capable engines.
    supports_streaming: bool = False
    supports_hints: bool = False

    @abstractmethod
    def load(self) -> None:
        """Bring the engine to a usable state. Must be idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model and any child process. Must be idempotent."""

    @abstractmethod
    def info(self) -> EngineInfo:
        """Return the static description of this engine and its model."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True once :meth:`load` has completed and no unload has followed."""

    def transcribe(
        self,
        wav_path: Path,
        language: str | None = None,
        hints: Hints | None = None,
        long_form: bool = False,
    ) -> Transcript:
        """Transcribe a WAV file. Raises :class:`EngineNotLoadedError` if unloaded.

        ``long_form=True`` means a multi-window import of a whole file, where the
        decoder may condition on the text it produced for earlier windows. The
        default, False, is dictation: every utterance stands on its own, so a
        stray hallucination cannot propagate.
        """
        assert wav_path is not None, "wav_path is required"
        self._ensure_loaded()
        return self._transcribe(wav_path, language=language, hints=hints, long_form=long_form)

    def stream(
        self,
        chunks: Iterable[bytes],
        language: str | None = None,
        hints: Hints | None = None,
    ) -> Iterator[Partial]:
        """Yield partials for a live audio stream.

        ``language`` and ``hints`` mean what they mean in :meth:`transcribe`, so
        one utterance is described the same way whichever path decodes it and no
        caller has to know which one ran.

        An engine may ignore either: Voxtral Realtime, the only streaming engine
        Murmur ships, has no biasing dial at all and says so in its own
        :meth:`_stream`. **That is why a live decode is not automatically the
        final transcript.** The app treats the streamed text as the answer only
        when the configured language is ``auto`` — see ``murmur.language_is_auto``
        — and otherwise keeps the partials for the pill and takes the batch
        result, which the engine *can* honour. That rule lives in those two
        places and nowhere else.

        Raises :class:`EngineCapabilityError` when ``supports_streaming`` is False.
        """
        if not self.supports_streaming:
            raise EngineCapabilityError(f"{type(self).__name__} does not support streaming")
        self._ensure_loaded()
        return self._stream(chunks, language=language, hints=hints)

    def runtime_summary(self) -> str:
        """One line describing what the engine actually loaded onto, for the log.

        Empty when the engine has nothing worth saying; the app logs it verbatim
        rather than reaching for engine-specific attributes it cannot know about.
        """
        return ""

    def _ensure_loaded(self) -> None:
        """Fail fast when the engine is not loaded."""
        if not self.is_loaded:
            raise EngineNotLoadedError(f"{type(self).__name__} is not loaded; call load() first")

    @abstractmethod
    def _transcribe(
        self,
        wav_path: Path,
        language: str | None,
        hints: Hints | None,
        long_form: bool,
    ) -> Transcript:
        """Engine-specific transcription; called only when loaded."""

    def _stream(
        self,
        chunks: Iterable[bytes],
        language: str | None = None,
        hints: Hints | None = None,
    ) -> Iterator[Partial]:
        """Engine-specific streaming; only reached when ``supports_streaming``.

        Implementations that cannot honour ``language`` or ``hints`` must still
        accept them and say in their docstring that they are ignored, rather
        than dropping them from the signature: a silent ``TypeError`` at the
        first live utterance is a worse answer than a documented no-op.
        """
        raise NotImplementedError(
            f"{type(self).__name__} advertises streaming but does not implement _stream()"
        )
