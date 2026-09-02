#!/usr/bin/env python3
"""Shared helpers for the Murmur integration suite.

The integration suite is discovered **separately** from the unit suite::

    python -m unittest discover -s tests/integration -p "test_*.py"

That separation is why this directory has no ``__init__.py``: ``unittest``
only recurses into a subdirectory of ``tests/`` when it is an importable
package, so ``discover -s tests`` walks straight past it and the unit suite
stays fast and runtime-free. Discovery puts *this* directory on ``sys.path``,
which is how the test modules import ``_support``; each one also puts the
repository root there so ``engines``/``cleanup``/``services`` import whatever
the working directory happens to be.

Two rules every module here obeys:

1. **Transcript text is never logged.** Not on success, not in an assertion
   message, not in a skip reason. :func:`log` exists so latencies, word counts
   and error rates have somewhere to go that isn't the transcript itself.
2. **A missing runtime is a skip, never a failure.** Every requirement (a
   built binary, an installed model, an importable backend, Apple Silicon) is
   probed at import time and turned into an explicit ``skipUnless`` reason, so
   a developer without the 3 GB of models still gets a green run that says
   exactly what it did not exercise.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: Repository root, so ``engines``, ``cleanup`` and ``services`` import from a
#: checkout rather than from whatever the caller's working directory is.
REPO_ROOT = _HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Where committed fixture clips live when someone has recorded them.
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "audio"

#: The languages the fixture set covers (see ``tests/fixtures/audio/README.md``).
FIXTURE_LANGUAGES: tuple[str, ...] = ("en", "fr", "nl", "de")

#: Prefix on every measurement line, so a CI log can be grepped for them.
LOG_PREFIX = "[integration]"


def log(message: str) -> None:
    """Print one measurement line to stderr.

    **Never** pass transcript text, a reference sentence or anything derived
    from user audio: latencies, durations, word counts and error rates only.
    """
    print(f"{LOG_PREFIX} {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Runtime probes
# ---------------------------------------------------------------------------


def is_macos() -> bool:
    """True on Darwin. ``say``, ``afconvert`` and Quartz all need it."""
    return sys.platform == "darwin"


def is_apple_silicon() -> bool:
    """True on an arm64 Mac, the only place MLX has wheels (decision D7)."""
    return is_macos() and platform.machine() == "arm64"


def module_available(name: str) -> bool:
    """True when ``name`` can be imported without actually importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve_optional_binary(env_var: str, vendor_relpath: str) -> Path | None:
    """A bundled helper binary, or None when it has not been built.

    Same order the engines use — the environment override first, then
    ``vendor/`` in the checkout — but a miss is None rather than an
    exception, because "not built here" is a skip and not an error.
    """
    override = os.environ.get(env_var)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    candidate = REPO_ROOT / vendor_relpath
    return candidate if candidate.is_file() else None


def whisper_server_binary() -> Path | None:
    """``whisper-server`` from ``MURMUR_WHISPER_SERVER`` or ``vendor/``."""
    return resolve_optional_binary(
        "MURMUR_WHISPER_SERVER", "vendor/whispercpp/whisper-server"
    )


def llama_server_binary() -> Path | None:
    """``llama-server`` from ``MURMUR_LLAMA_SERVER`` or ``vendor/``."""
    return resolve_optional_binary("MURMUR_LLAMA_SERVER", "vendor/llamacpp/llama-server")


def installed_model_path(model_id: str, catalog=None) -> Path | None:
    """The path an engine wants for ``model_id``, or None when it is not installed.

    Goes through :class:`engines.model_store.ModelStore` so the location, the
    single-file-vs-directory rule and the "installed" test are the store's,
    not a second copy of them here.
    """
    from engines.model_store import CATALOG, ModelStore, UnknownModelError

    store = ModelStore(catalog=CATALOG if catalog is None else catalog)
    try:
        if not store.is_installed(model_id):
            return None
        return store.engine_model_path(model_id)
    except UnknownModelError:
        return None


# ---------------------------------------------------------------------------
# Word error rate, borrowed from the bake-off harness
# ---------------------------------------------------------------------------

_BAKEOFF = None


def bakeoff():
    """The bake-off harness, loaded from its path (``scripts/`` is not a package).

    Reused rather than reimplemented: the suite must score transcripts the
    same way decision D1 did, normalisation included.
    """
    global _BAKEOFF
    if _BAKEOFF is None:
        path = REPO_ROOT / "scripts" / "tools" / "bakeoff.py"
        name = "murmur_bakeoff_integration"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None, f"cannot load {path}"
        module = importlib.util.module_from_spec(spec)
        # Registered before it is executed, because ``@dataclass`` resolves the
        # defining module out of ``sys.modules`` while the class body runs and
        # raises an unhelpful AttributeError when it is not there yet.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _BAKEOFF = module
    return _BAKEOFF


def word_error_rate(reference: str, hypothesis: str) -> float:
    """:func:`scripts.tools.bakeoff.word_error_rate`, by import not by copy."""
    return float(bakeoff().word_error_rate(reference, hypothesis))


# ---------------------------------------------------------------------------
# Fixture clips
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    """One clip and the sentence it says: enough to score a transcript."""

    language: str
    path: Path
    text: str


#: One dictation-style sentence per language, spoken by macOS ``say`` when no
#: recorded fixture is available.
#:
#: Two constraints shaped these. **No digits**: an engine is free to write
#: "five" or "5" and neither is wrong, but WER counts it as an error, which
#: would make the threshold measure formatting rather than recognition.
#: **Around twenty words**: at ~8 seconds they sit in the same range as the
#: recorded fixtures, and Voxtral Realtime returns an empty string for a clip
#: much shorter than that (measured on 2026-09-02 — a 3 s clip transcribes,
#: but there is no margin worth betting a test on).
SENTENCES: dict[str, str] = {
    "en": (
        "Hi team, this is a quick voice note about the Murmur project. "
        "Please push the staging build today so everyone can start testing it right away."
    ),
    "fr": (
        "Bonjour l'équipe, ceci est un petit message vocal à propos du projet Murmur. "
        "Merci de pousser la version de test aujourd'hui pour que chacun puisse "
        "commencer les essais."
    ),
    "nl": (
        "Hoi team, dit is een korte spraaknotitie over het Murmur project. "
        "Kunnen jullie de teststaging vandaag pushen, zodat iedereen meteen kan "
        "beginnen met testen?"
    ),
    "de": (
        "Hallo Team, das ist eine kurze Sprachnotiz zum Murmur Projekt. "
        "Bitte schiebt den Testbuild noch heute hoch, damit alle sofort mit dem "
        "Testen anfangen können."
    ),
}

#: Voices tried first per language, matched on the base name so
#: "Samantha (English (US))" counts as "Samantha", and accent-insensitively so
#: "Amelie" matches "Amélie".
PREFERRED_VOICES: dict[str, tuple[str, ...]] = {
    "en": ("Samantha", "Alex", "Ava", "Allison", "Karen", "Daniel", "Tessa"),
    "fr": ("Thomas", "Amelie", "Audrey", "Aurelie", "Jacques"),
    "nl": ("Xander", "Claire", "Ellen"),
    "de": ("Anna", "Markus", "Petra", "Helena"),
}

#: Sample rate, channel count and sample width every engine here requires.
SAMPLE_RATE_HZ = 16000


def installed_voices() -> list[tuple[str, str]]:
    """``(name, language)`` for every installed ``say`` voice; empty on failure.

    ``say -v '?'`` prints ``<name><spaces><locale>  # <example>``, and names
    contain spaces ("Bad News"), so the comment is cut first and the locale
    taken from the right-hand end of what is left.
    """
    try:
        completed = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    voices: list[tuple[str, str]] = []
    for line in completed.stdout.splitlines():
        left = line.split("#", 1)[0].rstrip()
        parts = left.rsplit(None, 1)
        if len(parts) != 2:
            continue
        name, locale = parts[0].strip(), parts[1].strip()
        language = locale.replace("-", "_").split("_")[0].lower()
        if name and len(language) == 2:
            voices.append((name, language))
    return voices


def _fold(text: str) -> str:
    """Case- and accent-insensitive key, so "Amelie" matches "Amélie"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def base_voice_name(name: str) -> str:
    """"Samantha (English (US))" -> "Samantha"."""
    return name.split(" (", 1)[0].strip()


def voice_for(language: str, voices: list[tuple[str, str]]) -> str | None:
    """The best installed ``say`` voice for ``language``, or None.

    Preferred names win. The fallback deliberately prefers a voice whose
    displayed name carries its locale in parentheses: on current macOS that is
    how the real localised voices are listed, while the legacy novelty voices
    ("Albert", "Zarvox", "Bad News") are listed as bare names — and picking one
    of those produces audio that Voxtral transcribes as an empty string, which
    looks exactly like an engine bug. Measured on 2026-09-02: "Albert" reads
    the English sentence, whisper.cpp copes, Voxtral returns "".
    """
    candidates = [name for name, voice_language in voices if voice_language == language]
    by_base = {_fold(base_voice_name(name)): name for name in reversed(candidates)}
    for preferred in PREFERRED_VOICES.get(language, ()):
        match = by_base.get(_fold(preferred))
        if match:
            return match
    for name in candidates:
        if "(" in name:
            return name
    return candidates[0] if candidates else None


def _say_clip(voice: str, sentence: str, destination: Path) -> bool:
    """Speak ``sentence`` to a 16 kHz mono 16-bit WAV. False when the tools fail.

    The same ``say`` + ``afconvert`` pair as
    ``scripts/tools/make_synthetic_fixtures.sh``. Synthetic speech is not real
    dictation and must never decide anything (see the fixtures README); here it
    only makes the suite self-sufficient on a Mac with no recordings.
    """
    aiff = destination.with_suffix(".aiff")
    try:
        subprocess.run(
            ["say", "-v", voice, "-o", str(aiff), sentence],
            check=True,
            capture_output=True,
            timeout=120,
        )
        subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{SAMPLE_RATE_HZ}",
                "-c",
                "1",
                str(aiff),
                str(destination),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        aiff.unlink(missing_ok=True)
    return destination.is_file()


def _clips_from_fixtures(languages: tuple[str, ...]) -> dict[str, Clip]:
    """First recorded clip per language from ``manifest.json``, when there is one."""
    manifest = FIXTURES_DIR / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        entries = bakeoff().load_manifest(manifest)
    except Exception:
        # A malformed or stale manifest is not this suite's problem to report;
        # falling through to `say` keeps the engines under test.
        return {}
    chosen: dict[str, Clip] = {}
    for entry in entries:
        if entry.language not in languages or entry.language in chosen:
            continue
        if Path(entry.path).is_file():
            chosen[entry.language] = Clip(entry.language, Path(entry.path), entry.text)
    return chosen


def clips_for(languages: tuple[str, ...], tmp_dir: str | Path) -> list[Clip]:
    """One clip per language: recorded fixtures first, macOS ``say`` otherwise.

    Returns only the languages it could actually produce, in ``languages``
    order — possibly empty, which callers turn into a skip. Generated clips
    land in ``tmp_dir`` and are the caller's to clean up.
    """
    chosen = _clips_from_fixtures(tuple(languages))
    missing = [language for language in languages if language not in chosen]
    if missing and is_macos():
        voices = installed_voices()
        for language in missing:
            sentence = SENTENCES.get(language)
            voice = voice_for(language, voices)
            if not sentence or not voice:
                continue
            destination = Path(tmp_dir) / f"{language}.wav"
            if _say_clip(voice, sentence, destination):
                chosen[language] = Clip(language, destination, sentence)
    return [chosen[language] for language in languages if language in chosen]


def wav_duration_s(path: Path) -> float:
    """Duration of a WAV in seconds, for the latency lines."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        return handle.getnframes() / float(rate) if rate else 0.0


def pcm_chunks(path: Path, chunk_ms: int) -> list[bytes]:
    """Split a 16 kHz mono 16-bit WAV into ``chunk_ms`` slices of raw PCM.

    Whole samples only, which is what
    :meth:`engines.voxtral_mlx.VoxtralMlxEngine._stream` requires; the last
    chunk is short when the clip does not divide evenly.
    """
    assert chunk_ms > 0, "chunk_ms must be positive"
    with wave.open(str(path), "rb") as handle:
        rate, channels, width = (
            handle.getframerate(),
            handle.getnchannels(),
            handle.getsampwidth(),
        )
        frames = handle.readframes(handle.getnframes())
    assert rate == SAMPLE_RATE_HZ, f"{path.name} is {rate} Hz, expected {SAMPLE_RATE_HZ}"
    assert channels == 1, f"{path.name} has {channels} channels, expected mono"
    assert width == 2, f"{path.name} is {width * 8}-bit, expected 16-bit"
    step = int(rate * chunk_ms / 1000) * width
    return [frames[start : start + step] for start in range(0, len(frames), step)]


__all__ = [
    "Clip",
    "FIXTURES_DIR",
    "FIXTURE_LANGUAGES",
    "REPO_ROOT",
    "SAMPLE_RATE_HZ",
    "SENTENCES",
    "clips_for",
    "installed_model_path",
    "is_apple_silicon",
    "is_macos",
    "llama_server_binary",
    "log",
    "module_available",
    "pcm_chunks",
    "wav_duration_s",
    "whisper_server_binary",
    "word_error_rate",
]
