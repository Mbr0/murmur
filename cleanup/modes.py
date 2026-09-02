#!/usr/bin/env python3
"""Cleanup modes and tones: prompts and UI copy stored as data, not code.

Five modes (Dictation, Message, Mail, Notes, Code) and four tones (Neutral,
Warm, Formal, Terse) live under ``cleanup/prompts/``: one ``<mode>.txt``
system-prompt template per mode, plus ``modes.json`` and ``tones.json``
manifests carrying the identity, UI copy, and behaviour flags a copywriter
or product person would want to edit without touching Python.

This module only wires that data into a small, testable API: frozen
dataclasses for ``Mode`` and ``Tone``, the ``MODES``/``TONES`` registries
built from the manifests at import time, prompt loading that fails loudly
when a file is missing rather than falling back to something silent, and
system-prompt rendering that fills in the three placeholders every template
shares: ``{tone_instruction}``, ``{language}``, and ``{vocabulary}``.

Dictation is a passthrough (``Mode.is_passthrough`` is True): the cleanup
pipeline that owns the actual LLM call — not this module — skips calling it
for Dictation entirely. ``dictation.txt`` still exists and still renders, so
every mode/tone pair is testable the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MODES_MANIFEST = PROMPTS_DIR / "modes.json"
TONES_MANIFEST = PROMPTS_DIR / "tones.json"

#: Config keys read by :func:`mode_from_config` / :func:`tone_from_config`.
CONFIG_MODE_KEY = "cleanup_mode"
CONFIG_TONE_KEY = "cleanup_tone"

DEFAULT_MODE_ID = "dictation"
DEFAULT_TONE_ID = "neutral"

#: Canonical ordering for golden tests and any UI listing.
MODE_IDS: tuple[str, ...] = ("dictation", "message", "mail", "notes", "code")
TONE_IDS: tuple[str, ...] = ("neutral", "warm", "formal", "terse")


class ModeConfigError(ValueError):
    """Base class for an unknown mode or tone id."""


class UnknownModeError(ModeConfigError):
    """Raised when a mode id has no entry in :data:`MODES`."""


class UnknownToneError(ModeConfigError):
    """Raised when a tone id has no entry in :data:`TONES`."""


class PromptFileMissingError(RuntimeError):
    """Raised when a mode's prompt template, or a manifest, cannot be read."""


@dataclass(frozen=True)
class Mode:
    """One cleanup mode: identity, UI copy, and behaviour, all manifest-driven."""

    id: str
    display_name: str
    description: str
    is_passthrough: bool
    default_tone_id: str


@dataclass(frozen=True)
class Tone:
    """One tone: identity, UI copy, and the one-sentence instruction folded into prompts."""

    id: str
    display_name: str
    description: str
    instruction: str


def _load_json_manifest(path: Path) -> dict:
    """Read and parse a JSON manifest. Fails loudly; never returns a default."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except OSError as error:
        raise PromptFileMissingError(f"Cannot read manifest {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise PromptFileMissingError(f"Invalid JSON in manifest {path}: {error}") from error


def _build_modes() -> dict[str, Mode]:
    manifest = _load_json_manifest(MODES_MANIFEST)
    return {
        mode_id: Mode(
            id=mode_id,
            display_name=fields["display_name"],
            description=fields["description"],
            is_passthrough=bool(fields["is_passthrough"]),
            default_tone_id=fields["default_tone"],
        )
        for mode_id, fields in manifest.items()
    }


def _build_tones() -> dict[str, Tone]:
    manifest = _load_json_manifest(TONES_MANIFEST)
    return {
        tone_id: Tone(
            id=tone_id,
            display_name=fields["display_name"],
            description=fields["description"],
            instruction=fields["instruction"],
        )
        for tone_id, fields in manifest.items()
    }


#: Populated once at import time from the JSON manifests.
MODES: dict[str, Mode] = _build_modes()
TONES: dict[str, Tone] = _build_tones()

_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(mode: "Mode | str") -> str:
    """Read a mode's raw system-prompt template, once, then serve it from cache.

    Accepts either a :class:`Mode` or a raw mode id string, so it can be used
    to probe an id that is not (yet) in :data:`MODES`. Raises
    :class:`PromptFileMissingError` if the file is absent or unreadable —
    there is no silent fallback to an inline default prompt.
    """
    mode_id = mode.id if isinstance(mode, Mode) else mode
    cached = _PROMPT_CACHE.get(mode_id)
    if cached is not None:
        return cached
    path = PROMPTS_DIR / f"{mode_id}.txt"
    try:
        with open(path, "r", encoding="utf-8") as file:
            template = file.read()
    except OSError as error:
        raise PromptFileMissingError(f"Missing prompt file for mode {mode_id!r}: {path}") from error
    _PROMPT_CACHE[mode_id] = template
    return template


def _mode_by_id(mode_id: str) -> Mode:
    try:
        return MODES[mode_id]
    except KeyError:
        raise UnknownModeError(
            f"Unknown cleanup mode {mode_id!r}; known modes: {', '.join(MODE_IDS)}"
        ) from None


def _tone_by_id(tone_id: str) -> Tone:
    try:
        return TONES[tone_id]
    except KeyError:
        raise UnknownToneError(
            f"Unknown cleanup tone {tone_id!r}; known tones: {', '.join(TONE_IDS)}"
        ) from None


def _vocabulary_clause(vocabulary: tuple[str, ...]) -> str:
    """Build the ``{vocabulary}`` value: a full clause, or "" when there is nothing to keep."""
    terms = [term.strip() for term in vocabulary if term and term.strip()]
    if not terms:
        return ""
    return "Keep these exact terms and their spelling unchanged: " + ", ".join(terms) + "."


def render_system_prompt(
    mode: "Mode | str",
    tone: "Tone | str",
    language: str | None,
    vocabulary: tuple[str, ...],
) -> str:
    """Render a mode's system prompt with tone, language, and vocabulary filled in.

    ``mode`` and ``tone`` accept either the dataclass or its id string.
    ``language`` may be None (auto-detect); ``vocabulary`` may be empty.
    """
    mode_obj = mode if isinstance(mode, Mode) else _mode_by_id(mode)
    tone_obj = tone if isinstance(tone, Tone) else _tone_by_id(tone)
    template = load_prompt(mode_obj)
    stripped = language.strip() if language else ""
    language_text = (
        "the same language as the dictation"
        if stripped in ("", "auto")
        else stripped
    )
    rendered = template.format(
        tone_instruction=tone_obj.instruction,
        language=language_text,
        vocabulary=_vocabulary_clause(vocabulary),
    )
    if mode_obj.id == "code":
        from cleanup.coding_mode import code_mode_llm_hint

        rendered = rendered.rstrip("\n") + "\n\n" + code_mode_llm_hint()
    return rendered


def default_tone_for(mode: "Mode | str") -> Tone:
    """The tone a mode starts with before the user picks one, per ``modes.json``."""
    mode_obj = mode if isinstance(mode, Mode) else _mode_by_id(mode)
    return TONES[mode_obj.default_tone_id]


def mode_from_config(config: dict) -> Mode:
    """Read ``cleanup_mode`` from config, defaulting to Dictation. Raises on an unknown id."""
    mode_id = config.get(CONFIG_MODE_KEY, DEFAULT_MODE_ID)
    return _mode_by_id(mode_id)


def tone_from_config(config: dict) -> Tone:
    """Read ``cleanup_tone`` from config, defaulting to Neutral. Raises on an unknown id."""
    tone_id = config.get(CONFIG_TONE_KEY, DEFAULT_TONE_ID)
    return _tone_by_id(tone_id)


__all__ = [
    "CONFIG_MODE_KEY",
    "CONFIG_TONE_KEY",
    "DEFAULT_MODE_ID",
    "DEFAULT_TONE_ID",
    "MODE_IDS",
    "MODES",
    "Mode",
    "ModeConfigError",
    "PromptFileMissingError",
    "TONE_IDS",
    "TONES",
    "Tone",
    "UnknownModeError",
    "UnknownToneError",
    "default_tone_for",
    "load_prompt",
    "mode_from_config",
    "render_system_prompt",
    "tone_from_config",
]
