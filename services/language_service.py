#!/usr/bin/env python3
"""Language selection: a default plus optional per-front-app overrides.

Two config keys drive this (see ``services/persistence_service.py``):

- ``language``: ``"auto"`` (detect) or an ISO code, the app-wide default.
- ``language_by_app``: ``{bundle_id: language_code}``, overriding the default
  while that app is frontmost. Populated by :func:`remember_language`, e.g.
  after the user picks a language once for an app; cleared per app by
  :func:`forget_language`.
"""

from __future__ import annotations

from typing import Any

from engines import LANGUAGE_AUTO, EngineInfo

DEFAULT_LANGUAGE = LANGUAGE_AUTO

#: Display names for the languages we expect to see most; anything else falls
#: back to its ISO code, upper-cased, in :func:`language_display_name`.
_LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "Français",
    "nl": "Nederlands",
    "de": "Deutsch",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
}


def resolve_language(config: dict[str, Any], bundle_id: str | None) -> str:
    """The language to transcribe in: the per-app override if one is set for
    ``bundle_id``, else the app-wide ``language`` default (``"auto"`` if unset).
    """
    assert config is not None, "config is required"
    by_app = config.get("language_by_app") or {}
    if bundle_id and bundle_id in by_app:
        return by_app[bundle_id]
    return config.get("language", DEFAULT_LANGUAGE)


def remember_language(config: dict[str, Any], bundle_id: str, language: str) -> dict[str, Any]:
    """Return a copy of ``config`` with ``bundle_id`` pinned to ``language``.

    Does not mutate ``config``.
    """
    assert config is not None, "config is required"
    assert bundle_id, "bundle_id is required"
    assert language, "language is required"
    by_app = dict(config.get("language_by_app") or {})
    by_app[bundle_id] = language
    return {**config, "language_by_app": by_app}


def forget_language(config: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    """Return a copy of ``config`` with any per-app override for ``bundle_id``
    removed, falling back to the app-wide default again. Does not mutate
    ``config``; a no-op (still a copy) when there was nothing to forget.
    """
    assert config is not None, "config is required"
    assert bundle_id, "bundle_id is required"
    by_app = dict(config.get("language_by_app") or {})
    by_app.pop(bundle_id, None)
    return {**config, "language_by_app": by_app}


def available_languages(engine_info: EngineInfo) -> tuple[str, ...]:
    """Languages to offer in the picker for an engine: ``"auto"`` first, then
    the engine's own ISO codes, deduplicated and sorted.
    """
    assert engine_info is not None, "engine_info is required"
    codes = {code for code in engine_info.languages if code != LANGUAGE_AUTO}
    return (LANGUAGE_AUTO, *sorted(codes))


def language_display_name(language_code: str) -> str:
    """Human-readable label for a language code, for the picker."""
    assert language_code, "language_code is required"
    if language_code == LANGUAGE_AUTO:
        return "Auto"
    return _LANGUAGE_DISPLAY_NAMES.get(language_code, language_code.upper())
