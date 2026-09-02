#!/usr/bin/env python3
"""Snippets: short spoken triggers that expand into stored text.

A snippet is a phrase the user says — "my address", "sign off" — and the text
Murmur types instead. It is the same shape of idea as a vocabulary
:class:`~cleanup.vocabulary.Replacement`, with two differences that earn it its
own module: a trigger is a *phrase*, not a word, and its expansion is usually
several lines rather than a corrected spelling.

Two rules keep :func:`expand_snippets` predictable:

* **Longest trigger first.** "my address" beats "address", so a specific
  phrase always wins over a shorter one it contains.
* **One pass, never its own output.** The replacement text is not rescanned,
  and text that already reads as an expansion is left alone — so expanding a
  transcript twice gives the same transcript, and a snippet whose text
  contains another trigger cannot cascade.

Serialises to a single config key, ``snippets`` (see
``services/persistence_service.py``): a list of ``{"trigger", "text"}``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

#: Free-tier cap on the number of snippets. Enforced by Wave 4's Pro gate and
#: surfaced by the Smart tab; this module has no notion of entitlement.
FREE_SNIPPET_LIMIT = 5

#: The one config key snippets live in.
CONFIG_SNIPPETS_KEY = "snippets"


class SnippetError(Exception):
    """Raised when snippet data cannot be read. Fails fast, never skips a row."""


@dataclass(frozen=True)
class Snippet:
    """One spoken ``trigger`` and the ``text`` it expands into."""

    trigger: str
    text: str


#: One word character, as ``\w`` defines it. Decides which end of a trigger can
#: carry a word boundary at all — the same rule ``cleanup/vocabulary.py`` uses,
#: so "C++" and ".NET" work as triggers too.
_WORD_CHAR = re.compile(r"\w")


def _pattern_source(trigger: str) -> str | None:
    """A regex for one trigger phrase, or None when there is nothing to match.

    Runs of whitespace inside the phrase become ``\\s+``: a trigger stored as
    "my address" must still fire when the engine wrote "my  address" or broke
    the line between the two words.
    """
    words = trigger.split()
    if not words:
        return None
    body = r"\s+".join(re.escape(word) for word in words)
    prefix = r"(?<!\w)" if _WORD_CHAR.match(words[0][:1]) else ""
    suffix = r"(?!\w)" if _WORD_CHAR.match(words[-1][-1:]) else ""
    return f"{prefix}{body}{suffix}"


def _expanded_spans(text: str, snippets: tuple[Snippet, ...]) -> tuple[tuple[int, int], ...]:
    """Where ``text`` already reads as one of the snippets' expansions.

    This is what makes :func:`expand_snippets` idempotent rather than merely
    single-pass. A snippet whose text contains its own trigger — "sig" →
    "sig: Matthieu" — would otherwise grow on every run. A trigger sitting
    inside a longer stretch that is already the finished expansion has clearly
    been expanded once, so it is left alone.
    """
    spans: list[tuple[int, int]] = []
    for snippet in snippets:
        expansion = snippet.text
        if not expansion.strip() or len(expansion) <= len(snippet.trigger.strip()):
            continue
        start = text.find(expansion)
        while start >= 0:
            spans.append((start, start + len(expansion)))
            start = text.find(expansion, start + 1)
    return tuple(spans)


def _is_inside(span: tuple[int, int], spans: tuple[tuple[int, int], ...]) -> bool:
    """Whether ``span`` falls wholly inside a strictly longer one."""
    start, end = span
    return any(
        outer_start <= start and end <= outer_end and (outer_end - outer_start) > (end - start)
        for outer_start, outer_end in spans
    )


def expand_snippets(text: str, snippets: Iterable[Snippet]) -> str:
    """Replace every spoken trigger in ``text`` with the snippet's text.

    Matching is whole-word and case-insensitive; the replacement is inserted
    exactly as stored, whatever case the trigger was said in. Longer triggers
    are tried first, and each part of ``text`` is consumed once, so no
    expansion is ever re-expanded.
    """
    assert text is not None, "text is required"
    assert snippets is not None, "snippets is required"

    usable = tuple(snippet for snippet in snippets if snippet.trigger.strip())
    if not usable:
        return text

    ordered = sorted(usable, key=lambda snippet: -len(snippet.trigger.strip()))
    parts: list[str] = []
    by_group: dict[str, str] = {}
    for index, snippet in enumerate(ordered):
        source = _pattern_source(snippet.trigger)
        if source is None:  # pragma: no cover - blank triggers are filtered above
            continue
        name = f"snippet{index}"
        parts.append(f"(?P<{name}>{source})")
        by_group[name] = snippet.text
    if not parts:  # pragma: no cover - defensive
        return text

    pattern = re.compile("|".join(parts), re.IGNORECASE)
    already = _expanded_spans(text, usable)

    def _replace(match: re.Match[str]) -> str:
        if _is_inside(match.span(), already):
            return match.group(0)
        return by_group[match.lastgroup]

    return pattern.sub(_replace, text)


def snippets_from_config(config: dict) -> tuple[Snippet, ...]:
    """Read ``snippets`` out of a config dict.

    A missing key means no snippets. Anything but a list of
    ``{"trigger", "text"}`` string pairs raises :class:`SnippetError` naming
    the offending row: no row is silently dropped.
    """
    assert config is not None, "config is required"

    raw = config.get(CONFIG_SNIPPETS_KEY, [])
    if not isinstance(raw, list):
        raise SnippetError(f"{CONFIG_SNIPPETS_KEY} must be a list")

    snippets: list[Snippet] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SnippetError(f"{CONFIG_SNIPPETS_KEY}[{index}] must be an object")
        trigger = item.get("trigger")
        text = item.get("text")
        if not isinstance(trigger, str) or not isinstance(text, str):
            raise SnippetError(f"{CONFIG_SNIPPETS_KEY}[{index}] has invalid field types")
        snippets.append(Snippet(trigger=trigger, text=text))
    return tuple(snippets)


def snippets_to_config(snippets: Iterable[Snippet]) -> dict:
    """Render ``snippets`` as the one config key :func:`snippets_from_config` reads."""
    assert snippets is not None, "snippets is required"
    return {
        CONFIG_SNIPPETS_KEY: [
            {"trigger": snippet.trigger, "text": snippet.text} for snippet in snippets
        ]
    }


__all__ = [
    "CONFIG_SNIPPETS_KEY",
    "FREE_SNIPPET_LIMIT",
    "Snippet",
    "SnippetError",
    "expand_snippets",
    "snippets_from_config",
    "snippets_to_config",
]
