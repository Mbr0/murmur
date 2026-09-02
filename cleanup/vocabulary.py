#!/usr/bin/env python3
"""User vocabulary: terms that bias transcription, and text replacements applied after.

Two independent lists make up a :class:`Vocabulary`:

- ``terms`` bias the engine's decode (see :func:`hints_from_vocabulary`), for
  proper nouns and jargon the engine would otherwise mis-hear.
- ``replacements`` run as a deterministic text pass after transcription (see
  :func:`apply_replacements`), for fixed corrections like expanding an
  abbreviation or fixing a name the engine consistently gets wrong.

Both (de)serialise to the on-disk config (:func:`vocabulary_from_config`,
:func:`vocabulary_to_config`) and to a portable CSV (:func:`export_csv`,
:func:`import_csv`) for backup and sharing between machines.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

from engines import Hints

#: Free-tier cap on the number of vocabulary terms. Enforced by Wave 4's Pro
#: gate, not here: this module has no notion of entitlement.
FREE_TERM_LIMIT = 20

#: Header row of the CSV format. ``kind`` is ``term`` or ``replacement``; a
#: term row leaves ``to`` and ``match_case`` blank.
CSV_HEADER = ("kind", "from", "to", "match_case")


class VocabularyError(Exception):
    """Raised when vocabulary data cannot be parsed. Fails fast, never skips a row."""


@dataclass(frozen=True)
class Replacement:
    """A whole-word text substitution applied after transcription."""

    from_text: str
    to_text: str
    match_case: bool


@dataclass(frozen=True)
class Vocabulary:
    """A user's terms (decode bias) and replacements (post-processing)."""

    terms: tuple[str, ...] = ()
    replacements: tuple[Replacement, ...] = ()


#: One word character, as ``\b`` and ``\w`` define it. Used to decide which end
#: of a term can carry a word boundary at all.
_WORD_CHAR = re.compile(r"\w")


def _replacement_pattern(replacement: Replacement) -> re.Pattern[str]:
    """Compile a term into a pattern guarded only where a guard can match.

    ``\\b`` asserts a word/non-word transition, so wrapping a term in ``\\b``
    breaks every term whose own edge is not a word character: "C++", ".NET" and
    "#tag" could never match anything, because the boundary was asserted on the
    wrong side of the punctuation. Each edge is therefore guarded with a
    lookaround only when that edge is a word character — "cat" still refuses
    "concatenate", while "C++" matches as written.
    """
    flags = 0 if replacement.match_case else re.IGNORECASE
    term = replacement.from_text
    prefix = r"(?<!\w)" if _WORD_CHAR.match(term[:1]) else ""
    suffix = r"(?!\w)" if _WORD_CHAR.match(term[-1:]) else ""
    return re.compile(rf"{prefix}{re.escape(term)}{suffix}", flags)


def apply_replacements(text: str, vocabulary: Vocabulary) -> str:
    """Apply every replacement in ``vocabulary`` to ``text``.

    Whole-word where the term has word edges: a replacement for "cat" never
    touches "category". A term that starts or ends in punctuation — "C++",
    ".NET", "#tag" — is guarded only on the end that is a word character, so
    it matches at all. Replacement text is inserted exactly as given,
    regardless of the case the match was found in.

    Order is deterministic: replacements with a longer ``from_text`` run
    first (so a specific phrase wins over a shorter word it contains), and
    replacements of equal length run in the order they were listed.
    """
    assert text is not None, "text is required"
    assert vocabulary is not None, "vocabulary is required"

    ordered = sorted(vocabulary.replacements, key=lambda r: -len(r.from_text))
    result = text
    for replacement in ordered:
        if not replacement.from_text:
            continue
        pattern = _replacement_pattern(replacement)
        to_text = replacement.to_text
        result = pattern.sub(lambda _match, value=to_text: value, result)
    return result


def hints_from_vocabulary(vocabulary: Vocabulary) -> Hints:
    """Build engine :class:`~engines.Hints` from a vocabulary's terms.

    Terms are stripped, emptied entries dropped, and duplicates removed while
    keeping the first occurrence's position. Replacements are not included:
    they run after transcription, not as decode bias.
    """
    assert vocabulary is not None, "vocabulary is required"
    seen: set[str] = set()
    ordered_terms: list[str] = []
    for term in vocabulary.terms:
        stripped = term.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        ordered_terms.append(stripped)
    return Hints(vocabulary=tuple(ordered_terms), initial_prompt=None)


def export_csv(vocabulary: Vocabulary) -> str:
    """Render ``vocabulary`` as CSV text: a header row, then one row per term
    and per replacement. Values are quoted by the CSV module as needed, so
    commas, quotes, and Unicode round-trip through :func:`import_csv`.
    """
    assert vocabulary is not None, "vocabulary is required"
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    for term in vocabulary.terms:
        writer.writerow(("term", term, "", ""))
    for replacement in vocabulary.replacements:
        writer.writerow(
            (
                "replacement",
                replacement.from_text,
                replacement.to_text,
                "true" if replacement.match_case else "false",
            )
        )
    return buffer.getvalue()


def _parse_match_case(raw_value: str, line_number: int) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0", ""):
        return False
    raise VocabularyError(f"Line {line_number}: invalid match_case value {raw_value!r}")


def import_csv(text: str) -> Vocabulary:
    """Parse CSV text produced by :func:`export_csv` (or hand-edited to match)
    back into a :class:`Vocabulary`.

    Fails fast: a missing/mismatched header, a row with the wrong number of
    fields, an unknown ``kind``, a blank ``from`` value, or an unparsable
    ``match_case`` all raise :class:`VocabularyError` naming the 1-based CSV
    row number. No row is silently dropped.
    """
    assert text is not None, "text is required"
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise VocabularyError("Line 1: missing header row")

    header = tuple(field.strip() for field in rows[0])
    if header != CSV_HEADER:
        raise VocabularyError(
            f"Line 1: expected header {','.join(CSV_HEADER)}, got {','.join(header)}"
        )

    terms: list[str] = []
    replacements: list[Replacement] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row:
            continue  # trailing blank line
        if len(row) != len(CSV_HEADER):
            raise VocabularyError(
                f"Line {line_number}: expected {len(CSV_HEADER)} fields, got {len(row)}"
            )
        kind = row[0].strip()
        from_text, to_text, match_case_raw = row[1], row[2], row[3]
        if kind == "term":
            if not from_text:
                raise VocabularyError(f"Line {line_number}: term row missing 'from' value")
            terms.append(from_text)
        elif kind == "replacement":
            if not from_text:
                raise VocabularyError(f"Line {line_number}: replacement row missing 'from' value")
            match_case = _parse_match_case(match_case_raw, line_number)
            replacements.append(Replacement(from_text=from_text, to_text=to_text, match_case=match_case))
        else:
            raise VocabularyError(f"Line {line_number}: unknown kind {kind!r}")

    return Vocabulary(terms=tuple(terms), replacements=tuple(replacements))


def vocabulary_from_config(config: dict) -> Vocabulary:
    """Read ``vocabulary_terms`` and ``vocabulary_replacements`` out of a config dict.

    Missing keys mean an empty list. Wrong types — anything but a list of
    strings for terms, or a list of ``{"from", "to", "match_case"}`` objects
    for replacements — raise :class:`VocabularyError`.
    """
    assert config is not None, "config is required"

    raw_terms = config.get("vocabulary_terms", [])
    if not isinstance(raw_terms, list) or not all(isinstance(term, str) for term in raw_terms):
        raise VocabularyError("vocabulary_terms must be a list of strings")

    raw_replacements = config.get("vocabulary_replacements", [])
    if not isinstance(raw_replacements, list):
        raise VocabularyError("vocabulary_replacements must be a list")

    replacements: list[Replacement] = []
    for index, item in enumerate(raw_replacements):
        if not isinstance(item, dict):
            raise VocabularyError(f"vocabulary_replacements[{index}] must be an object")
        from_text = item.get("from")
        to_text = item.get("to")
        match_case = item.get("match_case")
        if (
            not isinstance(from_text, str)
            or not isinstance(to_text, str)
            or not isinstance(match_case, bool)
        ):
            raise VocabularyError(f"vocabulary_replacements[{index}] has invalid field types")
        replacements.append(Replacement(from_text=from_text, to_text=to_text, match_case=match_case))

    return Vocabulary(terms=tuple(raw_terms), replacements=tuple(replacements))


def vocabulary_to_config(vocabulary: Vocabulary) -> dict:
    """Render ``vocabulary`` as the two config keys :func:`vocabulary_from_config` reads."""
    assert vocabulary is not None, "vocabulary is required"
    return {
        "vocabulary_terms": list(vocabulary.terms),
        "vocabulary_replacements": [
            {"from": r.from_text, "to": r.to_text, "match_case": r.match_case}
            for r in vocabulary.replacements
        ],
    }
