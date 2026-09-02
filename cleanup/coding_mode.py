#!/usr/bin/env python3
"""Rule-based spoken-to-code transforms for Murmur's Code mode.

Applied before any LLM cleanup step when the active mode is ``code``. Turns
dictation-style speech ("open paren", "camel case foo bar", "dash dash
force") into the literal characters a coding tool expects, so an optional
LLM step sees code-shaped text instead of spoken punctuation.

Transform order (single left-to-right pass over space-split word tokens;
each step below claims tokens the later ones never see):

1. Case commands ("camel case foo bar" -> "fooBarBaz") consume raw words
   until a punctuation-command word, "end case"/"stop", or end of text.
2. CLI dash-flags ("dash dash force" -> "--force", "dash v" -> "-v").
3. Spelled-out tool names ("n p m" -> "npm") - runs of up to four single
   letter words that spell a name from a small allowlist.
4. Spoken punctuation and symbols ("open paren" -> "(", "comma" -> ","),
   matched longest phrase first.
5. Anything left over passes through unchanged, in its original case.

Matching is whole-word and case-insensitive throughout. Numbers are left
alone: speech engines already emit digits, so there is no word-to-digit
table here.

Spacing rules (see ``_symbol_glue``): commas/periods/colons/semicolons and
closing brackets never get a space before them; opening brackets never get
a space after them; underscores, "at sign" and newline/tab glue on both
sides like an identifier character; quotes/backtick alternate open/close
each time they are spoken, like a smart-quote toggle. Binary and
comparison operators (equals, plus, less than, arrow, ...) keep a normal
space on both sides. This means "period"/"dot" get prose-style spacing (no
space before, a normal space after) rather than tight attribute-access
spacing (``self.name``) -- a deliberate simplification, not a code-parser.

The whole pass is idempotent: it only ever recognizes spoken *words*, never
the punctuation characters it produces, so running it twice on its own
output is a no-op.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# 1. Spoken punctuation and symbols
# ---------------------------------------------------------------------------

# Shown to the user in Settings. The *matcher* built below re-sorts these
# longest-phrase-first (by word count) so e.g. "double equals" is tried
# before "equals" regardless of this display order.
SPOKEN_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("open paren", "("),
    ("close paren", ")"),
    ("open bracket", "["),
    ("close bracket", "]"),
    ("open brace", "{"),
    ("close brace", "}"),
    ("comma", ","),
    ("period", "."),
    ("dot", "."),
    ("colon", ":"),
    ("semicolon", ";"),
    ("double equals", "=="),
    ("not equals", "!="),
    ("equals", "="),
    ("plus", "+"),
    ("minus", "-"),
    ("star", "*"),
    ("asterisk", "*"),
    ("slash", "/"),
    ("backslash", "\\"),
    ("pipe", "|"),
    ("ampersand", "&"),
    ("percent", "%"),
    ("caret", "^"),
    ("tilde", "~"),
    ("at sign", "@"),
    ("hash", "#"),
    ("dollar", "$"),
    ("question mark", "?"),
    ("exclamation mark", "!"),
    ("double quote", '"'),
    ("quote", '"'),
    ("single quote", "'"),
    ("apostrophe", "'"),
    ("backtick", "`"),
    ("underscore", "_"),
    ("fat arrow", "=>"),
    ("arrow", "->"),
    ("less than", "<"),
    ("greater than", ">"),
    ("new line", "\n"),
    ("tab", "\t"),
)

# Symbols that override the default "normal space on both sides" spacing.
# Anything not listed here (binary/comparison operators like "+", "==", "<",
# "->", and bare punctuation like "#", "$", "?", "!", "~", "^", "%") keeps a
# normal space on both sides, per "keep spaces around binary operators".
_OPEN_GLUE = {"(", "[", "{"}  # normal space before, glued to what follows
_CLOSE_GLUE = {")", "]", "}"}  # glued to what precedes, normal space after
_NO_SPACE_BEFORE = {",", ".", ":", ";"}  # "no space before commas/periods"
_GLUE_BOTH_SIDES = {"_", "@", "\n", "\t"}  # identifier glue / whitespace
_TOGGLE_QUOTES = {'"', "'", "`"}  # alternate open/close like a smart quote


# ---------------------------------------------------------------------------
# 2. CLI dash-flags
# ---------------------------------------------------------------------------
# Handled directly in the main loop below ("dash dash <word>" -> "--<word>",
# "dash <word>" -> "-<word>") because the following word is arbitrary, not a
# fixed phrase. "dash"/"dash dash" is only treated as a flag prefix when the
# next word is flag-shaped (letters/digits/hyphen/underscore only) and is
# neither a spoken-symbol keyword/phrase-starter nor a common prose
# stopword -- otherwise "dash" is left as a literal spoken word, so prose
# like "a dash of salt" and multi-word commands like "dash open paren" are
# not swallowed. The flag word's case is preserved (not lowercased).

_DASH_STOPWORDS = {
    "of", "the", "a", "an", "to", "in", "for", "and", "or", "is", "it", "on",
}
# First words of multi-word SPOKEN_SYMBOLS phrases, plus other symbol-ish
# trigger words, that must never be swallowed as a dash flag.
_DASH_SYMBOL_STARTERS = {
    "open", "close", "new", "double", "single", "less", "greater", "fat",
    "at", "question", "exclamation",
}
_FLAG_SHAPE_RE = re.compile(r"[A-Za-z0-9_-]+")


def _is_dash_flag_candidate(word_lower: str, word: str) -> bool:
    """Is ``word`` a plausible CLI flag name to attach after "dash"?"""
    if word_lower in _DASH_STOPWORDS or word_lower in _DASH_SYMBOL_STARTERS:
        return False
    if (word_lower,) in _SYMBOL_TABLE:
        return False
    return bool(_FLAG_SHAPE_RE.fullmatch(word))


# ---------------------------------------------------------------------------
# 3. Spelled-out tool names
# ---------------------------------------------------------------------------

# Small, deliberate allowlist: only these names are recognized when spelled
# out letter by letter, so ordinary single-letter dictation ("a", "i") is
# never mistaken for a command.
_KNOWN_SPELLED_TOOLS = {
    "npm", "ssh", "cd", "ls", "git", "cli", "api", "url", "sql", "css",
    "html", "http", "json",
}
# The spec's own examples ("n p m", "s s h") only need three letters, but
# the allowlist above includes four-letter names (html, http, json) that
# would otherwise be unreachable -- extended the cap from three to four to
# cover them; documented here as a deliberate deviation.
_MAX_SPELLED_RUN = 4


# ---------------------------------------------------------------------------
# Case commands
# ---------------------------------------------------------------------------

_CASE_COMMANDS: tuple[tuple[str, str], ...] = (
    ("screaming snake", "screaming_snake"),
    ("constant case", "screaming_snake"),
    # "all caps" renders the same way as "screaming snake" for multi-word
    # input; the spec's own example ("all caps foo" -> "FOO") is a single
    # word, which does not distinguish the two, so this is a judgment call.
    ("all caps", "screaming_snake"),
    ("camel case", "camel"),
    ("pascal case", "pascal"),
    ("snake case", "snake"),
    ("kebab case", "kebab"),
)

_CASE_TERMINATOR_WORDS: tuple[tuple[str, str], ...] = (
    ("end case", "terminator"),
    ("stop", "terminator"),
)


# ---------------------------------------------------------------------------
# Language support
# ---------------------------------------------------------------------------

# Small, deliberately narrow set of French spoken-command translations.
# Translated to their English equivalents *before* the main pipeline runs,
# so everything below only ever has to understand English trigger words.
# Kept intentionally short per the task's "keep small" instruction; English
# remains the default and the fully documented language.
_FRENCH_TRANSLATIONS: tuple[tuple[str, str], ...] = (
    ("parenthèse ouvrante", "open paren"),
    ("parenthèse fermante", "close paren"),
    ("tiret tiret", "dash dash"),
    ("virgule", "comma"),
    ("point", "period"),
    ("tiret", "dash"),
)


def _translate_french(text: str) -> str:
    """Rewrite the small set of French trigger phrases to their English form."""
    for phrase, replacement in _FRENCH_TRANSLATIONS:
        pattern = re.compile(
            r"(?i)\b" + r"\s+".join(re.escape(w) for w in phrase.split()) + r"\b"
        )
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Phrase matching helpers
# ---------------------------------------------------------------------------


def _build_phrase_table(
    pairs: tuple[tuple[str, str], ...],
) -> tuple[dict[tuple[str, ...], str], tuple[int, ...]]:
    """Index (phrase, value) pairs by lowercase word-tuple, longest first."""
    table: dict[tuple[str, ...], str] = {}
    for phrase, value in pairs:
        table[tuple(phrase.lower().split())] = value
    lengths = tuple(sorted({len(key) for key in table}, reverse=True))
    return table, lengths


_SYMBOL_TABLE, _SYMBOL_LENGTHS = _build_phrase_table(SPOKEN_SYMBOLS)
_CASE_TABLE, _CASE_LENGTHS = _build_phrase_table(_CASE_COMMANDS)
_CASE_TERMINATOR_TABLE, _CASE_TERMINATOR_LENGTHS = _build_phrase_table(
    _CASE_TERMINATOR_WORDS
)


def _match_phrase(
    words_lower: list[str],
    i: int,
    table: dict[tuple[str, ...], str],
    lengths: tuple[int, ...],
) -> tuple[str, int] | None:
    """Try each known phrase length (longest first) starting at index i."""
    for length in lengths:
        if i + length > len(words_lower):
            continue
        key = tuple(words_lower[i : i + length])
        if key in table:
            return table[key], length
    return None


def _match_spelled_tool(words_lower: list[str], i: int) -> tuple[str, int] | None:
    """Match a run of single-letter words spelling a known tool name."""
    n = len(words_lower)
    limit = min(_MAX_SPELLED_RUN, n - i)
    for run_len in range(limit, 0, -1):
        chunk = words_lower[i : i + run_len]
        if not all(len(word) == 1 and word.isalpha() for word in chunk):
            continue
        candidate = "".join(chunk)
        if candidate in _KNOWN_SPELLED_TOOLS:
            return candidate, run_len
    return None


def _render_case(kind: str, words: list[str]) -> str:
    """Join collected words per the requested case-command style."""
    words = [w.lower() for w in words if w]
    if not words:
        return ""
    if kind == "camel":
        head, *rest = words
        return head + "".join(w[:1].upper() + w[1:] for w in rest)
    if kind == "pascal":
        return "".join(w[:1].upper() + w[1:] for w in words)
    if kind == "snake":
        return "_".join(words)
    if kind == "kebab":
        return "-".join(words)
    if kind == "screaming_snake":
        return "_".join(w.upper() for w in words)
    raise AssertionError(f"unknown case kind: {kind}")  # pragma: no cover


def _symbol_glue(value: str, quote_state: dict[str, bool]) -> tuple[bool, bool]:
    """Return (glue_before, glue_after) for a resolved symbol character."""
    if value in _OPEN_GLUE:
        return False, True
    if value in _CLOSE_GLUE:
        return True, False
    if value in _NO_SPACE_BEFORE:
        return True, False
    if value in _GLUE_BOTH_SIDES:
        return True, True
    if value in _TOGGLE_QUOTES:
        opening = quote_state.get(value, True)
        quote_state[value] = not opening
        return (False, True) if opening else (True, False)
    return False, False


def _join(atoms: list[tuple[str, bool, bool]]) -> str:
    """Render (text, glue_before, glue_after) atoms into the final string."""
    if not atoms:
        return ""
    pieces = [atoms[0][0]]
    for idx in range(1, len(atoms)):
        text, glue_before, _ = atoms[idx]
        prev_glue_after = atoms[idx - 1][2]
        space = "" if (glue_before or prev_glue_after) else " "
        pieces.append(space + text)
    return "".join(pieces)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transform_spoken_code(text: str, *, language: str = "en") -> str:
    """Apply rule-based spoken-to-code transforms to ``text``.

    ``language`` selects the trigger vocabulary: "en" (default), ``None``,
    or "auto" (both treated as English), or "fr" for the small French phrase
    set in ``_FRENCH_TRANSLATIONS``. Runs of the space character are
    collapsed to one space (leading/trailing trimmed); other whitespace
    (e.g. an already-produced "\\n"/"\\t") is left alone. Raises
    :class:`ValueError` for any other unsupported language code.
    """
    assert isinstance(text, str), "text must be a string"
    if language in (None, "auto", "en"):
        pass
    elif language == "fr":
        text = _translate_french(text)
    else:
        raise ValueError(f"unsupported language: {language!r}")

    # Split on the literal space character only (collapsing repeats), never
    # on other whitespace: an already-produced "\n"/"\t" atom must survive
    # a second pass unchanged, or the transform would not be idempotent.
    words = [w for w in text.split(" ") if w]
    if not words:
        return text
    words_lower = [w.lower() for w in words]
    n = len(words)

    atoms: list[tuple[str, bool, bool]] = []
    quote_state: dict[str, bool] = {}

    i = 0
    while i < n:
        case_match = _match_phrase(words_lower, i, _CASE_TABLE, _CASE_LENGTHS)
        if case_match is not None:
            kind, consumed = case_match
            i += consumed
            collected: list[str] = []
            while i < n:
                terminator = _match_phrase(
                    words_lower, i, _CASE_TERMINATOR_TABLE, _CASE_TERMINATOR_LENGTHS
                )
                if terminator is not None:
                    i += terminator[1]
                    break
                if _match_phrase(words_lower, i, _SYMBOL_TABLE, _SYMBOL_LENGTHS):
                    break
                if _match_phrase(words_lower, i, _CASE_TABLE, _CASE_LENGTHS):
                    break
                collected.append(words_lower[i])
                i += 1
            rendered = _render_case(kind, collected)
            if rendered:
                atoms.append((rendered, False, False))
            continue

        if words_lower[i] == "dash":
            if i + 1 < n and words_lower[i + 1] == "dash":
                if i + 2 < n and _is_dash_flag_candidate(
                    words_lower[i + 2], words[i + 2]
                ):
                    atoms.append(("--" + words[i + 2], False, False))
                    i += 3
                    continue
                atoms.append((words[i], False, False))
                i += 1
                continue
            if i + 1 < n and _is_dash_flag_candidate(words_lower[i + 1], words[i + 1]):
                atoms.append(("-" + words[i + 1], False, False))
                i += 2
                continue
            atoms.append((words[i], False, False))
            i += 1
            continue

        spelled = _match_spelled_tool(words_lower, i)
        if spelled is not None:
            value, consumed = spelled
            atoms.append((value, False, False))
            i += consumed
            continue

        symbol_match = _match_phrase(words_lower, i, _SYMBOL_TABLE, _SYMBOL_LENGTHS)
        if symbol_match is not None:
            value, consumed = symbol_match
            glue_before, glue_after = _symbol_glue(value, quote_state)
            atoms.append((value, glue_before, glue_after))
            i += consumed
            continue

        atoms.append((words[i], False, False))
        i += 1

    return _join(atoms)


def code_mode_llm_hint() -> str:
    """One-sentence instruction for the LLM system prompt in Code mode."""
    return (
        "Code mode: leave code tokens, CLI flags, and identifiers exactly "
        "as written; do not rephrase, translate, or otherwise alter them."
    )
