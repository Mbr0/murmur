#!/usr/bin/env python3
"""Settings → Smart: cleanup, modes, context, vocabulary and snippets.

Everything Murmur does to a transcript *after* it has been heard lives on one
tab, and — like the other four — as a plain-Python model with an AppKit
rendering over it, so the rules are tested without a window server.

Five groups, one section model each:

* **Cleanup** — whether the local (or cloud) cleanup pass runs at all, and the
  mode and tone it runs with. ``cleanup_enabled`` is the one tri-state on the
  tab: ``None`` on disk means "nobody has decided yet", and the checkbox shows
  what this machine would do while saying so in a hint. Touching it writes a
  real ``True``/``False`` — see :meth:`SmartTabModel.set_cleanup_enabled`.
* **Per-app modes** — the ``mode_by_app`` overrides that always beat the
  built-in bundle table (``cleanup/context.py``), plus the switch that turns
  that built-in table on and off.
* **Vocabulary** — the Wave 1 terms box, replacements table and CSV
  import/export, ported here whole as :class:`VocabularySectionModel`.
* **Snippets** — spoken triggers that expand into stored text
  (``cleanup/snippets.py``).
* **Spoken symbols** — a read-only reference list of what Code mode
  understands, so the feature is discoverable rather than folklore.

Entitlement is asked of one injected callable and nothing else: the Pro gate in
``context.services["pro_gate"]``, called as ``is_pro_feature_enabled(feature)``.
This tab asks it three questions — :data:`FEATURE_CLEANUP`,
:data:`FEATURE_VOCABULARY_BEYOND_FREE` and :data:`FEATURE_SNIPPETS` — and never
reads a licence object. A missing gate is a closed gate.

Config keys owned here (documented in ``services/persistence_service.py``):

- ``cleanup_enabled``: ``True | False | None``; ``None`` means unresolved.
- ``cleanup_cloud``: run the cleanup pass in the cloud rather than on this Mac.
- ``cleanup_mode``, ``cleanup_tone``: the fallback mode and its tone.
- ``mode_by_app``: ``{bundle_id: mode}`` user overrides.
- ``context_awareness``: whether the built-in bundle table applies.
- ``include_selection``: whether the selected-text probe runs when capturing.
- ``vocabulary_terms``, ``vocabulary_replacements``: the vocabulary.
- ``snippets``: the snippet list.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace as dataclass_replace
from typing import Any

from cleanup.coding_mode import SPOKEN_SYMBOLS
from cleanup.context import (
    DEFAULT_MODE_BY_BUNDLE,
    MODE_OVERRIDES_CONFIG_KEY,
)
from cleanup.modes import (
    CONFIG_MODE_KEY,
    CONFIG_TONE_KEY,
    DEFAULT_MODE_ID,
    DEFAULT_TONE_ID,
    MODE_IDS,
    MODES,
    TONE_IDS,
    TONES,
)
from cleanup.snippets import (
    CONFIG_SNIPPETS_KEY,
    FREE_SNIPPET_LIMIT,
    Snippet,
    SnippetError,
    snippets_from_config,
    snippets_to_config,
)
from cleanup.vocabulary import (
    FREE_TERM_LIMIT,
    Replacement,
    Vocabulary,
    VocabularyError,
    export_csv,
    import_csv,
    vocabulary_from_config,
    vocabulary_to_config,
)
from services.persistence_service import (
    CONFIG_CLEANUP_CLOUD,
    CONFIG_CLEANUP_ENABLED,
    CONFIG_CLOUD_MODE,
    CLOUD_MODE_MURMUR,
    DEFAULT_CONFIG,
    resolve_cleanup_enabled,
)
from ui.settings import register_tab
from ui.settings.base import TAB_SMART, TabContext

logger = logging.getLogger(__name__)

# -- config keys --------------------------------------------------------------

CONFIG_CONTEXT_AWARENESS = "context_awareness"
CONFIG_INCLUDE_SELECTION = "include_selection"

#: Every key this tab may write, in the order :meth:`SmartTabModel.as_config`
#: builds them. Named so a test can assert the tab writes nothing else.
OWNED_CONFIG_KEYS: tuple[str, ...] = (
    CONFIG_CLEANUP_ENABLED,
    CONFIG_CLEANUP_CLOUD,
    CONFIG_MODE_KEY,
    CONFIG_TONE_KEY,
    MODE_OVERRIDES_CONFIG_KEY,
    CONFIG_CONTEXT_AWARENESS,
    CONFIG_INCLUDE_SELECTION,
    "vocabulary_terms",
    "vocabulary_replacements",
    CONFIG_SNIPPETS_KEY,
)

# -- the three questions this tab asks the Pro gate ---------------------------

FEATURE_CLEANUP = "cleanup"
FEATURE_VOCABULARY_BEYOND_FREE = "vocabulary_beyond_free"
FEATURE_SNIPPETS = "snippets"

#: Name of the injected ``is_pro_feature_enabled`` in ``TabContext.services``.
SERVICE_PRO_GATE = "pro_gate"

# -- copy ---------------------------------------------------------------------

SECTION_CLEANUP = "Cleanup"
SECTION_PER_APP = "Modes per app"
SECTION_VOCABULARY = "Vocabulary"
SECTION_SNIPPETS = "Snippets"
SECTION_SYMBOLS = "Spoken symbols in Code mode"

CLEANUP_CHECKBOX = "Tidy up what I dictate"
CLEANUP_HINT = (
    "A small local model removes filler words and fixes punctuation before the "
    "text is typed."
)
#: Shown while ``cleanup_enabled`` is still unresolved, so the checkbox state
#: reads as "what this Mac does" rather than "what you chose".
CLEANUP_DEFAULT_HINT = "Off below 16 GB by default"

CLOUD_CLEANUP_CHECKBOX = "Clean up in the cloud instead"
CLOUD_CLEANUP_HINT = "Available when transcription runs on Murmur Cloud."

MODE_LABEL = "Default mode"
TONE_LABEL = "Tone"

CONTEXT_CHECKBOX = "Pick the mode from the app I am dictating into"
CONTEXT_HINT = (
    "Mail apps get Mail, chat apps get Message, editors get Code. Your own "
    "per-app choices below always win."
)

SELECTION_CHECKBOX = "Read the text I have selected"
SELECTION_HINT = "Not used by any mode yet."

PRO_HINT = "Pro"
PRO_CLEANUP_HINT = "Pro — cleanup, modes and per-app choices are part of Murmur Pro."

VOCABULARY_HINT = (
    "Terms bias what the engine hears — names, jargon, product names. One per line."
)
REPLACEMENTS_HINT = "Replacements rewrite finished text, whole words only."
VOCABULARY_LIMIT_HINT = f"Only the first {FREE_TERM_LIMIT} terms are used on the free plan."

SNIPPETS_HINT = (
    "Say the trigger and Murmur types the text. Longer triggers win over "
    "shorter ones they contain."
)
SNIPPETS_LIMIT_HINT = f"Only the first {FREE_SNIPPET_LIMIT} snippets are used on the free plan."

SYMBOLS_HINT = "Code mode turns these spoken words into symbols."

#: Columns of the replacements table, in display order. Ported unchanged from
#: the Wave 1 window, along with the model below.
REPLACEMENT_COLUMNS: tuple[str, ...] = ("from", "to", "match_case")
REPLACEMENT_COLUMN_TITLES = {
    "from": "From",
    "to": "To",
    "match_case": "Match case",
}

#: Columns of the snippets table, in display order.
SNIPPET_COLUMNS: tuple[str, ...] = ("trigger", "text")
SNIPPET_COLUMN_TITLES = {"trigger": "When I say", "text": "Type this"}

#: Columns of the per-app overrides table, in display order.
OVERRIDE_COLUMNS: tuple[str, ...] = ("bundle_id", "mode")
OVERRIDE_COLUMN_TITLES = {"bundle_id": "App (bundle id)", "mode": "Mode"}

VOCABULARY_CSV_NAME = "murmur-vocabulary.csv"


def spoken_symbol_rows() -> tuple[tuple[str, str], ...]:
    """The Code-mode symbol table, deduplicated for a read-only reference list.

    ``SPOKEN_SYMBOLS`` lists several spellings of the same symbol ("star" and
    "asterisk" both give ``*``) because the recogniser needs every one of them.
    A reference list wants one row per symbol, naming all the ways to say it.
    """
    by_symbol: dict[str, list[str]] = {}
    for phrase, symbol in SPOKEN_SYMBOLS:
        by_symbol.setdefault(symbol, []).append(phrase)
    return tuple((symbol, " · ".join(phrases)) for symbol, phrases in by_symbol.items())


# -- Vocabulary ---------------------------------------------------------------


class VocabularySectionModel:
    """Editing state for the "Vocabulary" group.

    Ported whole from the Wave 1 single-page window (``settings_window.py``,
    archived by E3f) rather than rewritten: the editing rules — how a box of
    text becomes a term list, what a fresh replacement row holds, which row is
    selected after a removal, what actually gets saved — were already right and
    already tested. The AppKit code below is a rendering of this.
    """

    def __init__(self, config: dict) -> None:
        assert config is not None, "config is required"
        vocabulary = _vocabulary_or_empty(config)
        self.terms: list[str] = list(vocabulary.terms)
        self.replacements: list[Replacement] = list(vocabulary.replacements)

    # -- terms -----------------------------------------------------------

    @property
    def terms_text(self) -> str:
        """The terms box's contents: one term per line."""
        return "\n".join(self.terms)

    def set_terms_text(self, text: str) -> None:
        """Read the terms box back. One per line; blanks and repeats dropped.

        Deduplicating here rather than at save time means the box the user
        sees next is the list Murmur actually holds.
        """
        assert text is not None, "text is required"
        seen: set[str] = set()
        terms: list[str] = []
        for line in text.splitlines():
            term = line.strip()
            if not term or term in seen:
                continue
            seen.add(term)
            terms.append(term)
        self.terms = terms

    # -- replacements ----------------------------------------------------

    @property
    def row_count(self) -> int:
        """Rows in the replacements table, blank ones included."""
        return len(self.replacements)

    def value_for(self, row: int, column: str):
        """The value one table cell displays."""
        assert 0 <= row < len(self.replacements), f"row {row} is out of range"
        assert column in REPLACEMENT_COLUMNS, f"unknown column: {column!r}"
        replacement = self.replacements[row]
        if column == "from":
            return replacement.from_text
        if column == "to":
            return replacement.to_text
        return replacement.match_case

    def set_value(self, row: int, column: str, value) -> None:
        """Write one edited table cell back into the row."""
        assert 0 <= row < len(self.replacements), f"row {row} is out of range"
        assert column in REPLACEMENT_COLUMNS, f"unknown column: {column!r}"
        replacement = self.replacements[row]
        if column == "match_case":
            assert isinstance(value, bool), f"match_case must be a bool, got {value!r}"
            self.replacements[row] = dataclass_replace(replacement, match_case=value)
            return
        assert isinstance(value, str), f"{column} must be a string, got {value!r}"
        field = "from_text" if column == "from" else "to_text"
        self.replacements[row] = dataclass_replace(replacement, **{field: value})

    def add_replacement(self) -> int:
        """Append a blank row and return its index, so the table can edit it."""
        self.replacements.append(Replacement(from_text="", to_text="", match_case=False))
        return len(self.replacements) - 1

    def remove_replacement(self, row: int) -> int:
        """Delete a row; return the row to select next, or -1 when none is left."""
        assert 0 <= row < len(self.replacements), f"row {row} is out of range"
        del self.replacements[row]
        if not self.replacements:
            return -1
        return min(row, len(self.replacements) - 1)

    # -- persistence -----------------------------------------------------

    @property
    def vocabulary(self) -> Vocabulary:
        """What gets saved. A row with no ``from`` text replaces nothing, so
        the half-typed row left behind by a stray "+" is dropped rather than
        persisted as a rule that can never match.
        """
        return Vocabulary(
            terms=tuple(self.terms),
            replacements=tuple(item for item in self.replacements if item.from_text.strip()),
        )

    def to_config(self) -> dict:
        """The two config keys the vocabulary lives in."""
        return vocabulary_to_config(self.vocabulary)

    def export_text(self) -> str:
        """CSV for the Export button."""
        return export_csv(self.vocabulary)

    def import_text(self, text: str) -> None:
        """Replace both lists from CSV.

        Raises :class:`~cleanup.vocabulary.VocabularyError`, whose message
        names the offending line, and leaves the current lists untouched when
        it does — a bad file never half-imports.
        """
        imported = import_csv(text)
        self.terms = list(imported.terms)
        self.replacements = list(imported.replacements)


def _vocabulary_or_empty(config: dict) -> Vocabulary:
    """The stored vocabulary, or an empty one when the config is unreadable.

    A hand-edited config with the wrong shape must not stop Settings opening —
    the window is where a user goes to *fix* things. Nothing is overwritten:
    :meth:`SmartTabModel.apply` reports only keys the user changed, so an
    unreadable list stays on disk until it is edited.
    """
    try:
        return vocabulary_from_config(config)
    except VocabularyError as error:
        logger.warning("Ignoring unreadable vocabulary in the config: %s", error)
        return Vocabulary()


# -- Snippets -----------------------------------------------------------------


class SnippetsSectionModel:
    """Editing state for the "Snippets" table: the same shape as the
    replacements table above, so the two behave identically under the hand.
    """

    def __init__(self, config: dict) -> None:
        assert config is not None, "config is required"
        self.snippets: list[Snippet] = list(_snippets_or_empty(config))

    @property
    def row_count(self) -> int:
        """Rows in the table, blank ones included."""
        return len(self.snippets)

    def value_for(self, row: int, column: str) -> str:
        """The value one table cell displays."""
        assert 0 <= row < len(self.snippets), f"row {row} is out of range"
        assert column in SNIPPET_COLUMNS, f"unknown column: {column!r}"
        snippet = self.snippets[row]
        return snippet.trigger if column == "trigger" else snippet.text

    def set_value(self, row: int, column: str, value: str) -> None:
        """Write one edited table cell back into the row."""
        assert 0 <= row < len(self.snippets), f"row {row} is out of range"
        assert column in SNIPPET_COLUMNS, f"unknown column: {column!r}"
        assert isinstance(value, str), f"{column} must be a string, got {value!r}"
        field = "trigger" if column == "trigger" else "text"
        self.snippets[row] = dataclass_replace(self.snippets[row], **{field: value})

    def add_snippet(self) -> int:
        """Append a blank row and return its index, so the table can edit it."""
        self.snippets.append(Snippet(trigger="", text=""))
        return len(self.snippets) - 1

    def remove_snippet(self, row: int) -> int:
        """Delete a row; return the row to select next, or -1 when none is left."""
        assert 0 <= row < len(self.snippets), f"row {row} is out of range"
        del self.snippets[row]
        if not self.snippets:
            return -1
        return min(row, len(self.snippets) - 1)

    @property
    def saved(self) -> tuple[Snippet, ...]:
        """What gets saved: a snippet with no trigger can never fire, so the
        half-typed row left behind by a stray "+" is dropped."""
        return tuple(snippet for snippet in self.snippets if snippet.trigger.strip())

    def to_config(self) -> dict:
        """The one config key snippets live in."""
        return snippets_to_config(self.saved)


def _snippets_or_empty(config: dict) -> tuple[Snippet, ...]:
    """The stored snippets, or none when the config is unreadable.

    Same reasoning as :func:`_vocabulary_or_empty`: Settings must open.
    """
    try:
        return snippets_from_config(config)
    except SnippetError as error:
        logger.warning("Ignoring unreadable snippets in the config: %s", error)
        return ()


# -- Per-app modes ------------------------------------------------------------


@dataclass(frozen=True)
class OverrideRow:
    """One ``mode_by_app`` entry as the table draws it."""

    bundle_id: str
    mode: str

    @property
    def mode_index(self) -> int:
        """The mode's row in the popup cell; unknown ids show as the default."""
        return MODE_IDS.index(self.mode) if self.mode in MODE_IDS else 0

    @property
    def mode_display_name(self) -> str:
        return MODES[MODE_IDS[self.mode_index]].display_name


class OverridesSectionModel:
    """Editing state for the per-app mode table.

    The overrides live in config as a ``{bundle_id: mode}`` dict, but are edited
    as an ordered list of rows: a dict cannot hold the blank row a "+" makes,
    and rebuilding it on every keystroke would reorder the table under the
    cursor. :meth:`to_config` folds the list back down, dropping blank rows.
    """

    def __init__(self, config: dict) -> None:
        assert config is not None, "config is required"
        stored = config.get(MODE_OVERRIDES_CONFIG_KEY) or {}
        if not isinstance(stored, dict):
            logger.warning("Ignoring unreadable %s in the config", MODE_OVERRIDES_CONFIG_KEY)
            stored = {}
        self.rows: list[OverrideRow] = [
            OverrideRow(bundle_id=str(bundle_id), mode=_known_mode(mode))
            for bundle_id, mode in sorted(stored.items())
        ]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def value_for(self, row: int, column: str):
        """The value one table cell holds: the bundle id, or the mode's row.

        The mode column is drawn with a popup cell, whose object value is the
        index of the chosen item — so that is what this returns for it.
        """
        assert 0 <= row < len(self.rows), f"row {row} is out of range"
        assert column in OVERRIDE_COLUMNS, f"unknown column: {column!r}"
        entry = self.rows[row]
        return entry.bundle_id if column == "bundle_id" else entry.mode_index

    def set_value(self, row: int, column: str, value) -> None:
        """Write one edited table cell back into the row."""
        assert 0 <= row < len(self.rows), f"row {row} is out of range"
        assert column in OVERRIDE_COLUMNS, f"unknown column: {column!r}"
        if column == "bundle_id":
            assert isinstance(value, str), f"bundle_id must be a string, got {value!r}"
            self.rows[row] = dataclass_replace(self.rows[row], bundle_id=value.strip())
            return
        index = int(value)
        assert 0 <= index < len(MODE_IDS), f"mode row {index} is out of range"
        self.rows[row] = dataclass_replace(self.rows[row], mode=MODE_IDS[index])

    def add_override(self, bundle_id: str = "", mode: str = DEFAULT_MODE_ID) -> int:
        """Append a row and return its index, so the table can edit it."""
        self.rows.append(OverrideRow(bundle_id=bundle_id.strip(), mode=_known_mode(mode)))
        return len(self.rows) - 1

    def remove_override(self, row: int) -> int:
        """Delete a row; return the row to select next, or -1 when none is left."""
        assert 0 <= row < len(self.rows), f"row {row} is out of range"
        del self.rows[row]
        if not self.rows:
            return -1
        return min(row, len(self.rows) - 1)

    def to_config(self) -> dict:
        """The one config key the overrides live in.

        A blank bundle id is dropped rather than saved as an override on an app
        that cannot exist. The last row wins when a bundle id is listed twice,
        which is what the user sees happen as they retype one.
        """
        overrides = {
            entry.bundle_id: entry.mode for entry in self.rows if entry.bundle_id.strip()
        }
        return {MODE_OVERRIDES_CONFIG_KEY: overrides}


def _known_mode(mode: Any) -> str:
    """``mode`` when it is one Murmur knows, else the default.

    A config naming a mode that no longer exists shows as Dictation rather than
    taking the window down on the way up.
    """
    if mode in MODE_IDS:
        return str(mode)
    if mode is not None:
        logger.warning("Ignoring unknown cleanup mode %r; using %r", mode, DEFAULT_MODE_ID)
    return DEFAULT_MODE_ID


def built_in_mode_for(bundle_id: str) -> str | None:
    """What the shipped table would choose for ``bundle_id``, if anything.

    Only used to describe a row the user is editing; the actual resolution is
    :func:`cleanup.context.resolve_mode`'s job, never this tab's.
    """
    return DEFAULT_MODE_BY_BUNDLE.get(bundle_id)


def _known_tone(tone: Any) -> str:
    """``tone`` when it is one Murmur knows, else the default."""
    if tone in TONE_IDS:
        return str(tone)
    if tone is not None:
        logger.warning("Ignoring unknown cleanup tone %r; using %r", tone, DEFAULT_TONE_ID)
    return DEFAULT_TONE_ID


# -- The model ----------------------------------------------------------------


class SmartTabModel:
    """Editing state for Settings → Smart.

    Built from a config dict, which it never mutates, and from one injected
    ``pro_gate``. :meth:`apply` reports only the keys the user actually
    changed, so opening Settings and closing it again writes nothing at all.

    ``cleanup_probe`` answers "what would this Mac do?" for an unresolved
    ``cleanup_enabled``; the tab leaves it out and gets the real RAM probe,
    a test passes one and needs no cleanup runtime at all.
    """

    def __init__(
        self,
        config: dict,
        *,
        pro_gate: Callable[[str], bool] | None,
        engine_info: Any | None = None,
        cleanup_probe: Callable[[], bool] | None = None,
    ) -> None:
        assert config is not None, "config is required"
        self._config = config
        self._pro_gate = pro_gate
        self._cleanup_probe = cleanup_probe
        self.engine_info = engine_info
        self._read()
        self._original = self.as_config()

    def _read(self) -> None:
        """(Re-)build every field from the config dict this model was given."""
        config = self._config
        stored_cleanup = config.get(CONFIG_CLEANUP_ENABLED)
        #: ``None`` means "nobody has decided yet"; anything but a bool is read
        #: the same way, so a hand-edited "yes" asks the machine rather than
        #: being honoured as truthy.
        self._cleanup_enabled: bool | None = (
            stored_cleanup if isinstance(stored_cleanup, bool) else None
        )
        self.cleanup_cloud: bool = bool(
            config.get(CONFIG_CLEANUP_CLOUD, DEFAULT_CONFIG[CONFIG_CLEANUP_CLOUD])
        )
        self.cloud_mode: str = config.get(
            CONFIG_CLOUD_MODE, DEFAULT_CONFIG[CONFIG_CLOUD_MODE]
        )
        self.mode: str = _known_mode(config.get(CONFIG_MODE_KEY, DEFAULT_MODE_ID))
        self.tone: str = _known_tone(config.get(CONFIG_TONE_KEY, DEFAULT_TONE_ID))
        self.context_awareness: bool = bool(
            config.get(CONFIG_CONTEXT_AWARENESS, DEFAULT_CONFIG[CONFIG_CONTEXT_AWARENESS])
        )
        self.include_selection: bool = bool(
            config.get(CONFIG_INCLUDE_SELECTION, DEFAULT_CONFIG[CONFIG_INCLUDE_SELECTION])
        )
        self.overrides = OverridesSectionModel(config)
        self.vocabulary = VocabularySectionModel(config)
        self.snippets = SnippetsSectionModel(config)

    # -- entitlement -----------------------------------------------------

    def is_pro_feature_enabled(self, feature: str) -> bool:
        """Ask the injected Pro gate about one feature. The only such question.

        No gate means no Pro feature: a build that cannot ask must not assume
        yes, and a gate that falls over is a closed one.
        """
        assert feature, "feature is required"
        if self._pro_gate is None:
            return False
        try:
            return bool(self._pro_gate(feature))
        except Exception as error:  # noqa: BLE001 - a broken gate is a closed gate
            logger.warning("The Pro gate could not be asked about %s: %s", feature, error)
            return False

    @property
    def cleanup_entitled(self) -> bool:
        """Whether the cleanup group may be touched at all."""
        return self.is_pro_feature_enabled(FEATURE_CLEANUP)

    @property
    def cleanup_pro_hint(self) -> str | None:
        """Why the cleanup controls are dead, or None when they work."""
        return None if self.cleanup_entitled else PRO_CLEANUP_HINT

    # -- cleanup ---------------------------------------------------------

    @property
    def cleanup_enabled(self) -> bool:
        """What the checkbox shows: the stored choice, else this Mac's default."""
        if self._cleanup_enabled is not None:
            return self._cleanup_enabled
        return resolve_cleanup_enabled({}, probe=self._cleanup_probe)

    @property
    def cleanup_is_machine_default(self) -> bool:
        """Whether the checkbox is still showing a default nobody chose."""
        return self._cleanup_enabled is None

    @property
    def cleanup_default_hint(self) -> str | None:
        """The line that says the checkbox is a default, or None once it is a choice."""
        return CLEANUP_DEFAULT_HINT if self.cleanup_is_machine_default else None

    def set_cleanup_enabled(self, enabled: bool) -> None:
        """Record an explicit yes or no, replacing any machine default.

        Landing back on the value the machine would have chosen still writes a
        real bool: the user has now answered the question, and the answer is
        worth keeping when they next move the Mac's RAM or the default changes.
        """
        assert isinstance(enabled, bool), f"expected a bool, got {enabled!r}"
        if not self.cleanup_entitled:
            logger.info("Cleanup refused: the Pro gate does not allow it")
            return
        self._cleanup_enabled = enabled

    @property
    def cloud_cleanup_available(self) -> bool:
        """Whether cleaning up in the cloud can be chosen right now."""
        return self.cloud_mode == CLOUD_MODE_MURMUR and self.cleanup_entitled

    def set_cleanup_cloud(self, enabled: bool) -> None:
        """Choose where cleanup runs. Refused unless Murmur Cloud is on."""
        assert isinstance(enabled, bool), f"expected a bool, got {enabled!r}"
        if not self.cloud_cleanup_available:
            logger.info("Cloud cleanup refused: Murmur Cloud is not the active engine")
            return
        self.cleanup_cloud = enabled

    # -- mode and tone ---------------------------------------------------

    @property
    def mode_titles(self) -> tuple[str, ...]:
        return tuple(MODES[mode_id].display_name for mode_id in MODE_IDS)

    @property
    def mode_index(self) -> int:
        return MODE_IDS.index(self.mode)

    @property
    def mode_description(self) -> str:
        """The chosen mode's one-line description, from ``modes.json``."""
        return MODES[self.mode].description

    def set_mode(self, mode_id: str) -> None:
        assert mode_id in MODE_IDS, (
            f"Invalid cleanup mode {mode_id!r}; expected one of {', '.join(MODE_IDS)}"
        )
        if not self.cleanup_entitled:
            logger.info("Mode refused: the Pro gate does not allow cleanup")
            return
        self.mode = mode_id

    def set_mode_index(self, index: int) -> None:
        assert 0 <= index < len(MODE_IDS), f"mode row {index} is out of range"
        self.set_mode(MODE_IDS[index])

    @property
    def tone_titles(self) -> tuple[str, ...]:
        return tuple(TONES[tone_id].display_name for tone_id in TONE_IDS)

    @property
    def tone_index(self) -> int:
        return TONE_IDS.index(self.tone)

    @property
    def tone_description(self) -> str:
        return TONES[self.tone].description

    def set_tone(self, tone_id: str) -> None:
        assert tone_id in TONE_IDS, (
            f"Invalid cleanup tone {tone_id!r}; expected one of {', '.join(TONE_IDS)}"
        )
        if not self.cleanup_entitled:
            logger.info("Tone refused: the Pro gate does not allow cleanup")
            return
        self.tone = tone_id

    def set_tone_index(self, index: int) -> None:
        assert 0 <= index < len(TONE_IDS), f"tone row {index} is out of range"
        self.set_tone(TONE_IDS[index])

    # -- context ---------------------------------------------------------

    def set_context_awareness(self, enabled: bool) -> None:
        assert isinstance(enabled, bool), f"expected a bool, got {enabled!r}"
        if not self.cleanup_entitled:
            logger.info("Context awareness refused: the Pro gate does not allow cleanup")
            return
        self.context_awareness = enabled

    def set_include_selection(self, enabled: bool) -> None:
        """Turn the selected-text probe on or off.

        Deliberately not behind the Pro gate: this is a capture permission, not
        a cleanup feature, and a user must always be able to turn it off.
        """
        assert isinstance(enabled, bool), f"expected a bool, got {enabled!r}"
        self.include_selection = enabled

    # -- free-plan limits ------------------------------------------------

    @property
    def vocabulary_limit_hint(self) -> str | None:
        """The line about the free-plan term cap, or None when it does not bite.

        Nothing is truncated here: the stored list keeps every term the user
        typed, and Wave 4's pipeline takes the first :data:`FREE_TERM_LIMIT` of
        them at use. Losing terms on save would make an expiring plan destroy
        data the user could not get back.
        """
        if self.is_pro_feature_enabled(FEATURE_VOCABULARY_BEYOND_FREE):
            return None
        if len(self.vocabulary.terms) <= FREE_TERM_LIMIT:
            return None
        return VOCABULARY_LIMIT_HINT

    @property
    def snippets_limit_hint(self) -> str | None:
        """The line about the free-plan snippet cap, or None. Truncates nothing."""
        if self.is_pro_feature_enabled(FEATURE_SNIPPETS):
            return None
        if len(self.snippets.saved) <= FREE_SNIPPET_LIMIT:
            return None
        return SNIPPETS_LIMIT_HINT

    # -- persistence -----------------------------------------------------

    def as_config(self) -> dict:
        """Every key this tab owns, at its current value.

        ``cleanup_enabled`` may be ``None`` here, and that is the point: an
        untouched machine default differs from no value at all, so :meth:`apply`
        reports nothing and the file keeps saying "not decided yet".
        """
        data: dict[str, Any] = {
            CONFIG_CLEANUP_ENABLED: self._cleanup_enabled,
            CONFIG_CLEANUP_CLOUD: self.cleanup_cloud,
            CONFIG_MODE_KEY: self.mode,
            CONFIG_TONE_KEY: self.tone,
            CONFIG_CONTEXT_AWARENESS: self.context_awareness,
            CONFIG_INCLUDE_SELECTION: self.include_selection,
        }
        data.update(self.overrides.to_config())
        data.update(self.vocabulary.to_config())
        data.update(self.snippets.to_config())
        assert set(data) == set(OWNED_CONFIG_KEYS), (
            "the Smart tab wrote a key it does not own: "
            f"{sorted(set(data) ^ set(OWNED_CONFIG_KEYS))}"
        )
        return data

    def apply(self) -> dict:
        """The keys that differ from the config this model was built on."""
        current = self.as_config()
        return {key: value for key, value in current.items() if self._original[key] != value}

    def mark_saved(self) -> None:
        """Called once :meth:`apply`'s dict has been persisted."""
        self._original = self.as_config()

    def refresh(self) -> None:
        """Re-read every field from the config, e.g. after another tab wrote.

        The Engine tab turning Murmur Cloud on or off changes whether cloud
        cleanup can be chosen, so this tab has to look again.
        """
        self._read()
        self._original = self.as_config()


# -- AppKit ------------------------------------------------------------------
#
# Everything below draws :class:`SmartTabModel`. AppKit is imported inside the
# functions that need it (and inside the ``ui.settings.base`` helpers), so this
# module stays importable in a headless test run.


_TABLE_SOURCE_CLASS: Any = None


def _table_source_class() -> Any:
    """Define (once) the NSObject that forwards table calls to Python."""
    global _TABLE_SOURCE_CLASS
    if _TABLE_SOURCE_CLASS is not None:
        return _TABLE_SOURCE_CLASS

    import objc
    from Foundation import NSObject

    class MurmurSmartTableSource(NSObject):
        """An ``NSTableViewDataSource`` over three Python callables."""

        @objc.python_method
        def configure(self, row_count, value_for, set_value):
            self._row_count = row_count
            self._value_for = value_for
            self._set_value = set_value

        def numberOfRowsInTableView_(self, table_view):
            return self._row_count()

        def tableView_objectValueForTableColumn_row_(self, table_view, column, row):
            return self._value_for(int(row), str(column.identifier()))

        def tableView_setObjectValue_forTableColumn_row_(self, table_view, value, column, row):
            self._set_value(int(row), str(column.identifier()), value)

    _TABLE_SOURCE_CLASS = MurmurSmartTableSource
    return _TABLE_SOURCE_CLASS


def make_table_source(
    row_count: Callable[[], int],
    value_for: Callable[[int, str], Any],
    set_value: Callable[[int, str, Any], None],
) -> Any:
    """A data source for one table. The caller must keep it alive: AppKit
    holds a data source weakly, and a collected one crashes the table."""
    assert callable(row_count), "row_count must be callable"
    source = _table_source_class().alloc().init()
    source.configure(row_count, value_for, set_value)
    return source


@dataclass(frozen=True)
class ColumnSpec:
    """One table column: identifier, header, width, and what draws its cells.

    ``kind`` is ``"text"`` (editable field), ``"switch"`` (a checkbox, whose
    object value is a bool) or ``"popup"`` (a menu, whose object value is the
    chosen row) — the three the Smart tab's three tables need between them.
    """

    identifier: str
    title: str
    width: int
    kind: str = "text"
    titles: tuple[str, ...] = ()


def _make_table(
    specs: Iterable[ColumnSpec],
    theme: Any,
    *,
    accessibility: str,
    width: int,
    height: int,
) -> Any:
    """An ``NSTableView`` with the given columns, styled like the old window."""
    from Cocoa import NSButtonCell, NSMakeRect, NSPopUpButtonCell, NSTableColumn, NSTableView

    table = NSTableView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
    table.setUsesAlternatingRowBackgroundColors_(True)
    table.setAllowsMultipleSelection_(False)
    table.setAppearance_(theme.control_appearance())
    table.setAccessibilityLabel_(accessibility)
    for spec in specs:
        column = NSTableColumn.alloc().initWithIdentifier_(spec.identifier)
        column.headerCell().setStringValue_(spec.title)
        column.setWidth_(spec.width)
        if spec.kind == "switch":
            cell = NSButtonCell.alloc().init()
            cell.setButtonType_(3)  # NSSwitchButton
            cell.setTitle_("")
            column.setDataCell_(cell)
        elif spec.kind == "popup":
            cell = NSPopUpButtonCell.alloc().init()
            cell.setBordered_(False)
            for title in spec.titles:
                cell.addItemWithTitle_(title)
            column.setDataCell_(cell)
        else:
            column.dataCell().setEditable_(True)
        table.addTableColumn_(column)
    return table


def _make_scroll(document: Any, width: int, height: int) -> Any:
    """A bordered scroll view of a fixed size around ``document``.

    The size is pinned with constraints because an ``NSStackView`` gives its
    arranged subviews their intrinsic size, and a scroll view has none.
    """
    from Cocoa import NSBezelBorder, NSMakeRect, NSScrollView

    scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
    scroll.setHasVerticalScroller_(True)
    scroll.setHasHorizontalScroller_(False)
    scroll.setBorderType_(NSBezelBorder)
    scroll.setDocumentView_(document)
    scroll.setTranslatesAutoresizingMaskIntoConstraints_(False)
    scroll.widthAnchor().constraintEqualToConstant_(float(width)).setActive_(True)
    scroll.heightAnchor().constraintEqualToConstant_(float(height)).setActive_(True)
    return scroll


def _make_terms_view(text: str, theme: Any, width: int, height: int) -> Any:
    """The plain-text box holding one vocabulary term per line."""
    from Cocoa import NSFont, NSMakeRect, NSTextView

    view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
    view.setString_(text)
    view.setFont_(NSFont.systemFontOfSize_(12))
    view.setRichText_(False)
    view.setAutomaticQuoteSubstitutionEnabled_(False)
    view.setAppearance_(theme.control_appearance())
    view.setAccessibilityLabel_("Vocabulary terms, one per line")
    return view


def _make_spacer(height: int) -> Any:
    """A view that only takes up room — the bottom padding of a scrolled stack."""
    from Cocoa import NSMakeRect, NSView

    spacer = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 1, height))
    spacer.setTranslatesAutoresizingMaskIntoConstraints_(False)
    spacer.heightAnchor().constraintEqualToConstant_(float(height)).setActive_(True)
    return spacer


def _imported_theme() -> Any:
    """``ui_theme``, imported on demand, for a context that carries none."""
    import ui_theme

    return ui_theme


def _label(text: str, theme: Any) -> Any:
    """The ordinary 12 pt line that names a popup."""
    from ui.settings.base import make_label

    return make_label(text, theme, size=12)


def _show(view: Any, text: str | None) -> None:
    """Show ``view`` carrying ``text``, or hide it when there is nothing to say."""
    if view is None:
        return
    if text is None:
        view.setHidden_(True)
        return
    view.setStringValue_(text)
    view.setHidden_(False)


#: The Smart tab is the tallest of the five: cleanup, per-app modes, two
#: tables, a text box and a reference list do not fit in 660 points, so this
#: tab — alone among the five — scrolls its own content.
CONTENT_WIDTH = 440
TAB_WIDTH = 500
TAB_HEIGHT = 560


@register_tab
class SmartTab:
    """The AppKit rendering of :class:`SmartTabModel`.

    Like the other tabs there is no Save button: every control writes through
    :meth:`_commit` as soon as it changes, so a setting the user can see is a
    setting already on disk.
    """

    identifier = TAB_SMART
    title = "Smart"

    def __init__(self) -> None:
        self.context: TabContext | None = None
        self.model: SmartTabModel | None = None
        self._theme: Any = None
        self._view: Any = None
        self._cleanup_checkbox: Any = None
        self._cleanup_default_hint: Any = None
        self._cloud_checkbox: Any = None
        self._mode_popup: Any = None
        self._mode_hint: Any = None
        self._tone_popup: Any = None
        self._tone_hint: Any = None
        self._pro_hint: Any = None
        self._context_checkbox: Any = None
        self._selection_checkbox: Any = None
        self._overrides_table: Any = None
        self._terms_view: Any = None
        self._replacements_table: Any = None
        self._vocabulary_hint: Any = None
        self._snippets_table: Any = None
        self._snippets_hint: Any = None
        #: Controls that die with the Pro gate, and the table data sources,
        #: which AppKit holds weakly and this tab must keep alive.
        self._gated: list[Any] = []
        self._sources: list[Any] = []

    # -- building --------------------------------------------------------

    def _make_model(self, context: TabContext) -> SmartTabModel:
        return SmartTabModel(
            context.config,
            pro_gate=context.service(SERVICE_PRO_GATE),
            engine_info=context.engine_info,
        )

    def build(self, context: TabContext) -> Any:
        """Lay the tab out and return its scrolling content view."""
        assert context is not None, "context is required"
        from Cocoa import NSMakeRect, NSScrollView, NSView

        from ui.settings.base import CONTENT_MARGIN, ROW_SPACING, stack_vertical

        self.context = context
        self.model = self._make_model(context)
        self._theme = context.theme if context.theme is not None else _imported_theme()

        rows = [
            *self._build_cleanup(),
            *self._build_per_app(),
            *self._build_vocabulary(),
            *self._build_snippets(),
            *self._build_symbols(),
            _make_spacer(CONTENT_MARGIN),
        ]
        stack = stack_vertical(rows, spacing=ROW_SPACING)

        scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(0, 0, TAB_WIDTH, TAB_HEIGHT)
        )
        scroll.setHasVerticalScroller_(True)
        scroll.setHasHorizontalScroller_(False)
        scroll.setBorderType_(0)  # NSNoBorder
        scroll.setDrawsBackground_(False)
        scroll.setAutoresizingMask_(2 | 16)  # width | height sizable
        scroll.setDocumentView_(stack)
        clip = scroll.contentView()
        # Leading, trailing and top only: the stack's own height then decides
        # how far the view scrolls. A bottom constraint here would clamp the
        # content to the visible height instead, which is the bug this whole
        # scroll view exists to avoid.
        stack.leadingAnchor().constraintEqualToAnchor_constant_(
            clip.leadingAnchor(), CONTENT_MARGIN
        ).setActive_(True)
        stack.trailingAnchor().constraintEqualToAnchor_constant_(
            clip.trailingAnchor(), -CONTENT_MARGIN
        ).setActive_(True)
        stack.topAnchor().constraintEqualToAnchor_constant_(
            clip.topAnchor(), CONTENT_MARGIN
        ).setActive_(True)

        container = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, TAB_WIDTH, TAB_HEIGHT))
        container.addSubview_(scroll)
        self._view = container
        self.refresh()
        return container

    def _build_cleanup(self) -> list[Any]:
        """Cleanup on/off, where it runs, and the mode and tone it runs with."""
        from ui.settings.base import make_checkbox, make_hint, make_popup, make_section_title

        theme = self._theme
        self._cleanup_checkbox = make_checkbox(
            CLEANUP_CHECKBOX, self.model.cleanup_enabled, theme, self._cleanup_changed
        )
        self._cleanup_default_hint = make_hint(CLEANUP_DEFAULT_HINT, theme)
        self._cloud_checkbox = make_checkbox(
            CLOUD_CLEANUP_CHECKBOX, self.model.cleanup_cloud, theme, self._cloud_changed
        )
        self._mode_popup = make_popup(
            list(self.model.mode_titles), self.model.mode_index, theme, self._mode_changed
        )
        self._mode_hint = make_hint(self.model.mode_description, theme)
        self._tone_popup = make_popup(
            list(self.model.tone_titles), self.model.tone_index, theme, self._tone_changed
        )
        self._tone_hint = make_hint(self.model.tone_description, theme)
        self._pro_hint = make_hint(PRO_CLEANUP_HINT, theme)

        self._gated.extend(
            [self._cleanup_checkbox, self._mode_popup, self._tone_popup]
        )
        return [
            make_section_title(SECTION_CLEANUP, theme),
            self._cleanup_checkbox,
            make_hint(CLEANUP_HINT, theme),
            self._cleanup_default_hint,
            self._pro_hint,
            self._cloud_checkbox,
            make_hint(CLOUD_CLEANUP_HINT, theme),
            _label(MODE_LABEL, theme),
            self._mode_popup,
            self._mode_hint,
            _label(TONE_LABEL, theme),
            self._tone_popup,
            self._tone_hint,
        ]

    def _build_per_app(self) -> list[Any]:
        """The bundle table switch, the user's own overrides, and the selection probe."""
        from ui.settings.base import (
            make_button,
            make_checkbox,
            make_hint,
            make_section_title,
            stack_horizontal,
        )

        theme = self._theme
        self._context_checkbox = make_checkbox(
            CONTEXT_CHECKBOX, self.model.context_awareness, theme, self._context_changed
        )
        self._selection_checkbox = make_checkbox(
            SELECTION_CHECKBOX, self.model.include_selection, theme, self._selection_changed
        )
        self._overrides_table = _make_table(
            (
                ColumnSpec("bundle_id", OVERRIDE_COLUMN_TITLES["bundle_id"], 250),
                ColumnSpec(
                    "mode",
                    OVERRIDE_COLUMN_TITLES["mode"],
                    150,
                    kind="popup",
                    titles=self.model.mode_titles,
                ),
            ),
            theme,
            accessibility="Modes chosen per app",
            width=CONTENT_WIDTH,
            height=110,
        )
        self._bind_source(
            self._overrides_table,
            lambda: self.model.overrides.row_count,
            self._override_value,
            self._set_override_value,
        )
        add_button = make_button("+", theme, self._add_override, width=40)
        remove_button = make_button("−", theme, self._remove_override, width=40)
        self._gated.extend([self._context_checkbox, self._overrides_table, add_button, remove_button])

        return [
            make_section_title(SECTION_PER_APP, theme),
            self._context_checkbox,
            make_hint(CONTEXT_HINT, theme),
            _make_scroll(self._overrides_table, CONTENT_WIDTH, 110),
            stack_horizontal([add_button, remove_button]),
            self._selection_checkbox,
            make_hint(SELECTION_HINT, theme),
        ]

    def _build_vocabulary(self) -> list[Any]:
        """The Wave 1 vocabulary editor: terms box, replacements table, CSV."""
        from ui.settings.base import (
            make_button,
            make_hint,
            make_section_title,
            stack_horizontal,
        )

        theme = self._theme
        self._terms_view = _make_terms_view(
            self.model.vocabulary.terms_text, theme, CONTENT_WIDTH, 84
        )
        self._replacements_table = _make_table(
            (
                ColumnSpec("from", REPLACEMENT_COLUMN_TITLES["from"], 170),
                ColumnSpec("to", REPLACEMENT_COLUMN_TITLES["to"], 170),
                ColumnSpec("match_case", REPLACEMENT_COLUMN_TITLES["match_case"], 90, "switch"),
            ),
            theme,
            accessibility="Text replacements",
            width=CONTENT_WIDTH,
            height=110,
        )
        self._bind_source(
            self._replacements_table,
            lambda: self.model.vocabulary.row_count,
            self._replacement_value,
            self._set_replacement_value,
        )
        self._vocabulary_hint = make_hint(VOCABULARY_LIMIT_HINT, theme)

        return [
            make_section_title(SECTION_VOCABULARY, theme),
            make_hint(VOCABULARY_HINT, theme),
            _make_scroll(self._terms_view, CONTENT_WIDTH, 84),
            self._vocabulary_hint,
            make_hint(REPLACEMENTS_HINT, theme),
            _make_scroll(self._replacements_table, CONTENT_WIDTH, 110),
            stack_horizontal(
                [
                    make_button("+", theme, self._add_replacement, width=40),
                    make_button("−", theme, self._remove_replacement, width=40),
                    make_button("Import CSV…", theme, self._import_vocabulary, width=120),
                    make_button("Export CSV…", theme, self._export_vocabulary, width=120),
                ]
            ),
        ]

    def _build_snippets(self) -> list[Any]:
        """Trigger/text rows, and the +/− that edit them."""
        from ui.settings.base import (
            make_button,
            make_hint,
            make_section_title,
            stack_horizontal,
        )

        theme = self._theme
        self._snippets_table = _make_table(
            (
                ColumnSpec("trigger", SNIPPET_COLUMN_TITLES["trigger"], 160),
                ColumnSpec("text", SNIPPET_COLUMN_TITLES["text"], 270),
            ),
            theme,
            accessibility="Snippets",
            width=CONTENT_WIDTH,
            height=110,
        )
        self._bind_source(
            self._snippets_table,
            lambda: self.model.snippets.row_count,
            self._snippet_value,
            self._set_snippet_value,
        )
        self._snippets_hint = make_hint(SNIPPETS_LIMIT_HINT, theme)

        return [
            make_section_title(SECTION_SNIPPETS, theme),
            make_hint(SNIPPETS_HINT, theme),
            _make_scroll(self._snippets_table, CONTENT_WIDTH, 110),
            stack_horizontal(
                [
                    make_button("+", theme, self._add_snippet, width=40),
                    make_button("−", theme, self._remove_snippet, width=40),
                ]
            ),
            self._snippets_hint,
        ]

    def _build_symbols(self) -> list[Any]:
        """A read-only reference for Code mode's spoken symbols."""
        from ui.settings.base import make_hint, make_section_title

        theme = self._theme
        lines = "\n".join(f"{symbol}   {phrases}" for symbol, phrases in spoken_symbol_rows())
        return [
            make_section_title(SECTION_SYMBOLS, theme),
            make_hint(SYMBOLS_HINT, theme),
            _make_scroll(_make_terms_view(lines, theme, CONTENT_WIDTH, 120), CONTENT_WIDTH, 120),
        ]

    def _bind_source(self, table: Any, row_count, value_for, set_value) -> None:
        """Give one table its data source and keep the source alive."""
        source = make_table_source(row_count, value_for, set_value)
        self._sources.append(source)
        table.setDataSource_(source)

    # -- refreshing ------------------------------------------------------

    def refresh(self) -> None:
        """Re-read config into every control, e.g. after another tab wrote."""
        if self.context is None or self.model is None:
            return
        self.model.refresh()
        from Cocoa import NSOffState, NSOnState

        def _set(checkbox: Any, on: bool) -> None:
            if checkbox is not None:
                checkbox.setState_(NSOnState if on else NSOffState)

        _set(self._cleanup_checkbox, self.model.cleanup_enabled)
        _set(self._cloud_checkbox, self.model.cleanup_cloud)
        _set(self._context_checkbox, self.model.context_awareness)
        _set(self._selection_checkbox, self.model.include_selection)
        if self._mode_popup is not None:
            self._mode_popup.selectItemAtIndex_(self.model.mode_index)
            self._mode_hint.setStringValue_(self.model.mode_description)
        if self._tone_popup is not None:
            self._tone_popup.selectItemAtIndex_(self.model.tone_index)
            self._tone_hint.setStringValue_(self.model.tone_description)
        if self._terms_view is not None:
            self._terms_view.setString_(self.model.vocabulary.terms_text)
        for table in (self._overrides_table, self._replacements_table, self._snippets_table):
            if table is not None:
                table.reloadData()
        self._refresh_entitlement()

    def _refresh_entitlement(self) -> None:
        """Grey out what this plan cannot use, and say why.

        The per-app table and its buttons are disabled here rather than refused
        in the model: an override is only ever read through the cleanup path
        the gate already closes, so the control being dead is the whole of the
        enforcement this tab owes. The cleanup switch, mode, tone and context
        switch refuse in the model as well, because an entitlement can go away
        under an open window and those four write on their own.
        """
        entitled = self.model.cleanup_entitled
        for control in self._gated:
            control.setEnabled_(entitled)
        if self._cloud_checkbox is not None:
            self._cloud_checkbox.setEnabled_(self.model.cloud_cleanup_available)
        _show(self._pro_hint, self.model.cleanup_pro_hint)
        _show(self._cleanup_default_hint, self.model.cleanup_default_hint)
        _show(self._vocabulary_hint, self.model.vocabulary_limit_hint)
        _show(self._snippets_hint, self.model.snippets_limit_hint)

    # -- saving ----------------------------------------------------------

    def _read_terms(self) -> None:
        """Fold the terms box's text back into the model before saving."""
        if self._terms_view is not None:
            self.model.vocabulary.set_terms_text(str(self._terms_view.string()))

    def _commit(self) -> dict:
        """Persist what changed and redraw the lines that follow from it."""
        assert self.context is not None and self.model is not None
        self._read_terms()
        changed = self.model.apply()
        if changed:
            self.context.save(changed)
            self.model.mark_saved()
        self._refresh_entitlement()
        return changed

    # -- cleanup actions -------------------------------------------------

    def _cleanup_changed(self, sender) -> None:
        from ui.settings.base import checkbox_is_on

        self.model.set_cleanup_enabled(checkbox_is_on(sender))
        self._commit()

    def _cloud_changed(self, sender) -> None:
        from ui.settings.base import checkbox_is_on

        self.model.set_cleanup_cloud(checkbox_is_on(sender))
        self._commit()

    def _mode_changed(self, sender) -> None:
        self.model.set_mode_index(sender.indexOfSelectedItem())
        self._mode_hint.setStringValue_(self.model.mode_description)
        self._commit()

    def _tone_changed(self, sender) -> None:
        self.model.set_tone_index(sender.indexOfSelectedItem())
        self._tone_hint.setStringValue_(self.model.tone_description)
        self._commit()

    def _context_changed(self, sender) -> None:
        from ui.settings.base import checkbox_is_on

        self.model.set_context_awareness(checkbox_is_on(sender))
        self._commit()

    def _selection_changed(self, sender) -> None:
        from ui.settings.base import checkbox_is_on

        self.model.set_include_selection(checkbox_is_on(sender))
        self._commit()

    # -- table cells -----------------------------------------------------

    def _override_value(self, row: int, column: str) -> Any:
        return self.model.overrides.value_for(row, column)

    def _set_override_value(self, row: int, column: str, value: Any) -> None:
        if column == "mode":
            self.model.overrides.set_value(row, column, int(value))
        else:
            self.model.overrides.set_value(row, column, str(value))
        self._commit()

    def _replacement_value(self, row: int, column: str) -> Any:
        return self.model.vocabulary.value_for(row, column)

    def _set_replacement_value(self, row: int, column: str, value: Any) -> None:
        if column == "match_case":
            self.model.vocabulary.set_value(row, column, bool(int(value)))
        else:
            self.model.vocabulary.set_value(row, column, str(value))
        self._commit()

    def _snippet_value(self, row: int, column: str) -> Any:
        return self.model.snippets.value_for(row, column)

    def _set_snippet_value(self, row: int, column: str, value: Any) -> None:
        self.model.snippets.set_value(row, column, str(value))
        self._commit()

    # -- +/- buttons -----------------------------------------------------

    def _edit_new_row(self, table: Any, row: int) -> None:
        """Redraw, select the new row, and put the cursor in its first cell."""
        from Cocoa import NSIndexSet

        table.reloadData()
        table.selectRowIndexes_byExtendingSelection_(NSIndexSet.indexSetWithIndex_(row), False)
        table.editColumn_row_withEvent_select_(0, row, None, True)

    def _remove_selected(self, table: Any, remove: Callable[[int], int]) -> None:
        """Delete the selected row, select its neighbour, and save."""
        from Cocoa import NSIndexSet

        row = table.selectedRow()
        if row < 0:
            return
        next_row = remove(int(row))
        table.reloadData()
        if next_row >= 0:
            table.selectRowIndexes_byExtendingSelection_(
                NSIndexSet.indexSetWithIndex_(next_row), False
            )
        self._commit()

    def _add_override(self, sender) -> None:
        self._edit_new_row(self._overrides_table, self.model.overrides.add_override())

    def _remove_override(self, sender) -> None:
        self._remove_selected(self._overrides_table, self.model.overrides.remove_override)

    def _add_replacement(self, sender) -> None:
        self._edit_new_row(self._replacements_table, self.model.vocabulary.add_replacement())

    def _remove_replacement(self, sender) -> None:
        self._remove_selected(
            self._replacements_table, self.model.vocabulary.remove_replacement
        )

    def _add_snippet(self, sender) -> None:
        self._edit_new_row(self._snippets_table, self.model.snippets.add_snippet())

    def _remove_snippet(self, sender) -> None:
        self._remove_selected(self._snippets_table, self.model.snippets.remove_snippet)

    # -- CSV -------------------------------------------------------------

    def _import_vocabulary(self, sender) -> None:
        """Load terms and replacements from a CSV, replacing what is here."""
        from Cocoa import NSOpenPanel

        panel = NSOpenPanel.openPanel()
        panel.setTitle_("Import vocabulary")
        panel.setPrompt_("Import")
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(False)
        panel.setAllowedFileTypes_(["csv"])
        if panel.runModal() != 1:
            return
        urls = panel.URLs()
        if not urls:
            return
        try:
            with open(str(urls[0].path()), encoding="utf-8") as handle:
                self.model.vocabulary.import_text(handle.read())
        except VocabularyError as error:
            # The message names the offending line; that is the whole point of
            # importing strictly rather than skipping rows.
            self._alert(f"That file could not be imported.\n\n{error}")
            return
        except OSError as error:
            self._alert(f"That file could not be read.\n\n{error}")
            return
        self._terms_view.setString_(self.model.vocabulary.terms_text)
        self._replacements_table.reloadData()
        self._commit()

    def _export_vocabulary(self, sender) -> None:
        """Write the current terms and replacements out as CSV."""
        from Cocoa import NSSavePanel

        self._read_terms()
        panel = NSSavePanel.savePanel()
        panel.setTitle_("Export vocabulary")
        panel.setPrompt_("Export")
        panel.setNameFieldStringValue_(VOCABULARY_CSV_NAME)
        panel.setAllowedFileTypes_(["csv"])
        if panel.runModal() != 1:
            return
        url = panel.URL()
        if url is None:
            return
        try:
            with open(str(url.path()), "w", encoding="utf-8") as handle:
                handle.write(self.model.vocabulary.export_text())
        except OSError as error:
            self._alert(f"That file could not be written.\n\n{error}")

    def _alert(self, message: str) -> None:
        import ui_alerts

        ui_alerts.show_alert("Murmur", message)

    # -- closing ---------------------------------------------------------

    def close(self) -> None:
        """Save anything still in the terms box, then let the tables go.

        The terms box has no action of its own — it is a text view, not a
        control — so a window closed while it has focus would otherwise lose
        what was typed into it. Safe to call twice: the second call finds no
        text view and nothing to save.
        """
        if self._terms_view is not None and self.context is not None and self.model is not None:
            try:
                self._commit()
            except Exception as error:  # noqa: BLE001 - closing must not raise
                logger.warning("Could not save the vocabulary while closing: %s", error)
        self._terms_view = None
        for table in (self._overrides_table, self._replacements_table, self._snippets_table):
            if table is not None:
                table.setDataSource_(None)
        self._sources.clear()


__all__ = [
    "CLEANUP_DEFAULT_HINT",
    "CONFIG_CONTEXT_AWARENESS",
    "CONFIG_INCLUDE_SELECTION",
    "FEATURE_CLEANUP",
    "FEATURE_SNIPPETS",
    "FEATURE_VOCABULARY_BEYOND_FREE",
    "OWNED_CONFIG_KEYS",
    "PRO_CLEANUP_HINT",
    "SNIPPETS_LIMIT_HINT",
    "VOCABULARY_LIMIT_HINT",
    "ColumnSpec",
    "OverrideRow",
    "OverridesSectionModel",
    "SmartTab",
    "SmartTabModel",
    "SnippetsSectionModel",
    "VocabularySectionModel",
    "built_in_mode_for",
    "spoken_symbol_rows",
]
