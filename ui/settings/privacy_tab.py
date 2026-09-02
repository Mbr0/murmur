#!/usr/bin/env python3
"""Settings → Privacy: what is kept, what leaves, and how to erase it.

Three things live here:

* :class:`PrivacyTabModel` — the two toggles, the generated "what leaves the
  Mac" lines for the *current* configuration, and :meth:`PrivacyTabModel.apply`,
  which says exactly which config keys changed. Plain Python, no AppKit.
* :class:`PrivacyTab` — its AppKit rendering: two checkboxes, the generated
  text, and a "Delete all data…" button behind a confirmation.
* The delete action, which calls
  :meth:`services.persistence_service.PersistenceService.delete_all_data`.

The text is generated rather than written by hand on purpose: a privacy claim
that is typed into a label goes stale the moment the engine changes, and a
stale privacy claim is worse than none. Everything the tab says comes from
:func:`~services.persistence_service.what_leaves_the_mac`, reading the same
config the engine reads.

Config keys owned here (documented in ``services/persistence_service.py``):

- ``save_history``: keep transcriptions in the history window.
- ``save_audio``: keep the recording next to the transcription.
- ``privacy_mode``: derived, true when neither of those is on. It gates the
  detailed logging in ``should_log_sensitive``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from services.persistence_service import (
    CONFIG_HISTORY_ENABLED,
    CONFIG_KEEP_AUDIO,
    CONFIG_PRIVACY_MODE,
    DEFAULT_CONFIG,
    DeletionSummary,
    what_leaves_the_mac,
)
from ui.settings import register_tab
from ui.settings.base import TAB_PRIVACY, TabContext

logger = logging.getLogger(__name__)

#: Where recordings are kept. Mirrors ``murmur.AUDIO_DIR``; the tab must not
#: import ``murmur`` to find out, so the path is repeated and asserted in
#: ``tests/test_settings_privacy_tab.py``.
DEFAULT_AUDIO_DIR = os.path.expanduser("~/.murmur_audio")

#: Name of the injected ``PersistenceService`` in ``TabContext.services``.
SERVICE_PERSISTENCE = "persistence"
#: Name of the injected audio directory override in ``TabContext.services``.
SERVICE_AUDIO_DIR = "audio_dir"

HISTORY_CHECKBOX = "Keep a history of what I dictate"
KEEP_AUDIO_CHECKBOX = "Keep the audio recordings too"
SECTION_KEPT = "Kept on this Mac"
SECTION_LEAVES = "What leaves this Mac"
DELETE_BUTTON = "Delete all data…"

KEPT_HINT = (
    "Both are off to begin with. History and recordings never leave this Mac; "
    "they live in ~/.murmur_history.json and ~/.murmur_audio."
)

DELETE_TITLE = "Delete all your data?"
DELETE_BODY = (
    "This removes your history, your saved recordings, your vocabulary and the "
    "language and mode you chose per app. Your shortcut, speech engine and the "
    "rest of your settings are kept. This cannot be undone."
)
DELETE_CONFIRM = "Delete"
DELETE_CANCEL = "Cancel"
DELETED_TITLE = "Data deleted"

NO_SERVICE_TITLE = "Nothing to delete from here"
NO_SERVICE_BODY = (
    "Settings could not reach Murmur's local storage. Open Settings from the "
    "Murmur menu bar icon and try again."
)


class PrivacyTabModel:
    """The Privacy tab as plain data: two toggles and the generated text.

    ``config`` is the live dict the window loaded. The model reads it, keeps
    the toggles as its own state, and reports the difference in :meth:`apply`.
    The generated lines always reflect the *pending* toggles, so the text
    changes under the reader's hand rather than one save later.
    """

    def __init__(self, config: dict, *, engine_info: Any = None) -> None:
        assert config is not None, "config is required"
        self._config = config
        self.engine_info = engine_info
        self.history_enabled = bool(
            config.get(CONFIG_HISTORY_ENABLED, DEFAULT_CONFIG[CONFIG_HISTORY_ENABLED])
        )
        self.keep_audio = bool(
            config.get(CONFIG_KEEP_AUDIO, DEFAULT_CONFIG[CONFIG_KEEP_AUDIO])
        )
        self._original = self.as_config()

    # -- toggles ---------------------------------------------------------

    def set_history_enabled(self, enabled: bool) -> None:
        self.history_enabled = bool(enabled)

    def set_keep_audio(self, enabled: bool) -> None:
        self.keep_audio = bool(enabled)

    def as_config(self) -> dict:
        """The config keys this tab owns, at their pending values.

        ``privacy_mode`` is derived, never toggled directly: it means "nothing
        of what I said is on disk", which is exactly both switches being off.
        """
        return {
            CONFIG_HISTORY_ENABLED: self.history_enabled,
            CONFIG_KEEP_AUDIO: self.keep_audio,
            CONFIG_PRIVACY_MODE: not (self.history_enabled or self.keep_audio),
        }

    # -- generated text --------------------------------------------------

    @property
    def lines(self) -> list[str]:
        """Plain-language lines for the configuration as it stands right now."""
        return what_leaves_the_mac(
            {**self._config, **self.as_config()}, engine_info=self.engine_info
        )

    @property
    def summary_text(self) -> str:
        """:attr:`lines` as one bulleted block for a read-only text field."""
        return "\n".join(f"• {line}" for line in self.lines)

    # -- saving ----------------------------------------------------------

    def apply(self) -> dict:
        """The keys that differ from the config this model was built on."""
        current = self.as_config()
        return {key: value for key, value in current.items() if self._original[key] != value}

    def mark_saved(self) -> None:
        """Called once :meth:`apply`'s dict has been persisted."""
        self._original = self.as_config()

    # -- deletion --------------------------------------------------------

    def delete_all_data(self, persistence: Any, audio_dir: str) -> DeletionSummary:
        """Erase history, recordings and user content; keep the preferences.

        The live config is handed over so the user-content keys are cleared in
        the same pass, and the toggles are deliberately untouched: deleting
        what Murmur stored is not a request to change what it stores next.
        """
        assert persistence is not None, "a persistence service is required"
        assert audio_dir, "audio_dir is required"
        return persistence.delete_all_data(audio_dir, self._config)


def _resolved_theme(theme: Any) -> Any:
    """The context's theme, or ``ui.theme`` imported on demand."""
    if theme is not None:
        return theme
    from ui import theme as ui_theme

    return ui_theme


def _make_summary_field(text: str, theme: Any, width: int = 440) -> Any:
    """A read-only, wrapping text field — the generated privacy statement.

    Selectable but not editable: people copy this into a support thread, and
    nobody should be able to type a different promise over it.
    """
    from Cocoa import NSFont, NSMakeRect, NSTextField

    field = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, width, 140))
    field.setStringValue_(text)
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(True)
    field.setFont_(NSFont.systemFontOfSize_(12))
    field.setTextColor_(theme.primary_text_color())
    field.setLineBreakMode_(0)  # NSLineBreakByWordWrapping
    field.cell().setWraps_(True)
    field.setPreferredMaxLayoutWidth_(float(width))
    field.setTranslatesAutoresizingMaskIntoConstraints_(False)
    field.widthAnchor().constraintEqualToConstant_(float(width)).setActive_(True)
    return field


@register_tab
class PrivacyTab:
    """The AppKit rendering of :class:`PrivacyTabModel`.

    Like the other tabs there is no Save button: a toggle the user can see is
    a toggle already on disk, and the generated text below it is rewritten in
    the same breath.
    """

    identifier = TAB_PRIVACY
    title = "Privacy"

    def __init__(self) -> None:
        self.context: TabContext | None = None
        self.model: PrivacyTabModel | None = None
        self._view = None
        self._history_checkbox = None
        self._keep_audio_checkbox = None
        self._summary_field = None

    # -- building --------------------------------------------------------

    def build(self, context: TabContext) -> Any:
        """Lay the tab out and return its content view."""
        assert context is not None, "context is required"
        from Cocoa import NSMakeRect, NSView

        from ui.settings.base import (
            CONTENT_MARGIN,
            make_checkbox,
            make_hint,
            make_button,
            make_section_title,
            stack_vertical,
        )

        self.context = context
        self.model = PrivacyTabModel(context.config, engine_info=context.engine_info)
        theme = _resolved_theme(context.theme)

        self._history_checkbox = make_checkbox(
            HISTORY_CHECKBOX, self.model.history_enabled, theme, self._history_changed
        )
        self._keep_audio_checkbox = make_checkbox(
            KEEP_AUDIO_CHECKBOX, self.model.keep_audio, theme, self._keep_audio_changed
        )
        self._summary_field = _make_summary_field(self.model.summary_text, theme)

        rows = [
            make_section_title(SECTION_KEPT, theme),
            self._history_checkbox,
            self._keep_audio_checkbox,
            make_hint(KEPT_HINT, theme),
            make_section_title(SECTION_LEAVES, theme),
            self._summary_field,
            make_button(DELETE_BUTTON, theme, self._delete_clicked, width=180),
        ]

        container = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 480, 520))
        stack = stack_vertical(rows)
        container.addSubview_(stack)
        stack.leadingAnchor().constraintEqualToAnchor_constant_(
            container.leadingAnchor(), CONTENT_MARGIN
        ).setActive_(True)
        stack.topAnchor().constraintEqualToAnchor_constant_(
            container.topAnchor(), CONTENT_MARGIN
        ).setActive_(True)
        stack.trailingAnchor().constraintLessThanOrEqualToAnchor_constant_(
            container.trailingAnchor(), -CONTENT_MARGIN
        ).setActive_(True)
        self._view = container
        return container

    def refresh(self) -> None:
        """Re-read config into the controls, e.g. after another tab wrote.

        The engine tab switching to Murmur Cloud has to change what this tab
        promises, so the generated text is rebuilt here too.
        """
        if self.context is None:
            return
        self.model = PrivacyTabModel(
            self.context.config, engine_info=self.context.engine_info
        )
        from Cocoa import NSOffState, NSOnState

        if self._history_checkbox is not None:
            self._history_checkbox.setState_(
                NSOnState if self.model.history_enabled else NSOffState
            )
        if self._keep_audio_checkbox is not None:
            self._keep_audio_checkbox.setState_(
                NSOnState if self.model.keep_audio else NSOffState
            )
        self._redraw_summary()

    # -- actions ---------------------------------------------------------

    def _redraw_summary(self) -> None:
        if self._summary_field is not None and self.model is not None:
            self._summary_field.setStringValue_(self.model.summary_text)

    def _commit(self) -> None:
        """Persist what changed, then rewrite the generated text."""
        assert self.context is not None and self.model is not None
        changed = self.model.apply()
        if changed:
            self.context.save(changed)
            self.model.mark_saved()
        self._redraw_summary()

    def _history_changed(self, sender) -> None:
        from ui.settings.base import checkbox_is_on

        self.model.set_history_enabled(checkbox_is_on(sender))
        self._commit()

    def _keep_audio_changed(self, sender) -> None:
        from ui.settings.base import checkbox_is_on

        self.model.set_keep_audio(checkbox_is_on(sender))
        self._commit()

    # -- delete all data -------------------------------------------------

    def _persistence(self) -> Any:
        """The injected persistence service, or the window's own one."""
        assert self.context is not None
        injected = self.context.service(SERVICE_PERSISTENCE)
        if injected is not None:
            return injected
        from ui.settings.window import PERSISTENCE

        return PERSISTENCE

    def _audio_dir(self) -> str:
        assert self.context is not None
        return self.context.service(SERVICE_AUDIO_DIR, DEFAULT_AUDIO_DIR)

    def _delete_clicked(self, sender) -> None:
        """Confirm, erase, tell the user exactly what went."""
        from ui import alerts as ui_alerts

        if not ui_alerts.show_confirm(
            DELETE_TITLE, DELETE_BODY, ok=DELETE_CONFIRM, cancel=DELETE_CANCEL
        ):
            return

        persistence = self._persistence()
        if persistence is None:
            logger.error("Privacy tab has no persistence service; nothing deleted")
            ui_alerts.show_alert(NO_SERVICE_TITLE, NO_SERVICE_BODY)
            return

        summary = self.model.delete_all_data(persistence, self._audio_dir())
        logger.info(
            "Deleted all local data: %s history entries, %s recordings, keys %s",
            summary.history_entries,
            summary.audio_files,
            ", ".join(summary.config_keys) or "none",
        )

        app = self.context.app
        if app is not None and hasattr(app, "history"):
            # The running app holds the list in memory; leaving it there would
            # write the deleted entries straight back on the next dictation.
            app.history = []

        self.refresh()
        ui_alerts.show_alert(DELETED_TITLE, summary.describe())
