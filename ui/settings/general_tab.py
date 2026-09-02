#!/usr/bin/env python3
"""Settings → General: shortcut, shortcut behaviour, language, appearance,
launch at login.

:class:`GeneralTabModel` is the whole tab as plain Python — which keys it
owns, what the shortcut recorder produced, and, in :meth:`GeneralTabModel.apply`,
exactly which keys changed. :class:`GeneralTab` is a rendering of it, so the
rules are tested without a window server.

Config keys owned here (see ``services/persistence_service.py``):

- ``hotkey_keycode``, ``hotkey_command``, ``hotkey_option``, ``hotkey_control``,
  ``hotkey_shift``, ``hotkey_fn``, ``hotkey_label``: the recorded shortcut.
- ``hotkey_mode``: ``toggle | hold | auto``.
- ``language``: ``"auto"`` or an ISO code.
- ``appearance_mode``: ``system | dark | light``.
- ``launch_at_login``: new in Wave 3, ``False`` until the user turns it on.
"""

from __future__ import annotations

import logging
from typing import Any

from engines import LANGUAGE_AUTO
from services.hotkey_service import (
    DEFAULT_HOTKEY,
    HOTKEY_MODE_AUTO,
    HOTKEY_MODE_CONFIG_KEY,
    HOTKEY_MODE_HOLD,
    HOTKEY_MODE_TOGGLE,
    HOTKEY_MODES,
    HotkeyBinding,
    format_hotkey,
    hotkey_from_config,
    hotkey_mode_from_config,
    hotkey_to_config,
)
from services.language_service import available_languages, language_display_name
from ui.settings import register_tab
from ui.settings.base import TAB_GENERAL, TabContext

logger = logging.getLogger(__name__)

CONFIG_LANGUAGE = "language"
CONFIG_APPEARANCE = "appearance_mode"

#: New in Wave 3. Whether Murmur starts itself when the user logs in.
CONFIG_LAUNCH_AT_LOGIN = "launch_at_login"

#: Mirrors ``ui_theme.APPEARANCE_MODES``; kept here so the model needs no
#: AppKit. ``tests/test_settings_general_tab.py`` fails if the two drift.
APPEARANCE_MODES: tuple[str, ...] = ("system", "dark", "light")
APPEARANCE_LABELS = {"system": "System", "dark": "Dark", "light": "Light"}

#: What each shortcut behaviour is called, in ``HOTKEY_MODES`` order.
HOTKEY_MODE_LABELS = {
    HOTKEY_MODE_TOGGLE: "Toggle",
    HOTKEY_MODE_HOLD: "Hold to talk",
    HOTKEY_MODE_AUTO: "Automatic",
}

#: Languages offered when no engine is loaded to ask. Same shape as
#: ``available_languages``: auto first, ISO codes after it, sorted.
FALLBACK_LANGUAGES: tuple[str, ...] = (
    LANGUAGE_AUTO,
    "de",
    "en",
    "es",
    "fr",
    "it",
    "nl",
    "pt",
)

#: Keys whose change means the running app must re-register the shortcut.
HOTKEY_CONFIG_KEYS = frozenset(hotkey_to_config(DEFAULT_HOTKEY)) | {
    HOTKEY_MODE_CONFIG_KEY
}


def language_codes(engine_info: Any | None) -> tuple[str, ...]:
    """Languages to offer: the loaded engine's own list, else the known set.

    The one place the UI asks an engine anything — no engine branch anywhere
    else in the tabs. An engine that cannot answer is logged, not fatal.
    """
    if engine_info is None:
        return FALLBACK_LANGUAGES
    try:
        return available_languages(engine_info)
    except Exception as error:  # pragma: no cover - defensive around engine code
        logger.warning("Could not read languages from the engine: %s", error)
        return FALLBACK_LANGUAGES


def needs_hotkey_reload(changed: dict) -> bool:
    """Whether ``changed`` touches the shortcut, so the app must re-register."""
    assert changed is not None, "changed is required"
    return any(key in changed for key in HOTKEY_CONFIG_KEYS)


def _index_of(items: tuple[str, ...], value: Any, default: int = 0) -> int:
    """``value``'s position in ``items``, or ``default`` when it is not there."""
    if value in items:
        return items.index(value)
    return default


class GeneralTabModel:
    """Editing state for Settings → General.

    Built from a config dict, which it never mutates. :meth:`apply` reports
    only the keys the user actually changed, so opening Settings and closing
    it again writes nothing at all — the reason the twelve pre-v2 keys still
    round-trip untouched.
    """

    def __init__(self, config: dict, *, engine_info: Any | None = None) -> None:
        assert config is not None, "config is required"
        self.binding: HotkeyBinding = hotkey_from_config(config)
        self.hotkey_label: str | None = config.get("hotkey_label")
        self.hotkey_mode: str = hotkey_mode_from_config(config)
        self.language: str = config.get(CONFIG_LANGUAGE, LANGUAGE_AUTO)
        self.appearance: str = config.get(CONFIG_APPEARANCE, APPEARANCE_MODES[0])
        self.launch_at_login: bool = bool(config.get(CONFIG_LAUNCH_AT_LOGIN, False))
        self.engine_languages: tuple[str, ...] = language_codes(engine_info)
        self._original = self.as_config()

    # -- shortcut --------------------------------------------------------

    @property
    def shortcut_label(self) -> str:
        """What the recorder button shows, e.g. ``"⌥ Space"``."""
        return format_hotkey(self.binding, label=self.hotkey_label)

    def set_binding(self, binding: HotkeyBinding, label: str | None = None) -> None:
        """Record a captured shortcut."""
        assert binding is not None, "binding is required"
        self.binding = binding
        self.hotkey_label = label

    def reset_shortcut(self) -> None:
        """Restore the shipped ⌥ Space."""
        self.binding = DEFAULT_HOTKEY
        self.hotkey_label = "Space"

    # -- shortcut behaviour ----------------------------------------------

    @property
    def hotkey_mode_index(self) -> int:
        return HOTKEY_MODES.index(self.hotkey_mode)

    def set_hotkey_mode(self, mode: str) -> None:
        assert mode in HOTKEY_MODES, (
            f"Invalid hotkey mode {mode!r}; expected one of {', '.join(HOTKEY_MODES)}"
        )
        self.hotkey_mode = mode

    # -- language --------------------------------------------------------

    @property
    def language_choices(self) -> tuple[str, ...]:
        """Codes for the picker: the engine's list, plus whatever is stored.

        A stored language the current engine does not claim still shows, so
        switching engines never silently rewrites the user's choice.
        """
        if self.language in self.engine_languages:
            return self.engine_languages
        return (*self.engine_languages, self.language)

    @property
    def language_titles(self) -> tuple[str, ...]:
        return tuple(
            "Automatic" if code == LANGUAGE_AUTO else language_display_name(code)
            for code in self.language_choices
        )

    @property
    def language_index(self) -> int:
        return _index_of(self.language_choices, self.language)

    def set_language(self, code: str) -> None:
        assert code, "a language code is required"
        self.language = code

    # -- appearance and login --------------------------------------------

    @property
    def appearance_index(self) -> int:
        return _index_of(APPEARANCE_MODES, self.appearance)

    def set_appearance(self, mode: str) -> None:
        assert mode in APPEARANCE_MODES, (
            f"Invalid appearance {mode!r}; expected one of {', '.join(APPEARANCE_MODES)}"
        )
        self.appearance = mode

    def set_launch_at_login(self, enabled: bool) -> None:
        assert isinstance(enabled, bool), f"expected a bool, got {enabled!r}"
        self.launch_at_login = enabled

    # -- persistence -----------------------------------------------------

    def as_config(self) -> dict:
        """Every key this tab owns, at its current value."""
        data = dict(hotkey_to_config(self.binding, label=self.hotkey_label))
        data[HOTKEY_MODE_CONFIG_KEY] = self.hotkey_mode
        data[CONFIG_LANGUAGE] = self.language
        data[CONFIG_APPEARANCE] = self.appearance
        data[CONFIG_LAUNCH_AT_LOGIN] = self.launch_at_login
        return data

    def apply(self) -> dict:
        """The keys that differ from the config this model was built on."""
        current = self.as_config()
        return {key: value for key, value in current.items() if self._original[key] != value}

    def mark_saved(self) -> None:
        """Called once ``apply``'s dict has been persisted."""
        self._original = self.as_config()


@register_tab
class GeneralTab:
    """The AppKit rendering of :class:`GeneralTabModel`.

    Every control writes through :meth:`_commit` as soon as it changes: there
    is no Save button in the tabbed window, so a setting the user can see is a
    setting already on disk.
    """

    identifier = TAB_GENERAL
    title = "General"

    def __init__(self) -> None:
        self.context: TabContext | None = None
        self.model: GeneralTabModel | None = None
        self._view = None
        self._shortcut_button = None
        self._mode_popup = None
        self._language_popup = None
        self._appearance_popup = None
        self._launch_checkbox = None
        self._monitor = None
        self._capture_modifiers = 0

    # -- building --------------------------------------------------------

    def build(self, context: TabContext) -> Any:
        """Lay the tab out and return its content view."""
        assert context is not None, "context is required"
        from Cocoa import NSMakeRect, NSView

        from ui.settings.base import (
            CONTENT_MARGIN,
            make_button,
            make_checkbox,
            make_hint,
            make_popup,
            make_section_title,
            stack_horizontal,
            stack_vertical,
        )

        self.context = context
        self.model = GeneralTabModel(context.config, engine_info=context.engine_info)
        theme = context.theme

        self._shortcut_button = make_button(
            self.model.shortcut_label, theme, self._record_shortcut, width=220
        )
        reset_button = make_button(
            "Default (⌥ Space)", theme, self._reset_shortcut, width=160
        )
        self._mode_popup = make_popup(
            [HOTKEY_MODE_LABELS[mode] for mode in HOTKEY_MODES],
            self.model.hotkey_mode_index,
            theme,
            self._mode_changed,
        )
        self._language_popup = make_popup(
            list(self.model.language_titles),
            self.model.language_index,
            theme,
            self._language_changed,
        )
        self._appearance_popup = make_popup(
            [APPEARANCE_LABELS[mode] for mode in APPEARANCE_MODES],
            self.model.appearance_index,
            theme,
            self._appearance_changed,
            width=180,
        )
        self._launch_checkbox = make_checkbox(
            "Start Murmur when I log in",
            self.model.launch_at_login,
            theme,
            self._launch_changed,
        )

        rows = [
            make_section_title("Keyboard shortcut", theme),
            stack_horizontal([self._shortcut_button, reset_button]),
            make_section_title("Shortcut behaviour", theme),
            self._mode_popup,
            make_hint(
                "Toggle starts and stops on separate presses. Hold to talk "
                "records only while the keys are down. Automatic decides from "
                "how long you hold: a tap toggles, a hold talks.",
                theme,
            ),
            make_section_title("Language", theme),
            self._language_popup,
            make_hint(
                "Automatic detects the language for each recording. Picking "
                "one is faster and stops short phrases being mistaken for "
                "another language.",
                theme,
            ),
            make_section_title("Appearance", theme),
            self._appearance_popup,
            make_section_title("Startup", theme),
            self._launch_checkbox,
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
        """Re-read config into the controls, e.g. after another tab wrote."""
        if self.context is None:
            return
        self.model = GeneralTabModel(
            self.context.config, engine_info=self.context.engine_info
        )
        if self._shortcut_button is not None:
            self._shortcut_button.setTitle_(self.model.shortcut_label)
        if self._mode_popup is not None:
            self._mode_popup.selectItemAtIndex_(self.model.hotkey_mode_index)
        if self._language_popup is not None:
            self._language_popup.selectItemAtIndex_(self.model.language_index)
        if self._appearance_popup is not None:
            self._appearance_popup.selectItemAtIndex_(self.model.appearance_index)
        if self._launch_checkbox is not None:
            from Cocoa import NSOffState, NSOnState

            self._launch_checkbox.setState_(
                NSOnState if self.model.launch_at_login else NSOffState
            )

    # -- actions ---------------------------------------------------------

    def _commit(self) -> None:
        """Persist what changed, then tell the app if the shortcut moved."""
        assert self.context is not None and self.model is not None
        changed = self.model.apply()
        if not changed:
            return
        self.context.save(changed)
        self.model.mark_saved()
        if CONFIG_APPEARANCE in changed:
            self.context.theme.set_appearance_mode(changed[CONFIG_APPEARANCE])
        if needs_hotkey_reload(changed):
            if self.context.app is None:
                logger.info("Shortcut changed with no running app to reload")
            else:
                self.context.app_call("reload_hotkey", prompt=False)

    def _mode_changed(self, sender) -> None:
        self.model.set_hotkey_mode(HOTKEY_MODES[sender.indexOfSelectedItem()])
        self._commit()

    def _language_changed(self, sender) -> None:
        self.model.set_language(self.model.language_choices[sender.indexOfSelectedItem()])
        self._commit()

    def _appearance_changed(self, sender) -> None:
        self.model.set_appearance(APPEARANCE_MODES[sender.indexOfSelectedItem()])
        self._commit()

    def _launch_changed(self, sender) -> None:
        from ui.settings.base import checkbox_is_on

        enabled = checkbox_is_on(sender)
        self.model.set_launch_at_login(enabled)
        self._commit()
        if self.context.app is None:
            logger.info("Launch at login set to %s; no running app to apply it", enabled)
            return
        self.context.app_call("set_launch_at_login", enabled)

    # -- shortcut recorder -----------------------------------------------

    def _reset_shortcut(self, sender) -> None:
        self._stop_capture()
        self.model.reset_shortcut()
        self._shortcut_button.setTitle_(self.model.shortcut_label)
        self._commit()

    def _record_shortcut(self, sender) -> None:
        """Listen for the next real key combination and record it."""
        from Cocoa import NSEvent, NSEventMaskFlagsChanged, NSEventMaskKeyDown

        self._stop_capture()
        self._capture_modifiers = 0
        self._shortcut_button.setTitle_("Press shortcut…")
        self._monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            NSEventMaskKeyDown | NSEventMaskFlagsChanged,
            self._capture_event,
        )

    def _stop_capture(self) -> None:
        from Cocoa import NSEvent

        if self._monitor is not None:
            NSEvent.removeMonitor_(self._monitor)
            self._monitor = None
        self._capture_modifiers = 0
        if self._shortcut_button is not None and self.model is not None:
            self._shortcut_button.setTitle_(self.model.shortcut_label)

    def _capture_event(self, event):
        """One event during capture. Ported unchanged from ``settings_window``.

        Modifier-only presses are tracked but never accepted, and a plain key
        with no modifier is passed through: a global shortcut without a
        modifier would swallow ordinary typing.
        """
        from Cocoa import (
            NSEventModifierFlagCommand,
            NSEventModifierFlagControl,
            NSEventModifierFlagFunction,
            NSEventModifierFlagOption,
            NSEventModifierFlagShift,
            NSEventTypeFlagsChanged,
        )

        from services.hotkey_service import (
            MODIFIER_KEYCODES,
            binding_from_ns_flags,
            binding_has_modifier,
            capture_label_for_binding,
            ns_modifier_flags,
        )

        if event.type() == NSEventTypeFlagsChanged:
            self._capture_modifiers = ns_modifier_flags(event.modifierFlags())
            return event

        keycode = event.keyCode()
        if keycode in MODIFIER_KEYCODES:
            return event

        flags = ns_modifier_flags(event.modifierFlags() | self._capture_modifiers)
        wanted = (
            NSEventModifierFlagCommand
            | NSEventModifierFlagOption
            | NSEventModifierFlagControl
            | NSEventModifierFlagShift
            | NSEventModifierFlagFunction
        )
        if not flags & wanted:
            return event

        binding = binding_from_ns_flags(keycode, flags)
        if not binding_has_modifier(binding):
            return event

        characters = event.charactersIgnoringModifiers() or ""
        self.model.set_binding(
            binding, capture_label_for_binding(binding, characters=characters)
        )
        self._stop_capture()
        self._commit()
        return None
