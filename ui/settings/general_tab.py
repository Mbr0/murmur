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
from typing import Any, Callable

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
    permission_status_message,
)
from services.language_service import available_languages, language_display_name
from ui.settings import register_tab
from ui.settings.base import TAB_GENERAL, TabContext

logger = logging.getLogger(__name__)

CONFIG_LANGUAGE = "language"
CONFIG_APPEARANCE = "appearance_mode"

#: New in Wave 3. Whether Murmur starts itself when the user logs in. Only
#: written where something actually registers the login item — see
#: :func:`supports_launch_at_login`.
CONFIG_LAUNCH_AT_LOGIN = "launch_at_login"

#: Shown instead of a switch that would do nothing.
LAUNCH_AT_LOGIN_UNSUPPORTED = "Not available in this build"

#: Shown when the switch was turned on and the login item is registered, but
#: macOS is not acting on it yet — ``SMAppServiceStatusRequiresApproval``, or a
#: refusal from the framework. The box goes back off, because that is the truth.
LAUNCH_AT_LOGIN_NEEDS_APPROVAL = (
    "macOS has not allowed this yet. Turn Murmur on in System Settings › "
    "General › Login Items."
)

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


def supports_launch_at_login(app: Any | None) -> bool:
    """Whether the running app can actually register Murmur as a login item.

    The switch is only honest where something implements it: without
    ``set_launch_at_login`` the config key is written, kept across restarts,
    and Murmur still does not start itself.
    """
    if app is None:
        return False
    return callable(getattr(app, "set_launch_at_login", None))


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

    def __init__(
        self,
        config: dict,
        *,
        engine_info: Any | None = None,
        permission_status: Callable[[], str] | None = None,
        launch_at_login_supported: bool = False,
    ) -> None:
        assert config is not None, "config is required"
        self.binding: HotkeyBinding = hotkey_from_config(config)
        self.hotkey_label: str | None = config.get("hotkey_label")
        self.hotkey_mode: str = hotkey_mode_from_config(config)
        self.language: str = config.get(CONFIG_LANGUAGE, LANGUAGE_AUTO)
        self.appearance: str = config.get(CONFIG_APPEARANCE, APPEARANCE_MODES[0])
        # Off by default: a build that has not proved it can register a login
        # item does not get to write a key claiming it did.
        self.launch_at_login_supported: bool = bool(launch_at_login_supported)
        self.launch_at_login: bool = bool(config.get(CONFIG_LAUNCH_AT_LOGIN, False))
        #: The user asked for the login item and macOS has not granted it yet.
        self.launch_at_login_pending: bool = False
        self.engine_languages: tuple[str, ...] = language_codes(engine_info)
        self._permission_status = permission_status or permission_status_message
        self._original = self.as_config()

    @property
    def permission_status(self) -> str:
        """The Accessibility permission line shown under the shortcut section."""
        return self._permission_status()

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

    def set_engine_info(self, engine_info: Any | None) -> None:
        """Recompute the language choices for a newly-loaded engine.

        Called after a live engine swap. The configured language stays
        selected: :attr:`language_choices` already keeps a value the new
        engine does not claim as its own row.
        """
        self.engine_languages = language_codes(engine_info)

    # -- appearance and login --------------------------------------------

    @property
    def appearance_index(self) -> int:
        return _index_of(APPEARANCE_MODES, self.appearance)

    def set_appearance(self, mode: str) -> None:
        assert mode in APPEARANCE_MODES, (
            f"Invalid appearance {mode!r}; expected one of {', '.join(APPEARANCE_MODES)}"
        )
        self.appearance = mode

    @property
    def launch_at_login_hint(self) -> str | None:
        """Why the checkbox is dead or did not take, or None when all is well."""
        if not self.launch_at_login_supported:
            return LAUNCH_AT_LOGIN_UNSUPPORTED
        if self.launch_at_login_pending:
            return LAUNCH_AT_LOGIN_NEEDS_APPROVAL
        return None

    def set_launch_at_login(self, enabled: bool) -> None:
        """Record a state that is known to have taken effect."""
        self.set_launch_at_login_state(enabled, enabled)

    def set_launch_at_login_state(self, requested: bool, actual: bool) -> None:
        """Record what the system actually did with the switch.

        ``actual`` is what the login item reports *after* the call, which is not
        always what was asked for: a registration can sit at
        ``SMAppServiceStatusRequiresApproval`` until the user allows it. The
        model keeps the real state — so nothing false is written to config —
        and remembers that the request is outstanding, which is what
        :attr:`launch_at_login_hint` then explains.
        """
        assert isinstance(requested, bool), f"expected a bool, got {requested!r}"
        assert isinstance(actual, bool), f"expected a bool, got {actual!r}"
        if not self.launch_at_login_supported:
            logger.info("Launch at login is not available in this build; ignoring the switch")
            return
        self.launch_at_login = actual
        self.launch_at_login_pending = requested and not actual

    # -- persistence -----------------------------------------------------

    def as_config(self) -> dict:
        """Every key this tab owns, at its current value.

        ``launch_at_login`` is absent — and so never reported by :meth:`apply`,
        never written — in a build that cannot act on it. A key on disk from a
        build that could is left exactly as it is rather than rewritten.
        """
        data = dict(hotkey_to_config(self.binding, label=self.hotkey_label))
        data[HOTKEY_MODE_CONFIG_KEY] = self.hotkey_mode
        data[CONFIG_LANGUAGE] = self.language
        data[CONFIG_APPEARANCE] = self.appearance
        if self.launch_at_login_supported:
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
        self._launch_hint = None
        self._permission_hint = None
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
        self.model = self._make_model(context)
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
        self._launch_checkbox.setEnabled_(self.model.launch_at_login_supported)
        # Always made, even with nothing to say: the line has to be there to
        # carry the "macOS has not allowed this yet" answer when the switch is
        # flipped, and a row appearing under a checkbox would shift the layout.
        self._launch_hint = make_hint(self.model.launch_at_login_hint or " ", theme)
        permissions_button = make_button(
            "Open Privacy Settings", theme, self._open_privacy_settings
        )
        self._permission_hint = make_hint(self.model.permission_status, theme)

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
            permissions_button,
            self._permission_hint,
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
            self._launch_hint,
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

    def _make_model(self, context: TabContext) -> GeneralTabModel:
        """The model, told whether this build can register a login item."""
        return GeneralTabModel(
            context.config,
            engine_info=context.engine_info,
            launch_at_login_supported=supports_launch_at_login(context.app),
        )

    def refresh(self) -> None:
        """Re-read config into the controls, e.g. after another tab wrote."""
        if self.context is None:
            return
        self.model = self._make_model(self.context)
        if self._shortcut_button is not None:
            self._shortcut_button.setTitle_(self.model.shortcut_label)
        if self._mode_popup is not None:
            self._mode_popup.selectItemAtIndex_(self.model.hotkey_mode_index)
        if self._language_popup is not None:
            self._language_popup.removeAllItems()
            self._language_popup.addItemsWithTitles_(list(self.model.language_titles))
            self._language_popup.selectItemAtIndex_(self.model.language_index)
        if self._appearance_popup is not None:
            self._appearance_popup.selectItemAtIndex_(self.model.appearance_index)
        self._show_launch_at_login()
        if self._permission_hint is not None:
            self._permission_hint.setStringValue_(self.model.permission_status)

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

        self.set_launch_at_login(checkbox_is_on(sender))

    def set_launch_at_login(self, requested: bool) -> None:
        """Move the login item, then show and store what actually happened.

        The app is asked first and its answer — the state it read back off
        ``SMAppService`` — is what gets committed and what the checkbox is set
        to. Writing ``launch_at_login: True`` before the call and ignoring the
        result left the config, and the box, claiming something macOS had not
        agreed to; a registration awaiting approval in System Settings is the
        common way that happens.
        """
        assert self.context is not None and self.model is not None
        if not self.model.launch_at_login_supported:
            logger.info("Launch at login is not available in this build; ignoring the switch")
            self._show_launch_at_login()
            return
        self.model.set_launch_at_login_state(requested, self._request_launch_at_login(requested))
        self._commit()
        self._show_launch_at_login()

    def _request_launch_at_login(self, requested: bool) -> bool:
        """Ask the app to register or unregister; return the state afterwards.

        A refusal — ``LaunchAtLoginUnavailable``, or anything else the
        ServiceManagement bridge throws — means the switch did not move, so the
        honest answer is the state it was not asked to be in.
        """
        if self.context.app is None:
            logger.info("Launch at login set to %s; no running app to apply it", requested)
            return requested
        try:
            state = self.context.app_call("set_launch_at_login", requested)
        except Exception as error:  # noqa: BLE001 - the framework raises widely
            logger.warning("The login item did not move: %s", error)
            return not requested
        # ``app_call`` answers None for an app without the method; the switch is
        # only offered where it has one, so the request is taken at face value.
        return state if isinstance(state, bool) else requested

    def _show_launch_at_login(self) -> None:
        """Put the model's real state on the checkbox and its hint line."""
        if self.model is None:
            return
        if self._launch_checkbox is not None:
            self._set_checkbox(self.model.launch_at_login)
        if self._launch_hint is not None:
            self._launch_hint.setStringValue_(self.model.launch_at_login_hint or " ")

    def _set_checkbox(self, on: bool) -> None:
        """The one AppKit call in the launch-at-login path, on its own.

        Everything that decides *what* the box should say is plain Python above
        it, and testable without a window server.
        """
        from Cocoa import NSOffState, NSOnState

        self._launch_checkbox.setState_(NSOnState if on else NSOffState)

    def _open_privacy_settings(self, sender) -> None:
        from services.hotkey_service import open_privacy_settings

        open_privacy_settings()

    # -- closing ---------------------------------------------------------

    def close(self) -> None:
        """Give back the shortcut recorder's event monitor.

        Settings closed while "Press shortcut…" is showing would otherwise
        leave a local monitor installed for the life of the process, still
        swallowing keys for a window that is gone. Safe to call twice.
        """
        self._stop_capture()

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

    def _remove_monitor(self, monitor: Any) -> None:
        """Hand one local event monitor back to AppKit.

        The single AppKit call in the teardown path, on its own so closing a
        tab that never captured needs no window server at all.
        """
        from Cocoa import NSEvent

        NSEvent.removeMonitor_(monitor)

    def _stop_capture(self) -> None:
        if self._monitor is not None:
            monitor, self._monitor = self._monitor, None
            self._remove_monitor(monitor)
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
