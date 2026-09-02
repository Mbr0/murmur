#!/usr/bin/env python3
"""Global hotkey registration and configuration helpers."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

BUNDLE_ID = "com.canopystudio.murmur"
_LAST_PERMISSION_LOG: bool | None = None

from AppKit import (
    NSEvent,
    NSEventMaskKeyDown,
    NSEventMaskKeyUp,
    NSEventMaskFlagsChanged,
    NSEventModifierFlagCommand,
    NSEventModifierFlagControl,
    NSEventModifierFlagDeviceIndependentFlagsMask,
    NSEventModifierFlagFunction,
    NSEventModifierFlagOption,
    NSEventModifierFlagShift,
)
from ApplicationServices import AXIsProcessTrusted, AXIsProcessTrustedWithOptions
from Foundation import NSDictionary

SPACE_KEYCODE = 49
MODIFIER_KEYCODES = frozenset({54, 55, 56, 57, 58, 59, 60, 61, 62, 63})
ACCESSIBILITY_SETTINGS_URL = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
)

KEYCODE_LABELS: dict[int, str] = {
    0: "A",
    1: "S",
    2: "D",
    3: "F",
    4: "H",
    5: "G",
    6: "Z",
    7: "X",
    8: "C",
    9: "V",
    11: "B",
    12: "Q",
    13: "W",
    14: "E",
    15: "R",
    16: "Y",
    17: "T",
    18: "1",
    19: "2",
    20: "3",
    21: "4",
    22: "6",
    23: "5",
    24: "=",
    25: "9",
    26: "7",
    27: "-",
    28: "8",
    29: "0",
    30: "]",
    31: "O",
    32: "U",
    33: "[",
    34: "I",
    35: "P",
    37: "L",
    38: "J",
    39: "'",
    40: "K",
    41: ";",
    42: "\\",
    43: ",",
    44: "/",
    45: "N",
    46: "M",
    47: ".",
    49: "Space",
    36: "Return",
    48: "Tab",
    51: "Delete",
    53: "Escape",
    123: "Left",
    124: "Right",
    125: "Down",
    126: "Up",
    122: "F1",
    120: "F2",
    99: "F3",
    118: "F4",
    96: "F5",
    97: "F6",
    98: "F7",
    100: "F8",
    101: "F9",
    109: "F10",
    103: "F11",
    111: "F12",
}


@dataclass(frozen=True)
class HotkeyBinding:
    keycode: int
    command: bool = False
    option: bool = False
    control: bool = False
    shift: bool = False
    fn: bool = False


DEFAULT_HOTKEY = HotkeyBinding(keycode=SPACE_KEYCODE, option=True)

# --- Push-to-talk -----------------------------------------------------------

HOTKEY_MODE_TOGGLE = "toggle"
HOTKEY_MODE_HOLD = "hold"
HOTKEY_MODE_AUTO = "auto"
HOTKEY_MODES: tuple[str, ...] = (HOTKEY_MODE_TOGGLE, HOTKEY_MODE_HOLD, HOTKEY_MODE_AUTO)
DEFAULT_HOTKEY_MODE = HOTKEY_MODE_AUTO
HOTKEY_MODE_CONFIG_KEY = "hotkey_mode"

#: Presses at or above this many seconds count as a hold; shorter ones toggle.
HOLD_THRESHOLD_S = 0.3

#: Modes that need key-up delivered as well as key-down.
KEY_UP_MODES = frozenset({HOTKEY_MODE_HOLD, HOTKEY_MODE_AUTO})

ACTION_START = "start"
ACTION_STOP = "stop"
PRESS_ACTIONS: tuple[str, ...] = (ACTION_START, ACTION_STOP)


def hotkey_mode_from_config(config: dict[str, Any]) -> str:
    """Read the push-to-talk mode from persisted config.

    Missing key means the shipped default (``auto``). Anything else present but
    unrecognised is a misconfiguration and raises rather than silently defaulting.
    """
    if HOTKEY_MODE_CONFIG_KEY not in config:
        return DEFAULT_HOTKEY_MODE
    mode = config[HOTKEY_MODE_CONFIG_KEY]
    if not isinstance(mode, str) or mode not in HOTKEY_MODES:
        raise ValueError(
            f"Invalid {HOTKEY_MODE_CONFIG_KEY}: {mode!r}. "
            f"Expected one of {', '.join(HOTKEY_MODES)}."
        )
    return mode


class PressController:
    """Turn raw key-down and key-up events into ``start`` / ``stop`` decisions.

    Pure logic: no AppKit, no timers, no I/O. The caller passes the clock, so the
    hold threshold is exercised directly in tests.

    - ``toggle``: key-down flips recording on and off; key-up is ignored.
    - ``hold``: key-down starts, key-up stops.
    - ``auto``: key-down starts. On key-up, a press held for at least
      ``hold_threshold_s`` stops; a shorter press latches, so recording continues
      until the next key-down stops it.

    Key repeat (further key-down events while the key is already down) is ignored,
    and a key-up with no matching key-down is a no-op.

    ``key_up_available`` says whether key-up will actually be delivered. When it
    is False — the Carbon path without Accessibility, where Carbon reports only
    presses — ``hold`` and ``auto`` run as ``toggle``: without a release the
    controller would hold ``_key_down`` forever, read every later press as key
    repeat, and leave recording running with no way to stop it.
    """

    def __init__(
        self,
        mode: str = DEFAULT_HOTKEY_MODE,
        *,
        hold_threshold_s: float = HOLD_THRESHOLD_S,
        key_up_available: bool = True,
    ) -> None:
        if mode not in HOTKEY_MODES:
            raise ValueError(
                f"Invalid hotkey mode: {mode!r}. Expected one of {', '.join(HOTKEY_MODES)}."
            )
        if not hold_threshold_s > 0:
            raise ValueError(f"hold_threshold_s must be positive, got {hold_threshold_s!r}")
        self.mode = mode
        self.hold_threshold_s = float(hold_threshold_s)
        self._key_up_available = bool(key_up_available)
        self._recording = False
        self._key_down = False
        self._pressed_at: float | None = None
        self._latched = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def key_up_available(self) -> bool:
        """Whether key-up events reach this controller at all."""
        return self._key_up_available

    @property
    def effective_mode(self) -> str:
        """The mode actually in force, which is ``toggle`` when key-up is missing."""
        if self._key_up_available:
            return self.mode
        return HOTKEY_MODE_TOGGLE

    def set_key_up_available(self, available: bool) -> None:
        """Record whether key-up will be delivered, e.g. after registering.

        Changing it changes the mode in force, so the half-finished press the
        old mode was tracking is dropped. Recording itself is left alone; the
        caller reconciles that with :meth:`sync`.
        """
        available = bool(available)
        if available == self._key_up_available:
            return
        self._key_up_available = available
        self._key_down = False
        self._pressed_at = None
        self._latched = False

    @property
    def is_latched(self) -> bool:
        """True when a short ``auto`` press left recording running until the next press."""
        return self._latched

    def reset(self) -> None:
        """Drop all state, e.g. when the shortcut is re-registered."""
        self._recording = False
        self._key_down = False
        self._pressed_at = None
        self._latched = False

    def sync(self, is_recording: bool) -> None:
        """Reconcile with the app, which may have refused or ended a recording itself."""
        self._recording = bool(is_recording)
        if not self._recording:
            self._latched = False

    def on_key_down(self, now: float) -> str | None:
        mode = self.effective_mode
        if self._key_up_available:
            # Only a key-up can clear this, so it is never set when none is coming:
            # otherwise the second press would look like key repeat forever.
            if self._key_down:
                return None  # key repeat
            self._key_down = True
            self._pressed_at = now

        if mode == HOTKEY_MODE_TOGGLE:
            return self._flip()
        if mode == HOTKEY_MODE_HOLD:
            if self._recording:
                return None
            self._recording = True
            return ACTION_START

        # auto
        if self._latched:
            self._latched = False
            self._recording = False
            return ACTION_STOP
        if self._recording:
            return None
        self._recording = True
        return ACTION_START

    def on_key_up(self, now: float) -> str | None:
        mode = self.effective_mode
        if not self._key_down:
            return None  # release with no press we saw
        pressed_at = self._pressed_at
        self._key_down = False
        self._pressed_at = None

        if mode == HOTKEY_MODE_TOGGLE:
            return None
        if not self._recording:
            return None
        if mode == HOTKEY_MODE_HOLD:
            self._recording = False
            return ACTION_STOP

        # auto
        held = now - pressed_at if pressed_at is not None else 0.0
        if held >= self.hold_threshold_s:
            self._recording = False
            return ACTION_STOP
        self._latched = True
        return None

    def _flip(self) -> str:
        if self._recording:
            self._recording = False
            return ACTION_STOP
        self._recording = True
        return ACTION_START


@dataclass
class HotkeyRegistration:
    """Everything one registration owns, so ``unregister`` can undo all of it.

    Every monitor is recorded on this object as soon as it exists, including on
    the paths that go on to fail: an abandoned NSEvent monitor stays installed
    for the life of the process, and the hotkey retry timer would stack a fresh
    set on every attempt.
    """

    unregister_fn: Callable[[], None] | None = None
    global_monitor: Any = None
    local_monitor: Any = None
    global_flags_monitor: Any = None
    local_flags_monitor: Any = None
    global_key_up_monitor: Any = None
    local_key_up_monitor: Any = None
    handlers: list[Any] | None = None
    #: True only when key-up really reaches the app. False here means ``hold``
    #: and ``auto`` cannot work and must fall back to ``toggle``.
    key_up_available: bool = False


def hotkey_registration_active(registration: HotkeyRegistration | None) -> bool:
    """Return True when at least one event monitor or hotkey is active."""
    if registration is None:
        return False
    return any(
        (
            registration.unregister_fn is not None,
            registration.global_monitor is not None,
            registration.local_monitor is not None,
        )
    )


def is_bundled_app() -> bool:
    """Return True when running inside a PyInstaller .app bundle."""
    return hasattr(sys, "_MEIPASS")


def hotkey_permissions_ok() -> bool:
    """Best-effort Accessibility check (can lag behind System Settings)."""
    return bool(AXIsProcessTrusted())


def parse_codesign_details(output: str) -> dict[str, str]:
    """Parse `codesign -dv` text into a small signature summary."""
    details: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        details[key.strip()] = value.strip()
    return details


def executable_signature_info(executable: str | None = None) -> dict[str, str]:
    """Return code-signing details for the running binary."""
    target = executable or sys.executable
    info = {
        "executable": target,
        "signature": "unknown",
        "team_identifier": "unknown",
        "cdhash": "unknown",
        "identifier": BUNDLE_ID,
    }
    codesign = "/usr/bin/codesign"
    if not os.path.isfile(codesign):
        codesign = "codesign"
    try:
        result = subprocess.run(
            [codesign, "-dv", target],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return info

    details = parse_codesign_details(f"{result.stderr}\n{result.stdout}")
    info["signature"] = details.get("Signature", info["signature"])
    info["team_identifier"] = details.get("TeamIdentifier", info["team_identifier"])
    info["cdhash"] = details.get("CDHash", info["cdhash"])
    info["identifier"] = details.get("Identifier", info["identifier"])
    return info


def probe_global_monitor() -> bool:
    """Return True when macOS accepts a global key-down monitor for this binary."""
    monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
        NSEventMaskKeyDown,
        lambda _event: None,
    )
    if monitor is None:
        return False
    NSEvent.removeMonitor_(monitor)
    return True


def hotkey_diagnostics() -> dict[str, Any]:
    """Collect runtime facts used to debug shortcut permission mismatches."""
    signature = executable_signature_info()
    ax_trusted = hotkey_permissions_ok()
    monitor_ok = probe_global_monitor()
    return {
        "bundled": is_bundled_app(),
        "executable": sys.executable,
        "bundle_id": signature.get("identifier", BUNDLE_ID),
        "signature": signature.get("signature", "unknown"),
        "team_identifier": signature.get("team_identifier", "unknown"),
        "cdhash": signature.get("cdhash", "unknown"),
        "ax_trusted": ax_trusted,
        "global_monitor_ok": monitor_ok,
        "shortcut_effective": ax_trusted and monitor_ok,
    }


def format_hotkey_diagnostics(diagnostics: dict[str, Any] | None = None) -> str:
    """Return a compact, user-facing diagnostic summary."""
    data = diagnostics or hotkey_diagnostics()
    lines = [
        f"Executable: {data['executable']}",
        f"Bundle ID: {data['bundle_id']}",
        f"Signature: {data['signature']}",
        f"CDHash: {data['cdhash']}",
        f"Accessibility trusted: {data['ax_trusted']}",
        f"Global monitor accepted: {data['global_monitor_ok']}",
    ]
    if data.get("bundled") and data.get("signature") == "adhoc":
        lines.append(
            "This DMG build is ad-hoc signed. macOS ties Accessibility to the exact "
            "binary hash, so each reinstall needs a fresh permission grant."
        )
    return "\n".join(lines)


def log_hotkey_diagnostics(logger: Any, *, event: str) -> None:
    """Emit shortcut diagnostics at WARNING so bundled apps log them by default."""
    diagnostics = hotkey_diagnostics()
    logger.warning(
        "%s: bundled=%s executable=%s bundle_id=%s signature=%s cdhash=%s "
        "ax_trusted=%s global_monitor_ok=%s",
        event,
        diagnostics["bundled"],
        diagnostics["executable"],
        diagnostics["bundle_id"],
        diagnostics["signature"],
        diagnostics["cdhash"],
        diagnostics["ax_trusted"],
        diagnostics["global_monitor_ok"],
    )


def reset_accessibility_permission(bundle_id: str = BUNDLE_ID) -> bool:
    """Reset Accessibility TCC entries for Murmur (needed after DMG reinstall)."""
    try:
        result = subprocess.run(
            ["tccutil", "reset", "Accessibility", bundle_id],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except OSError:
        return False


def request_hotkey_permissions(logger: Any = None, *, prompt: bool = True) -> bool:
    """Ask macOS to grant Accessibility permission required for global shortcuts."""
    global _LAST_PERMISSION_LOG

    if hotkey_permissions_ok():
        _LAST_PERMISSION_LOG = True
        return True
    if prompt:
        options = NSDictionary.dictionaryWithObject_forKey_(True, "AXTrustedCheckOptionPrompt")
        AXIsProcessTrustedWithOptions(options)
    trusted = hotkey_permissions_ok()
    if logger is not None and not trusted and _LAST_PERMISSION_LOG is not False:
        log_hotkey_diagnostics(logger, event="Accessibility permission missing")
        _LAST_PERMISSION_LOG = False
    elif trusted:
        _LAST_PERMISSION_LOG = True
    return trusted


def open_privacy_settings() -> None:
    """Open macOS Privacy settings for Accessibility."""
    try:
        subprocess.run(["open", ACCESSIBILITY_SETTINGS_URL], check=False)
    except OSError:
        pass


def permission_status_message(*, diagnostics: dict[str, Any] | None = None) -> str:
    """User-facing summary of macOS permissions for Murmur."""
    data = diagnostics or hotkey_diagnostics()
    lines = [
        "Global shortcut (⌥ Space): works without extra permissions.",
        "",
        "Paste at cursor: enable Murmur in "
        "System Settings → Privacy & Security → Accessibility.",
        "",
        "After installing from a DMG:",
        "1. Open Murmur from /Applications.",
        "2. Press ⌥ Space to test the shortcut.",
        "3. Use Enable Shortcut Permission… to grant Accessibility for paste.",
        "4. Quit Murmur completely, then reopen it once.",
    ]
    if data.get("bundled") and data.get("signature") == "adhoc":
        lines.extend(
            [
                "",
                "Ad-hoc builds get a new binary hash on every reinstall. If paste "
                "stopped working, remove old Murmur entries from Accessibility and "
                "grant access again.",
                "",
                "For stable permissions across updates, distribute a Developer ID "
                "signed and notarized build (see RELEASE_SIGNING.md).",
            ]
        )
    lines.extend(["", format_hotkey_diagnostics(data)])
    return "\n".join(lines)


def hotkey_from_config(config: dict[str, Any]) -> HotkeyBinding:
    """Load a hotkey binding from persisted config."""
    keycode = config.get("hotkey_keycode", DEFAULT_HOTKEY.keycode)
    if isinstance(keycode, float) and keycode.is_integer():
        keycode = int(keycode)
    if not isinstance(keycode, int):
        keycode = DEFAULT_HOTKEY.keycode
    return HotkeyBinding(
        keycode=keycode,
        command=bool(config.get("hotkey_command", DEFAULT_HOTKEY.command)),
        option=bool(config.get("hotkey_option", DEFAULT_HOTKEY.option)),
        control=bool(config.get("hotkey_control", DEFAULT_HOTKEY.control)),
        shift=bool(config.get("hotkey_shift", DEFAULT_HOTKEY.shift)),
        fn=bool(config.get("hotkey_fn", DEFAULT_HOTKEY.fn)),
    )


def hotkey_to_config(binding: HotkeyBinding, *, label: str | None = None) -> dict[str, Any]:
    """Serialize a hotkey binding into config keys."""
    return {
        "hotkey_keycode": binding.keycode,
        "hotkey_command": binding.command,
        "hotkey_option": binding.option,
        "hotkey_control": binding.control,
        "hotkey_shift": binding.shift,
        "hotkey_fn": binding.fn,
        "hotkey_label": label,
    }


def ns_modifier_flags(raw_flags: int) -> int:
    """Normalize AppKit modifier flags for reliable comparisons."""
    return int(raw_flags) & NSEventModifierFlagDeviceIndependentFlagsMask


def binding_from_ns_flags(keycode: int, raw_flags: int) -> HotkeyBinding:
    """Build a binding from an AppKit key event."""
    flags = ns_modifier_flags(raw_flags)
    return HotkeyBinding(
        keycode=int(keycode),
        command=bool(flags & NSEventModifierFlagCommand),
        option=bool(flags & NSEventModifierFlagOption),
        control=bool(flags & NSEventModifierFlagControl),
        shift=bool(flags & NSEventModifierFlagShift),
        fn=bool(flags & NSEventModifierFlagFunction),
    )


def binding_has_modifier(binding: HotkeyBinding) -> bool:
    return any(
        (
            binding.command,
            binding.option,
            binding.control,
            binding.shift,
            binding.fn,
        )
    )


def capture_label_for_binding(binding: HotkeyBinding, *, characters: str = "") -> str | None:
    if binding.keycode == SPACE_KEYCODE:
        return "Space"
    if len(characters) == 1 and characters.isalpha():
        return characters.upper()
    if characters:
        return characters
    return None


def _modifier_labels(binding: HotkeyBinding) -> list[str]:
    labels: list[str] = []
    if binding.control:
        labels.append("⌃")
    if binding.option:
        labels.append("⌥")
    if binding.shift:
        labels.append("⇧")
    if binding.command:
        labels.append("⌘")
    if binding.fn:
        labels.append("fn")
    return labels


def _key_label(keycode: int) -> str:
    named = KEYCODE_LABELS.get(keycode)
    if named is not None:
        return named
    return f"Key {keycode}"


def format_hotkey(binding: HotkeyBinding, *, label: str | None = None) -> str:
    """Return a human-readable shortcut label."""
    parts = _modifier_labels(binding)
    parts.append(label or _key_label(binding.keycode))
    return " ".join(parts)


def format_hotkey_from_config(config: dict[str, Any]) -> str:
    """Format the shortcut stored in config."""
    return format_hotkey(
        hotkey_from_config(config),
        label=config.get("hotkey_label"),
    )


def binding_matches_modifiers(
    binding: HotkeyBinding,
    keycode: int,
    *,
    command: bool,
    option: bool,
    control: bool,
    shift: bool,
    fn: bool,
) -> bool:
    return (
        keycode == binding.keycode
        and command == binding.command
        and option == binding.option
        and control == binding.control
        and shift == binding.shift
        and fn == binding.fn
    )


def binding_matches_ns_event(
    binding: HotkeyBinding,
    keycode: int,
    raw_flags: int,
    *,
    tracked_modifiers: int = 0,
) -> bool:
    """Return True when an AppKit event matches the configured binding."""
    flags = ns_modifier_flags(raw_flags | tracked_modifiers)
    return binding_matches_modifiers(
        binding,
        keycode,
        command=bool(flags & NSEventModifierFlagCommand),
        option=bool(flags & NSEventModifierFlagOption),
        control=bool(flags & NSEventModifierFlagControl),
        shift=bool(flags & NSEventModifierFlagShift),
        fn=bool(flags & NSEventModifierFlagFunction),
    )


def binding_matches_ns_key_up(binding: HotkeyBinding, keycode: int) -> bool:
    """Return True when an AppKit key-up belongs to the configured binding.

    Key-up matches on the key code alone. Modifiers are routinely released before
    (or with) the key itself, so demanding the exact modifier mask that key-down
    required would drop most real push-to-talk releases and leave recording stuck.
    """
    return int(keycode) == binding.keycode


#: Keyword names a future quickmachotkey might use to expose kEventHotKeyReleased.
_CARBON_KEY_UP_PARAMETERS = ("onKeyUp", "onRelease", "released", "eventKinds")


def carbon_supports_key_up() -> bool:
    """Whether the Carbon hotkey library can deliver key-up (``kEventHotKeyReleased``).

    False for every quickmachotkey released so far. The library installs exactly one
    Carbon event handler at import time, hard-wired to ``kEventHotKeyPressed``, and its
    only entry point (``quickHotKey``) offers no way to ask for the released event.
    ``kEventHotKeyReleased`` exists in ``quickmachotkey.constants`` but nothing consumes
    it. So on the Carbon path Murmur registers key-down with Carbon and borrows an
    NSEvent monitor for key-up alone, which is why ``hold`` and ``auto`` need
    Accessibility even though ``toggle`` does not.

    This stays a probe rather than a hard-coded False so that a quickmachotkey which
    grows a released hook lights the Carbon path up without changes at the call sites.
    """
    try:
        from quickmachotkey import quickHotKey
    except Exception:
        return False
    try:
        parameters = inspect.signature(quickHotKey).parameters
    except (TypeError, ValueError):
        return False
    return any(name in parameters for name in _CARBON_KEY_UP_PARAMETERS)


def unregister_global_hotkey(registration: HotkeyRegistration | None) -> None:
    """Disable and detach a previously registered global hotkey."""
    if registration is None:
        return
    if registration.unregister_fn is not None:
        try:
            registration.unregister_fn()
        except Exception:
            pass
    for monitor in (
        registration.global_monitor,
        registration.local_monitor,
        registration.global_flags_monitor,
        registration.local_flags_monitor,
        registration.global_key_up_monitor,
        registration.local_key_up_monitor,
    ):
        if monitor is not None:
            try:
                NSEvent.removeMonitor_(monitor)
            except Exception:
                pass


def _add_key_up_monitors(
    binding: HotkeyBinding,
    on_key_up: Callable[[], None],
    on_error: Callable[[Exception], None],
) -> tuple[Any, Any, list[Any]]:
    """Attach NSEvent key-up monitors for the binding. Thin glue, no decisions."""

    def handle_key_up(event, *, swallow: bool):
        try:
            if not binding_matches_ns_key_up(binding, event.keyCode()):
                return event
            on_key_up()
            return None if swallow else event
        except Exception as error:  # pragma: no cover - UI callback defensive branch
            on_error(error)
            return event

    def global_key_up_handler(event):
        return handle_key_up(event, swallow=False)

    def local_key_up_handler(event):
        return handle_key_up(event, swallow=True)

    global_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
        NSEventMaskKeyUp,
        global_key_up_handler,
    )
    local_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
        NSEventMaskKeyUp,
        local_key_up_handler,
    )
    return global_monitor, local_monitor, [global_key_up_handler, local_key_up_handler]


def _attach_carbon_key_up_fallback(
    binding: HotkeyBinding,
    registration: HotkeyRegistration,
    on_key_up: Callable[[], None],
    on_error: Callable[[Exception], None],
    logger: Any,
) -> bool:
    """Give the Carbon path a key-up source, since Carbon here cannot provide one.

    See :func:`carbon_supports_key_up`. The degradation is logged, never silent: the
    user gets the shortcut either way, but hold and auto collapse into toggle
    behaviour when Accessibility is missing.
    """
    if logger is not None:
        logger.warning(
            "Carbon hotkeys cannot deliver key-up (quickmachotkey installs a "
            "kEventHotKeyPressed handler only). Using an NSEvent monitor for key-up "
            "on %s, which needs Accessibility.",
            format_hotkey(binding),
        )
    if not hotkey_permissions_ok():
        if logger is not None:
            logger.warning(
                "Accessibility is not granted, so key-up for %s cannot be observed. "
                "Hold and auto behave like toggle until it is granted.",
                format_hotkey(binding),
            )
        return False

    global_monitor, local_monitor, handlers = _add_key_up_monitors(
        binding, on_key_up, on_error
    )
    registration.global_key_up_monitor = global_monitor
    registration.local_key_up_monitor = local_monitor
    registration.handlers = handlers
    if global_monitor is None:
        if logger is not None:
            logger.warning(
                "macOS refused the key-up monitor for %s; hold and auto behave "
                "like toggle.",
                format_hotkey(binding),
            )
        return False
    registration.key_up_available = True
    return True


def register_global_hotkey(
    binding: HotkeyBinding,
    on_trigger: Callable[[], None],
    on_error: Callable[[Exception], None],
    logger: Any,
    *,
    on_key_up: Callable[[], None] | None = None,
    mode: str = HOTKEY_MODE_TOGGLE,
) -> HotkeyRegistration:
    """Register a global hotkey using Carbon RegisterEventHotKey (via quickmachotkey).

    ``on_trigger`` fires on key-down. ``on_key_up`` fires on key-up and is only wired
    up for the modes that need it (``hold`` and ``auto``); ``toggle`` keeps the
    key-down-only behaviour and stays free of Accessibility requirements.

    Falls back to NSEvent monitors (which require Accessibility) if Carbon fails
    or when the binding requires fn (Carbon has no fn modifier bit). Carbon itself
    cannot report key-up here (see :func:`carbon_supports_key_up`), so a Carbon
    registration that needs key-up borrows an NSEvent key-up monitor and logs why.

    Modifier-only bindings are not supported on either path, matching the existing
    behaviour: the flags monitors track modifier state but never trigger.
    """
    if mode not in HOTKEY_MODES:
        raise ValueError(
            f"Invalid hotkey mode: {mode!r}. Expected one of {', '.join(HOTKEY_MODES)}."
        )
    wants_key_up = on_key_up is not None and mode in KEY_UP_MODES
    # 1. Try Carbon hotkey first (does NOT require Accessibility permissions).
    # Carbon RegisterEventHotKey cannot express fn — skip so NSEvent can match it.
    if not binding.fn:
        try:
            from quickmachotkey import quickHotKey
            from quickmachotkey.constants import cmdKey, shiftKey, optionKey, controlKey

            # Map modifiers
            modifier_mask = 0
            if binding.command:
                modifier_mask |= cmdKey
            if binding.shift:
                modifier_mask |= shiftKey
            if binding.option:
                modifier_mask |= optionKey
            if binding.control:
                modifier_mask |= controlKey

            # Define the handler
            def trigger_handler() -> None:
                try:
                    on_trigger()
                except Exception as error:
                    on_error(error)

            # Register the hotkey
            registerable_handler = quickHotKey(
                virtualKey=binding.keycode,
                modifierMask=modifier_mask,
                immediately=True,
            )(trigger_handler)

            if logger is not None:
                if is_bundled_app():
                    logger.warning(
                        "Global Carbon hotkey registered successfully: %s (keycode=%d, mask=0x%x)",
                        format_hotkey(binding),
                        binding.keycode,
                        modifier_mask,
                    )
                else:
                    logger.info(
                        "Global Carbon hotkey registered successfully: %s (keycode=%d, mask=0x%x)",
                        format_hotkey(binding),
                        binding.keycode,
                        modifier_mask,
                    )

            registration = HotkeyRegistration(
                unregister_fn=registerable_handler.unregister,
            )
            # key_up_available stays False unless the NSEvent fallback below
            # really attaches: Carbon itself wires no released handler here.
            if wants_key_up and not carbon_supports_key_up():
                # Never let a key-up problem undo a working key-down registration.
                try:
                    _attach_carbon_key_up_fallback(
                        binding, registration, on_key_up, on_error, logger
                    )
                except Exception as key_up_error:
                    if logger is not None:
                        logger.warning(
                            "Could not attach the key-up monitor for %s: %s",
                            format_hotkey(binding),
                            key_up_error,
                        )
            return registration

        except Exception as carbon_error:
            if logger is not None:
                logger.warning(
                    "Failed to register Carbon hotkey (%s). Falling back to NSEvent monitors.",
                    carbon_error,
                )

    # 2. Fallback to NSEvent monitors (requires Accessibility permissions)
    if not hotkey_permissions_ok():
        raise RuntimeError(
            "Accessibility permission is not active yet for this Murmur process."
        )

    cooldown_state = {"active": False}
    modifier_state = {"flags": 0}

    def reset_cooldown() -> None:
        cooldown_state["active"] = False

    def maybe_trigger() -> None:
        if wants_key_up:
            # Push-to-talk needs every press: PressController already collapses key
            # repeat, and a debounce would swallow the press that ends a latched
            # recording.
            on_trigger()
            return
        if cooldown_state["active"]:
            return
        cooldown_state["active"] = True
        on_trigger()
        import threading

        threading.Timer(0.5, reset_cooldown).start()

    def update_modifiers(event) -> None:
        modifier_state["flags"] = ns_modifier_flags(event.modifierFlags())

    def handle_key_event(event, *, swallow: bool):
        try:
            keycode = event.keyCode()
            if keycode in MODIFIER_KEYCODES:
                return event
            if wants_key_up and bool(getattr(event, "isARepeat", bool)()):
                return event
            combined_flags = event.modifierFlags() | modifier_state["flags"]
            if not binding_matches_ns_event(binding, keycode, combined_flags):
                return event
            maybe_trigger()
            return None if swallow else event
        except Exception as error:  # pragma: no cover - UI callback defensive branch
            on_error(error)
            return event

    def global_key_handler(event):
        return handle_key_event(event, swallow=False)

    def local_key_handler(event):
        return handle_key_event(event, swallow=True)

    def global_flags_handler(event):
        update_modifiers(event)

    def local_flags_handler(event):
        update_modifiers(event)
        return event

    handlers = [
        global_key_handler,
        local_key_handler,
        global_flags_handler,
        local_flags_handler,
    ]

    # Every monitor is recorded on the registration as it is created, and the
    # registration is torn down on any failure: an NSEvent monitor that nobody
    # holds a reference to stays installed for the life of the process, and the
    # retry timer would add another set of them on every attempt.
    registration = HotkeyRegistration(handlers=handlers)
    try:
        registration.global_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSEventMaskKeyDown,
            global_key_handler,
        )
        registration.local_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            NSEventMaskKeyDown,
            local_key_handler,
        )
        registration.global_flags_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSEventMaskFlagsChanged,
            global_flags_handler,
        )
        registration.local_flags_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            NSEventMaskFlagsChanged,
            local_flags_handler,
        )

        if wants_key_up:
            (
                registration.global_key_up_monitor,
                registration.local_key_up_monitor,
                key_up_handlers,
            ) = _add_key_up_monitors(binding, on_key_up, on_error)
            handlers.extend(key_up_handlers)
            registration.key_up_available = registration.global_key_up_monitor is not None
            if not registration.key_up_available and logger is not None:
                logger.warning(
                    "macOS refused the key-up monitor for %s; hold and auto behave "
                    "like toggle.",
                    format_hotkey(binding),
                )

        if registration.global_monitor is None:
            diagnostics = hotkey_diagnostics()
            raise RuntimeError(
                "Failed to register the global shortcut. "
                "Enable Accessibility for the current Murmur.app in "
                "System Settings → Privacy & Security. "
                "After a DMG reinstall, remove old Murmur entries and grant access again. "
                f"CDHash={diagnostics['cdhash']} ax_trusted={diagnostics['ax_trusted']}"
            )
    except BaseException:
        unregister_global_hotkey(registration)
        raise

    if is_bundled_app():
        log_hotkey_diagnostics(logger, event=f"Global hotkey registered: {format_hotkey(binding)}")
    else:
        logger.info(
            "Global hotkey registered: %s (executable=%s, ax_trusted=%s)",
            format_hotkey(binding),
            sys.executable,
            hotkey_permissions_ok(),
        )
    return registration


def register_option_space_hotkey(
    on_trigger: Callable[[], None],
    on_error: Callable[[Exception], None],
    logger: Any,
) -> HotkeyRegistration:
    """Backward-compatible helper for the default Option+Space shortcut."""
    return register_global_hotkey(DEFAULT_HOTKEY, on_trigger, on_error, logger)
