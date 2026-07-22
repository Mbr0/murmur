#!/usr/bin/env python3
"""Global hotkey registration and configuration helpers."""

from __future__ import annotations

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


@dataclass
class HotkeyRegistration:
    unregister_fn: Callable[[], None] | None = None
    global_monitor: Any = None
    local_monitor: Any = None
    global_flags_monitor: Any = None
    local_flags_monitor: Any = None
    handlers: list[Any] | None = None


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
    ):
        if monitor is not None:
            try:
                NSEvent.removeMonitor_(monitor)
            except Exception:
                pass


def register_global_hotkey(
    binding: HotkeyBinding,
    on_trigger: Callable[[], None],
    on_error: Callable[[Exception], None],
    logger: Any,
) -> HotkeyRegistration:
    """Register a global hotkey using Carbon RegisterEventHotKey (via quickmachotkey).

    Falls back to NSEvent monitors (which require Accessibility) if Carbon fails
    or when the binding requires fn (Carbon has no fn modifier bit).
    """
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

            return HotkeyRegistration(
                unregister_fn=registerable_handler.unregister,
            )

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

    global_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
        NSEventMaskKeyDown,
        global_key_handler,
    )
    local_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
        NSEventMaskKeyDown,
        local_key_handler,
    )
    global_flags_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
        NSEventMaskFlagsChanged,
        global_flags_handler,
    )
    local_flags_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
        NSEventMaskFlagsChanged,
        local_flags_handler,
    )

    if global_monitor is None:
        diagnostics = hotkey_diagnostics()
        raise RuntimeError(
            "Failed to register the global shortcut. "
            "Enable Accessibility for the current Murmur.app in "
            "System Settings → Privacy & Security. "
            "After a DMG reinstall, remove old Murmur entries and grant access again. "
            f"CDHash={diagnostics['cdhash']} ax_trusted={diagnostics['ax_trusted']}"
        )

    if is_bundled_app():
        log_hotkey_diagnostics(logger, event=f"Global hotkey registered: {format_hotkey(binding)}")
    else:
        logger.info(
            "Global hotkey registered: %s (executable=%s, ax_trusted=%s)",
            format_hotkey(binding),
            sys.executable,
            hotkey_permissions_ok(),
        )
    return HotkeyRegistration(
        global_monitor=global_monitor,
        local_monitor=local_monitor,
        global_flags_monitor=global_flags_monitor,
        local_flags_monitor=local_flags_monitor,
        handlers=handlers,
    )


def register_option_space_hotkey(
    on_trigger: Callable[[], None],
    on_error: Callable[[Exception], None],
    logger: Any,
) -> HotkeyRegistration:
    """Backward-compatible helper for the default Option+Space shortcut."""
    return register_global_hotkey(DEFAULT_HOTKEY, on_trigger, on_error, logger)
