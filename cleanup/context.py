#!/usr/bin/env python3
"""Front-app context capture and bundle-id -> cleanup-mode mapping.

Canonical home for "what app is the user talking into?". `murmur.py` calls
`front_app_bundle_id()` / `capture_context()`; nothing else should reach for
`NSWorkspace` or the Accessibility API directly.

PRIVACY: window titles and selected text are user content. They are returned to
the caller and never logged; this module deliberately takes no logger.

PORTABILITY: every AppKit / ApplicationServices import happens inside a method
so the module imports on a machine without PyObjC and tests can inject fakes.
When the *default* provider is used and PyObjC is missing, the ImportError
propagates -- fail fast rather than silently degrading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# --- config keys -----------------------------------------------------------

# User's fallback mode when the front app is not in the table.
MODE_CONFIG_KEY = "cleanup_mode"
# Per-app user overrides: {bundle_id: mode}. Persisted by the settings/menu wiring.
MODE_OVERRIDES_CONFIG_KEY = "mode_by_app"

DEFAULT_MODE = "dictation"

MODE_DICTATION = "dictation"
MODE_MESSAGE = "message"
MODE_MAIL = "mail"
MODE_NOTES = "notes"
MODE_CODE = "code"

# Mirrors the mode slugs defined by cleanup/modes.py. Kept as a local tuple so
# this module stays importable on its own (no cross-module import cycle).
KNOWN_MODES: tuple[str, ...] = (
    MODE_DICTATION,
    MODE_MESSAGE,
    MODE_MAIL,
    MODE_NOTES,
    MODE_CODE,
)

# --- bundle-id table -------------------------------------------------------
#
# Ids marked (unverified) are not installed on the reference Mac and come from
# vendor documentation rather than a read of the app's Info.plist. They are
# listed in UNVERIFIED_BUNDLE_IDS below; confirm one before relying on it.

DEFAULT_MODE_BY_BUNDLE: dict[str, str] = {
    # Mail clients
    "com.apple.mail": MODE_MAIL,
    "com.microsoft.Outlook": MODE_MAIL,  # unverified
    "com.readdle.smartemail-Mac": MODE_MAIL,  # Spark
    "it.bloop.airmail2": MODE_MAIL,  # Airmail, unverified
    # Chat / messaging
    "com.apple.MobileSMS": MODE_MESSAGE,  # Messages
    "com.tinyspeck.slackmacgap": MODE_MESSAGE,  # unverified
    "com.hnc.Discord": MODE_MESSAGE,  # unverified
    "net.whatsapp.WhatsApp": MODE_MESSAGE,
    "ru.keepcoder.Telegram": MODE_MESSAGE,  # Telegram for macOS, unverified
    "org.telegram.desktop": MODE_MESSAGE,  # Telegram Desktop, unverified
    "com.microsoft.teams": MODE_MESSAGE,  # classic, unverified
    "com.microsoft.teams2": MODE_MESSAGE,  # new Teams, unverified
    # Notes / long-form
    "com.apple.Notes": MODE_NOTES,
    "com.apple.reminders": MODE_NOTES,
    "net.shinyfrog.bear": MODE_NOTES,  # unverified
    "md.obsidian": MODE_NOTES,  # unverified
    "notion.id": MODE_NOTES,  # unverified
    "com.lukilabs.lukiapp": MODE_NOTES,  # Craft, unverified
    # Terminals and editors
    "com.apple.Terminal": MODE_CODE,
    "com.googlecode.iterm2": MODE_CODE,  # unverified
    "com.mitchellh.ghostty": MODE_CODE,  # unverified
    "dev.warp.Warp-Stable": MODE_CODE,  # unverified
    "net.kovidgoyal.kitty": MODE_CODE,  # unverified
    "org.alacritty": MODE_CODE,  # unverified
    "com.microsoft.VSCode": MODE_CODE,  # unverified
    "com.todesktop.230313mzl4w4u92": MODE_CODE,  # Cursor
    "com.apple.dt.Xcode": MODE_CODE,
    "dev.zed.Zed": MODE_CODE,  # unverified
}

# Prefix rules, applied only when there is no exact match. Keys end with a dot.
DEFAULT_MODE_BY_BUNDLE_PREFIX: dict[str, str] = {
    "com.jetbrains.": MODE_CODE,  # IntelliJ, PyCharm, GoLand, ... unverified
}

# Read from the app's Info.plist on the reference Mac.
VERIFIED_BUNDLE_IDS: frozenset[str] = frozenset(
    {
        "com.apple.mail",
        "com.readdle.smartemail-Mac",
        "com.apple.MobileSMS",
        "net.whatsapp.WhatsApp",
        "com.apple.Notes",
        "com.apple.reminders",
        "com.apple.Terminal",
        "com.todesktop.230313mzl4w4u92",
        "com.apple.dt.Xcode",
    }
)

# App not installed on the reference Mac; id taken from vendor documentation.
UNVERIFIED_BUNDLE_IDS: frozenset[str] = frozenset(
    {
        "com.microsoft.Outlook",
        "it.bloop.airmail2",
        "com.tinyspeck.slackmacgap",
        "com.hnc.Discord",
        "ru.keepcoder.Telegram",
        "org.telegram.desktop",
        "com.microsoft.teams",
        "com.microsoft.teams2",
        "net.shinyfrog.bear",
        "md.obsidian",
        "notion.id",
        "com.lukilabs.lukiapp",
        "com.googlecode.iterm2",
        "com.mitchellh.ghostty",
        "dev.warp.Warp-Stable",
        "net.kovidgoyal.kitty",
        "org.alacritty",
        "com.microsoft.VSCode",
        "dev.zed.Zed",
        "com.jetbrains",  # prefix rule "com.jetbrains."
    }
)


@dataclass(frozen=True)
class AppContext:
    """What Murmur knows about the app the user is dictating into.

    `window_title` and `selected_text` are user content: never log them.
    """

    bundle_id: str | None
    app_name: str | None
    window_title: str | None
    selected_text: str | None


# --- injectable macOS providers -------------------------------------------


def _ax_chain(element: Any, attributes: tuple[Any, ...]) -> Any:
    """Walk a chain of AX attributes; return None on any AX failure."""
    from ApplicationServices import AXUIElementCopyAttributeValue

    current = element
    for attribute in attributes:
        if current is None:
            return None
        try:
            error, value = AXUIElementCopyAttributeValue(current, attribute, None)
        except Exception:
            return None
        if error != 0:
            return None
        current = value
    return current


class DefaultWorkspace:
    """Front-app lookup backed by NSWorkspace."""

    def frontmost_application(self) -> tuple[str | None, str | None, int | None]:
        """Return (bundle_id, app_name, pid); all None when there is no front app."""
        from AppKit import NSWorkspace  # lazy: keeps this module importable anywhere

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None, None, None
        return (
            app.bundleIdentifier() or None,
            app.localizedName() or None,
            int(app.processIdentifier()),
        )


class DefaultAccessibility:
    """Window title and selected text via the Accessibility API."""

    def is_trusted(self) -> bool:
        from ApplicationServices import AXIsProcessTrusted

        return bool(AXIsProcessTrusted())

    def window_title(self, pid: int) -> str | None:
        from ApplicationServices import (
            AXUIElementCreateApplication,
            kAXFocusedWindowAttribute,
            kAXTitleAttribute,
        )

        try:
            app_element = AXUIElementCreateApplication(pid)
        except Exception:
            return None
        value = _ax_chain(app_element, (kAXFocusedWindowAttribute, kAXTitleAttribute))
        return str(value) if value else None

    def selected_text(self, pid: int) -> str | None:
        from ApplicationServices import (
            AXUIElementCreateApplication,
            kAXFocusedUIElementAttribute,
            kAXSelectedTextAttribute,
        )

        try:
            app_element = AXUIElementCreateApplication(pid)
        except Exception:
            return None
        value = _ax_chain(
            app_element, (kAXFocusedUIElementAttribute, kAXSelectedTextAttribute)
        )
        return str(value) if value else None


# --- capture ---------------------------------------------------------------


def capture_context(
    *,
    include_selection: bool = False,
    workspace: Any = None,
    ax: Any = None,
) -> AppContext:
    """Describe the frontmost app.

    Window title and selected text are read only when Accessibility is granted;
    otherwise they are None. `include_selection` defaults to False -- the
    selected-text getter is never called unless a caller opts in -- because no
    prompt consumes selected text yet; it stays off until a mode uses it.
    """
    workspace = DefaultWorkspace() if workspace is None else workspace
    ax = DefaultAccessibility() if ax is None else ax

    bundle_id, app_name, pid = workspace.frontmost_application()
    if pid is None:
        return AppContext(bundle_id, app_name, None, None)

    if not ax.is_trusted():
        return AppContext(bundle_id, app_name, None, None)

    window_title = ax.window_title(pid)
    selected_text = ax.selected_text(pid) if include_selection else None
    return AppContext(bundle_id, app_name, window_title, selected_text)


def front_app_bundle_id(workspace: Any = None) -> str | None:
    """Bundle id of the frontmost app, or None."""
    workspace = DefaultWorkspace() if workspace is None else workspace
    return workspace.frontmost_application()[0]


# --- mode mapping ----------------------------------------------------------


def default_mode_for_bundle(bundle_id: str | None) -> str | None:
    """Table lookup only: exact match, then prefix rule. None when unknown."""
    if not bundle_id:
        return None
    mode = DEFAULT_MODE_BY_BUNDLE.get(bundle_id)
    if mode is not None:
        return mode
    for prefix, prefix_mode in DEFAULT_MODE_BY_BUNDLE_PREFIX.items():
        if bundle_id.startswith(prefix):
            return prefix_mode
    return None


def is_terminal_or_editor(bundle_id: str | None) -> bool:
    """True for terminals and code editors (used by the coding-mode module)."""
    return default_mode_for_bundle(bundle_id) == MODE_CODE


def resolve_mode(context: AppContext, config: dict[str, Any]) -> str:
    """Single entry point for resolving a cleanup mode; precedence, in order:

    1. ``mode_by_app`` user override for this app's bundle id (always applies,
       even when ``context_awareness`` is off -- the user set it explicitly).
    2. The built-in bundle table (exact match, then prefix), but only when
       ``config.get("context_awareness", True)`` is True.
    3. ``config.get("cleanup_mode", "dictation")``.

    Raises :class:`cleanup.modes.UnknownModeError` if the resolved mode is
    not a known mode.
    """
    bundle_id = context.bundle_id

    overrides = config.get(MODE_OVERRIDES_CONFIG_KEY) or {}
    if bundle_id:
        override = overrides.get(bundle_id)
        if override:
            _require_known_mode(override)
            return override

    if config.get("context_awareness", True):
        mode = default_mode_for_bundle(bundle_id)
        if mode is not None:
            return mode

    configured = config.get(MODE_CONFIG_KEY) or DEFAULT_MODE
    _require_known_mode(configured)
    return configured


def remember_mode(config: dict[str, Any], bundle_id: str, mode: str) -> dict[str, Any]:
    """Copy of `config` with `bundle_id` pinned to `mode`. Does not mutate."""
    if not bundle_id:
        raise ValueError("bundle_id is required to remember a mode")
    _require_known_mode(mode)

    updated = dict(config)
    overrides = dict(updated.get(MODE_OVERRIDES_CONFIG_KEY) or {})
    overrides[bundle_id] = mode
    updated[MODE_OVERRIDES_CONFIG_KEY] = overrides
    return updated


def forget_mode(config: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    """Copy of `config` with the override for `bundle_id` removed. Does not mutate."""
    if not bundle_id:
        raise ValueError("bundle_id is required to forget a mode")

    updated = dict(config)
    overrides = dict(updated.get(MODE_OVERRIDES_CONFIG_KEY) or {})
    overrides.pop(bundle_id, None)
    updated[MODE_OVERRIDES_CONFIG_KEY] = overrides
    return updated


def _require_known_mode(mode: str) -> None:
    from cleanup.modes import MODES, UnknownModeError

    if mode not in MODES:
        raise UnknownModeError(
            f"unknown cleanup mode: {mode!r} (expected one of {tuple(MODES)})"
        )
