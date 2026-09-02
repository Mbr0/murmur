#!/usr/bin/env python3
"""Text insertion service for clipboard + Cmd+V workflow.

Two seams, added in Wave 5 so ``tests/integration/test_paste_flow.py`` can
drive the whole paste path without touching the real pasteboard or posting a
real Cmd+V into whatever window happens to be focused: ``pasteboard`` and
``post_keystroke``. Both default to the real thing, so no caller changes.
"""

from __future__ import annotations

from typing import Any
import time

import pyperclip
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
)

# Delay after Cmd+V key-up before restoring the previous clipboard.
# Target apps often read the pasteboard asynchronously after the paste event;
# ~50ms is too early and can restore before the paste completes. 0.4s gives
# enough headroom without feeling sluggish on the happy path.
CLIPBOARD_RESTORE_DELAY_S = 0.4

# Delay after writing the transcript, before Cmd+V: the pasteboard write is
# asynchronous too, and pasting into an app that has not seen it yet inserts
# the *previous* clipboard.
PASTEBOARD_SETTLE_DELAY_S = 0.1

# How long Cmd+V is held between the key-down and key-up events.
KEY_HOLD_DELAY_S = 0.05

#: Virtual keycode of "V" on every keyboard layout (it is positional).
V_KEYCODE = 9


class SystemPasteboard:
    """Default pasteboard seam: the real system pasteboard, through pyperclip.

    ``pyperclip`` is looked up on each call rather than captured, so the unit
    tests that patch the module global keep working.
    """

    def read(self) -> str:
        """Whatever is on the pasteboard right now."""
        return pyperclip.paste()

    def write(self, text: str) -> None:
        """Replace the pasteboard contents with ``text``."""
        pyperclip.copy(text)


def post_paste_keystroke() -> None:
    """Default event-poster seam: one Cmd+V, as a CGEvent key-down/key-up pair."""
    event_down = CGEventCreateKeyboardEvent(None, V_KEYCODE, True)
    CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, event_down)

    time.sleep(KEY_HOLD_DELAY_S)

    event_up = CGEventCreateKeyboardEvent(None, V_KEYCODE, False)
    CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, event_up)


class TextInsertionService:
    """Pastes a transcript into the focused app and puts the clipboard back.

    ``pasteboard`` is anything with ``read()``/``write(text)`` and
    ``post_keystroke`` is any zero-argument callable; both default to the real
    macOS ones.
    """

    def __init__(
        self,
        logger: Any,
        pasteboard: Any = None,
        post_keystroke: Any = None,
    ):
        self._logger = logger
        self._pasteboard = pasteboard if pasteboard is not None else SystemPasteboard()
        self._post_keystroke = (
            post_keystroke if post_keystroke is not None else post_paste_keystroke
        )

    def paste_text(self, text: str) -> None:
        # PRODUCT: restore the user's previous clipboard after paste so we do not
        # permanently clobber it. Transcript remains pasted into the focused app.
        # If restore fails, leave the transcript on the clipboard.
        previous = self._pasteboard.read()
        self._pasteboard.write(text)
        time.sleep(PASTEBOARD_SETTLE_DELAY_S)

        try:
            self._post_keystroke()
            self._logger.info("Paste sent via CGEvent")
        finally:
            time.sleep(CLIPBOARD_RESTORE_DELAY_S)
            try:
                self._pasteboard.write(previous)
            except Exception as restore_error:
                self._logger.warning(
                    "Clipboard restore failed; leaving transcript on clipboard: %s",
                    restore_error,
                )
