#!/usr/bin/env python3
"""Text insertion service for clipboard + Cmd+V workflow."""

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


class TextInsertionService:
    def __init__(self, logger: Any):
        self._logger = logger

    def paste_text(self, text: str) -> None:
        # PRODUCT: restore the user's previous clipboard after paste so we do not
        # permanently clobber it. Transcript remains pasted into the focused app.
        # If restore fails, leave the transcript on the clipboard.
        previous = pyperclip.paste()
        pyperclip.copy(text)
        time.sleep(0.1)

        try:
            v_keycode = 9
            event_down = CGEventCreateKeyboardEvent(None, v_keycode, True)
            CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, event_down)

            time.sleep(0.05)

            event_up = CGEventCreateKeyboardEvent(None, v_keycode, False)
            CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, event_up)

            self._logger.info("Paste sent via CGEvent")
        finally:
            time.sleep(CLIPBOARD_RESTORE_DELAY_S)
            try:
                pyperclip.copy(previous)
            except Exception as restore_error:
                self._logger.warning(
                    "Clipboard restore failed; leaving transcript on clipboard: %s",
                    restore_error,
                )
