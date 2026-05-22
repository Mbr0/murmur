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


class TextInsertionService:
    def __init__(self, logger: Any):
        self._logger = logger

    def paste_text(self, text: str) -> None:
        pyperclip.copy(text)
        time.sleep(0.1)

        v_keycode = 9
        event_down = CGEventCreateKeyboardEvent(None, v_keycode, True)
        CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event_down)

        time.sleep(0.05)

        event_up = CGEventCreateKeyboardEvent(None, v_keycode, False)
        CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event_up)

        self._logger.info("Paste sent via CGEvent")
