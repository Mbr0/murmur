#!/usr/bin/env python3
"""Global hotkey registration helpers."""

from __future__ import annotations

from typing import Any, Callable

from Quartz import (
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetCurrent,
    CGEventGetFlags,
    CGEventGetIntegerValueField,
    CGEventMaskBit,
    CGEventTapCreate,
    CGEventTapEnable,
    kCFRunLoopCommonModes,
    kCGEventFlagMaskAlternate,
    kCGEventKeyDown,
    kCGEventKeyUp,
    kCGEventFlagsChanged,
    kCGHeadInsertEventTap,
    kCGKeyboardEventKeycode,
    kCGSessionEventTap,
    kCGEventTapOptionDefault,
)


def register_option_space_hotkey(
    on_trigger: Callable[[], None],
    on_error: Callable[[Exception], None],
    logger: Any,
) -> Any:
    space_keycode = 49
    cooldown_state = {"active": False}

    def reset_cooldown():
        cooldown_state["active"] = False

    def callback(proxy, event_type, event, refcon):
        try:
            flags = CGEventGetFlags(event)
            has_option = (flags & kCGEventFlagMaskAlternate) != 0
            if event_type == kCGEventKeyDown:
                keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
                if keycode == space_keycode and has_option and not cooldown_state["active"]:
                    cooldown_state["active"] = True
                    on_trigger()
                    import threading

                    threading.Timer(0.5, reset_cooldown).start()
                    return None
        except Exception as error:  # pragma: no cover - UI callback defensive branch
            on_error(error)
        return event

    event_mask = CGEventMaskBit(kCGEventKeyDown) | CGEventMaskBit(kCGEventKeyUp) | CGEventMaskBit(
        kCGEventFlagsChanged
    )
    tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionDefault,
        event_mask,
        callback,
        None,
    )
    if tap is None:
        raise RuntimeError("Failed to create event tap. Check Accessibility permissions.")

    run_loop_source = CFMachPortCreateRunLoopSource(None, tap, 0)
    CFRunLoopAddSource(CFRunLoopGetCurrent(), run_loop_source, kCFRunLoopCommonModes)
    CGEventTapEnable(tap, True)
    logger.info("Global hotkey (Option+Space) registered via CGEventTap")
    return tap
