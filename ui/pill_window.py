#!/usr/bin/env python3
"""Floating pill shown near the text cursor while Murmur is dictating.

Three layers, deliberately separable:

1. **State** — :class:`PillState` and :class:`PillController` are plain Python.
   They own the phase, the one line of text and the fade deadline, and they are
   the only thing the tests need. Engines that cannot stream (whisper.cpp) drive
   ``listening -> working -> done`` and the pill shows state only; engines that
   can (Voxtral MLX) also push :class:`~engines.base.Partial` objects and the
   pill shows the live text.
2. **AppKit** — :class:`PillWindow` renders that state in a borderless,
   non-activating panel. Every AppKit import is lazy so this module still
   imports on a machine without PyObjC.
3. **Presenter** — :class:`PillPresenter` is the public API for the wiring step.
   Every method is safe to call from a background thread; the work hops to the
   main thread the way the rest of the app does it.

Transcript text lives only in the state object. It is never logged.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from engines.base import Partial

APP_NAME = "Murmur"

PHASE_IDLE = "idle"
PHASE_LISTENING = "listening"
PHASE_WORKING = "working"
PHASE_DONE = "done"
PHASE_ERROR = "error"
PHASES = (PHASE_IDLE, PHASE_LISTENING, PHASE_WORKING, PHASE_DONE, PHASE_ERROR)

#: One line, newest words first: longer text is cut at the *start*.
MAX_TEXT_CHARS = 60
ELLIPSIS = "…"

DONE_HIDE_AFTER_S = 1.2
ERROR_HIDE_AFTER_S = 3.0
#: Only the disappearance is animated, and only when reduce-motion is off.
FADE_DURATION_S = 0.15

#: Points between the caret (or pointer) and the pill.
ANCHOR_GAP = 10.0
PILL_SIZE = (260.0, 34.0)
PILL_CORNER_RADIUS = 17.0
PILL_H_PADDING = 12.0

#: Text glyph per phase — no image assets, no animation.
GLYPHS = {
    PHASE_IDLE: "",
    PHASE_LISTENING: "●",  # ●
    PHASE_WORKING: "…",  # …
    PHASE_DONE: "✓",  # ✓
    PHASE_ERROR: "!",
}

_ACCESSIBILITY_LABELS = {
    PHASE_IDLE: APP_NAME,
    PHASE_LISTENING: f"{APP_NAME} listening",
    PHASE_WORKING: f"{APP_NAME} working",
    PHASE_DONE: f"{APP_NAME} done",
}


def _one_line(text: object) -> str:
    """Collapse any whitespace run to a single space. The pill is one line."""
    return " ".join(str(text or "").split())


def ellipsise_start(text: object, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Shorten to ``max_chars`` by dropping the *oldest* words.

    The newest words are the ones the speaker just said, so those are the ones
    worth keeping; the cut end gets the ellipsis.
    """
    assert max_chars >= 2, "max_chars must leave room for the ellipsis"
    line = _one_line(text)
    if len(line) <= max_chars:
        return line
    return ELLIPSIS + line[-(max_chars - 1) :]


def partial_buffer_text(partial: object) -> str:
    """The pill's whole buffer after ``partial``, which simply *is* its text.

    :class:`~engines.base.Partial` is cumulative: every partial carries the
    utterance so far, never a delta, so the newest one replaces the buffer
    outright. There is deliberately no delta-vs-cumulative heuristic here —
    guessing appended a second copy of the sentence the moment a partial
    revised a word it had already emitted.
    """
    return _one_line(getattr(partial, "text", ""))


@dataclass(frozen=True)
class PillState:
    """What the pill shows right now. ``text`` is already display-ready."""

    phase: str = PHASE_IDLE
    text: str = ""
    visible: bool = False

    @property
    def glyph(self) -> str:
        return GLYPHS.get(self.phase, "")


class PillController:
    """The pill's state machine. No AppKit, no clock of its own beyond ``now``."""

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        assert callable(now), "now must be callable"
        self._now = now
        self._state = PillState()
        # Full text, kept only so the caller can read the finished utterance.
        self._buffer = ""
        self._message = ""
        self._hide_at: float | None = None
        self.final_text_len = 0

    @property
    def state(self) -> PillState:
        return self._state

    @property
    def text(self) -> str:
        """The full (un-ellipsised) text so far. Never write this to a log."""
        return self._buffer

    @property
    def hide_deadline(self) -> float | None:
        """Monotonic time the pill should fade at, or None while it stays up."""
        return self._hide_at

    def on_listening(self) -> PillState:
        """Recording started: clear the previous utterance and show the pill."""
        self._buffer = ""
        self._message = ""
        self.final_text_len = 0
        self._hide_at = None
        self._state = PillState(phase=PHASE_LISTENING, text="", visible=True)
        return self._state

    def on_partial(self, partial: Partial) -> PillState:
        """Fold a streamed partial in. ``is_final`` moves the pill to *done*."""
        assert partial is not None, "partial is required"
        self._buffer = partial_buffer_text(partial)
        if getattr(partial, "is_final", False):
            return self.on_done(len(self._buffer))
        self._hide_at = None
        self._state = PillState(
            phase=PHASE_LISTENING, text=ellipsise_start(self._buffer), visible=True
        )
        return self._state

    def on_working(self, label: str | None = None) -> PillState:
        """Audio is in, the engine (or cleanup) is thinking.

        ``label`` replaces the displayed line for a wait the user would
        otherwise read as a hang — the cleanup server's first, multi-second
        start. It is display only: the buffered transcript is untouched, so the
        *done* state still shows the words that were actually pasted.
        """
        self._hide_at = None
        line = _one_line(label) if label else self._buffer
        self._state = PillState(
            phase=PHASE_WORKING, text=ellipsise_start(line), visible=True
        )
        return self._state

    def on_done(self, final_text_len: int = 0) -> PillState:
        """Text landed: show the checkmark state, then fade after 1.2 s."""
        assert final_text_len >= 0, "final_text_len cannot be negative"
        self.final_text_len = int(final_text_len)
        self._hide_at = self._now() + DONE_HIDE_AFTER_S
        self._state = PillState(
            phase=PHASE_DONE, text=ellipsise_start(self._buffer), visible=True
        )
        return self._state

    def on_error(self, message: object) -> PillState:
        """Show a short failure message for 3 s. Drops any buffered transcript."""
        self._buffer = ""
        self._message = _one_line(message)
        self._hide_at = self._now() + ERROR_HIDE_AFTER_S
        self._state = PillState(
            phase=PHASE_ERROR, text=ellipsise_start(self._message), visible=True
        )
        return self._state

    def tick(self, now: float | None = None) -> bool:
        """Return True exactly once, when the fade deadline has been reached."""
        if self._hide_at is None:
            return False
        moment = self._now() if now is None else now
        if moment < self._hide_at:
            return False
        self.hide()
        return True

    def hide(self) -> PillState:
        """Drop back to idle and forget the transcript."""
        self._buffer = ""
        self._message = ""
        self._hide_at = None
        self._state = PillState()
        return self._state

    def accessibility_label(self) -> str:
        """VoiceOver string. State only — the transcript is never announced."""
        if self._state.phase == PHASE_ERROR:
            return f"{APP_NAME} error: {self._message}" if self._message else f"{APP_NAME} error"
        return _ACCESSIBILITY_LABELS.get(self._state.phase, APP_NAME)


def place_pill(
    anchor_rect: tuple[float, float, float, float],
    pill_size: tuple[float, float],
    screen_frame: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Bottom-left origin for the pill, in AppKit screen coordinates (y up).

    The pill sits just below the anchor and centred on it, flips above when
    there is no room below, and is clamped to ``screen_frame`` either way.
    """
    assert anchor_rect is not None, "anchor_rect is required"
    anchor_x, anchor_y, anchor_w, anchor_h = (float(v) for v in anchor_rect)
    pill_w, pill_h = (float(v) for v in pill_size)
    screen_x, screen_y, screen_w, screen_h = (float(v) for v in screen_frame)

    x = anchor_x + anchor_w / 2.0 - pill_w / 2.0
    y = anchor_y - ANCHOR_GAP - pill_h
    if y < screen_y:
        above = anchor_y + anchor_h + ANCHOR_GAP
        if above + pill_h <= screen_y + screen_h:
            y = above

    x = max(screen_x, min(x, screen_x + screen_w - pill_w))
    y = max(screen_y, min(y, screen_y + screen_h - pill_h))
    return (x, y)


def resolve_anchor(
    caret_provider: Callable[[], tuple[float, float, float, float] | None] | None = None,
    mouse_provider: Callable[[], tuple[float, float] | None] | None = None,
) -> tuple[float, float, float, float] | None:
    """Anchor rect for the pill: the text caret when known, else the pointer.

    ``caret_provider`` is injectable and optional; the AppKit default reads the
    focused element's selected-text bounds through the Accessibility API, which
    fails (and returns None) when the permission has not been granted.
    """
    if caret_provider is not None:
        try:
            rect = caret_provider()
        except Exception:
            rect = None
        if rect:
            x, y, w, h = rect
            return (float(x), float(y), float(w), float(h))
    if mouse_provider is None:
        return None
    point = mouse_provider()
    if point is None:
        return None
    x, y = point
    return (float(x), float(y), 0.0, 0.0)


# ---------------------------------------------------------------------------
# AppKit layer. Every AppKit import below is inside a function on purpose: this
# module has to import on a machine without PyObjC (Linux CI, plain unit tests).
# ---------------------------------------------------------------------------


#: Collection-behaviour flags the pill's panel needs, by AppKit constant name.
#:
#: ``FullScreenAuxiliary`` is the load-bearing one: without it the panel is not
#: allowed over a full-screen app, so the pill would vanish exactly when the
#: user is most focused — which is when they dictate. ``IgnoresCycle`` keeps a
#: non-activating helper out of Cmd-`. The first two keep it on whatever space
#: is in front and stop it scrolling away with one.
COLLECTION_BEHAVIOR_FLAGS = (
    "NSWindowCollectionBehaviorCanJoinAllSpaces",
    "NSWindowCollectionBehaviorStationary",
    "NSWindowCollectionBehaviorFullScreenAuxiliary",
    "NSWindowCollectionBehaviorIgnoresCycle",
)


def collection_behavior_mask(ns: object) -> int:
    """OR of :data:`COLLECTION_BEHAVIOR_FLAGS` read off ``ns`` (the AppKit module).

    Pure and injectable so the flags can be asserted without PyObjC. A constant
    this AppKit does not define is skipped rather than fatal: losing one
    behaviour is a worse-placed pill, losing the panel is no pill at all.
    """
    mask = 0
    for name in COLLECTION_BEHAVIOR_FLAGS:
        value = getattr(ns, name, None)
        if value is None:
            continue
        mask |= int(value)
    return mask


def run_on_main_thread(func: Callable[[], None]) -> None:
    """Hop to the main thread, the way ``murmur.py`` does. Safe from anywhere."""
    if threading.current_thread() is threading.main_thread():
        func()
        return
    from PyObjCTools import AppHelper

    AppHelper.callAfter(func)


def reduce_motion() -> bool:
    """True when the system asks for no animation — and when we cannot tell.

    The design manifest says no animation by default, so an unreadable setting
    resolves to "do not animate" rather than to a fade nobody asked for.
    """
    try:
        from AppKit import NSWorkspace

        return bool(NSWorkspace.sharedWorkspace().accessibilityDisplayShouldReduceMotion())
    except Exception:
        return True


def mouse_point() -> tuple[float, float] | None:
    """Pointer location in screen coordinates, or None without AppKit."""
    try:
        from AppKit import NSEvent

        point = NSEvent.mouseLocation()
        return (float(point.x), float(point.y))
    except Exception:
        return None


def caret_anchor_rect() -> tuple[float, float, float, float] | None:
    """Screen bounds of the insertion point in the focused text element.

    Returns None whenever Accessibility is not granted, the focused element is
    not a text area, or anything else goes sideways — the caller then falls back
    to the pointer.
    """
    try:
        from ApplicationServices import (
            AXUIElementCopyAttributeValue,
            AXUIElementCopyParameterizedAttributeValue,
            AXUIElementCreateSystemWide,
            AXValueGetValue,
            kAXBoundsForRangeParameterizedAttribute,
            kAXFocusedUIElementAttribute,
            kAXSelectedTextRangeAttribute,
            kAXValueCGRectType,
        )

        system = AXUIElementCreateSystemWide()
        err, focused = AXUIElementCopyAttributeValue(system, kAXFocusedUIElementAttribute, None)
        if err != 0 or focused is None:
            return None
        err, text_range = AXUIElementCopyAttributeValue(
            focused, kAXSelectedTextRangeAttribute, None
        )
        if err != 0 or text_range is None:
            return None
        err, bounds = AXUIElementCopyParameterizedAttributeValue(
            focused, kAXBoundsForRangeParameterizedAttribute, text_range, None
        )
        if err != 0 or bounds is None:
            return None
        ok, rect = AXValueGetValue(bounds, kAXValueCGRectType, None)
        if not ok or rect is None:
            return None
        # Accessibility reports a top-left origin; AppKit windows use bottom-left.
        from AppKit import NSScreen

        main = NSScreen.screens()[0].frame()
        flipped_y = float(main.size.height) - float(rect.origin.y) - float(rect.size.height)
        return (
            float(rect.origin.x),
            flipped_y,
            float(rect.size.width),
            float(rect.size.height),
        )
    except Exception:
        return None


def visible_frame_for(x: float, y: float) -> tuple[float, float, float, float]:
    """Visible frame of the screen holding ``(x, y)``, else the main screen's."""
    from AppKit import NSScreen

    for screen in NSScreen.screens():
        frame = screen.frame()
        if (
            frame.origin.x <= x <= frame.origin.x + frame.size.width
            and frame.origin.y <= y <= frame.origin.y + frame.size.height
        ):
            visible = screen.visibleFrame()
            break
    else:
        visible = NSScreen.mainScreen().visibleFrame()
    return (
        float(visible.origin.x),
        float(visible.origin.y),
        float(visible.size.width),
        float(visible.size.height),
    )


class PillWindow:
    """Borderless, non-activating floating panel that renders a :class:`PillState`.

    Every method must run on the main thread; :class:`PillPresenter` guarantees
    that. The panel takes no clicks and never becomes key, so it cannot steal
    focus from whatever the user is dictating into.
    """

    def __init__(
        self,
        size: tuple[float, float] = PILL_SIZE,
        caret_provider: Callable[[], tuple[float, float, float, float] | None] | None = None,
        mouse_provider: Callable[[], tuple[float, float] | None] | None = None,
    ) -> None:
        self._size = (float(size[0]), float(size[1]))
        self._caret_provider = caret_anchor_rect if caret_provider is None else caret_provider
        self._mouse_provider = mouse_point if mouse_provider is None else mouse_provider
        self._panel = None
        self._content = None
        self._glyph_field = None
        self._text_field = None

    # -- construction -------------------------------------------------------

    def _ensure_panel(self):
        if self._panel is None:
            self._build()
        return self._panel

    def _build(self) -> None:
        import AppKit
        from AppKit import (
            NSBackingStoreBuffered,
            NSColor,
            NSFloatingWindowLevel,
            NSFont,
            NSPanel,
            NSTextField,
            NSView,
            NSWindowStyleMaskBorderless,
            NSWindowStyleMaskNonactivatingPanel,
        )
        from Foundation import NSMakeRect

        import ui_theme

        width, height = self._size
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setOpaque_(False)
        panel.setHasShadow_(False)  # depth without drama, per the manifest
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setIgnoresMouseEvents_(True)
        panel.setReleasedWhenClosed_(False)
        panel.setHidesOnDeactivate_(False)
        panel.setCollectionBehavior_(collection_behavior_mask(AppKit))

        content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        content.setWantsLayer_(True)
        layer = content.layer()
        layer.setCornerRadius_(PILL_CORNER_RADIUS)
        layer.setMasksToBounds_(True)
        layer.setBackgroundColor_(ui_theme.card_background_color())
        layer.setBorderWidth_(1.0)
        layer.setBorderColor_(ui_theme.separator_color())
        panel.setContentView_(content)

        glyph_width = 16.0
        glyph = NSTextField.alloc().initWithFrame_(
            NSMakeRect(PILL_H_PADDING, (height - 18.0) / 2.0, glyph_width, 18.0)
        )
        text_x = PILL_H_PADDING + glyph_width + 6.0
        text = NSTextField.alloc().initWithFrame_(
            NSMakeRect(text_x, (height - 18.0) / 2.0, width - text_x - PILL_H_PADDING, 18.0)
        )
        for field in (glyph, text):
            field.setBezeled_(False)
            field.setDrawsBackground_(False)
            field.setEditable_(False)
            field.setSelectable_(False)
            field.setStringValue_("")
            content.addSubview_(field)
        glyph.setFont_(NSFont.systemFontOfSize_weight_(12.0, 0.3))
        glyph.setTextColor_(ui_theme.brand_accent_color())
        text.setFont_(NSFont.systemFontOfSize_(12.0))
        text.setTextColor_(ui_theme.primary_text_color())
        try:
            text.cell().setUsesSingleLineMode_(True)
        except Exception:
            pass

        self._panel = panel
        self._content = content
        self._glyph_field = glyph
        self._text_field = text

    # -- rendering ----------------------------------------------------------

    def show(self, state: PillState, anchor_rect=None) -> None:
        """Position the pill at the anchor and order it in without activating."""
        panel = self._ensure_panel()
        self._apply(state)
        self._move(panel, anchor_rect)
        panel.setAlphaValue_(1.0)
        panel.orderFrontRegardless()

    def update(self, state: PillState) -> None:
        """Refresh glyph, text and label in place. The pill does not move."""
        if self._panel is None:
            self.show(state)
            return
        self._apply(state)

    def hide(self) -> None:
        """Order out. Fades over 0.15 s unless the system asks for less motion."""
        panel = self._panel
        if panel is None:
            return
        if reduce_motion():
            panel.orderOut_(None)
            panel.setAlphaValue_(1.0)
            return

        from AppKit import NSAnimationContext

        def finished():
            panel.orderOut_(None)
            panel.setAlphaValue_(1.0)

        NSAnimationContext.beginGrouping()
        context = NSAnimationContext.currentContext()
        context.setDuration_(FADE_DURATION_S)
        context.setCompletionHandler_(finished)
        panel.animator().setAlphaValue_(0.0)
        NSAnimationContext.endGrouping()

    def close(self) -> None:
        if self._panel is not None:
            self._panel.orderOut_(None)
            self._panel.close()
        self._panel = None
        self._content = None
        self._glyph_field = None
        self._text_field = None

    # -- internals ----------------------------------------------------------

    def _apply(self, state: PillState) -> None:
        import ui_theme

        if self._glyph_field is not None:
            self._glyph_field.setStringValue_(state.glyph)
        if self._text_field is not None:
            self._text_field.setStringValue_(state.text)
            colour = (
                ui_theme.muted_text_color()
                if state.phase == PHASE_ERROR
                else ui_theme.primary_text_color()
            )
            self._text_field.setTextColor_(colour)
        self._apply_accessibility(state)

    def _apply_accessibility(self, state: PillState) -> None:
        """VoiceOver sees a static-text element labelled with the state only."""
        content = self._content
        if content is None:
            return
        label = _ACCESSIBILITY_LABELS.get(state.phase, APP_NAME)
        try:
            content.setAccessibilityLabel_(label)
            from AppKit import NSAccessibilityStaticTextRole

            content.setAccessibilityRole_(NSAccessibilityStaticTextRole)
        except Exception:
            pass

    def _move(self, panel, anchor_rect) -> None:
        from Foundation import NSMakePoint

        rect = anchor_rect
        if rect is None:
            rect = resolve_anchor(self._caret_provider, self._mouse_provider)
        if rect is None:
            panel.center()
            return
        screen = visible_frame_for(rect[0], rect[1])
        origin = place_pill(rect, self._size, screen)
        panel.setFrameOrigin_(NSMakePoint(*origin))

    def set_accessibility_label(self, label: str) -> None:
        """Override the VoiceOver label (used for the error phase's message)."""
        if self._content is None:
            return
        try:
            self._content.setAccessibilityLabel_(str(label))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API for the wiring step (Wave 2 E2f).
# ---------------------------------------------------------------------------


class PillPresenter:
    """Thread-safe façade over the controller and the panel.

    Every method may be called from the recording or transcription thread; the
    AppKit work is handed to the main thread with the app's usual hop. The
    window is only built once, and only on the main thread.
    """

    def __init__(
        self,
        window_factory: Callable[[], object] = PillWindow,
        *,
        controller: PillController | None = None,
        main_thread: Callable[[Callable[[], None]], None] | None = None,
        now: Callable[[], float] = time.monotonic,
        timer_factory: Callable[[float, Callable[[], None]], object] | None = None,
    ) -> None:
        assert callable(window_factory), "window_factory must be callable"
        self._window_factory = window_factory
        self._controller = controller if controller is not None else PillController(now=now)
        self._hop = run_on_main_thread if main_thread is None else main_thread
        self._timer_factory = threading.Timer if timer_factory is None else timer_factory
        self._now = now
        self._lock = threading.RLock()
        self._window = None
        self._timer = None
        self._shown = False

    # -- state ---------------------------------------------------------------

    @property
    def state(self) -> PillState:
        return self._controller.state

    @property
    def text(self) -> str:
        """Full text of the utterance in flight. Never write this to a log."""
        return self._controller.text

    def accessibility_label(self) -> str:
        return self._controller.accessibility_label()

    # -- the six calls the app makes -----------------------------------------

    def listening(self) -> PillState:
        self._cancel_timer()
        with self._lock:
            state = self._controller.on_listening()
        self._present(state)
        return state

    def partial(self, partial: Partial) -> PillState:
        with self._lock:
            state = self._controller.on_partial(partial)
        if state.phase == PHASE_DONE:
            self._schedule_hide()
        self._present(state)
        return state

    def working(self, label: str | None = None) -> PillState:
        self._cancel_timer()
        with self._lock:
            state = self._controller.on_working(label)
        self._present(state)
        return state

    def done(self, text_len: int = 0) -> PillState:
        self._cancel_timer()
        with self._lock:
            state = self._controller.on_done(text_len)
        self._schedule_hide()
        self._present(state)
        return state

    def error(self, message: object) -> PillState:
        self._cancel_timer()
        with self._lock:
            state = self._controller.on_error(message)
        self._schedule_hide()
        self._present(state)
        return state

    def hide(self) -> None:
        self._cancel_timer()
        with self._lock:
            self._controller.hide()
            self._shown = False
        self._hop(self._hide_window)

    def close(self) -> None:
        """Tear the panel down entirely (app quit)."""
        self._cancel_timer()

        def render():
            window, self._window = self._window, None
            if window is not None:
                window.close()

        with self._lock:
            self._shown = False
        self._hop(render)

    # -- streaming -----------------------------------------------------------

    def feed_stream(
        self,
        partials: Iterable[Partial] | Iterator[Partial],
        *,
        cancelled: threading.Event | None = None,
    ) -> str:
        """Pump ``engine.stream()`` output into the pill; return the full text.

        The returned text is the complete utterance, not the shortened line the
        pill displays. Engines without ``supports_streaming`` never reach here —
        they use :meth:`listening`, :meth:`working` and :meth:`done` and the pill
        shows state only.

        ``cancelled`` is how an *abandoned* decoder lets go. The app gives up on
        a stream that overruns its join timeout and moves on to the next
        utterance, but the generator keeps running — and every partial it pushes
        after that lands on a pill that now belongs to somebody else's words.
        Once the event is set this stops presenting anything at all, including
        the error state: the failure of a stream nobody is waiting for is not
        news the user needs mid-sentence.

        An engine that raises mid-stream while still current leaves the pill
        visible with no hide deadline of its own, so the failure is turned into
        the error phase — which drops the half-transcript and schedules the 3 s
        fade — before the exception carries on to the caller.
        """
        assert partials is not None, "partials is required"

        def abandoned() -> bool:
            return cancelled is not None and cancelled.is_set()

        text = ""
        try:
            for partial in partials:
                if abandoned():
                    break
                self.partial(partial)
                text = partial_buffer_text(partial)
        except Exception:
            if not abandoned():
                self.error("Transcription failed")
            raise
        return text

    # -- timing --------------------------------------------------------------

    def tick(self, now: float | None = None) -> bool:
        """Hide the pill when its deadline has passed. Returns True if it did."""
        with self._lock:
            hidden = self._controller.tick(now)
            if hidden:
                self._shown = False
        if hidden:
            self._hop(self._hide_window)
        return hidden

    def _schedule_hide(self) -> None:
        deadline = self._controller.hide_deadline
        if deadline is None:
            return
        delay = max(0.0, deadline - self._now())
        timer = self._timer_factory(delay, self.tick)
        try:
            timer.daemon = True
        except Exception:
            pass
        with self._lock:
            self._timer = timer
        timer.start()

    def _cancel_timer(self) -> None:
        with self._lock:
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    # -- rendering -----------------------------------------------------------

    def _present(self, state: PillState) -> None:
        with self._lock:
            first = not self._shown
            self._shown = True

        def render():
            window = self._ensure_window()
            if window is None:
                return
            if first:
                window.show(state)
            else:
                window.update(state)

        self._hop(render)

    def _hide_window(self) -> None:
        window = self._window
        if window is not None:
            window.hide()

    def _ensure_window(self):
        """Build the panel on first use. Only ever called inside the hop."""
        if self._window is None:
            self._window = self._window_factory()
        return self._window
