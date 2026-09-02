"""Pure-state tests for the floating pill (ui/pill_window.py).

Everything here runs without AppKit: the state machine, the text shortener, the
placement maths and the presenter (driven with a fake window and a fake
main-thread hop). No test asserts on transcript text reaching a log.
"""

import unittest

from engines import Partial

from ui import pill_window
from ui.pill_window import (
    DONE_HIDE_AFTER_S,
    ERROR_HIDE_AFTER_S,
    MAX_TEXT_CHARS,
    PHASE_DONE,
    PHASE_ERROR,
    PHASE_IDLE,
    PHASE_LISTENING,
    PHASE_WORKING,
    PillController,
    PillPresenter,
    PillState,
    collection_behavior_mask,
    ellipsise_start,
    partial_buffer_text,
    place_pill,
    resolve_anchor,
)


class FakeClock:
    """Monotonic clock the tests advance by hand."""

    def __init__(self, start: float = 100.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> float:
        self.value += seconds
        return self.value


class FakeWindow:
    """Records what the presenter asks the window to do."""

    def __init__(self) -> None:
        self.calls = []

    def show(self, state, anchor_rect=None):
        self.calls.append(("show", state.phase, state.text))

    def update(self, state):
        self.calls.append(("update", state.phase, state.text))

    def hide(self):
        self.calls.append(("hide", None, None))

    def close(self):
        self.calls.append(("close", None, None))


class RecordingHop:
    """Main-thread hop that queues work instead of running it inline."""

    def __init__(self, auto=True) -> None:
        self.auto = auto
        self.pending = []
        self.count = 0

    def __call__(self, func):
        self.count += 1
        if self.auto:
            func()
        else:
            self.pending.append(func)

    def flush(self):
        pending, self.pending = self.pending, []
        for func in pending:
            func()


class FakeTimer:
    """Stand-in for threading.Timer that never touches a real thread."""

    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.function()


class FakeTimerFactory:
    def __init__(self):
        self.timers = []

    def __call__(self, interval, function):
        timer = FakeTimer(interval, function)
        self.timers.append(timer)
        return timer

    @property
    def last(self):
        return self.timers[-1] if self.timers else None


class EllipsiseTests(unittest.TestCase):
    def test_short_text_is_untouched(self):
        self.assertEqual(ellipsise_start("hello there"), "hello there")

    def test_text_is_collapsed_to_one_line(self):
        self.assertEqual(ellipsise_start("hello\n there\tworld  "), "hello there world")

    def test_long_text_keeps_the_newest_words(self):
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima"
        shortened = ellipsise_start(text)
        self.assertEqual(len(shortened), MAX_TEXT_CHARS)
        self.assertTrue(shortened.startswith("…"))
        self.assertTrue(text.endswith(shortened[1:]))

    def test_exactly_at_the_limit_is_not_ellipsised(self):
        text = "x" * MAX_TEXT_CHARS
        self.assertEqual(ellipsise_start(text), text)

    def test_one_over_the_limit_is_ellipsised(self):
        shortened = ellipsise_start("y" * (MAX_TEXT_CHARS + 1))
        self.assertEqual(shortened, "…" + "y" * (MAX_TEXT_CHARS - 1))

    def test_empty_and_none(self):
        self.assertEqual(ellipsise_start(""), "")
        self.assertEqual(ellipsise_start(None), "")


class PartialBufferTextTests(unittest.TestCase):
    """Partials are cumulative (see engines.base.Partial): the newest one wins.

    There is no delta-vs-cumulative guessing left: guessing appended a whole
    second copy of the utterance whenever a partial revised an earlier word.
    """

    def test_the_partial_text_is_the_whole_buffer(self):
        self.assertEqual(
            partial_buffer_text(Partial(text="hello there", is_final=False)), "hello there"
        )

    def test_the_text_is_collapsed_to_one_line(self):
        self.assertEqual(
            partial_buffer_text(Partial(text=" hello\n there ", is_final=False)),
            "hello there",
        )

    def test_an_empty_partial_gives_an_empty_buffer(self):
        self.assertEqual(partial_buffer_text(Partial(text="", is_final=True)), "")

    def test_a_partial_without_text_is_tolerated(self):
        self.assertEqual(partial_buffer_text(object()), "")


class PillControllerTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.controller = PillController(now=self.clock)

    def test_starts_idle_and_hidden(self):
        state = self.controller.state
        self.assertIsInstance(state, PillState)
        self.assertEqual(state.phase, PHASE_IDLE)
        self.assertEqual(state.text, "")
        self.assertFalse(state.visible)
        self.assertIsNone(self.controller.hide_deadline)

    def test_listening_shows_an_empty_pill(self):
        state = self.controller.on_listening()
        self.assertEqual(state.phase, PHASE_LISTENING)
        self.assertEqual(state.text, "")
        self.assertTrue(state.visible)
        self.assertIsNone(self.controller.hide_deadline)

    def test_partials_accumulate_while_listening(self):
        self.controller.on_listening()
        self.controller.on_partial(Partial(text="hello", is_final=False))
        state = self.controller.on_partial(Partial(text="hello there", is_final=False))
        self.assertEqual(state.phase, PHASE_LISTENING)
        self.assertEqual(state.text, "hello there")
        self.assertEqual(self.controller.text, "hello there")

    def test_a_correction_replaces_the_buffer_instead_of_doubling_it(self):
        # The decoder revised a word it had already emitted. The new partial is
        # still the whole utterance, so it replaces; appending would show the
        # sentence twice.
        self.controller.on_listening()
        self.controller.on_partial(Partial(text="the quick brown", is_final=False))
        state = self.controller.on_partial(Partial(text="the quick brawn fox", is_final=False))
        self.assertEqual(self.controller.text, "the quick brawn fox")
        self.assertEqual(state.text, "the quick brawn fox")

    def test_a_repeated_word_survives_the_next_partial(self):
        self.controller.on_listening()
        self.controller.on_partial(Partial(text="the quick", is_final=False))
        self.controller.on_partial(Partial(text="quick the quick", is_final=False))
        self.assertEqual(self.controller.text, "quick the quick")

    def test_a_shorter_correction_shrinks_the_buffer(self):
        self.controller.on_listening()
        self.controller.on_partial(Partial(text="hello there world", is_final=False))
        state = self.controller.on_partial(Partial(text="hello there", is_final=False))
        self.assertEqual(self.controller.text, "hello there")
        self.assertEqual(state.text, "hello there")

    def test_partial_display_text_is_ellipsised_but_buffer_is_not(self):
        self.controller.on_listening()
        long_text = "word " * 40
        state = self.controller.on_partial(Partial(text=long_text, is_final=False))
        self.assertEqual(len(state.text), MAX_TEXT_CHARS)
        self.assertGreater(len(self.controller.text), MAX_TEXT_CHARS)

    def test_final_partial_moves_to_done_and_schedules_the_hide(self):
        self.controller.on_listening()
        self.controller.on_partial(Partial(text="hello", is_final=False))
        state = self.controller.on_partial(Partial(text="hello there", is_final=True))
        self.assertEqual(state.phase, PHASE_DONE)
        self.assertTrue(state.visible)
        self.assertEqual(self.controller.text, "hello there")
        self.assertAlmostEqual(
            self.controller.hide_deadline, self.clock.value + DONE_HIDE_AFTER_S
        )

    def test_working_keeps_the_text_and_clears_any_deadline(self):
        self.controller.on_listening()
        self.controller.on_partial(Partial(text="hello", is_final=True))
        state = self.controller.on_working()
        self.assertEqual(state.phase, PHASE_WORKING)
        self.assertEqual(state.text, "hello")
        self.assertTrue(state.visible)
        self.assertIsNone(self.controller.hide_deadline)

    def test_state_only_flow_for_engines_without_streaming(self):
        # whisper.cpp never calls on_partial: listening -> working -> done.
        self.assertEqual(self.controller.on_listening().text, "")
        self.assertEqual(self.controller.on_working().text, "")
        state = self.controller.on_done(42)
        self.assertEqual(state.phase, PHASE_DONE)
        self.assertEqual(state.text, "")
        self.assertTrue(state.visible)
        self.assertEqual(self.controller.final_text_len, 42)

    def test_done_rejects_a_negative_length(self):
        with self.assertRaises(AssertionError):
            self.controller.on_done(-1)

    def test_error_shows_the_message_and_hides_later(self):
        state = self.controller.on_error("Microphone unavailable")
        self.assertEqual(state.phase, PHASE_ERROR)
        self.assertEqual(state.text, "Microphone unavailable")
        self.assertTrue(state.visible)
        self.assertAlmostEqual(
            self.controller.hide_deadline, self.clock.value + ERROR_HIDE_AFTER_S
        )

    def test_error_clears_any_buffered_transcript(self):
        self.controller.on_listening()
        self.controller.on_partial(Partial(text="secret words", is_final=False))
        self.controller.on_error("boom")
        self.assertEqual(self.controller.text, "")

    def test_listening_clears_the_previous_transcript(self):
        self.controller.on_listening()
        self.controller.on_partial(Partial(text="first run", is_final=True))
        state = self.controller.on_listening()
        self.assertEqual(self.controller.text, "")
        self.assertEqual(state.text, "")


class PillControllerTickTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.controller = PillController(now=self.clock)

    def test_tick_does_not_hide_before_the_done_deadline(self):
        self.controller.on_listening()
        self.controller.on_done(5)
        self.clock.advance(DONE_HIDE_AFTER_S - 0.01)
        self.assertFalse(self.controller.tick(self.clock.value))
        self.assertTrue(self.controller.state.visible)

    def test_tick_hides_once_at_the_done_deadline(self):
        self.controller.on_listening()
        self.controller.on_done(5)
        self.clock.advance(DONE_HIDE_AFTER_S)
        self.assertTrue(self.controller.tick(self.clock.value))
        state = self.controller.state
        self.assertFalse(state.visible)
        self.assertEqual(state.phase, PHASE_IDLE)
        self.assertEqual(state.text, "")
        # A second tick reports nothing more to do.
        self.assertFalse(self.controller.tick(self.clock.value))

    def test_error_hides_after_three_seconds_not_after_one(self):
        self.controller.on_error("boom")
        self.clock.advance(DONE_HIDE_AFTER_S)
        self.assertFalse(self.controller.tick(self.clock.value))
        self.clock.advance(ERROR_HIDE_AFTER_S - DONE_HIDE_AFTER_S)
        self.assertTrue(self.controller.tick(self.clock.value))
        self.assertFalse(self.controller.state.visible)

    def test_tick_uses_the_injected_clock_when_no_time_is_given(self):
        self.controller.on_done(1)
        self.clock.advance(DONE_HIDE_AFTER_S)
        self.assertTrue(self.controller.tick())

    def test_tick_is_a_no_op_while_listening(self):
        self.controller.on_listening()
        self.clock.advance(600)
        self.assertFalse(self.controller.tick(self.clock.value))
        self.assertTrue(self.controller.state.visible)


class AccessibilityLabelTests(unittest.TestCase):
    def setUp(self):
        self.controller = PillController(now=FakeClock())

    def test_labels_name_the_app_and_the_state(self):
        self.controller.on_listening()
        self.assertEqual(self.controller.accessibility_label(), "Murmur listening")
        self.controller.on_working()
        self.assertEqual(self.controller.accessibility_label(), "Murmur working")
        self.controller.on_done(3)
        self.assertEqual(self.controller.accessibility_label(), "Murmur done")

    def test_idle_label(self):
        self.assertEqual(self.controller.accessibility_label(), "Murmur")

    def test_error_label_carries_the_message(self):
        self.controller.on_error("Microphone unavailable")
        self.assertEqual(
            self.controller.accessibility_label(), "Murmur error: Microphone unavailable"
        )

    def test_label_never_leaks_the_transcript(self):
        self.controller.on_listening()
        self.controller.on_partial(Partial(text="my private sentence", is_final=False))
        self.assertEqual(self.controller.accessibility_label(), "Murmur listening")


SCREEN = (0.0, 0.0, 1440.0, 900.0)
PILL = (260.0, 34.0)


class PlacePillTests(unittest.TestCase):
    def test_sits_below_the_caret_centred_on_it(self):
        origin = place_pill((700.0, 500.0, 2.0, 20.0), PILL, SCREEN)
        self.assertAlmostEqual(origin[0], 701.0 - 130.0)
        self.assertAlmostEqual(origin[1], 500.0 - pill_window.ANCHOR_GAP - 34.0)

    def test_flips_above_the_caret_near_the_bottom_of_the_screen(self):
        origin = place_pill((700.0, 5.0, 2.0, 20.0), PILL, SCREEN)
        self.assertAlmostEqual(origin[1], 5.0 + 20.0 + pill_window.ANCHOR_GAP)

    def test_clamped_to_the_right_edge(self):
        origin = place_pill((1435.0, 500.0, 2.0, 20.0), PILL, SCREEN)
        self.assertAlmostEqual(origin[0], 1440.0 - 260.0)

    def test_clamped_to_the_left_edge(self):
        origin = place_pill((2.0, 500.0, 2.0, 20.0), PILL, SCREEN)
        self.assertAlmostEqual(origin[0], 0.0)

    def test_clamped_to_the_top_edge(self):
        origin = place_pill((700.0, 899.0, 2.0, 20.0), PILL, SCREEN)
        self.assertLessEqual(origin[1] + PILL[1], 900.0)

    def test_respects_a_screen_frame_with_a_non_zero_origin(self):
        screen = (1440.0, 25.0, 1440.0, 875.0)
        origin = place_pill((1441.0, 30.0, 2.0, 20.0), PILL, screen)
        self.assertGreaterEqual(origin[0], 1440.0)
        self.assertGreaterEqual(origin[1], 25.0)
        self.assertLessEqual(origin[0] + PILL[0], 1440.0 + 1440.0)
        self.assertLessEqual(origin[1] + PILL[1], 25.0 + 875.0)

    def test_a_pill_wider_than_the_screen_pins_to_the_left(self):
        origin = place_pill((100.0, 100.0, 2.0, 20.0), (2000.0, 34.0), SCREEN)
        self.assertAlmostEqual(origin[0], 0.0)


class ResolveAnchorTests(unittest.TestCase):
    def test_prefers_the_caret_rect_when_the_provider_has_one(self):
        rect = resolve_anchor(
            caret_provider=lambda: (10.0, 20.0, 2.0, 18.0),
            mouse_provider=lambda: (900.0, 700.0),
        )
        self.assertEqual(rect, (10.0, 20.0, 2.0, 18.0))

    def test_falls_back_to_the_mouse_when_there_is_no_caret(self):
        rect = resolve_anchor(caret_provider=lambda: None, mouse_provider=lambda: (900.0, 700.0))
        self.assertEqual(rect, (900.0, 700.0, 0.0, 0.0))

    def test_falls_back_to_the_mouse_when_the_caret_provider_raises(self):
        def boom():
            raise RuntimeError("accessibility not granted")

        rect = resolve_anchor(caret_provider=boom, mouse_provider=lambda: (5.0, 6.0))
        self.assertEqual(rect, (5.0, 6.0, 0.0, 0.0))

    def test_no_provider_at_all_returns_none(self):
        self.assertIsNone(resolve_anchor(caret_provider=None, mouse_provider=None))


class PillPresenterTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.window = FakeWindow()
        self.hop = RecordingHop()
        self.timers = FakeTimerFactory()
        self.presenter = PillPresenter(
            window_factory=lambda: self.window,
            main_thread=self.hop,
            now=self.clock,
            timer_factory=self.timers,
        )

    def test_listening_shows_the_window_through_the_main_thread_hop(self):
        self.presenter.listening()
        self.assertEqual(self.hop.count, 1)
        self.assertEqual(self.window.calls, [("show", PHASE_LISTENING, "")])

    def test_nothing_touches_the_window_until_the_hop_runs(self):
        deferred = RecordingHop(auto=False)
        presenter = PillPresenter(
            window_factory=lambda: self.window,
            main_thread=deferred,
            now=self.clock,
            timer_factory=self.timers,
        )
        presenter.listening()
        presenter.partial(Partial(text="hello", is_final=False))
        self.assertEqual(self.window.calls, [])
        self.assertEqual(deferred.count, 2)
        deferred.flush()
        self.assertEqual(
            self.window.calls,
            [("show", PHASE_LISTENING, ""), ("update", PHASE_LISTENING, "hello")],
        )

    def test_every_public_call_hops_to_the_main_thread(self):
        self.presenter.listening()
        self.presenter.partial(Partial(text="hi", is_final=False))
        self.presenter.working()
        self.presenter.done(2)
        self.presenter.error("boom")
        self.presenter.hide()
        self.assertEqual(self.hop.count, 6)

    def test_done_then_error_then_hide_reach_the_window(self):
        self.presenter.listening()
        self.presenter.working()
        self.presenter.done(4)
        self.presenter.hide()
        phases = [call[1] for call in self.window.calls]
        self.assertEqual(phases, [PHASE_LISTENING, PHASE_WORKING, PHASE_DONE, None])
        self.assertEqual(self.window.calls[-1][0], "hide")

    def test_done_schedules_a_timer_that_hides_the_pill(self):
        self.presenter.listening()
        self.presenter.done(4)
        timer = self.timers.last
        self.assertIsNotNone(timer)
        self.assertTrue(timer.started)
        self.assertAlmostEqual(timer.interval, DONE_HIDE_AFTER_S)
        self.clock.advance(DONE_HIDE_AFTER_S)
        timer.fire()
        self.assertEqual(self.window.calls[-1][0], "hide")
        self.assertFalse(self.presenter.state.visible)

    def test_error_schedules_a_three_second_timer(self):
        self.presenter.error("no microphone")
        self.assertAlmostEqual(self.timers.last.interval, ERROR_HIDE_AFTER_S)
        self.assertEqual(self.window.calls[-1], ("show", PHASE_ERROR, "no microphone"))

    def test_a_new_utterance_cancels_the_pending_hide(self):
        self.presenter.done(4)
        pending = self.timers.last
        self.presenter.listening()
        self.assertTrue(pending.cancelled)

    def test_feed_stream_pumps_partials_and_returns_the_final_text(self):
        partials = [
            Partial(text="hello", is_final=False),
            Partial(text="hello there", is_final=False),
            Partial(text="hello there world", is_final=True),
        ]
        self.presenter.listening()
        final_text = self.presenter.feed_stream(iter(partials))
        self.assertEqual(final_text, "hello there world")
        self.assertEqual(self.window.calls[-1][1], PHASE_DONE)
        self.assertEqual(self.window.calls[-1][2], "hello there world")

    def test_feed_stream_on_an_empty_stream_returns_an_empty_string(self):
        self.assertEqual(self.presenter.feed_stream(iter([])), "")

    def test_feed_stream_returns_the_last_partial_not_a_concatenation(self):
        partials = [
            Partial(text="the quick brown", is_final=False),
            Partial(text="the quick brawn fox", is_final=True),
        ]
        self.presenter.listening()
        self.assertEqual(self.presenter.feed_stream(iter(partials)), "the quick brawn fox")

    def test_feed_stream_shows_an_error_and_re_raises_when_the_engine_fails(self):
        # An engine that dies mid-stream must not leave the pill sitting in
        # `listening` with half an utterance and no hide timer.
        def dying_stream():
            yield Partial(text="hello", is_final=False)
            raise RuntimeError("engine died")

        self.presenter.listening()
        with self.assertRaises(RuntimeError):
            self.presenter.feed_stream(dying_stream())

        self.assertEqual(self.presenter.state.phase, PHASE_ERROR)
        self.assertTrue(self.presenter.state.visible)
        self.assertEqual(self.window.calls[-1][1], PHASE_ERROR)
        self.assertAlmostEqual(self.timers.last.interval, ERROR_HIDE_AFTER_S)
        self.assertTrue(self.timers.last.started)

    def test_feed_stream_drops_the_partial_transcript_when_the_engine_fails(self):
        def dying_stream():
            yield Partial(text="my private sentence", is_final=False)
            raise RuntimeError("engine died")

        self.presenter.listening()
        with self.assertRaises(RuntimeError):
            self.presenter.feed_stream(dying_stream())

        self.assertEqual(self.presenter.text, "")
        self.assertNotIn("private", self.presenter.state.text)

    def test_accessibility_label_is_exposed_for_the_window(self):
        self.presenter.listening()
        self.assertEqual(self.presenter.accessibility_label(), "Murmur listening")

    def test_window_is_created_once_and_reused(self):
        made = []

        def factory():
            window = FakeWindow()
            made.append(window)
            return window

        presenter = PillPresenter(
            window_factory=factory,
            main_thread=self.hop,
            now=self.clock,
            timer_factory=self.timers,
        )
        presenter.listening()
        presenter.working()
        self.assertEqual(len(made), 1)


#: The four AppKit flag values the pill reads, with their real bit positions.
CAN_JOIN_ALL_SPACES = 1 << 0
STATIONARY = 1 << 4
IGNORES_CYCLE = 1 << 6
FULL_SCREEN_AUXILIARY = 1 << 8


class FakeAppKit:
    """Stands in for the AppKit module: just the constants the helper reads."""

    NSWindowCollectionBehaviorCanJoinAllSpaces = CAN_JOIN_ALL_SPACES
    NSWindowCollectionBehaviorStationary = STATIONARY
    NSWindowCollectionBehaviorIgnoresCycle = IGNORES_CYCLE
    NSWindowCollectionBehaviorFullScreenAuxiliary = FULL_SCREEN_AUXILIARY


class CollectionBehaviorTests(unittest.TestCase):
    def test_the_pill_is_visible_over_a_full_screen_app(self):
        # Without FullScreenAuxiliary the panel is hidden exactly when the user
        # is most focused, which is when they dictate.
        mask = collection_behavior_mask(FakeAppKit)
        self.assertTrue(mask & FULL_SCREEN_AUXILIARY)

    def test_the_pill_follows_every_space_and_does_not_scroll_with_one(self):
        mask = collection_behavior_mask(FakeAppKit)
        self.assertTrue(mask & CAN_JOIN_ALL_SPACES)
        self.assertTrue(mask & STATIONARY)

    def test_the_pill_stays_out_of_the_window_cycle(self):
        self.assertTrue(collection_behavior_mask(FakeAppKit) & IGNORES_CYCLE)

    def test_the_mask_is_exactly_those_four_flags(self):
        self.assertEqual(
            collection_behavior_mask(FakeAppKit),
            CAN_JOIN_ALL_SPACES | STATIONARY | IGNORES_CYCLE | FULL_SCREEN_AUXILIARY,
        )

    def test_a_flag_this_appkit_does_not_define_is_skipped_not_fatal(self):
        class OlderAppKit:
            NSWindowCollectionBehaviorCanJoinAllSpaces = CAN_JOIN_ALL_SPACES
            NSWindowCollectionBehaviorStationary = STATIONARY

        self.assertEqual(
            collection_behavior_mask(OlderAppKit), CAN_JOIN_ALL_SPACES | STATIONARY
        )


class LazyAppKitImportTests(unittest.TestCase):
    def test_module_does_not_bind_appkit_symbols_at_import_time(self):
        # The module must import on a machine without PyObjC (Linux CI).
        for symbol in ("NSPanel", "NSTextField", "NSWorkspace", "AppHelper"):
            self.assertNotIn(symbol, vars(pill_window))


if __name__ == "__main__":
    unittest.main()
