#!/usr/bin/env python3
"""The paste path end to end, with a fake pasteboard and a fake event poster.

This is the one integration test with no model and no binary behind it, and
the reason it is here rather than in ``tests/`` is that it exercises the whole
of :meth:`services.text_insertion_service.TextInsertionService.paste_text` —
real sleeps, real ordering, real ``finally`` — against stand-ins for the only
two things that must not run for real: the system pasteboard (it belongs to
the user) and the Cmd+V event (it would land in whatever window is focused,
including the terminal running the suite).

``services.text_insertion_service`` imports Quartz at module scope, so this
skips off macOS.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _support import is_macos, log  # noqa: E402

_SKIP_REASON = "" if is_macos() else "needs macOS: the service imports Quartz"

if is_macos():
    from services.text_insertion_service import (  # noqa: E402
        CLIPBOARD_RESTORE_DELAY_S,
        PASTEBOARD_SETTLE_DELAY_S,
        TextInsertionService,
    )

#: Stand-in for a transcript. Invented, and never printed by this module.
TRANSCRIPT = "Send the release notes to the team before the meeting."

#: What the user had on their clipboard before dictating.
PREVIOUS_CLIPBOARD = "https://example.invalid/whatever-they-were-copying"


class FakePasteboard:
    """A pasteboard that lives in a variable instead of in the window server.

    ``fail_on_write`` is 1-based: 1 fails the transcript write, 2 fails the
    restore. A failed write is recorded but does not change the contents,
    which is how the real one behaves when the pasteboard is held by another
    process.
    """

    def __init__(self, initial: str = "", fail_on_write: int | None = None) -> None:
        self.value = initial
        self.writes: list[str] = []
        self._fail_on_write = fail_on_write

    def read(self) -> str:
        return self.value

    def write(self, text: str) -> None:
        self.writes.append(text)
        if self._fail_on_write == len(self.writes):
            raise RuntimeError("pasteboard write failed")
        self.value = text


class FakeEventPoster:
    """Records each Cmd+V and what the pasteboard held at that instant."""

    def __init__(self, pasteboard: FakePasteboard, error: Exception | None = None) -> None:
        self._pasteboard = pasteboard
        self._error = error
        self.calls = 0
        self.pasteboard_at_paste: list[str] = []

    def __call__(self) -> None:
        self.calls += 1
        self.pasteboard_at_paste.append(self._pasteboard.read())
        if self._error is not None:
            raise self._error


class _Recorder:
    """The smallest logger the service needs, without pulling in ``logging``."""

    def __init__(self) -> None:
        self.info_calls: list[tuple] = []
        self.warning_calls: list[tuple] = []

    def info(self, message, *args) -> None:
        self.info_calls.append((message, args))

    def warning(self, message, *args) -> None:
        self.warning_calls.append((message, args))


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON)
class PasteFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pasteboard = FakePasteboard(PREVIOUS_CLIPBOARD)
        self.poster = FakeEventPoster(self.pasteboard)
        self.logger = _Recorder()

    def _service(self) -> "TextInsertionService":
        return TextInsertionService(
            logger=self.logger,
            pasteboard=self.pasteboard,
            post_keystroke=self.poster,
        )

    def test_transcript_is_on_the_pasteboard_when_the_keystroke_is_posted(self) -> None:
        started_at = time.monotonic()
        self._service().paste_text(TRANSCRIPT)
        elapsed_s = time.monotonic() - started_at

        self.assertEqual(self.poster.calls, 1, "Cmd+V was not posted exactly once")
        self.assertEqual(
            self.poster.pasteboard_at_paste,
            [TRANSCRIPT],
            "the keystroke was posted before the transcript reached the pasteboard",
        )
        log(f"paste flow: {elapsed_s:.2f}s of real delays per paste")
        # The two sleeps are the whole reason the path is slow; assert they are
        # actually being waited out rather than optimised away by a stub.
        self.assertGreaterEqual(
            elapsed_s, PASTEBOARD_SETTLE_DELAY_S + CLIPBOARD_RESTORE_DELAY_S - 0.05
        )

    def test_previous_clipboard_is_restored_after_the_paste(self) -> None:
        self._service().paste_text(TRANSCRIPT)

        self.assertEqual(
            self.pasteboard.writes,
            [TRANSCRIPT, PREVIOUS_CLIPBOARD],
            "the pasteboard was not written transcript-then-restore",
        )
        self.assertEqual(
            self.pasteboard.read(),
            PREVIOUS_CLIPBOARD,
            "the user's clipboard was left clobbered",
        )
        self.assertEqual(len(self.logger.info_calls), 1)
        self.assertEqual(self.logger.warning_calls, [])

    def test_clipboard_is_restored_even_when_the_keystroke_fails(self) -> None:
        self.poster = FakeEventPoster(self.pasteboard, RuntimeError("CGEventPost failed"))

        with self.assertRaises(RuntimeError):
            self._service().paste_text(TRANSCRIPT)

        self.assertEqual(self.pasteboard.writes, [TRANSCRIPT, PREVIOUS_CLIPBOARD])
        self.assertEqual(self.pasteboard.read(), PREVIOUS_CLIPBOARD)

    def test_failed_restore_leaves_the_transcript_and_warns(self) -> None:
        # Fail the *second* write, the restore: the first one must succeed or
        # there is no transcript to paste in the first place.
        self.pasteboard = FakePasteboard(PREVIOUS_CLIPBOARD, fail_on_write=2)
        self.poster = FakeEventPoster(self.pasteboard)

        self._service().paste_text(TRANSCRIPT)

        self.assertEqual(self.pasteboard.writes, [TRANSCRIPT, PREVIOUS_CLIPBOARD])
        self.assertEqual(
            self.pasteboard.read(),
            TRANSCRIPT,
            "a failed restore must leave the transcript on the clipboard",
        )
        self.assertEqual(len(self.logger.warning_calls), 1)


if __name__ == "__main__":
    unittest.main()
