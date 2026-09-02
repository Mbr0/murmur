#!/usr/bin/env python3
"""Cleanup end to end: a real ``llama-server`` rewriting a filler sentence.

Needs the ``llama-server`` binary (``MURMUR_LLAMA_SERVER`` or
``vendor/llamacpp/llama-server``, built by ``scripts/tools/fetch_llama.sh``)
and the installed cleanup GGUF (about 2.1 GB). Both missing is the normal CI
outcome — the model costs more to download than the job is worth — and the
skip says so.

The input here is a fixed, invented filler sentence with no user content in
it, so it is safe to hold in the source; the *output* is never logged, and the
assertions read lengths and timings only.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _support import installed_model_path, llama_server_binary, log  # noqa: E402

from cleanup.llama_server import (  # noqa: E402
    CLEANUP_MODEL_ID,
    CLEANUP_MODEL_SPEC,
    CleanupClient,
    LlamaServer,
    timeout_for_text,
)
from cleanup.modes import render_system_prompt  # noqa: E402
from engines.model_store import CATALOG  # noqa: E402

#: Dictation with the artefacts cleanup exists to remove. Invented, not a real
#: transcript, so committing it leaks nothing.
FILLER_SENTENCE = (
    "so um I was thinking that uh we should probably you know send the notes "
    "to the team before the meeting starts"
)

#: A 2 GB GGUF loads slowly the first time the page cache is cold; the default
#: 60 s is a dictation-path budget, not a cold-start one.
STARTUP_TIMEOUT_S = 180.0

_BINARY = llama_server_binary()
_MODEL = installed_model_path(
    CLEANUP_MODEL_ID, catalog=CATALOG + (CLEANUP_MODEL_SPEC,)
)

if _BINARY is None:
    _SKIP_REASON = (
        "llama-server not built; run `bash scripts/tools/fetch_llama.sh` "
        "or set MURMUR_LLAMA_SERVER"
    )
elif _MODEL is None:
    _SKIP_REASON = (
        f"cleanup model {CLEANUP_MODEL_ID} is not installed in the Murmur model store"
    )
else:
    _SKIP_REASON = ""


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON)
class CleanupEndToEndTests(unittest.TestCase):
    """One server for the class, started in setUp and stopped in tearDown."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = LlamaServer(
            model_path=_MODEL, binary=_BINARY, startup_timeout_s=STARTUP_TIMEOUT_S
        )
        started_at = time.monotonic()
        cls.server.start()
        log(f"llama-server start: {time.monotonic() - started_at:.2f}s")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def test_message_mode_cleanup_runs_within_its_budget(self) -> None:
        system_prompt = render_system_prompt(
            "message", "neutral", "en", vocabulary=("Murmur",)
        )
        budget_s = timeout_for_text(FILLER_SENTENCE)

        result = CleanupClient(self.server).cleanup(FILLER_SENTENCE, system_prompt)

        # ``reason`` is written by CleanupClient and never contains transcript
        # text, so it is safe in an assertion message.
        self.assertIs(
            result.skipped,
            False,
            f"cleanup skipped instead of running: {result.reason}",
        )
        self.assertTrue(result.text.strip(), "cleanup returned an empty rewrite")
        log(
            f"cleanup message mode: {result.elapsed_s:.2f}s "
            f"(budget {budget_s:.1f}s), {len(FILLER_SENTENCE)} chars in, "
            f"{len(result.text)} chars out"
        )
        self.assertLess(
            result.elapsed_s,
            budget_s,
            f"cleanup took {result.elapsed_s:.2f}s, over its {budget_s:.1f}s budget",
        )

    def test_server_reports_itself_running(self) -> None:
        """A guard on the fixture: a dead child would make the test above lie."""
        self.assertTrue(self.server.is_running)
        self.assertIsNone(self.server.exit_code)


if __name__ == "__main__":
    unittest.main()
