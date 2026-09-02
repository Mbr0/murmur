#!/usr/bin/env python3
"""whisper.cpp end to end: a real ``whisper-server`` over real audio.

Needs three things, and skips with an explicit reason when any is absent:

* the ``whisper-server`` binary (``MURMUR_WHISPER_SERVER`` or
  ``vendor/whispercpp/whisper-server``, built by
  ``scripts/tools/fetch_whispercpp.sh``),
* the installed ``whispercpp-large-v3-turbo-q5_0`` model, located through
  :class:`engines.model_store.ModelStore`,
* one clip per language, either recorded in ``tests/fixtures/audio`` or
  generated here with macOS ``say`` — hence the macOS requirement.

Transcript text is never printed; only latencies and word error rates are.
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _support import (  # noqa: E402
    FIXTURE_LANGUAGES,
    clips_for,
    installed_model_path,
    is_macos,
    log,
    wav_duration_s,
    whisper_server_binary,
    word_error_rate,
)

from engines.base import Hints  # noqa: E402
from engines.whispercpp import WhisperCppEngine  # noqa: E402

#: The quantised turbo model, the one CI can afford to download (574 MB).
MODEL_ID = "whispercpp-large-v3-turbo-q5_0"

#: Above this, the transcript is not a recognisable rendering of the sentence.
#: Loose on purpose: this suite proves the pipeline works end to end, it does
#: not re-litigate decision D1 — that is the bake-off harness's job.
MAX_WER = 0.5

#: MEASURED, not assumed (whisper.cpp v1.7.5 on 2026-09-02): ``verbose_json``
#: reports the language as its English *name* — "english", "french", "dutch",
#: "german" — not the ISO code the request carries. So ``Transcript.language``
#: from this engine is not comparable to the code the caller asked for, nor to
#: what ``engines.voxtral_mlx`` returns, which is the code. This suite accepts
#: either form rather than pretend one of them does not happen.
LANGUAGE_NAMES = {"en": "english", "fr": "french", "nl": "dutch", "de": "german"}


def _identifies(reported: str | None, language: str) -> bool:
    """True when ``reported`` names ``language``, as a code or as a name."""
    if not reported:
        return False
    return reported.strip().casefold() in {language, LANGUAGE_NAMES.get(language, "")}

_BINARY = whisper_server_binary()
_MODEL = installed_model_path(MODEL_ID)

if not is_macos():
    _SKIP_REASON = "needs macOS: the fixture clips are made with `say`/`afconvert`"
elif _BINARY is None:
    _SKIP_REASON = (
        "whisper-server not built; run `bash scripts/tools/fetch_whispercpp.sh` "
        "or set MURMUR_WHISPER_SERVER"
    )
elif _MODEL is None:
    _SKIP_REASON = f"model {MODEL_ID} is not installed in the Murmur model store"
else:
    _SKIP_REASON = ""


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON)
class WhisperCppEndToEndTests(unittest.TestCase):
    """One loaded server for the whole class; every test reuses it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="murmur-integration-whisper-")
        cls.clips = clips_for(FIXTURE_LANGUAGES, cls._tmp.name)
        if not cls.clips:
            cls._tmp.cleanup()
            raise unittest.SkipTest(
                "no fixture clips: none recorded and `say` produced none "
                "(no voice installed for any of "
                f"{', '.join(FIXTURE_LANGUAGES)})"
            )
        cls.engine = WhisperCppEngine(model_path=_MODEL, binary=_BINARY)
        started_at = time.monotonic()
        cls.engine.load()
        log(f"whispercpp load: {time.monotonic() - started_at:.2f}s")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.unload()
        cls._tmp.cleanup()

    def test_transcribes_every_available_language(self) -> None:
        """Non-empty text, the language back, and WER under the threshold."""
        for clip in self.clips:
            with self.subTest(language=clip.language):
                started_at = time.monotonic()
                transcript = self.engine.transcribe(clip.path, language=clip.language)
                elapsed_s = time.monotonic() - started_at

                self.assertTrue(
                    transcript.text.strip(),
                    f"{clip.language}: whisper-server returned an empty transcript",
                )
                self.assertEqual(transcript.engine_id, "whispercpp")
                self.assertTrue(
                    _identifies(transcript.language, clip.language),
                    f"{clip.language}: server reported language "
                    f"{transcript.language!r}, which names neither "
                    f"{clip.language!r} nor {LANGUAGE_NAMES[clip.language]!r}",
                )

                wer = word_error_rate(clip.text, transcript.text)
                # Latency is logged, never asserted: it is a property of the
                # machine, and a slow CI runner is not a broken pipeline.
                log(
                    f"whispercpp {clip.language}: {elapsed_s:.2f}s for "
                    f"{wav_duration_s(clip.path):.1f}s of audio, wer {wer:.2f}"
                )
                self.assertLess(
                    wer,
                    MAX_WER,
                    f"{clip.language}: word error rate {wer:.2f} is at or above "
                    f"{MAX_WER}",
                )

    def test_language_is_detected_when_none_is_requested(self) -> None:
        """With no ``language``, the server detects one and reports it."""
        clip = self.clips[0]
        transcript = self.engine.transcribe(clip.path)

        self.assertTrue(transcript.text.strip(), "auto-detect returned no text")
        self.assertTrue(
            transcript.language,
            "auto-detect returned no language at all",
        )
        log(f"whispercpp auto-detect on {clip.language}: {transcript.language}")

    def test_hints_are_applied(self) -> None:
        """A prompt built from hints reaches the decoder, and it says so."""
        clip = self.clips[0]
        hints = Hints(vocabulary=("Murmur", "Canopy Studio"))

        transcript = self.engine.transcribe(
            clip.path, language=clip.language, hints=hints
        )

        self.assertIs(
            transcript.hints_applied,
            True,
            "whisper.cpp advertises hints but did not report applying them",
        )
        self.assertTrue(transcript.text.strip(), "hinted pass returned no text")


if __name__ == "__main__":
    unittest.main()
