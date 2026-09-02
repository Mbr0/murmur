#!/usr/bin/env python3
"""Voxtral Mini 4B Realtime end to end: real MLX weights, batch and streaming.

Needs an Apple Silicon Mac, an importable ``mlx_audio`` and the installed
``voxtral-mini-4b-realtime-4bit`` model (about 3.1 GB). Any of those missing is
a skip with a reason, which is the normal outcome on CI: decision D7 already
says MLX is Apple Silicon only, and a 3 GB download per run is more than a
hosted runner should spend to learn what a local run already proves.

Transcript text is never printed; only latencies, partial counts and WER are.
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
    clips_for,
    installed_model_path,
    is_apple_silicon,
    log,
    module_available,
    pcm_chunks,
    wav_duration_s,
    word_error_rate,
)

from engines.voxtral_mlx import VoxtralMlxEngine  # noqa: E402

MODEL_ID = "voxtral-mini-4b-realtime-4bit"

#: Same loose threshold as the whisper.cpp suite, for the same reason.
MAX_WER = 0.5

#: Chunk size fed to the streaming session. 320 ms is four of the model's
#: 80 ms audio tokens, close to what a live capture callback delivers.
CHUNK_MS = 320

#: The clip both halves of this suite use. One language keeps the wall clock
#: honest: a 4B model loads once and decodes twice here as it is.
LANGUAGE = "en"

_MODEL = installed_model_path(MODEL_ID)

if not is_apple_silicon():
    _SKIP_REASON = "needs Apple Silicon: MLX has no wheels elsewhere (decision D7)"
elif not module_available("mlx_audio"):
    _SKIP_REASON = "mlx_audio is not importable; install it with `pip install -r requirements.txt`"
elif _MODEL is None:
    _SKIP_REASON = f"model {MODEL_ID} is not installed in the Murmur model store"
else:
    _SKIP_REASON = ""


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON)
class VoxtralEndToEndTests(unittest.TestCase):
    """One loaded model for the whole class; batch and streaming share it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="murmur-integration-voxtral-")
        clips = clips_for((LANGUAGE,), cls._tmp.name)
        if not clips:
            cls._tmp.cleanup()
            raise unittest.SkipTest(
                f"no {LANGUAGE} fixture clip: none recorded and `say` produced none"
            )
        cls.clip = clips[0]
        cls.engine = VoxtralMlxEngine(model_path=_MODEL)
        started_at = time.monotonic()
        cls.engine.load()
        log(f"voxtral load: {time.monotonic() - started_at:.2f}s")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.unload()
        cls._tmp.cleanup()

    def test_batch_transcription_matches_the_spoken_sentence(self) -> None:
        started_at = time.monotonic()
        transcript = self.engine.transcribe(self.clip.path, language=LANGUAGE)
        elapsed_s = time.monotonic() - started_at

        self.assertTrue(transcript.text.strip(), "Voxtral returned an empty transcript")
        self.assertEqual(transcript.engine_id, "voxtral_mlx")

        wer = word_error_rate(self.clip.text, transcript.text)
        log(
            f"voxtral batch {LANGUAGE}: {elapsed_s:.2f}s for "
            f"{wav_duration_s(self.clip.path):.1f}s of audio, wer {wer:.2f}"
        )
        self.assertLess(
            wer, MAX_WER, f"word error rate {wer:.2f} is at or above {MAX_WER}"
        )

    def test_streaming_emits_partials_before_the_final_one(self) -> None:
        """The same clip, fed live: revisions first, then one final partial."""
        chunks = pcm_chunks(self.clip.path, CHUNK_MS)
        self.assertGreater(len(chunks), 1, "clip is shorter than one chunk")

        started_at = time.monotonic()
        partials = list(self.engine.stream(chunks, language=LANGUAGE))
        elapsed_s = time.monotonic() - started_at

        self.assertTrue(partials, "streaming produced no partials at all")
        final = partials[-1]
        self.assertTrue(final.is_final, "the last partial is not marked final")
        self.assertTrue(final.text.strip(), "the final partial carries no text")
        non_final = [item for item in partials[:-1] if not item.is_final]
        self.assertTrue(
            non_final,
            "no non-final partial was emitted before the final one; "
            "the pill would have nothing to show while the user speaks",
        )
        self.assertEqual(
            [item for item in partials[:-1] if item.is_final],
            [],
            "a partial before the last one claimed to be final",
        )

        log(
            f"voxtral stream {LANGUAGE}: {elapsed_s:.2f}s, {len(chunks)} chunks of "
            f"{CHUNK_MS}ms, {len(non_final)} non-final partials"
        )


if __name__ == "__main__":
    unittest.main()
