#!/usr/bin/env python3
"""Tests for scripts/tools/bakeoff.py: the engine bake-off harness."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import subprocess
import sys
import unittest
import wave
from collections import OrderedDict
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "tools"))
bakeoff = importlib.import_module("bakeoff")

import engines


class WordErrorRateTests(unittest.TestCase):
    def test_identical_text_is_zero(self):
        self.assertEqual(bakeoff.word_error_rate("send it to the team", "send it to the team"), 0.0)

    def test_one_substitution_in_four_words(self):
        self.assertEqual(bakeoff.word_error_rate("send it to accounting", "send it to marketing"), 0.25)

    def test_one_insertion(self):
        # hypothesis has one extra word versus a four-word reference.
        self.assertEqual(bakeoff.word_error_rate("send it to accounting", "please send it to accounting"), 0.25)

    def test_one_deletion(self):
        # hypothesis is missing one word from a four-word reference.
        self.assertEqual(bakeoff.word_error_rate("send it to accounting", "send it accounting"), 0.25)

    def test_punctuation_and_case_are_normalised(self):
        self.assertEqual(
            bakeoff.word_error_rate("Send it, please.", "send it please"),
            0.0,
        )

    def test_empty_reference_raises(self):
        with self.assertRaises(ValueError):
            bakeoff.word_error_rate("   ...  ", "send it")


class ManifestTests(unittest.TestCase):
    def _write_manifest(self, tmp_dir: Path, clips: list) -> Path:
        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"clips": clips}))
        return manifest_path

    def test_missing_manifest_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            bakeoff.load_manifest(Path("/nonexistent/dir/manifest.json"))

    def test_bad_language_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            manifest_path = self._write_manifest(
                tmp_dir,
                [{"path": "xx/001.wav", "language": "xx", "text": "hello"}],
            )
            with self.assertRaises(bakeoff.ManifestError):
                bakeoff.load_manifest(manifest_path)

    def test_valid_manifest_loads_clips(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            manifest_path = self._write_manifest(
                tmp_dir,
                [{"path": "en/001.wav", "language": "en", "text": "send it now"}],
            )
            clips = bakeoff.load_manifest(manifest_path)
            self.assertEqual(len(clips), 1)
            self.assertEqual(clips[0].language, "en")
            self.assertEqual(clips[0].text, "send it now")
            self.assertEqual(clips[0].path, tmp_dir / "en/001.wav")


class RenderTableTests(unittest.TestCase):
    def test_renders_header_and_rows(self):
        results = {
            "whispercpp": {
                "model": "large-v3-turbo",
                "wer": {"en": 0.1, "fr": 0.2, "nl": 0.05, "de": 0.15},
                "latency_median_s": 1.234,
                "peak_ram_gb": 2.5,
            },
            "voxtral_mlx": {"unavailable": "MLX not installed on this machine"},
        }
        table = bakeoff.render_table(results)
        lines = table.splitlines()
        self.assertEqual(
            lines[0],
            "| Engine | Model | EN WER | FR WER | NL WER | DE WER "
            "| Median latency (10 s clip) | Peak RAM |",
        )
        self.assertIn("whispercpp", lines[2])
        self.assertIn("large-v3-turbo", lines[2])
        self.assertIn("10.0%", lines[2])
        self.assertIn("20.0%", lines[2])
        self.assertIn("1.23s", lines[2])
        self.assertIn("2.50 GB", lines[2])
        self.assertIn("voxtral_mlx", lines[3])
        self.assertIn("unavailable: MLX not installed on this machine", lines[3])


def _write_silent_wav(path: Path, duration_s: float = 1.0, rate: int = 16000) -> None:
    """Write a mono 16-bit PCM WAV of silence, for run-loop tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(duration_s * rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


class _CannedEngine:
    """Fake Engine returning a fixed transcript, standing in for a real backend."""

    def __init__(self, model_path=None, **_kwargs):
        self.model_path = model_path
        self._loaded = False

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def info(self):
        return {"id": "fake"}

    @property
    def is_loaded(self):
        return self._loaded

    def transcribe(self, wav_path, language=None, hints=None):
        from engines.base import Transcript

        return Transcript(
            text="send it now", language=language, duration_s=1.0, segments=(), engine_id="fake"
        )


class RunBakeoffTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixtures_dir = Path(self._tmp.name)

        _write_silent_wav(self.fixtures_dir / "en" / "001.wav", duration_s=1.0)
        _write_silent_wav(self.fixtures_dir / "fr" / "001.wav", duration_s=1.0)
        manifest = {
            "clips": [
                {"path": "en/001.wav", "language": "en", "text": "send it now"},
                {"path": "fr/001.wav", "language": "fr", "text": "envoie le maintenant"},
            ]
        }
        (self.fixtures_dir / "manifest.json").write_text(json.dumps(manifest))

        engines.register_engine("fake-bakeoff", _CannedEngine)
        self.addCleanup(engines.unregister_engine, "fake-bakeoff")

    def test_run_loop_aggregates_results(self):
        results = bakeoff.run_bakeoff(["fake-bakeoff=unused-spec"], self.fixtures_dir)
        row = results["fake-bakeoff"]
        self.assertNotIn("unavailable", row)
        self.assertEqual(row["model"], "unused-spec")
        # "send it now" vs reference "send it now" -> WER 0.0 for en.
        self.assertEqual(row["wer"]["en"], 0.0)
        # "send it now" vs reference "envoie le maintenant" -> full mismatch.
        self.assertEqual(row["wer"]["fr"], 1.0)
        self.assertEqual(len(row["per_clip"]), 2)
        # Both clips are 1 s, outside the 8-12 s latency bucket.
        self.assertIsNone(row["latency_median_s"])
        self.assertGreaterEqual(row["peak_ram_gb"], 0.0)

    def test_unavailable_engine_becomes_a_row_and_run_continues(self):
        def _unavailable_factory(**_kwargs):
            raise engines.base.EngineUnavailableError("runtime missing on this machine")

        engines.register_engine("fake-unavailable", _unavailable_factory)
        self.addCleanup(engines.unregister_engine, "fake-unavailable")

        results = bakeoff.run_bakeoff(
            ["fake-unavailable=unused", "fake-bakeoff=unused-spec"], self.fixtures_dir
        )
        self.assertEqual(results["fake-unavailable"], {"unavailable": "runtime missing on this machine"})
        self.assertNotIn("unavailable", results["fake-bakeoff"])


class MergeEngineResultsTests(unittest.TestCase):
    """Merging the per-engine sidecars produced by isolated child runs."""

    def test_merges_two_per_engine_result_dicts_in_order(self):
        first = {"whispercpp": {"model": "large-v3-turbo", "peak_ram_gb": 1.5}}
        second = {"voxtral_mlx": {"model": "voxtral-mini-4b-realtime-4bit", "peak_ram_gb": 6.25}}
        merged = bakeoff.merge_engine_results([first, second])
        self.assertEqual(list(merged), ["whispercpp", "voxtral_mlx"])
        # Each engine keeps the RAM figure measured in its own process; the
        # second must not inherit the first's high-water mark.
        self.assertEqual(merged["whispercpp"]["peak_ram_gb"], 1.5)
        self.assertEqual(merged["voxtral_mlx"]["peak_ram_gb"], 6.25)

    def test_merges_unavailable_rows_unchanged(self):
        merged = bakeoff.merge_engine_results(
            [{"voxtral_mlx": {"unavailable": "model directory not found"}}, {}]
        )
        self.assertEqual(merged["voxtral_mlx"], {"unavailable": "model directory not found"})

    def test_empty_input_merges_to_empty_mapping(self):
        self.assertEqual(dict(bakeoff.merge_engine_results([])), {})


class ChildCommandTests(unittest.TestCase):
    """Argument construction for the per-engine child process."""

    def test_command_carries_engine_spec_and_json_out(self):
        command = bakeoff.build_child_command(
            "voxtral_mlx=/models/voxtral-mini-4b-realtime-4bit",
            Path("/fixtures/audio"),
            runs=3,
            json_out=Path("/tmp/voxtral.json"),
        )
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(
            command[command.index("--engine") + 1],
            "voxtral_mlx=/models/voxtral-mini-4b-realtime-4bit",
        )
        self.assertEqual(command[command.index("--json-out") + 1], "/tmp/voxtral.json")
        self.assertEqual(command[command.index("--fixtures") + 1], "/fixtures/audio")
        self.assertEqual(command[command.index("--runs") + 1], "3")
        # Exactly one --engine: the child must never fan out again.
        self.assertEqual(command.count("--engine"), 1)
        self.assertNotIn("--show-transcripts", command)

    def test_command_runs_this_script(self):
        command = bakeoff.build_child_command(
            "whispercpp=/models/ggml.bin", Path("/fixtures"), runs=1, json_out=Path("/tmp/w.json")
        )
        self.assertEqual(Path(command[1]).name, "bakeoff.py")

    def test_show_transcripts_is_forwarded(self):
        command = bakeoff.build_child_command(
            "whispercpp=/models/ggml.bin",
            Path("/fixtures"),
            runs=1,
            json_out=Path("/tmp/w.json"),
            show_transcripts=True,
        )
        self.assertIn("--show-transcripts", command)


class FanOutTests(unittest.TestCase):
    """Isolated per-engine runs, with ``subprocess.run`` mocked out."""

    def _fake_run_writing_sidecars(self, calls, peak_ram_by_call=None):
        def fake_run(command, **_kwargs):
            calls.append(command)
            raw = command[command.index("--engine") + 1]
            engine_id, spec = bakeoff.parse_engine_arg(raw)
            json_out = Path(command[command.index("--json-out") + 1])
            peak = (peak_ram_by_call or {}).get(len(calls), float(len(calls)))
            json_out.write_text(
                json.dumps({engine_id: {"model": spec, "wer": {}, "peak_ram_gb": peak}})
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        return fake_run

    def test_each_engine_runs_in_its_own_child_process(self):
        calls = []
        with mock.patch.object(
            bakeoff.subprocess, "run", self._fake_run_writing_sidecars(calls)
        ):
            results = bakeoff.run_bakeoff_isolated(
                ["whispercpp=/models/ggml.bin", "voxtral_mlx=/models/voxtral"],
                Path("tests/fixtures/audio"),
                runs=2,
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(list(results), ["whispercpp", "voxtral_mlx"])
        self.assertEqual(results["whispercpp"]["peak_ram_gb"], 1.0)
        self.assertEqual(results["voxtral_mlx"]["peak_ram_gb"], 2.0)
        # Each child got its own sidecar path.
        first_json = calls[0][calls[0].index("--json-out") + 1]
        second_json = calls[1][calls[1].index("--json-out") + 1]
        self.assertNotEqual(first_json, second_json)

    def test_nonzero_child_exit_aborts_with_child_stderr(self):
        def fake_run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 2, "", "Traceback: boom in child")

        with mock.patch.object(bakeoff.subprocess, "run", fake_run):
            with self.assertRaises(bakeoff.ChildRunError) as caught:
                bakeoff.run_bakeoff_isolated(
                    ["whispercpp=/models/ggml.bin", "voxtral_mlx=/models/voxtral"],
                    Path("tests/fixtures/audio"),
                )
        self.assertIn("boom in child", str(caught.exception))

    def test_failure_stops_before_running_later_engines(self):
        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 1, "", "child failed")

        with mock.patch.object(bakeoff.subprocess, "run", fake_run):
            with self.assertRaises(bakeoff.ChildRunError):
                bakeoff.run_bakeoff_isolated(
                    ["whispercpp=/a.bin", "voxtral_mlx=/b", "whisper_openai=turbo"],
                    Path("tests/fixtures/audio"),
                )
        self.assertEqual(len(calls), 1)

    def test_bad_engine_arg_is_rejected_before_spawning(self):
        calls = []
        with mock.patch.object(
            bakeoff.subprocess, "run", self._fake_run_writing_sidecars(calls)
        ):
            with self.assertRaises(ValueError):
                bakeoff.run_bakeoff_isolated(["no-equals-sign"], Path("tests/fixtures/audio"))
        self.assertEqual(calls, [])


class MainDispatchTests(unittest.TestCase):
    """``main`` fans out only when more than one --engine is given."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)
        self.fixtures_dir = self.tmp_dir / "fixtures"
        self.fixtures_dir.mkdir()
        self.out_md = self.tmp_dir / "table.md"

        _write_silent_wav(self.fixtures_dir / "en" / "001.wav", duration_s=1.0)
        (self.fixtures_dir / "manifest.json").write_text(
            json.dumps({"clips": [{"path": "en/001.wav", "language": "en", "text": "send it now"}]})
        )

        engines.register_engine("fake-bakeoff", _CannedEngine)
        self.addCleanup(engines.unregister_engine, "fake-bakeoff")

    def _main(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = bakeoff.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_single_engine_runs_in_process(self):
        with mock.patch.object(bakeoff, "run_bakeoff_isolated") as isolated:
            code, _out, _err = self._main(
                [
                    "--engine",
                    "fake-bakeoff=unused-spec",
                    "--fixtures",
                    str(self.fixtures_dir),
                    "--out",
                    str(self.out_md),
                ]
            )
        self.assertEqual(code, 0)
        isolated.assert_not_called()
        self.assertIn("fake-bakeoff", self.out_md.read_text())

    def test_multiple_engines_fan_out(self):
        merged = OrderedDict(
            [
                ("whispercpp", {"model": "ggml", "wer": {}, "peak_ram_gb": 1.5}),
                ("voxtral_mlx", {"model": "voxtral", "wer": {}, "peak_ram_gb": 6.0}),
            ]
        )
        with mock.patch.object(
            bakeoff, "run_bakeoff_isolated", return_value=merged
        ) as isolated:
            code, _out, _err = self._main(
                [
                    "--engine",
                    "whispercpp=/a.bin",
                    "--engine",
                    "voxtral_mlx=/b",
                    "--fixtures",
                    str(self.fixtures_dir),
                    "--out",
                    str(self.out_md),
                ]
            )
        self.assertEqual(code, 0)
        isolated.assert_called_once()
        table = self.out_md.read_text()
        self.assertIn("1.50 GB", table)
        self.assertIn("6.00 GB", table)

    def test_child_failure_returns_nonzero_and_reports_child_stderr(self):
        with mock.patch.object(
            bakeoff,
            "run_bakeoff_isolated",
            side_effect=bakeoff.ChildRunError("child boom detail"),
        ):
            code, _out, err = self._main(
                [
                    "--engine",
                    "whispercpp=/a.bin",
                    "--engine",
                    "voxtral_mlx=/b",
                    "--fixtures",
                    str(self.fixtures_dir),
                    "--out",
                    str(self.out_md),
                ]
            )
        self.assertNotEqual(code, 0)
        self.assertIn("child boom detail", err)

    def test_json_out_holds_full_result_rows(self):
        json_out = self.tmp_dir / "results.json"
        code, _out, _err = self._main(
            [
                "--engine",
                "fake-bakeoff=unused-spec",
                "--fixtures",
                str(self.fixtures_dir),
                "--out",
                str(self.out_md),
                "--json-out",
                str(json_out),
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(json_out.read_text())
        self.assertIn("peak_ram_gb", payload["fake-bakeoff"])
        self.assertEqual(payload["fake-bakeoff"]["model"], "unused-spec")
        self.assertEqual(len(payload["fake-bakeoff"]["per_clip"]), 1)


if __name__ == "__main__":
    unittest.main()
