"""Tests for the bundled whisper.cpp HTTP engine.

No binary and no network: ``spawn`` is faked and the "server" is a local
``http.server`` on loopback implementing the two routes the engine uses.
"""

import io
import json
import os
import subprocess
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from engines import EngineError, EngineNotLoadedError, EngineUnavailableError, Hints, create_engine
from engines.whispercpp import (
    BINARY_ENV_VAR,
    ENGINE_CLASS,
    WHISPER_CPP_TAG,
    WhisperCppEngine,
    resolve_whisper_server_binary,
)

CANNED_RESPONSE = {
    "task": "transcribe",
    "language": "dutch",
    "duration": 2.5,
    "text": " Murmur schrijft mee. ",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.2, "text": " Murmur "},
        {"id": 1, "start": 1.2, "end": 2.5, "text": " schrijft mee. "},
    ],
}


def _parse_multipart(body: bytes, content_type: str) -> dict:
    """Minimal multipart reader: returns ``{field_name: value_or_bytes}``."""
    boundary = content_type.split("boundary=", 1)[1].strip()
    marker = f"--{boundary}".encode()
    fields = {}
    for part in body.split(marker):
        if not part.strip() or part.strip() == b"--":
            continue
        head, _, payload = part.partition(b"\r\n\r\n")
        headers = head.decode("utf-8", errors="replace")
        if 'name="' not in headers:
            continue
        name = headers.split('name="', 1)[1].split('"', 1)[0]
        payload = payload.rstrip(b"\r\n")
        fields[name] = payload if "filename=" in headers else payload.decode("utf-8")
    return fields


class _Recorder:
    """Shared state between the test and the handler class."""

    def __init__(self):
        self.requests = []
        self.inference_status = 200
        self.healthy = True


class _Handler(BaseHTTPRequestHandler):
    recorder: _Recorder = None  # set per-server

    def log_message(self, *args):  # silence the test output
        pass

    def _send(self, status, payload: bytes, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health" and self.recorder.healthy:
            self._send(200, b'{"status":"ok"}')
        else:
            self._send(503, b'{"error":"loading"}')

    def do_POST(self):
        if self.path != "/inference":
            self._send(404, b'{"error":"no such route"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.recorder.requests.append(
            _parse_multipart(body, self.headers.get("Content-Type", ""))
        )
        status = self.recorder.inference_status
        if status != 200:
            self._send(status, b'{"error":"failed to process audio"}')
            return
        self._send(200, json.dumps(CANNED_RESPONSE).encode("utf-8"))


class FakeProcess:
    """Stands in for ``subprocess.Popen``; records lifecycle calls."""

    def __init__(self, exit_code=None):
        self.returncode = exit_code
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(b"whisper_init: loading model\nerror: boom\n")
        self.terminated = False
        self.killed = False
        self.waits = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.returncode


class NoisyProcess(FakeProcess):
    """A child that floods a *real* pipe with far more than one pipe buffer.

    ``os.pipe`` blocks the writer once ~64 KiB sits unread, so if nothing
    drains ``stderr`` the writer never finishes — which is exactly how the
    real ``whisper-server`` stalls its own ``POST /inference``.
    """

    LINES = 4000  # ~136 KiB, comfortably past any pipe buffer

    def __init__(self, exit_code=None):
        super().__init__(exit_code)
        read_fd, write_fd = os.pipe()
        self.stderr = os.fdopen(read_fd, "rb")
        self.wrote_everything = threading.Event()
        self._writer = threading.Thread(
            target=self._write, args=(write_fd,), daemon=True
        )
        self._writer.start()

    def _write(self, write_fd):
        try:
            with os.fdopen(write_fd, "wb") as handle:
                for index in range(self.LINES):
                    handle.write(f"whisper log line {index:05d} ..........\n".encode())
        except OSError:
            return
        self.wrote_everything.set()


class FakeSpawn:
    """Records the argv the engine builds and hands back a :class:`FakeProcess`."""

    def __init__(self, exit_code=None, process_class=FakeProcess):
        self.exit_code = exit_code
        self.process_class = process_class
        self.calls = []
        self.process = None

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        self.process = self.process_class(self.exit_code)
        return self.process


class ServerBackedTestCase(unittest.TestCase):
    """Runs a real loopback HTTP server the engine is pointed at."""

    def setUp(self):
        self.recorder = _Recorder()
        handler = type("BoundHandler", (_Handler,), {"recorder": self.recorder})
        server_class = type("BoundServer", (ThreadingHTTPServer,), {"daemon_threads": True})
        self.server = server_class(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.model_path = self.tmp / "ggml-large-v3-turbo.bin"
        self.model_path.write_bytes(b"ggml" * 8)
        self.binary = self.tmp / "whisper-server"
        self.binary.write_text("#!/bin/sh\n")
        self.wav = self.tmp / "clip.wav"
        self.wav.write_bytes(b"RIFF....WAVEfmt ")

        port_patch = patch("engines.whispercpp._free_port", return_value=self.port)
        port_patch.start()
        self.addCleanup(port_patch.stop)

    def make_engine(self, **overrides):
        kwargs = dict(
            model_path=self.model_path,
            binary=self.binary,
            spawn=FakeSpawn(),
            http_open=urllib.request.urlopen,
            startup_timeout_s=5.0,
        )
        kwargs.update(overrides)
        engine = WhisperCppEngine(**kwargs)
        self.addCleanup(engine.unload)
        return engine


class LifecycleTests(ServerBackedTestCase):
    def test_load_spawns_the_server_with_model_host_and_port(self):
        spawn = FakeSpawn()
        engine = self.make_engine(spawn=spawn)
        self.assertFalse(engine.is_loaded)

        engine.load()

        self.assertTrue(engine.is_loaded)
        argv, _kwargs = spawn.calls[0]
        self.assertEqual(argv[0], str(self.binary))
        self.assertEqual(argv[argv.index("-m") + 1], str(self.model_path))
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--port") + 1], str(self.port))
        # Server-side default must be auto, since an omitted request field
        # otherwise falls back to whisper.cpp's own "en".
        self.assertEqual(argv[argv.index("-l") + 1], "auto")

    def test_load_is_idempotent(self):
        spawn = FakeSpawn()
        engine = self.make_engine(spawn=spawn)
        engine.load()
        engine.load()
        self.assertEqual(len(spawn.calls), 1)

    def test_child_stdout_is_discarded_and_stderr_is_piped(self):
        # Nothing reads the child's stdout, so a PIPE there is a deadlock
        # waiting to happen; stderr stays piped because the drain reads it.
        spawn = FakeSpawn()
        engine = self.make_engine(spawn=spawn)
        engine.load()
        _argv, kwargs = spawn.calls[0]
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.PIPE)

    def test_noisy_child_stderr_never_blocks_the_engine(self):
        spawn = FakeSpawn(process_class=NoisyProcess)
        engine = self.make_engine(spawn=spawn)

        engine.load()
        transcript = engine.transcribe(self.wav)

        self.assertEqual(transcript.text, "Murmur schrijft mee.")
        self.assertTrue(
            spawn.process.wrote_everything.wait(timeout=5),
            "the child blocked writing stderr: nothing drained the pipe",
        )

    def test_error_tail_keeps_only_the_last_stderr_lines(self):
        self.recorder.healthy = False
        spawn = FakeSpawn(process_class=NoisyProcess)
        engine = self.make_engine(spawn=spawn, startup_timeout_s=0.2)

        with self.assertRaises(EngineError) as ctx:
            engine.load()

        message = str(ctx.exception)
        self.assertLess(len(message), 2500)
        self.assertIn(f"whisper log line {NoisyProcess.LINES - 1:05d}", message)
        self.assertNotIn("whisper log line 00000", message)

    def test_unload_closes_the_child_pipes(self):
        spawn = FakeSpawn()
        engine = self.make_engine(spawn=spawn)
        engine.load()

        engine.unload()

        self.assertTrue(spawn.process.stdout.closed)
        self.assertTrue(spawn.process.stderr.closed)

    def test_unload_terminates_and_is_idempotent(self):
        spawn = FakeSpawn()
        engine = self.make_engine(spawn=spawn)
        engine.load()
        engine.unload()
        self.assertFalse(engine.is_loaded)
        self.assertTrue(spawn.process.terminated)
        self.assertIn(3.0, spawn.process.waits)
        engine.unload()

    def test_unload_kills_when_terminate_does_not_stop_it(self):
        class StubbornProcess(FakeProcess):
            def wait(self, timeout=None):
                self.waits.append(timeout)
                if not self.killed:
                    raise TimeoutError("still running")
                return self.returncode

        spawn = FakeSpawn()
        engine = self.make_engine(spawn=spawn)
        engine.load()
        spawn.process.__class__ = StubbornProcess
        engine.unload()
        self.assertTrue(spawn.process.killed)

    def test_startup_timeout_kills_the_process_and_raises(self):
        self.recorder.healthy = False
        spawn = FakeSpawn()
        engine = self.make_engine(spawn=spawn, startup_timeout_s=0.2)

        with self.assertRaises(EngineError) as ctx:
            engine.load()

        self.assertFalse(engine.is_loaded)
        self.assertTrue(spawn.process.terminated or spawn.process.killed)
        self.assertIn("/health", str(ctx.exception))

    def test_startup_error_carries_bounded_stderr_and_no_transcript(self):
        self.recorder.healthy = False
        engine = self.make_engine(startup_timeout_s=0.1)
        with self.assertRaises(EngineError) as ctx:
            engine.load()
        message = str(ctx.exception)
        self.assertIn("error: boom", message)
        self.assertLess(len(message), 2500)

    def test_process_that_dies_during_startup_raises_engine_error(self):
        engine = self.make_engine(spawn=FakeSpawn(exit_code=1), startup_timeout_s=1.0)
        with self.assertRaises(EngineError) as ctx:
            engine.load()
        self.assertIn("exited with code 1", str(ctx.exception))

    def test_missing_binary_raises_engine_unavailable(self):
        engine = self.make_engine(binary=self.tmp / "absent" / "whisper-server")
        with self.assertRaises(EngineUnavailableError) as ctx:
            engine.load()
        self.assertIn("fetch_whispercpp.sh", str(ctx.exception))

    def test_info_reports_model_stem_and_capabilities(self):
        info = self.make_engine().info()
        self.assertEqual(info.id, "whispercpp")
        self.assertEqual(info.model_id, "ggml-large-v3-turbo")
        self.assertEqual(info.languages, ("auto",))
        self.assertFalse(info.supports_streaming)
        self.assertTrue(info.supports_hints)
        self.assertFalse(info.requires_apple_silicon)
        self.assertEqual(info.size_bytes, self.model_path.stat().st_size)


class TranscriptionTests(ServerBackedTestCase):
    def test_transcribe_before_load_raises_not_loaded(self):
        engine = self.make_engine()
        with self.assertRaises(EngineNotLoadedError):
            engine.transcribe(self.wav)

    def test_load_transcribe_unload_round_trip(self):
        engine = self.make_engine()
        engine.load()
        transcript = engine.transcribe(self.wav, language="nl")

        self.assertEqual(transcript.text, "Murmur schrijft mee.")
        self.assertEqual(transcript.language, "dutch")
        self.assertEqual(transcript.duration_s, 2.5)
        self.assertEqual(transcript.engine_id, "whispercpp")
        self.assertEqual([s.text for s in transcript.segments], ["Murmur", "schrijft mee."])
        self.assertEqual(transcript.segments[1].start, 1.2)
        self.assertEqual(transcript.segments[1].end, 2.5)

        engine.unload()
        with self.assertRaises(EngineNotLoadedError):
            engine.transcribe(self.wav)

    def test_request_sends_the_wav_and_verbose_json(self):
        engine = self.make_engine()
        engine.load()
        engine.transcribe(self.wav)
        sent = self.recorder.requests[0]
        self.assertEqual(sent["file"], self.wav.read_bytes())
        self.assertEqual(sent["response_format"], "verbose_json")

    def test_explicit_language_is_sent(self):
        engine = self.make_engine()
        engine.load()
        engine.transcribe(self.wav, language="fr")
        self.assertEqual(self.recorder.requests[0]["language"], "fr")

    def test_language_is_omitted_for_auto_and_none(self):
        engine = self.make_engine()
        engine.load()
        engine.transcribe(self.wav, language="auto")
        engine.transcribe(self.wav, language=None)
        for sent in self.recorder.requests:
            self.assertNotIn("language", sent)

    def test_prompt_carries_initial_prompt_then_vocabulary(self):
        engine = self.make_engine()
        engine.load()
        engine.transcribe(
            self.wav,
            hints=Hints(vocabulary=("Murmur", "Boske"), initial_prompt="Dictation."),
        )
        prompt = self.recorder.requests[0]["prompt"]
        self.assertTrue(prompt.startswith("Dictation."))
        self.assertIn("Murmur, Boske", prompt)

    def test_no_prompt_field_without_hints(self):
        engine = self.make_engine()
        engine.load()
        engine.transcribe(self.wav, hints=Hints())
        self.assertNotIn("prompt", self.recorder.requests[0])

    def test_hints_applied_is_true_when_a_prompt_was_sent(self):
        engine = self.make_engine()
        engine.load()
        transcript = engine.transcribe(self.wav, hints=Hints(vocabulary=("Murmur",)))
        self.assertIs(transcript.hints_applied, True)

    def test_hints_applied_is_unknown_when_no_prompt_was_sent(self):
        engine = self.make_engine()
        engine.load()
        self.assertIsNone(engine.transcribe(self.wav).hints_applied)
        self.assertIsNone(engine.transcribe(self.wav, hints=Hints()).hints_applied)

    def test_long_form_is_accepted_and_changes_nothing(self):
        engine = self.make_engine()
        engine.load()
        engine.transcribe(self.wav, long_form=True)
        engine.transcribe(self.wav)
        self.assertEqual(self.recorder.requests[0], self.recorder.requests[1])

    def test_non_2xx_response_raises_engine_error_with_status(self):
        engine = self.make_engine()
        engine.load()
        self.recorder.inference_status = 500
        with self.assertRaises(EngineError) as ctx:
            engine.transcribe(self.wav)
        self.assertIn("500", str(ctx.exception))

    def test_unreadable_wav_raises_engine_error(self):
        engine = self.make_engine()
        engine.load()
        with self.assertRaises(EngineError):
            engine.transcribe(self.tmp / "missing.wav")


class BinaryResolutionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        env_patch = patch.dict("os.environ", {}, clear=False)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        import os as _os

        _os.environ.pop(BINARY_ENV_VAR, None)
        root_patch = patch("engines.whispercpp._repo_root", return_value=self.tmp / "repo")
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_env_override_wins(self):
        override = self.tmp / "custom-whisper-server"
        override.write_text("#!/bin/sh\n")
        with patch.dict("os.environ", {BINARY_ENV_VAR: str(override)}):
            self.assertEqual(resolve_whisper_server_binary(), override)

    def test_frozen_bundle_path_is_used(self):
        import sys as _sys

        bundled = self.tmp / "meipass" / "bin" / "whisper-server"
        bundled.parent.mkdir(parents=True)
        bundled.write_text("#!/bin/sh\n")
        with patch.object(_sys, "_MEIPASS", str(self.tmp / "meipass"), create=True):
            self.assertEqual(resolve_whisper_server_binary(), bundled)

    def test_repo_vendor_path_is_used(self):
        vendored = self.tmp / "repo" / "vendor" / "whispercpp" / "whisper-server"
        vendored.parent.mkdir(parents=True)
        vendored.write_text("#!/bin/sh\n")
        self.assertEqual(resolve_whisper_server_binary(), vendored)

    def test_missing_binary_names_the_fetch_script(self):
        with self.assertRaises(EngineUnavailableError) as ctx:
            resolve_whisper_server_binary()
        self.assertIn("scripts/tools/fetch_whispercpp.sh", str(ctx.exception))


class RegistryIntegrationTests(ServerBackedTestCase):
    def test_create_engine_builds_a_whispercpp_engine(self):
        engine = create_engine(
            "whispercpp",
            model_path=self.model_path,
            binary=self.binary,
            spawn=FakeSpawn(),
            http_open=urllib.request.urlopen,
        )
        self.addCleanup(engine.unload)
        self.assertIsInstance(engine, WhisperCppEngine)
        self.assertIs(ENGINE_CLASS, WhisperCppEngine)

    def test_pinned_tag_matches_the_fetch_script(self):
        script = Path(__file__).resolve().parent.parent / "scripts" / "tools" / "fetch_whispercpp.sh"
        self.assertIn(f'WHISPER_CPP_TAG:-{WHISPER_CPP_TAG}', script.read_text())


if __name__ == "__main__":
    unittest.main()
