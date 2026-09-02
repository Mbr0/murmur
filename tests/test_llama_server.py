"""Tests for the bundled ``llama-server`` cleanup runtime.

No binary and no network: ``spawn`` is faked and the "server" is a local
``http.server`` on loopback implementing the two routes the client uses
(``GET /health`` and ``POST /v1/chat/completions``), mirroring
``tests/test_engine_whispercpp.py``.
"""

import io
import json
import re
import subprocess
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cleanup.llama_server import (
    BINARY_ENV_VAR,
    BINARY_NAME,
    CLEANUP_MIN_RAM_GB,
    CLEANUP_MODEL_SPEC,
    CHAT_PATH,
    CONTEXT_RESERVE_TOKENS,
    DEFAULT_CONTEXT_TOKENS,
    HEALTH_PATH,
    LLAMA_CPP_TAG,
    MAX_TOKENS_HEADROOM,
    MAX_TOKENS_PER_WORD,
    MIN_OUTPUT_TOKENS,
    CleanupClient,
    CleanupResult,
    LlamaServer,
    LlamaServerError,
    estimate_prompt_tokens,
    max_tokens_for_text,
    resolve_llama_server_binary,
    should_enable_cleanup_by_default,
    timeout_for_text,
)
from engines.base import EngineUnavailableError
from engines.model_store import ModelSpec, ModelStore

CLEANED = "Murmur writes it down, cleanly."


def _chat_response(content: str) -> dict:
    """A minimal OpenAI-compatible non-streaming chat completion body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class _Recorder:
    """Shared state between a test and its handler class."""

    def __init__(self):
        self.requests = []
        self.chat_status = 200
        self.healthy = True
        self.health_hits = 0
        self.healthy_after = 0
        self.chat_delay_s = 0.0
        self.chat_content = CLEANED
        self.chat_body = None  # raw override, for non-JSON bodies


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
        if self.path != HEALTH_PATH:
            self._send(404, b'{"error":"no such route"}')
            return
        self.recorder.health_hits += 1
        ready = self.recorder.healthy and (
            self.recorder.health_hits > self.recorder.healthy_after
        )
        if ready:
            self._send(200, b'{"status":"ok"}')
        else:
            # Verified against the pinned tag's tools/server/README.md.
            self._send(
                503,
                b'{"error":{"code":503,"message":"Loading model",'
                b'"type":"unavailable_error"}}',
            )

    def do_POST(self):
        if self.path != CHAT_PATH:
            self._send(404, b'{"error":"no such route"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.recorder.requests.append(json.loads(body.decode("utf-8")))
        if self.recorder.chat_delay_s:
            time.sleep(self.recorder.chat_delay_s)
        status = self.recorder.chat_status
        if status != 200:
            self._send(status, b'{"error":{"message":"server is busy"}}')
            return
        if self.recorder.chat_body is not None:
            self._send(200, self.recorder.chat_body)
            return
        payload = json.dumps(_chat_response(self.recorder.chat_content))
        self._send(200, payload.encode("utf-8"))


class FakeProcess:
    """Stands in for ``subprocess.Popen``; records lifecycle calls."""

    def __init__(self, exit_code=None):
        self.returncode = exit_code
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(b"load_backend: loaded Metal\nerror: boom\n")
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def crash(self, code=137):
        """The child dies under us — an OOM kill is code 137. poll() flips."""
        self.returncode = code

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class FakeSpawn:
    """Records the argv the server builds and hands back a :class:`FakeProcess`."""

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
    """Runs a real loopback HTTP server the LlamaServer is pointed at."""

    def setUp(self):
        self.recorder = _Recorder()
        handler = type("BoundHandler", (_Handler,), {"recorder": self.recorder})
        server_class = type(
            "BoundServer", (ThreadingHTTPServer,), {"daemon_threads": True}
        )
        self.http = server_class(("127.0.0.1", 0), handler)
        self.port = self.http.server_address[1]
        thread = threading.Thread(
            target=self.http.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.model_path = self.tmp / "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
        self.model_path.write_bytes(b"GGUF" * 8)
        self.binary = self.tmp / BINARY_NAME
        self.binary.write_text("#!/bin/sh\n")

        port_patch = patch("cleanup.llama_server._free_port", return_value=self.port)
        port_patch.start()
        self.addCleanup(port_patch.stop)

    def make_server(self, **overrides):
        kwargs = dict(
            model_path=self.model_path,
            binary=self.binary,
            spawn=FakeSpawn(),
            http_open=urllib.request.urlopen,
            startup_timeout_s=5.0,
        )
        kwargs.update(overrides)
        server = LlamaServer(**kwargs)
        self.addCleanup(server.stop)
        return server

    def started_client(self, **overrides):
        server = self.make_server()
        server.start()
        return CleanupClient(server, **overrides)


class LifecycleTests(ServerBackedTestCase):
    def test_start_spawns_the_server_with_model_host_port_and_context(self):
        spawn = FakeSpawn()
        server = self.make_server(spawn=spawn, context_tokens=8192)
        self.assertFalse(server.is_running)

        server.start()

        self.assertTrue(server.is_running)
        self.assertEqual(server.port, self.port)
        argv, _kwargs = spawn.calls[0]
        self.assertEqual(argv[0], str(self.binary))
        self.assertEqual(argv[argv.index("-m") + 1], str(self.model_path))
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--port") + 1], str(self.port))
        self.assertEqual(argv[argv.index("-c") + 1], "8192")
        # Both flags exist in the pinned tag's option table; the Web UI is
        # dead weight in a background helper and its logs would hold user text.
        self.assertIn("--log-disable", argv)
        self.assertIn("--no-webui", argv)

    def test_start_polls_through_the_503_the_server_sends_while_loading(self):
        # llama-server answers 503 "Loading model" for the whole load window,
        # so a 503 must read as "not yet", never as a failure.
        self.recorder.healthy_after = 3
        server = self.make_server()

        server.start()

        self.assertTrue(server.is_running)
        self.assertGreater(self.recorder.health_hits, 3)

    def test_start_is_idempotent(self):
        spawn = FakeSpawn()
        server = self.make_server(spawn=spawn)
        server.start()
        server.start()
        self.assertEqual(len(spawn.calls), 1)

    def test_child_stdout_is_discarded_and_stderr_is_piped(self):
        spawn = FakeSpawn()
        server = self.make_server(spawn=spawn)
        server.start()
        _argv, kwargs = spawn.calls[0]
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.PIPE)

    def test_start_raises_when_health_never_answers(self):
        self.recorder.healthy = False
        server = self.make_server(startup_timeout_s=0.2)

        with self.assertRaises(LlamaServerError) as ctx:
            server.start()

        self.assertIn(HEALTH_PATH, str(ctx.exception))
        self.assertFalse(server.is_running)

    def test_start_raises_when_the_child_exits_during_startup(self):
        spawn = FakeSpawn(exit_code=1)
        server = self.make_server(spawn=spawn)

        with self.assertRaises(LlamaServerError) as ctx:
            server.start()

        self.assertIn("exited with code 1", str(ctx.exception))
        self.assertIn("error: boom", str(ctx.exception))

    def test_stop_terminates_closes_pipes_and_is_idempotent(self):
        spawn = FakeSpawn()
        server = self.make_server(spawn=spawn)
        server.start()

        server.stop()
        server.stop()

        self.assertFalse(server.is_running)
        self.assertIsNone(server.port)
        self.assertTrue(spawn.process.terminated)
        self.assertTrue(spawn.process.stdout.closed)
        self.assertTrue(spawn.process.stderr.closed)


class InterruptedStartTests(ServerBackedTestCase):
    """Quitting during the model load must not leave a 2 GB child behind."""

    def test_stop_during_a_start_aborts_it_and_kills_the_child(self):
        # The realistic case: the user quits while the first cleanup is still
        # loading the GGUF. start() holds no lock stop() needs, and the child
        # it already spawned is terminated rather than orphaned.
        self.recorder.healthy = False  # /health never comes good
        spawn = FakeSpawn()
        server = self.make_server(spawn=spawn, startup_timeout_s=30.0)
        failures = []

        def run():
            try:
                server.start()
            except LlamaServerError as error:
                failures.append(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and spawn.process is None:
            time.sleep(0.01)
        self.assertIsNotNone(spawn.process, "the child was never spawned")

        started_at = time.monotonic()
        server.stop()
        elapsed = time.monotonic() - started_at

        worker.join(5.0)
        self.assertFalse(worker.is_alive())
        self.assertLess(elapsed, 2.0)
        self.assertTrue(spawn.process.terminated)
        self.assertFalse(server.is_running)
        self.assertEqual(len(failures), 1)
        self.assertIn("stopped", str(failures[0]).lower())

    def test_a_start_after_a_stop_is_allowed_again(self):
        # stop() must not poison the object: CleanupRuntime builds a fresh
        # server per attempt, but a restart of the same one has to work.
        server = self.make_server()
        server.stop()
        server.start()
        self.assertTrue(server.is_running)


SYSTEM_PROMPT = "Clean up the transcript. Return only the cleaned text."


class CrashedChildTests(ServerBackedTestCase):
    """An OOM-killed child must not leave the client skipping forever."""

    def test_is_running_goes_false_when_the_child_dies_under_us(self):
        spawn = FakeSpawn()
        server = self.make_server(spawn=spawn)
        server.start()
        self.assertTrue(server.is_running)

        spawn.process.crash(137)

        self.assertFalse(server.is_running)
        self.assertIsNone(server.port)
        self.assertEqual(server.exit_code, 137)

    def test_a_deliberate_stop_is_not_reported_as_a_crash(self):
        server = self.make_server()
        server.start()

        server.stop()

        self.assertFalse(server.is_running)
        self.assertIsNone(server.exit_code)

    def test_a_crashed_server_starts_again_and_forgets_the_exit_code(self):
        spawn = FakeSpawn()
        server = self.make_server(spawn=spawn)
        server.start()
        spawn.process.crash(137)
        self.assertFalse(server.is_running)

        server.start()

        self.assertTrue(server.is_running)
        self.assertEqual(len(spawn.calls), 2)
        self.assertIsNone(server.exit_code)

    def test_cleanup_raises_naming_the_exit_code_instead_of_skipping(self):
        spawn = FakeSpawn()
        server = self.make_server(spawn=spawn)
        client = CleanupClient(server)
        server.start()
        spawn.process.crash(137)

        with self.assertRaises(LlamaServerError) as ctx:
            client.cleanup("murmur writes", SYSTEM_PROMPT)

        self.assertIn("137", str(ctx.exception))
        self.assertIn("exited", str(ctx.exception))
        self.assertEqual(self.recorder.requests, [])


class CleanupHappyPathTests(ServerBackedTestCase):
    def test_cleanup_parses_the_first_choice_message_content(self):
        client = self.started_client()

        result = client.cleanup("murmur writes it down cleanly", SYSTEM_PROMPT)

        self.assertIsInstance(result, CleanupResult)
        self.assertEqual(result.text, CLEANED)
        self.assertFalse(result.skipped)
        self.assertIsNone(result.reason)
        self.assertGreaterEqual(result.elapsed_s, 0.0)

    def test_cleanup_posts_system_and_user_messages_without_streaming(self):
        client = self.started_client()

        client.cleanup("hello there", SYSTEM_PROMPT, temperature=0.3)

        sent = self.recorder.requests[0]
        self.assertEqual(
            sent["messages"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "hello there"},
            ],
        )
        self.assertFalse(sent["stream"])
        self.assertAlmostEqual(sent["temperature"], 0.3)

    def test_cleanup_sends_a_max_tokens_budget_derived_from_the_input(self):
        client = self.started_client()
        text = " ".join(["word"] * 200)

        client.cleanup(text, SYSTEM_PROMPT)

        self.assertEqual(self.recorder.requests[0]["max_tokens"], 300 + MAX_TOKENS_HEADROOM)
        self.assertEqual(max_tokens_for_text(text), 300 + MAX_TOKENS_HEADROOM)

    def test_cleanup_strips_surrounding_whitespace_from_the_reply(self):
        self.recorder.chat_content = f"\n  {CLEANED}\n"
        client = self.started_client()

        result = client.cleanup("murmur writes", SYSTEM_PROMPT)

        self.assertEqual(result.text, CLEANED)
        self.assertFalse(result.skipped)


class CleanupFallbackTests(ServerBackedTestCase):
    """Every request-time failure returns the original text, visibly skipped."""

    def test_timeout_returns_the_original_text_and_a_reason(self):
        self.recorder.chat_delay_s = 1.0
        client = self.started_client(timeout_policy=lambda text: 0.05)
        original = "murmur writes it down cleanly"

        result = client.cleanup(original, SYSTEM_PROMPT)

        self.assertEqual(result.text, original)
        self.assertTrue(result.skipped)
        self.assertIn("timed out", result.reason)

    def test_server_error_returns_the_original_text_and_a_reason(self):
        self.recorder.chat_status = 503
        client = self.started_client()
        original = "murmur writes it down cleanly"

        result = client.cleanup(original, SYSTEM_PROMPT)

        self.assertEqual(result.text, original)
        self.assertTrue(result.skipped)
        self.assertIn("503", result.reason)

    def test_empty_reply_returns_the_original_text_and_a_reason(self):
        self.recorder.chat_content = "   \n  "
        client = self.started_client()
        original = "murmur writes it down cleanly"

        result = client.cleanup(original, SYSTEM_PROMPT)

        self.assertEqual(result.text, original)
        self.assertTrue(result.skipped)
        self.assertIn("empty", result.reason)

    def test_unreadable_reply_returns_the_original_text_and_a_reason(self):
        # A malformed body must not cost the user their transcript either.
        self.recorder.chat_body = b"not json at all"
        client = self.started_client()
        original = "murmur writes it down cleanly"

        result = client.cleanup(original, SYSTEM_PROMPT)

        self.assertEqual(result.text, original)
        self.assertTrue(result.skipped)
        self.assertIn("unreadable", result.reason)

    def test_a_skip_reason_never_carries_the_transcript_text(self):
        self.recorder.chat_status = 500
        client = self.started_client()
        secret = "the quick brown fox jumped over the lazy dog"

        result = client.cleanup(secret, SYSTEM_PROMPT)

        self.assertNotIn("quick brown fox", result.reason)

    def test_cleanup_raises_when_the_server_is_not_running(self):
        # A stopped server is the caller's bug, not the model's bad day.
        server = self.make_server()
        client = CleanupClient(server)

        with self.assertRaises(LlamaServerError):
            client.cleanup("murmur writes", SYSTEM_PROMPT)

        server.start()
        server.stop()
        with self.assertRaises(LlamaServerError):
            client.cleanup("murmur writes", SYSTEM_PROMPT)


class ContextWindowTests(ServerBackedTestCase):
    """The output budget has to fit inside the ``-c`` window, or nothing runs."""

    def test_text_too_long_for_the_window_skips_without_a_request(self):
        client = self.started_client()
        original = " ".join(["word"] * 4000)

        result = client.cleanup(original, SYSTEM_PROMPT)

        self.assertEqual(result.text, original)
        self.assertTrue(result.skipped)
        self.assertEqual(result.reason, "too long for cleanup")
        self.assertEqual(self.recorder.requests, [])

    def test_the_request_never_asks_for_more_tokens_than_the_window_holds(self):
        client = self.started_client()
        text = " ".join(["word"] * 2000)

        client.cleanup(text, SYSTEM_PROMPT)

        sent = self.recorder.requests[0]
        prompt_tokens = estimate_prompt_tokens(text, SYSTEM_PROMPT)
        self.assertLessEqual(
            prompt_tokens + sent["max_tokens"],
            DEFAULT_CONTEXT_TOKENS - CONTEXT_RESERVE_TOKENS,
        )
        # The cap really bit: this is less than the uncapped word-based budget.
        self.assertLess(sent["max_tokens"], int(2000 * MAX_TOKENS_PER_WORD))

    def test_the_budget_follows_the_window_the_server_was_started_with(self):
        server = self.make_server(context_tokens=32768)
        server.start()
        client = CleanupClient(server)
        text = " ".join(["word"] * 2000)

        client.cleanup(text, SYSTEM_PROMPT)

        self.assertEqual(server.context_tokens, 32768)
        # A window that size leaves the uncapped budget untouched.
        self.assertEqual(
            self.recorder.requests[0]["max_tokens"],
            int(2000 * MAX_TOKENS_PER_WORD) + MAX_TOKENS_HEADROOM,
        )


class ContextBudgetTests(unittest.TestCase):
    def test_short_text_keeps_the_uncapped_budget(self):
        text = " ".join(["word"] * 100)
        self.assertEqual(max_tokens_for_text(text, DEFAULT_CONTEXT_TOKENS), 214)

    def test_a_long_input_is_capped_to_what_the_window_leaves(self):
        text = " ".join(["word"] * 2000)
        prompt_tokens = estimate_prompt_tokens(text)
        budget = max_tokens_for_text(text, DEFAULT_CONTEXT_TOKENS, prompt_tokens)

        self.assertLess(budget, int(2000 * MAX_TOKENS_PER_WORD) + MAX_TOKENS_HEADROOM)
        self.assertEqual(prompt_tokens + budget, DEFAULT_CONTEXT_TOKENS - CONTEXT_RESERVE_TOKENS)

    def test_an_input_that_fills_the_window_leaves_no_room_to_answer(self):
        text = " ".join(["word"] * 4000)
        budget = max_tokens_for_text(text, DEFAULT_CONTEXT_TOKENS, estimate_prompt_tokens(text))
        self.assertLess(budget, MIN_OUTPUT_TOKENS)

    def test_the_prompt_estimate_counts_the_system_prompt_too(self):
        text = " ".join(["word"] * 10)
        self.assertGreater(
            estimate_prompt_tokens(text, SYSTEM_PROMPT), estimate_prompt_tokens(text)
        )

    def test_the_prompt_estimate_never_under_counts_the_words(self):
        # 1.4 tokens per word is the over-estimate the cap is built on.
        for words in (0, 1, 10, 500):
            with self.subTest(words=words):
                text = " ".join(["word"] * words)
                self.assertGreaterEqual(estimate_prompt_tokens(text), words)


class TimeoutPolicyTests(unittest.TestCase):
    """The plan's policy: 2 s per 100 words, minimum 3 s, cap 20 s."""

    def test_timeout_table(self):
        cases = [
            (0, 3.0),      # nothing to do still gets the floor
            (10, 3.0),     # 0.2 s of budget, floored
            (149, 3.0),    # 2.98 s, still under the floor
            (150, 3.0),    # exactly the floor
            (200, 4.0),    # 2 s per 100 words, plainly
            (500, 10.0),
            (999, 19.98),
            (1000, 20.0),  # exactly the cap
            (5000, 20.0),  # capped
        ]
        for words, expected in cases:
            with self.subTest(words=words):
                text = " ".join(["word"] * words)
                self.assertAlmostEqual(timeout_for_text(text), expected, places=6)

    def test_whitespace_only_text_gets_the_floor_not_a_crash(self):
        self.assertEqual(timeout_for_text("   \n\t "), 3.0)

    def test_max_tokens_table(self):
        for words, expected in [(0, 64), (10, 79), (100, 214), (1000, 1564)]:
            with self.subTest(words=words):
                text = " ".join(["word"] * words)
                self.assertEqual(max_tokens_for_text(text), expected)


class DefaultEnableTests(unittest.TestCase):
    """Risk register: cleanup is off by default below 16 GB."""

    def test_default_enable_table(self):
        cases = [
            (None, False),  # probe failed: assume the weaker machine
            (4, False),
            (8, False),
            (15, False),
            (16, True),
            (18, True),
            (32, True),
            (64, True),
        ]
        for ram_gb, expected in cases:
            with self.subTest(ram_gb=ram_gb):
                self.assertIs(should_enable_cleanup_by_default(ram_gb), expected)

    def test_threshold_matches_the_documented_constant(self):
        self.assertEqual(CLEANUP_MIN_RAM_GB, 16)
        self.assertFalse(should_enable_cleanup_by_default(CLEANUP_MIN_RAM_GB - 1))
        self.assertTrue(should_enable_cleanup_by_default(CLEANUP_MIN_RAM_GB))


class ModelSpecTests(unittest.TestCase):
    """Decision D3 requires an Apache-2.0 model of about 3B parameters."""

    def test_spec_is_apache_licensed_and_about_two_gigabytes(self):
        self.assertIsInstance(CLEANUP_MODEL_SPEC, ModelSpec)
        self.assertEqual(CLEANUP_MODEL_SPEC.engine, "llama_cleanup")
        self.assertEqual(CLEANUP_MODEL_SPEC.license, "Apache-2.0")
        self.assertIn("huggingface.co", CLEANUP_MODEL_SPEC.source)
        self.assertLess(CLEANUP_MODEL_SPEC.size_bytes, 3 * 1000**3)
        self.assertGreater(CLEANUP_MODEL_SPEC.size_bytes, 1 * 1000**3)

    def test_spec_is_a_single_verifiable_gguf(self):
        self.assertEqual(len(CLEANUP_MODEL_SPEC.files), 1)
        item = CLEANUP_MODEL_SPEC.files[0]
        self.assertTrue(item.name.endswith(".gguf"))
        self.assertIn("Q4_K_M", item.name)
        # A blank digest would be honest ignorance the store refuses to
        # certify; this one was read from Hugging Face LFS blob metadata.
        self.assertRegex(item.sha256, r"^[0-9a-f]{64}$")
        self.assertTrue(item.url.startswith("https://huggingface.co/"))
        self.assertIn(item.name, item.url)

    def test_spec_composes_into_the_store_catalog(self):
        # The app wires it as CATALOG + (CLEANUP_MODEL_SPEC,); the store must
        # accept it without engines/model_store.py knowing about cleanup.
        store = ModelStore(root=Path("/nonexistent"), catalog=(CLEANUP_MODEL_SPEC,))
        self.assertIs(store.spec(CLEANUP_MODEL_SPEC.id), CLEANUP_MODEL_SPEC)
        self.assertEqual(
            store.engine_model_path(CLEANUP_MODEL_SPEC.id).name,
            CLEANUP_MODEL_SPEC.files[0].name,
        )


class BinaryResolutionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_environment_override_wins(self):
        binary = self.tmp / BINARY_NAME
        binary.write_text("#!/bin/sh\n")
        with patch.dict("os.environ", {BINARY_ENV_VAR: str(binary)}):
            self.assertEqual(resolve_llama_server_binary(), binary)

    def test_missing_binary_names_the_fetch_script_and_the_env_var(self):
        with patch.dict("os.environ", {BINARY_ENV_VAR: str(self.tmp / "absent")}), patch(
            "cleanup.llama_server._repo_root", return_value=self.tmp
        ):
            with self.assertRaises(EngineUnavailableError) as ctx:
                resolve_llama_server_binary()

        message = str(ctx.exception)
        self.assertIn("fetch_llama.sh", message)
        self.assertIn(BINARY_ENV_VAR, message)

    def test_explicit_binary_that_does_not_exist_fails_before_spawning(self):
        spawn = FakeSpawn()
        server = LlamaServer(
            model_path=self.tmp / "model.gguf",
            binary=self.tmp / "absent",
            spawn=spawn,
        )

        with self.assertRaises(EngineUnavailableError):
            server.start()

        self.assertEqual(spawn.calls, [])


class PinnedTagTests(unittest.TestCase):
    def test_module_and_fetch_script_pin_the_same_llama_cpp_tag(self):
        script = (
            Path(__file__).resolve().parent.parent / "scripts" / "tools" / "fetch_llama.sh"
        ).read_text()
        match = re.search(r'LLAMA_CPP_TAG="\$\{LLAMA_CPP_TAG:-([^}"]+)\}"', script)
        self.assertIsNotNone(match, "fetch_llama.sh must pin LLAMA_CPP_TAG")
        self.assertEqual(match.group(1), LLAMA_CPP_TAG)

    def test_fetch_script_builds_the_server_target_into_vendor_llamacpp(self):
        script = (
            Path(__file__).resolve().parent.parent / "scripts" / "tools" / "fetch_llama.sh"
        ).read_text()
        self.assertIn("--target llama-server", script)
        self.assertIn("vendor/llamacpp", script)
        self.assertIn("-DGGML_METAL=ON", script)


if __name__ == "__main__":
    unittest.main()
