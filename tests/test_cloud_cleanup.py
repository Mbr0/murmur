"""Tests for cleanup through the Boske proxy chat endpoint.

No real network: the "proxy" is a local ``http.server`` on loopback that
implements the one route the client uses and records what it was sent — the
same fixture shape as ``tests/test_engine_cloud.py``.

The contract under test is the *same* as the local client's
(:class:`cleanup.llama_server.CleanupClient`), because the wiring swaps one for
the other: everything transient degrades to a visible skip carrying the
original text, and only the two cases the wiring must act on — a bad lease and
a spent allowance — raise.
"""

import json
import logging
import threading
import unittest
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cleanup.cloud_cleanup import (
    CHAT_PATH,
    CONFIG_CLOUD_CLEANUP_KEY,
    CONTEXT_TOKENS,
    MODEL_ID,
    RATE_LIMITED_REASON,
    CloudCleanupClient,
    should_use_cloud_cleanup,
    words_in,
)
from cleanup.llama_server import CleanupResult, max_tokens_for_text, timeout_for_text
from cleanup.modes import render_system_prompt
from engines.cloud import CloudAllowanceExhausted, CloudAuthError

LEASE = "eyJhbGciOiJFZERTQSJ9.cleanup-lease-do-not-log"

SYSTEM_PROMPT = "Clean up the dictation. Keep the meaning."

#: Deliberately messy, and deliberately distinctive: the log-hygiene test looks
#: for these exact words in every record the client emits.
TRANSCRIPT = "um so the boske proxy uh cleans this up right"

CLEANED = " The Boske proxy cleans this up. "


def _completion(content: str) -> bytes:
    """A chat completion body in the shape the proxy answers with."""
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        }
    ).encode("utf-8")


class _Recorder:
    """Shared state between a test and its handler class."""

    def __init__(self):
        self.requests = []
        #: Scripted statuses for consecutive POSTs; 200 once exhausted.
        self.statuses = deque()
        #: Body returned with a non-200 status.
        self.error_body = b'{"error": {"message": "upstream said no"}}'
        #: Overrides the canned 200 body.
        self.body = None
        #: When set, the chat route answers 302 pointing here.
        self.redirect_to = None

    def next_status(self) -> int:
        return self.statuses.popleft() if self.statuses else 200


class _Handler(BaseHTTPRequestHandler):
    recorder: _Recorder = None  # set per-server

    def log_message(self, *args):  # silence the test output
        pass

    def _send(self, status, payload: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = None
        self.recorder.requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "idempotency_key": self.headers.get("Idempotency-Key"),
                "content_type": self.headers.get("Content-Type"),
                "body": body,
            }
        )
        if self.path != CHAT_PATH:
            self._send(404, b'{"error": {"message": "no such route"}}')
            return
        if self.recorder.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.recorder.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status = self.recorder.next_status()
        if status != 200:
            self._send(status, self.recorder.error_body)
            return
        self._send(200, self.recorder.body if self.recorder.body is not None else _completion(CLEANED))


class ProxyBackedTestCase(unittest.TestCase):
    """Runs a real loopback HTTP server standing in for the Boske proxy."""

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

        # Skips log at INFO; keep them out of the suite output. assertLogs
        # installs its own handler, so LogHygieneTests still sees the records.
        log = logging.getLogger("cleanup.cloud_cleanup")
        log.addHandler(logging.NullHandler())
        propagated, log.propagate = log.propagate, False
        self.addCleanup(setattr, log, "propagate", propagated)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def make_client(self, **overrides) -> CloudCleanupClient:
        kwargs = dict(
            base_url=self.base_url,
            lease_provider=lambda: LEASE,
            http_open=urllib.request.urlopen,
        )
        kwargs.update(overrides)
        return CloudCleanupClient(**kwargs)

    def clean(self, text: str = TRANSCRIPT, **overrides) -> CleanupResult:
        return self.make_client(**overrides).cleanup(text, SYSTEM_PROMPT)

    @property
    def last_request(self) -> dict:
        self.assertTrue(self.recorder.requests, "the proxy was never called")
        return self.recorder.requests[-1]


class ConstructionTests(ProxyBackedTestCase):
    def test_a_plain_http_base_url_is_refused_before_a_client_exists(self):
        # The lease is a bearer token, so a mistyped endpoint must fail at
        # construction rather than on the first dictation.
        for bad in ("", "   ", "http://proxy.invalid", "proxy.invalid", "ftp://proxy.invalid"):
            with self.subTest(base_url=bad):
                with self.assertRaises(ValueError):
                    CloudCleanupClient(base_url=bad, lease_provider=lambda: LEASE)

    def test_an_https_base_url_is_accepted(self):
        client = CloudCleanupClient(
            base_url="https://proxy.boske.test/", lease_provider=lambda: LEASE
        )
        self.assertEqual(client.chat_url, "https://proxy.boske.test" + CHAT_PATH)

    def test_the_model_is_a_documented_constant(self):
        self.assertEqual(MODEL_ID, "ministral-3b-latest")
        self.assertEqual(self.make_client().model, MODEL_ID)
        self.assertEqual(self.make_client(model="other-model").model, "other-model")


class RequestShapeTests(ProxyBackedTestCase):
    def test_the_request_carries_the_route_lease_and_an_idempotency_key(self):
        self.clean()

        request = self.last_request
        self.assertEqual(request["path"], CHAT_PATH)
        self.assertEqual(request["authorization"], f"Bearer {LEASE}")
        self.assertIn("application/json", request["content_type"])
        self.assertTrue(request["idempotency_key"])

    def test_each_call_gets_its_own_idempotency_key(self):
        client = self.make_client()
        client.cleanup(TRANSCRIPT, SYSTEM_PROMPT)
        client.cleanup(TRANSCRIPT, SYSTEM_PROMPT)

        keys = {request["idempotency_key"] for request in self.recorder.requests}
        self.assertEqual(len(keys), 2)

    def test_the_body_is_the_openai_chat_shape(self):
        self.clean()

        body = self.last_request["body"]
        self.assertEqual(body["model"], MODEL_ID)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["temperature"], 0.1)
        self.assertEqual(
            body["messages"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": TRANSCRIPT},
            ],
        )

    def test_max_tokens_comes_from_the_shared_budget(self):
        self.clean()

        self.assertEqual(
            self.last_request["body"]["max_tokens"],
            max_tokens_for_text(TRANSCRIPT, CONTEXT_TOKENS),
        )

    def test_the_temperature_is_the_callers(self):
        self.make_client().cleanup(TRANSCRIPT, SYSTEM_PROMPT, temperature=0.4)

        self.assertEqual(self.last_request["body"]["temperature"], 0.4)

    def test_a_wave_two_mode_prompt_travels_verbatim(self):
        # The cloud path must send exactly the prompts the local path sends.
        prompt = render_system_prompt("mail", "formal", "en", ("Boske",))

        self.make_client().cleanup(TRANSCRIPT, prompt)

        self.assertEqual(self.last_request["body"]["messages"][0]["content"], prompt)

    def test_the_timeout_is_the_shared_policy(self):
        seen = []

        def recording_open(request, timeout=None):
            seen.append(timeout)
            return urllib.request.urlopen(request, timeout=timeout)

        self.clean(http_open=recording_open)

        self.assertEqual(seen, [timeout_for_text(TRANSCRIPT)])


class HappyPathTests(ProxyBackedTestCase):
    def test_the_cleaned_text_is_parsed_and_stripped(self):
        result = self.clean()

        self.assertEqual(result.text, CLEANED.strip())
        self.assertFalse(result.skipped)
        self.assertIsNone(result.reason)
        self.assertGreaterEqual(result.elapsed_s, 0.0)

    def test_the_clock_seam_measures_the_call(self):
        ticks = iter([10.0, 12.5])
        result = self.clean(clock=lambda: next(ticks))

        self.assertAlmostEqual(result.elapsed_s, 2.5)


class SkipTests(ProxyBackedTestCase):
    """Everything transient hands the original text back, visibly."""

    def assert_skipped(self, result: CleanupResult, contains: str = ""):
        self.assertTrue(result.skipped)
        self.assertEqual(result.text, TRANSCRIPT)
        self.assertTrue(result.reason)
        if contains:
            self.assertIn(contains, result.reason)

    def test_a_timeout_skips_rather_than_losing_the_transcript(self):
        def timing_out(request, timeout=None):
            raise TimeoutError("timed out")

        self.assert_skipped(self.clean(http_open=timing_out), "timed out")

    def test_a_timeout_wrapped_in_a_urlerror_also_skips(self):
        def timing_out(request, timeout=None):
            raise urllib.error.URLError(TimeoutError("timed out"))

        self.assert_skipped(self.clean(http_open=timing_out), "timed out")

    def test_an_unreachable_proxy_skips(self):
        def refused(request, timeout=None):
            raise urllib.error.URLError(ConnectionRefusedError("nope"))

        self.assert_skipped(self.clean(http_open=refused), "unreachable")

    def test_a_server_error_skips(self):
        for status in (500, 502, 503):
            with self.subTest(status=status):
                self.recorder.statuses.append(status)
                self.assert_skipped(self.clean(), str(status))

    def test_a_five_hundred_is_not_retried(self):
        self.recorder.statuses.append(500)

        self.clean()

        self.assertEqual(len(self.recorder.requests), 1)

    def test_an_empty_or_whitespace_reply_skips(self):
        for content in ("", "   \n\t "):
            with self.subTest(content=repr(content)):
                self.recorder.requests.clear()
                self.recorder.body = _completion(content)
                self.assert_skipped(self.clean(), "empty")

    def test_an_unparsable_body_skips(self):
        for body in (b"not json at all", b"[]", b'{"choices": []}', b'{"choices": [{}]}'):
            with self.subTest(body=body):
                self.recorder.requests.clear()
                self.recorder.body = body
                self.assert_skipped(self.clean(), "unreadable")

    def test_a_plain_rate_limit_skips_and_is_not_retried(self):
        self.recorder.statuses.append(429)

        result = self.clean()

        self.assert_skipped(result)
        self.assertEqual(result.reason, RATE_LIMITED_REASON)
        self.assertEqual(len(self.recorder.requests), 1)

    def test_a_client_error_skips(self):
        self.recorder.statuses.append(400)

        self.assert_skipped(self.clean(), "400")


class AuthTests(ProxyBackedTestCase):
    def test_a_missing_lease_raises_before_any_request(self):
        for provider in (lambda: None, lambda: ""):
            with self.subTest(provider=provider()):
                with self.assertRaises(CloudAuthError):
                    self.clean(lease_provider=provider)
                self.assertEqual(self.recorder.requests, [])

    def test_a_lease_provider_that_raises_is_an_auth_error(self):
        def boom():
            raise RuntimeError("keychain is locked: secret-value")

        with self.assertRaises(CloudAuthError) as ctx:
            self.clean(lease_provider=boom)

        self.assertEqual(self.recorder.requests, [])
        # Only the type; a provider is free to put a token in its message.
        self.assertIn("RuntimeError", str(ctx.exception))
        self.assertNotIn("secret-value", str(ctx.exception))

    def test_a_401_raises_so_the_wiring_can_re_link(self):
        self.recorder.statuses.append(401)

        with self.assertRaises(CloudAuthError):
            self.clean()


class AllowanceTests(ProxyBackedTestCase):
    """The one deliberate product fallback: cloud → local, with a notice."""

    ALLOWANCE_BODY = b'{"error": {"code": "allowance_exhausted", "message": "spent"}}'

    def test_the_allowance_code_raises_at_any_status(self):
        for status in (402, 429, 200):
            with self.subTest(status=status):
                self.recorder.requests.clear()
                self.recorder.error_body = self.ALLOWANCE_BODY
                self.recorder.body = self.ALLOWANCE_BODY
                self.recorder.statuses.append(status)
                with self.assertRaises(CloudAllowanceExhausted):
                    self.clean()

    def test_a_top_level_allowance_code_is_understood_too(self):
        self.recorder.error_body = b'{"code": "allowance_exhausted"}'
        self.recorder.statuses.append(429)

        with self.assertRaises(CloudAllowanceExhausted):
            self.clean()

    def test_the_allowance_message_never_mentions_an_http_status(self):
        self.recorder.error_body = self.ALLOWANCE_BODY
        self.recorder.statuses.append(429)

        with self.assertRaises(CloudAllowanceExhausted) as ctx:
            self.clean()

        message = str(ctx.exception)
        for noise in ("429", "HTTP", "402"):
            self.assertNotIn(noise, message)


class RedirectTests(ProxyBackedTestCase):
    def test_a_cross_host_redirect_is_refused_and_skips(self):
        # Refusing is what keeps the lease from reaching another host; skipping
        # is what keeps a misconfigured proxy from costing the user their text.
        self.recorder.redirect_to = "http://localhost:1" + CHAT_PATH

        result = self.clean()

        self.assertTrue(result.skipped)
        self.assertEqual(result.text, TRANSCRIPT)
        self.assertIn("redirect", result.reason)
        self.assertEqual(len(self.recorder.requests), 1)


class WordsInTests(unittest.TestCase):
    def test_words_are_whitespace_separated(self):
        self.assertEqual(words_in("one two three"), 3)
        self.assertEqual(words_in("  padded \n words\t here "), 3)

    def test_nothing_counts_as_nothing(self):
        for empty in ("", "   ", None):
            with self.subTest(text=empty):
                self.assertEqual(words_in(empty), 0)


class ShouldUseCloudCleanupTests(unittest.TestCase):
    """Pure decision: config on, cloud is the live route, and Pro allows it."""

    def decide(self, config, pro=True, cloud_active=True):
        return should_use_cloud_cleanup(
            config, pro_gate=lambda feature: pro, cloud_engine_active=cloud_active
        )

    def test_the_table(self):
        on = {CONFIG_CLOUD_CLEANUP_KEY: True}
        off = {CONFIG_CLOUD_CLEANUP_KEY: False}
        cases = [
            (on, True, True, True),
            (off, True, True, False),
            ({}, True, True, False),
            ({CONFIG_CLOUD_CLEANUP_KEY: None}, True, True, False),
            (on, False, True, False),
            (on, True, False, False),
            (on, False, False, False),
            (off, False, False, False),
        ]
        for config, pro, cloud_active, expected in cases:
            with self.subTest(config=config, pro=pro, cloud_active=cloud_active):
                self.assertIs(self.decide(config, pro, cloud_active), expected)

    def test_the_gate_is_asked_for_the_cleanup_feature(self):
        asked = []

        should_use_cloud_cleanup(
            {CONFIG_CLOUD_CLEANUP_KEY: True},
            pro_gate=lambda feature: asked.append(feature) or True,
            cloud_engine_active=True,
        )

        self.assertEqual(asked, ["cleanup"])

    def test_the_gate_is_not_consulted_when_the_setting_is_off(self):
        asked = []

        result = should_use_cloud_cleanup(
            {CONFIG_CLOUD_CLEANUP_KEY: False},
            pro_gate=lambda feature: asked.append(feature) or True,
            cloud_engine_active=True,
        )

        self.assertFalse(result)
        self.assertEqual(asked, [])

    def test_a_missing_config_is_not_a_crash(self):
        self.assertFalse(self.decide(None))


class LogHygieneTests(ProxyBackedTestCase):
    """No transcript text, no model output and no lease in any record."""

    def assert_clean(self, captured):
        blob = "\n".join(captured.output)
        for secret in (LEASE, "cleanup-lease-do-not-log", TRANSCRIPT, CLEANED.strip(), "boske"):
            self.assertNotIn(secret, blob)

    def test_a_successful_cleanup_logs_only_lengths_and_timings(self):
        with self.assertLogs("cleanup.cloud_cleanup", level=logging.DEBUG) as captured:
            self.clean()

        self.assert_clean(captured)

    def test_a_skip_logs_only_its_reason(self):
        self.recorder.statuses.append(500)

        with self.assertLogs("cleanup.cloud_cleanup", level=logging.DEBUG) as captured:
            result = self.clean()

        self.assert_clean(captured)
        # The reason is shown to the user, so it must not carry model output.
        self.assertNotIn(CLEANED.strip(), result.reason)

    def test_an_error_body_never_reaches_the_reason(self):
        self.recorder.error_body = b'{"error": {"message": "echo: um so the boske proxy"}}'
        self.recorder.statuses.append(500)

        result = self.clean()

        self.assertNotIn("boske", result.reason)
        self.assertNotIn("echo", result.reason)


if __name__ == "__main__":
    unittest.main()
