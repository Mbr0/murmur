"""Tests for the Murmur Cloud proxy client.

No real network: the "proxy" is a local ``http.server`` on loopback that
implements the two routes the engine uses and records what it was sent.
Sleeps and the clock are injected, so retry tests run instantly.
"""

import json
import logging
import threading
import unittest
import urllib.error
import urllib.request
import wave
from collections import deque
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from engines import EngineError, EngineNotLoadedError, Hints, create_engine
from engines.cloud import (
    ENGINE_CLASS,
    ENGINE_ID,
    MAX_ATTEMPTS,
    MODEL_ID,
    RESPONSE_FORMAT,
    RETRY_AFTER_CAP_S,
    TRANSCRIPTIONS_PATH,
    USAGE_PATH,
    CloudAllowanceExhausted,
    CloudAuthError,
    CloudEngine,
    Usage,
    fetch_usage,
)

LEASE = "eyJhbGciOiJFZERTQSJ9.lease-token-do-not-log"

CANNED_RESPONSE = {
    "task": "transcribe",
    "language": "fr",
    "duration": 2.5,
    "text": " Murmur écrit avec moi. ",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.2, "text": " Murmur "},
        {"id": 1, "start": 1.2, "end": 2.5, "text": " écrit avec moi. "},
    ],
}

CANNED_USAGE = {
    "minutes_used": 42.5,
    "minutes_allowance": 300.0,
    "words": 9120,
    "period_end": "2026-10-01T00:00:00Z",
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


def _write_wav(path: Path, seconds: float, framerate: int = 16000, sampwidth: int = 2) -> Path:
    """Write a silent WAV of ``seconds``. A low framerate keeps long clips tiny."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(sampwidth)
        handle.setframerate(framerate)
        handle.writeframes(b"\x00" * (sampwidth * int(seconds * framerate)))
    return path


class _Recorder:
    """Shared state between a test and its handler class."""

    def __init__(self):
        self.requests = []
        self.usage_requests = []
        #: Scripted statuses for consecutive POSTs; 200 once exhausted.
        self.statuses = deque()
        self.error_body = b'{"error": {"message": "upstream said no"}}'
        #: Overrides the canned 200 body for the transcription route.
        self.transcribe_body = None
        self.usage_status = 200
        self.usage_body = json.dumps(CANNED_USAGE).encode("utf-8")
        #: Sent as ``Retry-After`` on every non-200 answer when set.
        self.retry_after = None
        #: When set, the transcription route answers 302 to this location.
        self.redirect_to = None

    def next_status(self) -> int:
        return self.statuses.popleft() if self.statuses else 200


class _Handler(BaseHTTPRequestHandler):
    recorder: _Recorder = None  # set per-server

    def log_message(self, *args):  # silence the test output
        pass

    def _send(self, status, payload: bytes, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if status != 200 and self.recorder.retry_after is not None:
            self.send_header("Retry-After", str(self.recorder.retry_after))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path != USAGE_PATH:
            self._send(404, b'{"error": {"message": "no such route"}}')
            return
        self.recorder.usage_requests.append({"authorization": self.headers.get("Authorization")})
        status = self.recorder.usage_status
        payload = self.recorder.usage_body if status == 200 else self.recorder.error_body
        self._send(status, payload)

    def do_POST(self):
        if self.path != TRANSCRIPTIONS_PATH:
            self._send(404, b'{"error": {"message": "no such route"}}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.recorder.requests.append(
            {
                "authorization": self.headers.get("Authorization"),
                "idempotency_key": self.headers.get("Idempotency-Key"),
                "fields": _parse_multipart(body, self.headers.get("Content-Type", "")),
            }
        )
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
        body = self.recorder.transcribe_body
        self._send(200, body if body is not None else json.dumps(CANNED_RESPONSE).encode("utf-8"))


class FakeSleep:
    """Records the backoff delays instead of waiting for them."""

    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


class FakeClock:
    """Monotonic clock that advances one second per read."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 1.0
        return self.now


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

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.wav = _write_wav(self.tmp / "clip.wav", 2.0)
        self.sleep = FakeSleep()

        # Retries log a warning; keep it out of the suite output. assertLogs
        # installs its own handler, so LogHygieneTests still sees the records.
        cloud_logger = logging.getLogger("engines.cloud")
        cloud_logger.addHandler(logging.NullHandler())
        propagated, cloud_logger.propagate = cloud_logger.propagate, False
        self.addCleanup(setattr, cloud_logger, "propagate", propagated)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def make_engine(self, loaded=True, **overrides):
        kwargs = dict(
            base_url=self.base_url,
            lease_provider=lambda: LEASE,
            http_open=urllib.request.urlopen,
            clock=FakeClock(),
            sleep=self.sleep,
        )
        kwargs.update(overrides)
        engine = CloudEngine(**kwargs)
        if loaded:
            engine.load()
        self.addCleanup(engine.unload)
        return engine


class LifecycleTests(ProxyBackedTestCase):
    def test_load_validates_configuration_without_touching_the_network(self):
        engine = self.make_engine(loaded=False)
        self.assertFalse(engine.is_loaded)

        engine.load()

        self.assertTrue(engine.is_loaded)
        self.assertEqual(self.recorder.requests, [])
        self.assertEqual(self.recorder.usage_requests, [])

    def test_load_does_not_ask_for_a_lease(self):
        # Fetching a lease can hit the Keychain or the network; load() must not.
        calls = []

        def provider():
            calls.append(1)
            return LEASE

        self.make_engine(lease_provider=provider)
        self.assertEqual(calls, [])

    def test_load_is_idempotent_and_unload_is_repeatable(self):
        engine = self.make_engine()
        engine.load()
        self.assertTrue(engine.is_loaded)
        engine.unload()
        engine.unload()
        self.assertFalse(engine.is_loaded)

    def test_construction_rejects_an_empty_or_non_https_base_url(self):
        # The lease is a bearer token, so the endpoint is refused before the
        # engine exists rather than at load() — and plain http is refused too.
        for bad in ("", "   ", "ftp://proxy.invalid", "proxy.invalid", "http://proxy.invalid"):
            with self.subTest(base_url=bad):
                with self.assertRaises(ValueError):
                    CloudEngine(base_url=bad, lease_provider=lambda: LEASE)

    def test_an_https_base_url_is_accepted(self):
        engine = CloudEngine(base_url="https://proxy.boske.test/", lease_provider=lambda: LEASE)
        engine.load()
        self.assertTrue(engine.is_loaded)
        self.assertIn("https://proxy.boske.test", engine.runtime_summary())

    def test_transcribe_before_load_raises_not_loaded(self):
        engine = self.make_engine(loaded=False)
        with self.assertRaises(EngineNotLoadedError):
            engine.transcribe(self.wav)
        self.assertEqual(self.recorder.requests, [])

    def test_info_describes_the_hosted_engine(self):
        info = self.make_engine().info()
        self.assertEqual(info.id, ENGINE_ID)
        self.assertEqual(info.id, "cloud")
        self.assertEqual(info.name, "Murmur Cloud")
        self.assertEqual(info.model_id, MODEL_ID)
        self.assertFalse(info.requires_apple_silicon)
        self.assertFalse(info.supports_streaming)
        self.assertTrue(info.supports_hints)
        self.assertEqual(
            info.languages, ("auto", "en", "fr", "nl", "de", "es", "it", "pt")
        )

    def test_registry_creates_the_cloud_engine(self):
        engine = create_engine(
            "cloud", base_url=self.base_url, lease_provider=lambda: LEASE
        )
        self.assertIsInstance(engine, CloudEngine)
        self.assertIs(ENGINE_CLASS, CloudEngine)


class RequestShapeTests(ProxyBackedTestCase):
    def test_posts_multipart_with_the_lease_as_bearer(self):
        engine = self.make_engine()

        engine.transcribe(self.wav, language="fr")

        self.assertEqual(len(self.recorder.requests), 1)
        request = self.recorder.requests[0]
        self.assertEqual(request["authorization"], f"Bearer {LEASE}")
        fields = request["fields"]
        self.assertEqual(fields["model"], MODEL_ID)
        self.assertEqual(fields["model"], "voxtral-mini-latest")
        self.assertEqual(fields["response_format"], RESPONSE_FORMAT)
        self.assertEqual(fields["language"], "fr")
        self.assertEqual(fields["file"], self.wav.read_bytes())

    def test_language_none_or_auto_omits_the_language_field(self):
        for language in (None, "auto"):
            with self.subTest(language=language):
                self.recorder.requests.clear()
                self.make_engine().transcribe(self.wav, language=language)
                self.assertNotIn("language", self.recorder.requests[0]["fields"])

    def test_hints_become_the_prompt_field_and_are_reported_applied(self):
        hints = Hints(vocabulary=("Boske", "Murmur"), initial_prompt="Notes de réunion")

        transcript = self.make_engine().transcribe(self.wav, hints=hints)

        self.assertEqual(
            self.recorder.requests[0]["fields"]["prompt"], hints.as_prompt_text()
        )
        self.assertTrue(transcript.hints_applied)

    def test_no_hints_means_no_prompt_and_no_claim_either_way(self):
        transcript = self.make_engine().transcribe(self.wav)
        self.assertNotIn("prompt", self.recorder.requests[0]["fields"])
        self.assertIsNone(transcript.hints_applied)

    def test_empty_hints_send_no_prompt(self):
        transcript = self.make_engine().transcribe(self.wav, hints=Hints())
        self.assertNotIn("prompt", self.recorder.requests[0]["fields"])
        self.assertIsNone(transcript.hints_applied)

    def test_base_url_trailing_slash_does_not_double_the_separator(self):
        engine = self.make_engine(base_url=self.base_url + "/")
        engine.transcribe(self.wav)
        self.assertEqual(len(self.recorder.requests), 1)


class ResponseParsingTests(ProxyBackedTestCase):
    def test_verbose_json_becomes_a_transcript(self):
        transcript = self.make_engine().transcribe(self.wav)

        self.assertEqual(transcript.text, "Murmur écrit avec moi.")
        self.assertEqual(transcript.language, "fr")
        self.assertEqual(transcript.duration_s, 2.5)
        self.assertEqual(transcript.engine_id, "cloud")
        self.assertEqual(len(transcript.segments), 2)
        self.assertEqual(transcript.segments[1].text, "écrit avec moi.")
        self.assertEqual(transcript.segments[1].start, 1.2)

    def test_a_body_that_is_not_json_raises_engine_error(self):
        self.recorder.transcribe_body = b"<html>gateway</html>"
        with self.assertRaises(EngineError):
            self.make_engine().transcribe(self.wav)

    def test_a_200_carrying_an_error_object_raises(self):
        self.recorder.transcribe_body = json.dumps(
            {"error": {"message": "model unavailable"}}
        ).encode("utf-8")
        with self.assertRaises(EngineError):
            self.make_engine().transcribe(self.wav)

    def test_a_200_carrying_the_allowance_code_still_maps_to_the_fallback(self):
        self.recorder.transcribe_body = json.dumps(
            {"error": {"code": "allowance_exhausted"}}
        ).encode("utf-8")
        with self.assertRaises(CloudAllowanceExhausted):
            self.make_engine().transcribe(self.wav)


class RetryTests(ProxyBackedTestCase):
    def test_two_server_errors_then_success(self):
        self.recorder.statuses.extend([500, 503])

        transcript = self.make_engine().transcribe(self.wav)

        self.assertEqual(transcript.text, "Murmur écrit avec moi.")
        self.assertEqual(len(self.recorder.requests), 3)
        self.assertEqual(self.sleep.calls, [0.5, 1.0])

    def test_gives_up_after_the_attempt_cap(self):
        self.recorder.statuses.extend([500, 500, 500])

        with self.assertRaises(EngineError) as ctx:
            self.make_engine().transcribe(self.wav)

        self.assertNotIsInstance(ctx.exception, CloudAuthError)
        self.assertEqual(len(self.recorder.requests), MAX_ATTEMPTS)
        self.assertEqual(self.sleep.calls, [0.5, 1.0])

    def test_connection_errors_are_retried(self):
        failures = []

        def flaky_open(request, timeout=None):
            if len(failures) < 2:
                failures.append(1)
                raise urllib.error.URLError("connection refused")
            return urllib.request.urlopen(request, timeout=timeout)

        transcript = self.make_engine(http_open=flaky_open).transcribe(self.wav)

        self.assertEqual(transcript.text, "Murmur écrit avec moi.")
        self.assertEqual(len(self.recorder.requests), 1)
        self.assertEqual(self.sleep.calls, [0.5, 1.0])

    def test_timeouts_are_retried(self):
        failures = []

        def slow_open(request, timeout=None):
            if not failures:
                failures.append(1)
                raise TimeoutError("timed out")
            return urllib.request.urlopen(request, timeout=timeout)

        self.make_engine(http_open=slow_open).transcribe(self.wav)
        self.assertEqual(self.sleep.calls, [0.5])

    def test_client_errors_are_not_retried(self):
        self.recorder.statuses.append(400)

        with self.assertRaises(EngineError):
            self.make_engine().transcribe(self.wav)

        self.assertEqual(len(self.recorder.requests), 1)
        self.assertEqual(self.sleep.calls, [])


class ErrorMappingTests(ProxyBackedTestCase):
    def test_missing_lease_fails_before_any_request(self):
        engine = self.make_engine(lease_provider=lambda: None)

        with self.assertRaises(CloudAuthError):
            engine.transcribe(self.wav)

        self.assertEqual(self.recorder.requests, [])

    def test_a_lease_provider_that_raises_is_an_auth_error(self):
        def provider():
            raise RuntimeError("keychain locked")

        engine = self.make_engine(lease_provider=provider)
        with self.assertRaises(CloudAuthError):
            engine.transcribe(self.wav)
        self.assertEqual(self.recorder.requests, [])

    def test_a_lease_provider_failure_reports_its_type_not_its_message(self):
        # A provider exception may quote a token or a provider body; only the
        # exception's class name is safe to repeat back to the user.
        def provider():
            raise RuntimeError(f"refresh rejected token {LEASE}")

        engine = self.make_engine(lease_provider=provider)
        with self.assertRaises(CloudAuthError) as ctx:
            engine.transcribe(self.wav)
        message = str(ctx.exception)
        self.assertIn("RuntimeError", message)
        self.assertNotIn(LEASE, message)
        self.assertNotIn("refresh rejected token", message)

    def test_401_is_an_auth_error_and_is_not_retried(self):
        self.recorder.statuses.append(401)

        with self.assertRaises(CloudAuthError):
            self.make_engine().transcribe(self.wav)

        self.assertEqual(len(self.recorder.requests), 1)
        self.assertEqual(self.sleep.calls, [])

    def test_a_plain_429_is_a_retryable_rate_limit_not_the_allowance(self):
        # Only the ``allowance_exhausted`` code means "your minutes are spent".
        # A bare 429 is the proxy asking us to slow down; treating it as the
        # allowance would silently drop a paying user onto the local engine.
        self.recorder.statuses.extend([429, 429])

        transcript = self.make_engine().transcribe(self.wav)

        self.assertEqual(transcript.text, "Murmur écrit avec moi.")
        self.assertEqual(len(self.recorder.requests), 3)

    def test_a_429_that_never_clears_ends_as_a_plain_rate_limit_error(self):
        self.recorder.statuses.extend([429, 429, 429])

        with self.assertRaises(EngineError) as ctx:
            self.make_engine().transcribe(self.wav)

        self.assertNotIsInstance(ctx.exception, CloudAllowanceExhausted)
        self.assertIn("rate limited", str(ctx.exception))
        self.assertEqual(len(self.recorder.requests), MAX_ATTEMPTS)

    def test_retry_after_seconds_replaces_the_backoff_and_is_capped(self):
        self.recorder.statuses.append(429)
        self.recorder.retry_after = 3
        self.make_engine().transcribe(self.wav)
        self.assertEqual(self.sleep.calls, [3.0])

        self.recorder.requests.clear()
        self.sleep.calls.clear()
        self.recorder.statuses.append(429)
        self.recorder.retry_after = 900
        self.make_engine().transcribe(self.wav)
        self.assertEqual(self.sleep.calls, [RETRY_AFTER_CAP_S])

    def test_retry_after_as_an_http_date_is_understood(self):
        self.recorder.statuses.append(429)
        self.recorder.retry_after = format_datetime(
            datetime.now(timezone.utc) + timedelta(seconds=4), usegmt=True
        )

        self.make_engine().transcribe(self.wav)

        self.assertEqual(len(self.sleep.calls), 1)
        self.assertGreater(self.sleep.calls[0], 0.0)
        self.assertLessEqual(self.sleep.calls[0], RETRY_AFTER_CAP_S)

    def test_the_allowance_code_at_429_is_still_the_allowance(self):
        self.recorder.statuses.append(429)
        self.recorder.error_body = json.dumps(
            {"error": {"code": "allowance_exhausted"}}
        ).encode("utf-8")

        with self.assertRaises(CloudAllowanceExhausted) as ctx:
            self.make_engine().transcribe(self.wav)

        self.assertNotIn("429", str(ctx.exception))
        self.assertEqual(len(self.recorder.requests), 1)
        self.assertEqual(self.sleep.calls, [])

    def test_allowance_exhausted_code_in_the_body_wins_over_the_status(self):
        self.recorder.statuses.append(403)
        self.recorder.error_body = json.dumps(
            {"error": {"code": "allowance_exhausted", "message": "out of minutes"}}
        ).encode("utf-8")

        with self.assertRaises(CloudAllowanceExhausted):
            self.make_engine().transcribe(self.wav)

    def test_flat_allowance_exhausted_code_is_recognised_too(self):
        self.recorder.statuses.append(402)
        self.recorder.error_body = json.dumps(
            {"code": "allowance_exhausted", "message": "out of minutes"}
        ).encode("utf-8")

        with self.assertRaises(CloudAllowanceExhausted):
            self.make_engine().transcribe(self.wav)

    def test_allowance_exhausted_is_an_engine_error_subclass(self):
        for error in (CloudAuthError, CloudAllowanceExhausted):
            self.assertTrue(issubclass(error, EngineError))


class DurationCapTests(ProxyBackedTestCase):
    def test_a_clip_over_the_cap_is_refused_before_the_upload(self):
        long_wav = _write_wav(self.tmp / "long.wav", 3601, framerate=10, sampwidth=1)

        with self.assertRaises(EngineError) as ctx:
            self.make_engine().transcribe(long_wav)

        self.assertIn("60", str(ctx.exception))
        self.assertEqual(self.recorder.requests, [])

    def test_a_clip_at_the_cap_is_uploaded(self):
        at_cap = _write_wav(self.tmp / "at_cap.wav", 3600, framerate=10, sampwidth=1)
        self.make_engine().transcribe(at_cap)
        self.assertEqual(len(self.recorder.requests), 1)

    def test_the_cap_is_configurable(self):
        clip = _write_wav(self.tmp / "five.wav", 300, framerate=10, sampwidth=1)
        with self.assertRaises(EngineError):
            self.make_engine(max_minutes=1).transcribe(clip)
        self.assertEqual(self.recorder.requests, [])

    def test_a_file_that_is_not_a_wav_fails_fast(self):
        broken = self.tmp / "broken.wav"
        broken.write_bytes(b"RIFF....WAVEfmt ")
        with self.assertRaises(EngineError):
            self.make_engine().transcribe(broken)
        self.assertEqual(self.recorder.requests, [])

    def test_a_missing_file_fails_fast(self):
        with self.assertRaises(EngineError):
            self.make_engine().transcribe(self.tmp / "absent.wav")
        self.assertEqual(self.recorder.requests, [])


class UsageEndpointTests(ProxyBackedTestCase):
    def test_usage_is_parsed_from_the_voice_usage_route(self):
        usage = fetch_usage(self.base_url, LEASE, http_open=urllib.request.urlopen)

        self.assertIsInstance(usage, Usage)
        self.assertEqual(usage.minutes_used, 42.5)
        self.assertEqual(usage.minutes_allowance, 300.0)
        self.assertEqual(usage.words, 9120)
        self.assertEqual(usage.period_end, "2026-10-01T00:00:00Z")
        self.assertAlmostEqual(usage.fraction_used, 0.1416666, places=5)
        self.assertEqual(
            self.recorder.usage_requests[0]["authorization"], f"Bearer {LEASE}"
        )

    def test_an_unknown_allowance_is_not_a_spent_allowance(self):
        # None means "we do not know", which is not the same as "nothing left".
        usage = Usage(minutes_used=0.0, minutes_allowance=0.0, words=0, period_end=None)
        self.assertIsNone(usage.fraction_used)
        negative = Usage(minutes_used=1.0, minutes_allowance=-5.0, words=0, period_end=None)
        self.assertIsNone(negative.fraction_used)

    def test_a_usage_body_without_an_allowance_is_refused(self):
        self.recorder.usage_body = json.dumps({"minutes_used": 3}).encode("utf-8")
        with self.assertRaises(EngineError) as ctx:
            fetch_usage(self.base_url, LEASE, http_open=urllib.request.urlopen)
        self.assertIn("usage response missing minutes_allowance", str(ctx.exception))

    def test_a_usage_body_without_minutes_used_is_refused(self):
        self.recorder.usage_body = json.dumps({"minutes_allowance": 300}).encode("utf-8")
        with self.assertRaises(EngineError) as ctx:
            fetch_usage(self.base_url, LEASE, http_open=urllib.request.urlopen)
        self.assertIn("usage response missing minutes_used", str(ctx.exception))

    def test_the_truly_optional_fields_still_default(self):
        self.recorder.usage_body = json.dumps(
            {"minutes_used": 3, "minutes_allowance": 300}
        ).encode("utf-8")
        usage = fetch_usage(self.base_url, LEASE, http_open=urllib.request.urlopen)
        self.assertEqual(usage.minutes_used, 3.0)
        self.assertEqual(usage.minutes_allowance, 300.0)
        self.assertEqual(usage.words, 0)
        self.assertIsNone(usage.period_end)

    def test_usage_without_a_lease_is_an_auth_error(self):
        with self.assertRaises(CloudAuthError):
            fetch_usage(self.base_url, None, http_open=urllib.request.urlopen)
        self.assertEqual(self.recorder.usage_requests, [])

    def test_usage_over_plain_http_to_a_real_host_is_refused(self):
        with self.assertRaises(ValueError):
            fetch_usage("http://proxy.invalid", LEASE, http_open=urllib.request.urlopen)
        self.assertEqual(self.recorder.usage_requests, [])

    def test_usage_401_is_an_auth_error(self):
        self.recorder.usage_status = 401
        with self.assertRaises(CloudAuthError):
            fetch_usage(self.base_url, LEASE, http_open=urllib.request.urlopen)

    def test_usage_429_is_a_rate_limit_not_an_allowance_error(self):
        self.recorder.usage_status = 429
        with self.assertRaises(EngineError) as ctx:
            fetch_usage(self.base_url, LEASE, http_open=urllib.request.urlopen)
        self.assertNotIsInstance(ctx.exception, CloudAllowanceExhausted)
        self.assertIn("rate limited", str(ctx.exception))

    def test_usage_carrying_the_allowance_code_is_an_allowance_error(self):
        self.recorder.usage_status = 429
        self.recorder.error_body = json.dumps({"code": "allowance_exhausted"}).encode("utf-8")
        with self.assertRaises(CloudAllowanceExhausted):
            fetch_usage(self.base_url, LEASE, http_open=urllib.request.urlopen)


class IdempotencyTests(ProxyBackedTestCase):
    """A retried upload must not be billed twice."""

    def test_every_attempt_of_one_call_carries_the_same_idempotency_key(self):
        self.recorder.statuses.extend([500, 503])

        self.make_engine().transcribe(self.wav)

        keys = [request["idempotency_key"] for request in self.recorder.requests]
        self.assertEqual(len(keys), 3)
        self.assertTrue(all(keys))
        self.assertEqual(len(set(keys)), 1)

    def test_a_second_transcription_gets_a_fresh_key(self):
        engine = self.make_engine()
        engine.transcribe(self.wav)
        engine.transcribe(self.wav)
        first, second = (request["idempotency_key"] for request in self.recorder.requests)
        self.assertNotEqual(first, second)

    def test_a_timeout_retry_reuses_the_key_of_the_call_that_timed_out(self):
        # The lost response may have been a success: the proxy needs the key to
        # recognise the second POST as the same clip and not meter it again.
        failures = []

        def slow_open(request, timeout=None):
            if not failures:
                failures.append(1)
                raise TimeoutError("timed out")
            return urllib.request.urlopen(request, timeout=timeout)

        self.make_engine(http_open=slow_open).transcribe(self.wav)
        self.assertEqual(len(self.recorder.requests), 1)
        self.assertTrue(self.recorder.requests[0]["idempotency_key"])


class RedirectTests(ProxyBackedTestCase):
    def test_a_redirect_to_another_host_is_refused_before_the_lease_is_re_sent(self):
        self.recorder.redirect_to = "http://localhost:1/v1/audio/transcriptions"

        with self.assertRaises(EngineError) as ctx:
            self.make_engine().transcribe(self.wav)

        self.assertIn("redirect refused", str(ctx.exception))
        self.assertEqual(len(self.recorder.requests), 1)


class LogHygieneTests(ProxyBackedTestCase):
    def test_neither_the_lease_nor_the_transcript_reaches_the_log(self):
        self.recorder.statuses.append(500)
        engine = self.make_engine()

        with self.assertLogs("engines.cloud", level=logging.DEBUG) as captured:
            transcript = engine.transcribe(self.wav, hints=Hints(vocabulary=("Boske",)))

        blob = "\n".join(captured.output)
        self.assertNotIn(LEASE, blob)
        self.assertNotIn("lease-token-do-not-log", blob)
        self.assertNotIn(transcript.text, blob)
        self.assertNotIn("écrit avec moi", blob)
        self.assertNotIn("Boske", blob)


if __name__ == "__main__":
    unittest.main()
