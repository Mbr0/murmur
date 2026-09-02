"""Tests for the own-key (BYOK) engine.

No network and no real API key: a loopback ``http.server`` stands in for both
``api.mistral.ai`` and ``api.openai.com``, and the engine is pointed at it with
``base_url``. The handler records the headers and multipart fields it received,
which is how the per-provider contract is asserted.
"""

import json
import logging
import socket
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from engines import (
    ENGINE_IDS,
    EngineError,
    EngineNotLoadedError,
    EngineUnavailableError,
    Hints,
    create_engine,
)
from engines.byok import (
    ENGINE_CLASS,
    ENGINE_ID,
    PROVIDERS,
    TRANSCRIPTIONS_PATH,
    ByokAuthError,
    ByokEngine,
    ByokRateLimited,
    provider_from_config,
)

#: A value that must never leak into an error message or a log record.
SECRET_KEY = "sk-murmur-test-0123456789-do-not-leak"

MISTRAL_RESPONSE = {
    "model": "voxtral-mini-latest",
    "text": " Murmur schrijft mee. ",
    "language": "nl",
    "segments": [
        {"text": " Murmur ", "start": 0.0, "end": 1.2, "type": "transcription_segment"},
        {"text": " schrijft mee. ", "start": 1.2, "end": 2.5, "type": "transcription_segment"},
    ],
    "usage": {"prompt_audio_seconds": 3, "prompt_tokens": 4},
}

OPENAI_JSON_RESPONSE = {"text": "Murmur writes along."}

OPENAI_VERBOSE_RESPONSE = {
    "task": "transcribe",
    "language": "english",
    "duration": 2.5,
    "text": "Murmur writes along.",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": " Murmur "},
        {"id": 1, "start": 1.0, "end": 2.5, "text": " writes along. "},
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
        self.requests = []          # list of (path, headers dict, fields dict)
        self.status = 200
        self.payload = MISTRAL_RESPONSE
        self.retry_after = None
        #: When set, the route answers 302 to this location instead.
        self.redirect_to = None
        #: Body returned for a non-2xx status. Deliberately echoes the key, so
        #: the tests can prove the engine never repeats a provider body.
        self.error_body = json.dumps({"error": f"invalid key {SECRET_KEY}"}).encode()


class _Handler(BaseHTTPRequestHandler):
    recorder: _Recorder = None  # set per-server

    def log_message(self, *args):  # silence the test output
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.recorder.requests.append(
            (
                self.path,
                {key.lower(): value for key, value in self.headers.items()},
                _parse_multipart(body, self.headers.get("Content-Type", "")),
            )
        )
        if self.recorder.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.recorder.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status = self.recorder.status
        if status != 200:
            payload = self.recorder.error_body
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            if self.recorder.retry_after is not None:
                self.send_header("Retry-After", str(self.recorder.retry_after))
            self.end_headers()
            self.wfile.write(payload)
            return
        payload = json.dumps(self.recorder.payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class FlakyOpen:
    """``urlopen`` refusing the connection the first ``failures`` times.

    The reason is a real :class:`ConnectionRefusedError`, the way urllib wraps
    a socket failure: the engine looks at it to decide whether the request body
    could possibly have reached the provider.
    """

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def __call__(self, request, timeout=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise urllib.error.URLError(ConnectionRefusedError("connection refused"))
        return urllib.request.urlopen(request, timeout=timeout)


class ServerBackedTestCase(unittest.TestCase):
    """Runs a real loopback HTTP server both fake providers are pointed at."""

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
        self.wav = self.tmp / "clip.wav"
        self.wav.write_bytes(b"RIFF....WAVEfmt " + b"\x00" * 64)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def make_engine(self, provider="mistral", key=SECRET_KEY, **overrides):
        kwargs = dict(
            provider=provider,
            key_provider=lambda: key,
            base_url=self.base_url,
            http_open=urllib.request.urlopen,
        )
        kwargs.update(overrides)
        engine = ByokEngine(**kwargs)
        self.addCleanup(engine.unload)
        return engine

    def loaded(self, provider="mistral", **overrides):
        engine = self.make_engine(provider=provider, **overrides)
        engine.load()
        return engine

    def last_request(self):
        self.assertTrue(self.recorder.requests, "no request reached the fake provider")
        return self.recorder.requests[-1]


class LifecycleTests(ServerBackedTestCase):
    def test_load_requires_a_stored_key(self):
        for missing in (None, "", "   "):
            with self.subTest(key=missing):
                engine = self.make_engine(key=missing)
                with self.assertRaises(EngineUnavailableError) as caught:
                    engine.load()
                self.assertIn("No API key stored for mistral", str(caught.exception))
                self.assertFalse(engine.is_loaded)

    def test_load_is_idempotent_and_unload_clears_the_flag(self):
        engine = self.make_engine()
        self.assertFalse(engine.is_loaded)
        engine.load()
        engine.load()
        self.assertTrue(engine.is_loaded)
        engine.unload()
        engine.unload()
        self.assertFalse(engine.is_loaded)

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.make_engine(provider="anthropic")
        self.assertIn("anthropic", str(caught.exception))
        self.assertIn("mistral", str(caught.exception))

    def test_transcribe_before_load_raises(self):
        engine = self.make_engine()
        with self.assertRaises(EngineNotLoadedError):
            engine.transcribe(self.wav)

    def test_load_never_calls_the_provider(self):
        self.loaded()
        self.assertEqual(self.recorder.requests, [])

    def test_the_key_is_read_per_request_and_never_cached_on_the_engine(self):
        keys = iter([SECRET_KEY, "sk-second-key"])
        engine = self.loaded(key_provider=lambda: next(keys))
        engine.transcribe(self.wav)
        _path, headers, _fields = self.last_request()
        self.assertEqual(headers["authorization"], "Bearer sk-second-key")
        self.assertNotIn(SECRET_KEY, repr(engine.__dict__))

    def test_info_describes_the_provider_without_apple_silicon(self):
        info = self.make_engine(provider="openai").info()
        self.assertEqual(info.id, "byok")
        self.assertEqual(info.name, "Own key (OpenAI)")
        self.assertEqual(info.model_id, "gpt-4o-transcribe")
        self.assertFalse(info.requires_apple_silicon)
        self.assertFalse(info.supports_streaming)
        self.assertTrue(info.supports_hints)
        self.assertEqual(info.languages[0], "auto")
        self.assertIn("en", info.languages)

    def test_mistral_does_not_advertise_hints(self):
        info = self.make_engine(provider="mistral").info()
        self.assertEqual(info.name, "Own key (Mistral)")
        self.assertEqual(info.model_id, "voxtral-mini-latest")
        self.assertFalse(info.supports_hints)

    def test_model_override_wins(self):
        info = self.make_engine(provider="openai", model="whisper-1").info()
        self.assertEqual(info.model_id, "whisper-1")


class MistralRequestTests(ServerBackedTestCase):
    def test_url_headers_and_fields(self):
        engine = self.loaded("mistral")
        engine.transcribe(self.wav)

        path, headers, fields = self.last_request()
        self.assertEqual(path, f"/v1{TRANSCRIPTIONS_PATH}")
        # The audio docs authenticate with x-api-key while every other Mistral
        # route documents a bearer token; the client sends both.
        self.assertEqual(headers["authorization"], f"Bearer {SECRET_KEY}")
        self.assertEqual(headers["x-api-key"], SECRET_KEY)
        self.assertEqual(fields["model"], "voxtral-mini-latest")
        self.assertEqual(fields["timestamp_granularities"], "segment")
        self.assertNotIn("response_format", fields)
        self.assertEqual(fields["file"], self.wav.read_bytes())

    def test_language_is_omitted_for_none_and_auto(self):
        engine = self.loaded("mistral")
        for language in (None, "auto"):
            with self.subTest(language=language):
                engine.transcribe(self.wav, language=language)
                self.assertNotIn("language", self.last_request()[2])

    def test_language_is_sent_when_given(self):
        engine = self.loaded("mistral")
        engine.transcribe(self.wav, language="nl")
        self.assertEqual(self.last_request()[2]["language"], "nl")

    def test_hints_are_not_sent_and_are_reported_as_unapplied(self):
        engine = self.loaded("mistral")
        result = engine.transcribe(self.wav, hints=Hints(vocabulary=("Murmur", "Boske")))
        self.assertNotIn("prompt", self.last_request()[2])
        self.assertIs(result.hints_applied, False)

    def test_hints_applied_is_none_without_hints(self):
        engine = self.loaded("mistral")
        self.assertIsNone(engine.transcribe(self.wav).hints_applied)

    def test_response_is_parsed_with_segments_and_audio_seconds(self):
        engine = self.loaded("mistral")
        result = engine.transcribe(self.wav)
        self.assertEqual(result.text, "Murmur schrijft mee.")
        self.assertEqual(result.language, "nl")
        self.assertEqual(result.duration_s, 3.0)
        self.assertEqual(result.engine_id, ENGINE_ID)
        self.assertEqual([segment.text for segment in result.segments],
                         ["Murmur", "schrijft mee."])
        self.assertEqual(result.segments[1].start, 1.2)
        self.assertEqual(result.segments[1].end, 2.5)


class OpenAiRequestTests(ServerBackedTestCase):
    def setUp(self):
        super().setUp()
        self.recorder.payload = OPENAI_JSON_RESPONSE

    def test_url_headers_and_fields(self):
        engine = self.loaded("openai")
        engine.transcribe(self.wav)

        path, headers, fields = self.last_request()
        self.assertEqual(path, f"/v1{TRANSCRIPTIONS_PATH}")
        self.assertEqual(headers["authorization"], f"Bearer {SECRET_KEY}")
        self.assertNotIn("x-api-key", headers)
        self.assertEqual(fields["model"], "gpt-4o-transcribe")
        # gpt-4o-transcribe accepts only json or text.
        self.assertEqual(fields["response_format"], "json")
        self.assertNotIn("timestamp_granularities", fields)

    def test_prompt_carries_the_hints_and_is_reported_as_applied(self):
        engine = self.loaded("openai")
        result = engine.transcribe(
            self.wav, hints=Hints(initial_prompt="Dictation.", vocabulary=("Murmur",))
        )
        self.assertEqual(self.last_request()[2]["prompt"], "Dictation. Murmur")
        self.assertIs(result.hints_applied, True)
        self.assertEqual(result.text, "Murmur writes along.")
        self.assertEqual(result.segments, ())

    def test_whisper_1_asks_for_verbose_json_and_parses_segments(self):
        self.recorder.payload = OPENAI_VERBOSE_RESPONSE
        engine = self.loaded("openai", model="whisper-1")
        result = engine.transcribe(self.wav)

        self.assertEqual(self.last_request()[2]["response_format"], "verbose_json")
        self.assertEqual(result.duration_s, 2.5)
        self.assertEqual(result.language, "english")
        self.assertEqual([segment.text for segment in result.segments],
                         ["Murmur", "writes along."])


class ErrorMappingTests(ServerBackedTestCase):
    def test_401_and_403_become_auth_errors(self):
        for status in (401, 403):
            with self.subTest(status=status):
                self.recorder.status = status
                engine = self.loaded("openai")
                with self.assertRaises(ByokAuthError) as caught:
                    engine.transcribe(self.wav)
                message = str(caught.exception)
                self.assertIn(str(status), message)
                self.assertIn("OpenAI", message)
                self.assertIsInstance(caught.exception, EngineError)

    def test_429_becomes_rate_limited_with_retry_after(self):
        self.recorder.status = 429
        self.recorder.retry_after = 7
        engine = self.loaded("mistral")
        with self.assertRaises(ByokRateLimited) as caught:
            engine.transcribe(self.wav)
        self.assertEqual(caught.exception.retry_after_s, 7.0)
        self.assertIsInstance(caught.exception, EngineError)

    def test_429_without_a_retry_after_header(self):
        self.recorder.status = 429
        engine = self.loaded("mistral")
        with self.assertRaises(ByokRateLimited) as caught:
            engine.transcribe(self.wav)
        self.assertIsNone(caught.exception.retry_after_s)

    def test_other_statuses_carry_the_code_but_never_the_body(self):
        self.recorder.status = 500
        engine = self.loaded("mistral")
        with self.assertRaises(EngineError) as caught:
            engine.transcribe(self.wav)
        message = str(caught.exception)
        self.assertIn("500", message)
        self.assertNotIn("invalid key", message)
        self.assertNotIn(SECRET_KEY, message)
        self.assertNotIsInstance(caught.exception, (ByokAuthError, ByokRateLimited))

    def test_a_body_that_is_not_json_fails_loudly(self):
        self.recorder.payload = "not-a-mapping"
        engine = self.loaded("mistral")
        with self.assertRaises(EngineError):
            engine.transcribe(self.wav)

    def test_unreadable_wav_fails_before_any_request(self):
        engine = self.loaded("mistral")
        with self.assertRaises(EngineError):
            engine.transcribe(self.tmp / "missing.wav")
        self.assertEqual(self.recorder.requests, [])


class RetryTests(ServerBackedTestCase):
    def test_one_connection_failure_is_retried(self):
        flaky = FlakyOpen(failures=1)
        engine = self.loaded("mistral", http_open=flaky)
        result = engine.transcribe(self.wav)
        self.assertEqual(flaky.calls, 2)
        self.assertEqual(result.text, "Murmur schrijft mee.")

    def test_a_second_failure_gives_up(self):
        flaky = FlakyOpen(failures=5)
        engine = self.loaded("mistral", http_open=flaky)
        with self.assertRaises(EngineError):
            engine.transcribe(self.wav)
        self.assertEqual(flaky.calls, 2)

    def test_a_timeout_is_never_retried(self):
        # By the time a request times out the audio has been sent, so the clip
        # may already be transcribing — and be billed — on the user's own
        # account. A second POST could pay for the same clip twice.
        class TimingOut:
            def __init__(self):
                self.calls = 0

            def __call__(self, request, timeout=None):
                self.calls += 1
                raise TimeoutError("timed out")

        opener = TimingOut()
        engine = self.loaded("mistral", http_open=opener)
        with self.assertRaises(EngineError):
            engine.transcribe(self.wav)
        self.assertEqual(opener.calls, 1)

    def test_a_timeout_wrapped_in_a_url_error_is_not_retried_either(self):
        class TimingOut:
            def __init__(self):
                self.calls = 0

            def __call__(self, request, timeout=None):
                self.calls += 1
                raise urllib.error.URLError(TimeoutError("timed out"))

        opener = TimingOut()
        engine = self.loaded("mistral", http_open=opener)
        with self.assertRaises(EngineError):
            engine.transcribe(self.wav)
        self.assertEqual(opener.calls, 1)

    def test_a_dns_failure_is_retried_because_nothing_was_sent(self):
        class Unresolvable:
            def __init__(self):
                self.calls = 0

            def __call__(self, request, timeout=None):
                self.calls += 1
                raise urllib.error.URLError(socket.gaierror("nodename nor servname provided"))

        opener = Unresolvable()
        engine = self.loaded("mistral", http_open=opener)
        with self.assertRaises(EngineError):
            engine.transcribe(self.wav)
        self.assertEqual(opener.calls, 2)

    def test_an_http_status_is_never_retried(self):
        self.recorder.status = 500
        engine = self.loaded("mistral")
        with self.assertRaises(EngineError):
            engine.transcribe(self.wav)
        self.assertEqual(len(self.recorder.requests), 1)


class BaseUrlTests(ServerBackedTestCase):
    def test_a_custom_base_url_must_be_https(self):
        for bad in ("http://api.mistral.invalid/v1", "ftp://api.mistral.invalid", "api.mistral.ai"):
            with self.subTest(base_url=bad):
                with self.assertRaises(ValueError):
                    self.make_engine(base_url=bad)

    def test_an_https_custom_base_url_is_accepted(self):
        engine = self.make_engine(base_url="https://gateway.internal/v1/")
        self.assertEqual(engine.url, f"https://gateway.internal/v1{TRANSCRIPTIONS_PATH}")

    def test_the_shipped_provider_endpoints_are_https(self):
        for provider in PROVIDERS.values():
            with self.subTest(provider=provider.id):
                self.assertTrue(provider.base_url.startswith("https://"))


class RedirectTests(ServerBackedTestCase):
    def test_a_redirect_to_another_host_is_refused_before_the_key_is_re_sent(self):
        self.recorder.redirect_to = "http://localhost:1/v1/audio/transcriptions"
        engine = self.loaded("mistral")

        with self.assertRaises(EngineError) as caught:
            engine.transcribe(self.wav)

        self.assertIn("redirect refused", str(caught.exception))
        self.assertNotIn(SECRET_KEY, str(caught.exception))
        self.assertEqual(len(self.recorder.requests), 1)


class SecretHygieneTests(ServerBackedTestCase):
    def test_the_key_never_reaches_an_exception_message_or_a_log_record(self):
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture()
        root = logging.getLogger()
        root.addHandler(handler)
        previous = root.level
        root.setLevel(logging.DEBUG)
        self.addCleanup(root.setLevel, previous)
        self.addCleanup(root.removeHandler, handler)

        failures = []
        for status in (401, 403, 429, 500):
            self.recorder.status = status
            engine = self.loaded("openai")
            try:
                engine.transcribe(self.wav)
            except EngineError as exc:
                failures.append(exc)

        self.recorder.status = 200
        engine = self.loaded("mistral", http_open=FlakyOpen(failures=5))
        try:
            engine.transcribe(self.wav)
        except EngineError as exc:
            failures.append(exc)

        self.assertEqual(len(failures), 5)
        for exc in failures:
            self.assertNotIn(SECRET_KEY, str(exc))
            self.assertNotIn(SECRET_KEY, repr(exc))
            cause = exc.__cause__
            self.assertNotIn(SECRET_KEY, str(cause) if cause is not None else "")
        for record in records:
            self.assertNotIn(SECRET_KEY, record.getMessage())


class ProviderDataTests(unittest.TestCase):
    def test_known_providers_and_their_verified_endpoints(self):
        self.assertEqual(sorted(PROVIDERS), ["mistral", "openai"])
        self.assertEqual(
            PROVIDERS["mistral"].base_url + TRANSCRIPTIONS_PATH,
            "https://api.mistral.ai/v1/audio/transcriptions",
        )
        self.assertEqual(
            PROVIDERS["openai"].base_url + TRANSCRIPTIONS_PATH,
            "https://api.openai.com/v1/audio/transcriptions",
        )
        self.assertEqual(PROVIDERS["mistral"].default_model, "voxtral-mini-latest")
        self.assertEqual(PROVIDERS["openai"].default_model, "gpt-4o-transcribe")
        self.assertFalse(PROVIDERS["mistral"].supports_prompt)
        self.assertTrue(PROVIDERS["openai"].supports_prompt)
        self.assertTrue(PROVIDERS["mistral"].supports_segments("voxtral-mini-latest"))
        self.assertTrue(PROVIDERS["openai"].supports_segments("whisper-1"))
        self.assertFalse(PROVIDERS["openai"].supports_segments("gpt-4o-transcribe"))

    def test_provider_from_config_defaults_to_mistral(self):
        self.assertEqual(provider_from_config({}), ("mistral", None))

    def test_provider_from_config_reads_both_keys(self):
        config = {"byok_provider": "openai", "byok_model": "whisper-1"}
        self.assertEqual(provider_from_config(config), ("openai", "whisper-1"))

    def test_provider_from_config_ignores_a_blank_model(self):
        self.assertEqual(
            provider_from_config({"byok_provider": "openai", "byok_model": "  "}),
            ("openai", None),
        )

    def test_provider_from_config_rejects_an_unknown_provider(self):
        with self.assertRaises(ValueError):
            provider_from_config({"byok_provider": "anthropic"})


class RegistryTests(ServerBackedTestCase):
    def test_byok_is_a_known_engine_id(self):
        self.assertIn("byok", ENGINE_IDS)

    def test_engine_class_is_exported(self):
        self.assertIs(ENGINE_CLASS, ByokEngine)

    def test_create_engine_builds_a_working_byok_engine(self):
        engine = create_engine(
            "byok",
            provider="openai",
            key_provider=lambda: SECRET_KEY,
            base_url=self.base_url,
        )
        self.addCleanup(engine.unload)
        self.recorder.payload = OPENAI_JSON_RESPONSE
        engine.load()
        result = engine.transcribe(self.wav)
        self.assertEqual(result.engine_id, ENGINE_ID)
        self.assertEqual(result.text, "Murmur writes along.")


if __name__ == "__main__":
    unittest.main()
