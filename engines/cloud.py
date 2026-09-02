#!/usr/bin/env python3
"""Murmur Cloud: transcription through the Boske LLM proxy.

Decision D6 of the Murmur v2 plan: the user never pastes a key. Murmur obtains
an Ed25519 lease JWT through the same device-linking flow as the Boske desktop
app and sends it as a bearer token. This module is the HTTP client only; the
lease itself is produced by ``services/license_service.py`` and handed in as
``lease_provider``.

Proxy contract assumed (the OpenAI-compatible shape Boske already speaks; keep
this list in step with ``decisions.md`` D6):

* ``base`` must be an ``https://`` URL — the lease is a bearer token, so an
  ``http://`` endpoint is refused at construction. Loopback is the one
  exception, for the test servers.
* ``POST {base}/v1/audio/transcriptions`` — ``multipart/form-data`` with
  ``file`` (``audio/wav``), ``model`` (:data:`MODEL_ID`), ``response_format``
  (:data:`RESPONSE_FORMAT`), optional ``language`` (ISO-639-1; omitted means
  auto-detect) and optional ``prompt`` (the caller's hints, folded by
  :meth:`engines.base.Hints.as_prompt_text`).
* ``Idempotency-Key``: a fresh UUID per :meth:`CloudEngine.transcribe` call,
  **repeated unchanged on every retry of that call**. A transcription that
  times out may well have been received and billed, so the proxy must key on
  this header and return the first result rather than transcribe and meter the
  same clip again. One clip is charged once, however many attempts it took.
* The 200 body is ``verbose_json``: ``{"text", "language", "duration",
  "segments": [{"start", "end", "text"}]}`` — the same shape whisper.cpp
  returns, so :class:`~engines.base.Transcript` maps straight onto it.
* ``GET {base}/v1/voice/usage`` — ``{"minutes_used", "minutes_allowance",
  "words", "period_end"}``. ``minutes_used`` and ``minutes_allowance`` are
  **required**: an absent allowance is unknown, not zero, and guessing zero
  would read as "exhausted" and push every user onto the local engine.
  ``period_end`` is an ISO-8601 string or null.
* Errors are JSON. The code may sit at the top level or under ``error``:
  ``{"error": {"code": "allowance_exhausted", "message": ...}}``.
  ``401`` means the lease is invalid or expired and the caller must re-link.
  ``code == "allowance_exhausted"`` — at *any* status — means the metered
  allowance is spent; the router then switches to the local engine and shows
  the one-time notice, and the user never sees an HTTP status for that case.
  A bare ``429`` is only a rate limit: it is retried, honouring ``Retry-After``
  (seconds or an HTTP date, capped at :data:`RETRY_AFTER_CAP_S`), and ends as
  an ordinary transient :class:`~engines.base.EngineError`. Treating it as the
  allowance would silently drop a paying user to the local engine.
* Redirects are refused. A 30x to another host would re-send the lease to
  whatever that host is; see :mod:`engines._http`.

Privacy: transcript text, the prompt built from hints and the lease itself are
never logged, and neither is any exception message from the lease provider —
only its type. Only attempt counts and status codes are.
"""

from __future__ import annotations

import contextlib
import email.utils
import json
import logging
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engines._http import open_no_cross_host_redirect, require_https_base_url
from engines.base import (
    LANGUAGE_AUTO,
    Engine,
    EngineError,
    EngineInfo,
    Hints,
    Segment,
    Transcript,
)

logger = logging.getLogger(__name__)

#: Engine id, also the value of :attr:`engines.base.Transcript.engine_id`.
ENGINE_ID = "cloud"

#: Name shown in Settings and in the "what leaves the Mac" copy.
ENGINE_NAME = "Murmur Cloud"

#: The single model Murmur Cloud bills for. Constant on purpose: the tier is
#: sold as "Murmur Cloud", not as a model picker, and per-modality cost is
#: pinned on the Boske side against exactly this id.
MODEL_ID = "voxtral-mini-latest"

#: Routes on the proxy.
TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"
USAGE_PATH = "/v1/voice/usage"

#: Only ``verbose_json`` carries segments and duration.
RESPONSE_FORMAT = "verbose_json"

#: Languages the tier advertises; ``auto`` asks the proxy to detect.
LANGUAGES = (LANGUAGE_AUTO, "en", "fr", "nl", "de", "es", "it", "pt")

#: Hard cap on a single upload, in minutes. Checked from the WAV header before
#: a byte is sent, so an over-long import fails locally and instantly.
MAX_MINUTES = 60

#: Total attempts for one transcription, including the first.
MAX_ATTEMPTS = 3

#: Backoff before retry *n* is ``BACKOFF_BASE_S * 2 ** (n - 1)``: 0.5 s, 1 s,
#: 2 s, … With :data:`MAX_ATTEMPTS` at 3 the schedule stops after 0.5 s and 1 s.
BACKOFF_BASE_S = 0.5

#: Error code meaning "the metered allowance is spent"; the one deliberate
#: product fallback (cloud → local, with a visible notice) hangs off this.
#: **Only** this code, never a bare status: a 429 is a rate limit.
ALLOWANCE_EXHAUSTED_CODE = "allowance_exhausted"

#: Longest ``Retry-After`` we will actually wait. A proxy asking for ten
#: minutes gets one attempt, not a frozen dictation.
RETRY_AFTER_CAP_S = 10.0

_TRANSCRIBE_TIMEOUT_S = 600.0
_USAGE_TIMEOUT_S = 15.0
_ERROR_DETAIL_CHARS = 200


class CloudAuthError(EngineError):
    """The lease is missing, invalid or expired; the user must re-link."""


class CloudAllowanceExhausted(EngineError):
    """The metered allowance is spent. The router falls back to local."""


@dataclass(frozen=True)
class Usage:
    """What ``GET /v1/voice/usage`` reports for the current billing period."""

    minutes_used: float
    minutes_allowance: float
    words: int
    period_end: str | None

    @property
    def fraction_used(self) -> float | None:
        """Share of the allowance consumed, or None when there is no allowance.

        An allowance of zero or less is **unknown**, not spent. Reading it as
        "fully used" is how a truncated or unexpected body silently moves a
        paying user onto the local engine; the caller must decide what to do
        with "we do not know" instead of being told a number that is wrong.
        """
        if self.minutes_allowance <= 0:
            return None
        return self.minutes_used / self.minutes_allowance


def _encode_multipart(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    """Encode ``fields`` plus one file part; return ``(body, content_type)``."""
    boundary = f"----MurmurCloud{uuid.uuid4().hex}"
    marker = f"--{boundary}".encode()
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.append(marker + b"\r\n")
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.append(str(value).encode("utf-8") + b"\r\n")

    chunks.append(marker + b"\r\n")
    chunks.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode()
    )
    chunks.append(file_bytes + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())

    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def wav_duration_seconds(wav_path: Path) -> float:
    """Duration of a WAV from its header. Raises :class:`EngineError` if unreadable."""
    try:
        with contextlib.closing(wave.open(str(wav_path), "rb")) as handle:
            frame_rate = handle.getframerate()
            frames = handle.getnframes()
    except (OSError, wave.Error, EOFError) as exc:
        raise EngineError(f"cannot read {wav_path}: {exc}") from exc
    if frame_rate <= 0:
        raise EngineError(f"{wav_path} declares a frame rate of {frame_rate}")
    return frames / float(frame_rate)


def _maybe_json(payload: bytes) -> Any:
    """Decode a JSON body, or None when the body is not JSON at all."""
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _error_code(data: Any) -> str | None:
    """Read ``code`` from either the top level or an ``error`` object."""
    if not isinstance(data, dict):
        return None
    code = data.get("code")
    error = data.get("error")
    if code is None and isinstance(error, dict):
        code = error.get("code")
    return str(code) if code else None


def _error_message(data: Any, payload: bytes) -> str:
    """A short, safe detail for an error message; never the whole body."""
    if isinstance(data, dict):
        error = data.get("error")
        message = error.get("message") if isinstance(error, dict) else data.get("message")
        if message:
            return str(message)[:_ERROR_DETAIL_CHARS]
    return payload.decode("utf-8", errors="replace")[:_ERROR_DETAIL_CHARS]


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a JSON number, tolerating a string or a missing field."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a JSON integer, tolerating a string or a missing field."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _require_float(data: dict, name: str) -> float:
    """A number the usage contract makes mandatory, or :class:`EngineError`."""
    value = data.get(name)
    if isinstance(value, bool) or value is None:
        raise EngineError(f"{ENGINE_NAME} usage response missing {name}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise EngineError(f"{ENGINE_NAME} usage response missing {name}") from None


def _bearer_headers(lease: str) -> dict[str, str]:
    """Authorization header for the lease. The value is never logged."""
    return {"Authorization": f"Bearer {lease}"}


#: Product wording for the allowance case. Deliberately free of any HTTP
#: status: the user is told the allowance is spent, never that they got a 429.
ALLOWANCE_MESSAGE = (
    f"{ENGINE_NAME} allowance is used up for this period; continuing on the local engine"
)


def retry_after_seconds(headers: Any) -> float | None:
    """``Retry-After`` as a delay in seconds, capped, or None.

    RFC 9110 allows either a number of seconds or an HTTP date; both are
    understood. The result is clamped to ``[0, RETRY_AFTER_CAP_S]`` so a proxy
    cannot park a dictation for ten minutes, and a date already in the past
    reads as "retry now".
    """
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:  # noqa: BLE001 - a header bag that does not behave like one
        return None
    if raw is None:
        return None
    raw = str(raw).strip()

    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        try:
            when = email.utils.parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        seconds = when.timestamp() - time.time()

    return max(0.0, min(RETRY_AFTER_CAP_S, seconds))


class _Attempt(Exception):
    """Internal: one failed HTTP attempt, classified.

    ``error`` is what the caller should see, ``retryable`` says whether another
    attempt could help, ``reason`` is the short, sensitive-data-free string
    that may go into the log, and ``retry_after_s`` is the server's own delay
    request when it made one.
    """

    def __init__(
        self,
        error: EngineError,
        retryable: bool,
        reason: str,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(reason)
        self.error = error
        self.retryable = retryable
        self.reason = reason
        self.retry_after_s = retry_after_s


def _classify(status: int, payload: bytes, headers: Any = None) -> _Attempt:
    """Map a non-2xx (or error-carrying) response onto the contract's failures."""
    data = _maybe_json(payload)
    code = _error_code(data)
    detail = _error_message(data, payload)

    # The allowance is a product state the body declares, never a status code.
    if code == ALLOWANCE_EXHAUSTED_CODE:
        return _Attempt(CloudAllowanceExhausted(ALLOWANCE_MESSAGE), False, "allowance exhausted")
    if status == 401:
        return _Attempt(
            CloudAuthError(f"{ENGINE_NAME} rejected the lease; link this Mac again: {detail}"),
            False,
            "HTTP 401",
        )
    if status == 429:
        return _Attempt(
            EngineError(f"{ENGINE_NAME} rate limited this request (HTTP 429)"),
            True,
            "HTTP 429",
            retry_after_seconds(headers),
        )
    if 200 <= status < 300:
        # A success status carrying an error object: nothing to retry.
        return _Attempt(EngineError(f"{ENGINE_NAME} reported: {detail}"), False, "error body")
    return _Attempt(
        EngineError(f"{ENGINE_NAME} returned HTTP {status}: {detail}"),
        status >= 500,
        f"HTTP {status}",
        retry_after_seconds(headers) if status >= 500 else None,
    )


def _send(http_open: Callable, request: urllib.request.Request, timeout_s: float) -> bytes:
    """One HTTP round trip. Raises :class:`_Attempt` for every failure.

    The request goes through :func:`engines._http.open_no_cross_host_redirect`,
    so a 30x to another host raises rather than handing the lease over.
    """
    try:
        with open_no_cross_host_redirect(request, timeout_s, http_open) as response:
            status = int(getattr(response, "status", 200) or 200)
            payload = response.read()
            headers = getattr(response, "headers", None)
    except urllib.error.HTTPError as exc:
        body = b""
        with contextlib.suppress(Exception):
            body = exc.read() or b""
        failure = _classify(int(exc.code), body, getattr(exc, "headers", None))
        failure.__cause__ = exc
        raise failure
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        failure = _Attempt(
            EngineError(f"{ENGINE_NAME} is unreachable: {exc}"), True, "connection error"
        )
        failure.__cause__ = exc
        raise failure

    if not 200 <= status < 300:
        raise _classify(status, payload, headers)

    # A 2xx may still carry an error object; the allowance case has been seen
    # answered as 200 by proxies that queue rather than reject.
    data = _maybe_json(payload)
    if isinstance(data, dict) and "text" not in data and ("error" in data or _error_code(data)):
        raise _classify(200, payload, headers)
    return payload


def _with_retries(
    call: Callable[[], bytes],
    sleep: Callable[[float], Any],
    clock: Callable[[], float],
    max_attempts: int,
    what: str,
) -> tuple[bytes, int]:
    """Run ``call`` with backoff on retryable failures; return ``(payload, attempts)``.

    Only the attempt number, the short reason and the delay are logged.
    """
    started = clock()
    for attempt in range(1, max_attempts + 1):
        try:
            return call(), attempt
        except _Attempt as failure:
            if not failure.retryable:
                raise failure.error from failure.__cause__
            if attempt >= max_attempts:
                elapsed = clock() - started
                raise EngineError(
                    f"{what} failed after {attempt} attempts in {elapsed:.1f}s: {failure.error}"
                ) from failure.__cause__
            # A server that told us when to come back knows better than our
            # own schedule; ``retry_after_s`` is already clamped.
            delay = failure.retry_after_s
            if delay is None:
                delay = BACKOFF_BASE_S * (2 ** (attempt - 1))
            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                what,
                attempt,
                max_attempts,
                failure.reason,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable: the retry loop always returns or raises")


def fetch_usage(
    base_url: str,
    lease: str | None,
    http_open: Callable = urllib.request.urlopen,
    timeout_s: float = _USAGE_TIMEOUT_S,
) -> Usage:
    """Read the metered usage for the current period from the proxy.

    ``GET {base}/v1/voice/usage`` with the lease as bearer, answering
    ``{"minutes_used", "minutes_allowance", "words", "period_end"}``.

    ``minutes_used`` and ``minutes_allowance`` are required and a body without
    them raises. ``words`` and ``period_end`` are cosmetic and may be absent.
    Defaulting a missing allowance to zero is what made a truncated body look
    like an exhausted account, so the numbers the fallback decision rests on
    must be present or the reading is refused outright.
    """
    if not lease:
        raise CloudAuthError(f"{ENGINE_NAME} has no lease; link this Mac first")
    url = require_https_base_url(base_url, what=f"{ENGINE_NAME} base_url") + USAGE_PATH
    request = urllib.request.Request(url, headers=_bearer_headers(lease), method="GET")

    try:
        payload = _send(http_open, request, timeout_s)
    except _Attempt as failure:
        raise failure.error from failure.__cause__

    data = _maybe_json(payload)
    if not isinstance(data, dict):
        raise EngineError(f"{ENGINE_NAME} {USAGE_PATH} did not return a JSON object")

    period_end = data.get("period_end")
    return Usage(
        minutes_used=_require_float(data, "minutes_used"),
        minutes_allowance=_require_float(data, "minutes_allowance"),
        words=_as_int(data.get("words")),
        period_end=str(period_end) if period_end else None,
    )


def _parse_response(payload: bytes, hints_applied: bool | None) -> Transcript:
    """Turn a ``verbose_json`` body into a :class:`Transcript`."""
    data = _maybe_json(payload)
    if not isinstance(data, dict):
        raise EngineError(f"{ENGINE_NAME} returned a body that is not a JSON object")

    segments: list[Segment] = []
    for raw in data.get("segments") or ():
        if not isinstance(raw, dict):
            continue
        segments.append(
            Segment(
                start=_as_float(raw.get("start")),
                end=_as_float(raw.get("end")),
                text=str(raw.get("text", "")).strip(),
            )
        )

    duration = data.get("duration")
    language = data.get("language")
    return Transcript(
        text=str(data.get("text", "")).strip(),
        language=str(language) if language else None,
        duration_s=float(duration) if duration is not None else None,
        segments=tuple(segments),
        engine_id=ENGINE_ID,
        hints_applied=hints_applied,
    )


class CloudEngine(Engine):
    """Transcribes through the Boske proxy with a lease as bearer token.

    ``http_open``, ``clock`` and ``sleep`` are seams: tests point a real
    ``urlopen`` at a loopback server and inject a recording sleep, so the retry
    schedule is asserted without waiting for it.
    """

    supports_streaming = False
    supports_hints = True

    def __init__(
        self,
        base_url: str,
        lease_provider: Callable[[], str | None],
        http_open: Callable = urllib.request.urlopen,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = time.sleep,
        max_minutes: int = MAX_MINUTES,
    ) -> None:
        assert base_url is not None, "base_url is required"
        assert callable(lease_provider), "lease_provider must be callable"
        assert callable(http_open), "http_open must be callable"
        assert callable(clock), "clock must be callable"
        assert callable(sleep), "sleep must be callable"
        assert max_minutes > 0, "max_minutes must be positive"
        # The endpoint is checked here, not in load(): a mistyped or plain-http
        # base URL must fail before an object exists that could send a lease.
        self._base_url = require_https_base_url(base_url, what=f"{ENGINE_NAME} base_url")
        self._lease_provider = lease_provider
        self._http_open = http_open
        self._clock = clock
        self._sleep = sleep
        self._max_minutes = max_minutes
        self._base: str | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._base is not None

    def load(self) -> None:
        """Mark the engine ready. No network, and no lease is fetched.

        There is nothing to warm up on a hosted engine, and asking for a lease
        here would hit the Keychain (or the linking flow) at app start. The
        endpoint was already validated by ``__init__``.
        """
        if self.is_loaded:
            return
        self._base = self._base_url

    def unload(self) -> None:
        """Forget the validated configuration. Idempotent; nothing to release."""
        self._base = None

    def info(self) -> EngineInfo:
        """Static description. ``size_bytes`` is zero: nothing is downloaded."""
        return EngineInfo(
            id=ENGINE_ID,
            name=ENGINE_NAME,
            model_id=MODEL_ID,
            size_bytes=0,
            languages=LANGUAGES,
            supports_streaming=self.supports_streaming,
            supports_hints=self.supports_hints,
            requires_apple_silicon=False,
        )

    def runtime_summary(self) -> str:
        """What the app may log about this engine. Never the lease."""
        return f"endpoint={self._base or self._base_url} model={MODEL_ID}"

    # -- transcription -----------------------------------------------------

    def _lease(self) -> str:
        """The current lease, or :class:`CloudAuthError` when there is none.

        Any provider failure (a locked Keychain, a refresh that could not
        complete) is an auth problem from here: the caller re-links. Only the
        exception's *type* is repeated: a provider is free to put a token, or a
        proxy's response body, into the message it raises.
        """
        try:
            lease = self._lease_provider()
        except Exception as exc:  # noqa: BLE001 - any provider failure is "no lease"
            raise CloudAuthError(
                f"{ENGINE_NAME} could not obtain a lease ({type(exc).__name__}); "
                f"link this Mac again"
            ) from exc
        if not lease:
            raise CloudAuthError(f"{ENGINE_NAME} has no lease; link this Mac first")
        return str(lease)

    def _check_duration(self, wav_path: Path) -> float:
        """Refuse a clip past the cap before a single byte is uploaded."""
        duration_s = wav_duration_seconds(wav_path)
        if duration_s > self._max_minutes * 60:
            raise EngineError(
                f"{ENGINE_NAME} accepts at most {self._max_minutes} minutes per clip; "
                f"this one is {duration_s / 60:.1f} minutes"
            )
        return duration_s

    def _transcribe(
        self,
        wav_path: Path,
        language: str | None,
        hints: Hints | None,
        long_form: bool,
    ) -> Transcript:
        """Upload the WAV and parse ``verbose_json``.

        ``long_form`` is accepted and ignored: the proxy owns its own windowing
        and exposes no per-request switch for conditioning on previous text.
        """
        wav_path = Path(wav_path)
        self._check_duration(wav_path)
        lease = self._lease()

        try:
            audio = wav_path.read_bytes()
        except OSError as exc:
            raise EngineError(f"cannot read {wav_path}: {exc}") from exc

        fields: dict[str, str] = {"model": MODEL_ID, "response_format": RESPONSE_FORMAT}
        if language is not None and language != LANGUAGE_AUTO:
            fields["language"] = language
        prompt = hints.as_prompt_text() if hints is not None else None
        if prompt:
            fields["prompt"] = prompt

        body, content_type = _encode_multipart(fields, "file", wav_path.name, audio)
        # One key per call, deliberately built outside the retry loop and sent
        # on every attempt: a timed-out upload may already have been received
        # and metered, and the proxy uses this to charge the clip once.
        idempotency_key = str(uuid.uuid4())
        request = urllib.request.Request(
            f"{self._base}{TRANSCRIPTIONS_PATH}",
            data=body,
            headers={
                "Content-Type": content_type,
                "Idempotency-Key": idempotency_key,
                **_bearer_headers(lease),
            },
            method="POST",
        )

        payload, attempts = _with_retries(
            lambda: _send(self._http_open, request, _TRANSCRIBE_TIMEOUT_S),
            self._sleep,
            self._clock,
            MAX_ATTEMPTS,
            f"{ENGINE_NAME} {TRANSCRIPTIONS_PATH}",
        )
        logger.info("%s transcription succeeded on attempt %d", ENGINE_NAME, attempts)
        return _parse_response(payload, hints_applied=True if prompt else None)


#: Consumed by :func:`engines.create_engine`.
ENGINE_CLASS = CloudEngine
