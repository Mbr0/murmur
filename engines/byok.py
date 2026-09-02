#!/usr/bin/env python3
"""Own-key (BYOK) engine: the user's own Mistral or OpenAI transcription account.

Decision D6 of the Murmur v2 plan keeps this separate from Murmur Cloud: the key
belongs to the user, lives in the Keychain, and is handed to this engine by a
callable. Nothing here meters, counts or reports usage — the user is billed by
their provider directly.

Contract, VERIFIED on 2026-09-02 against the providers' own documentation and a
live probe of both hosts:

Mistral (`mistralai/platform-docs-public`, ``.../audio/speech_to_text/
offline_transcription.md``, and ``mistralai/client-python``
``docs/models/audiotranscriptionrequest.md``)

* ``POST https://api.mistral.ai/v1/audio/transcriptions``, ``multipart/form-data``.
* The documented audio header is ``x-api-key: $MISTRAL_API_KEY``; every other
  route in the same document authenticates with ``Authorization: Bearer``, which
  is also the security scheme the official SDK sends. A probe of
  ``GET /v1/models`` answers ``401 {"detail":"Invalid API Key"}`` for a bogus key
  under either header, so it cannot tell the two apart. This client therefore
  sends **both**; they carry the same value and there is nothing to lose.
* Request fields: ``file``, ``file_url``, ``file_id``, ``model``, ``language``,
  ``temperature``, ``stream``, ``diarize``, ``context_bias``,
  ``timestamp_granularities``. There is **no** ``prompt`` field, so hints cannot
  be applied here and :attr:`Transcript.hints_applied` is reported False when a
  caller supplies them. (``context_bias``, a list of bias strings, is the nearest
  equivalent; wiring vocabulary into it is deliberately left for later.)
* Response: ``{"model", "text", "language", "segments": [{"text", "start",
  "end", "speaker_id", "type"}], "usage": {"prompt_audio_seconds", ...}}``.
  There is no ``duration``; ``usage.prompt_audio_seconds`` is the clip length.
* Current transcription model: ``voxtral-mini-latest`` (Voxtral Mini Transcribe).

OpenAI (``developers.openai.com`` API reference, *Create transcription*)

* ``POST https://api.openai.com/v1/audio/transcriptions``, ``multipart/form-data``,
  ``Authorization: Bearer $OPENAI_API_KEY``. A probe with a bogus key answers
  ``401 invalid_api_key``.
* Request fields used here: ``file``, ``model``, ``language`` (ISO-639-1),
  ``prompt``, ``response_format``.
* Format support is per model: ``gpt-4o-transcribe`` accepts only ``json`` and
  ``text``, while ``whisper-1`` also accepts ``verbose_json`` — the only format
  carrying ``duration`` and ``segments``. So the default model is
  ``gpt-4o-transcribe`` for accuracy, asking for ``json``, and ``whisper-1``
  (an explicit override) is the way to get timings back.

The key itself is read from :attr:`ByokEngine._key_provider` for each request and
never stored on the instance, never put in a URL, and never included in an error
message. Provider response bodies are dropped for the same reason: a provider is
free to echo the credential it rejected.

Two further rules protect the key and the user's own bill:

* The endpoint must be ``https://`` (loopback excepted, for the tests), and a
  redirect off that origin is refused rather than followed — otherwise a 30x
  would hand the key to whatever host the ``Location`` names. See
  :mod:`engines._http`.
* A request is retried **only** when it failed before the audio was sent: a
  refused connection or a DNS failure. A timeout is never retried, because the
  upload may have arrived and be transcribing, and this endpoint is billed to
  the user's own account — a blind second POST could pay twice for one clip.
  There is no idempotency key to lean on here; neither provider offers one for
  this route.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from engines._http import open_no_cross_host_redirect, require_https_base_url
from engines.base import (
    LANGUAGE_AUTO,
    Engine,
    EngineError,
    EngineInfo,
    EngineUnavailableError,
    Hints,
    Segment,
    Transcript,
)

#: Engine id, also the value of :attr:`engines.base.Transcript.engine_id`.
ENGINE_ID = "byok"

#: Route both providers expose, appended to the provider's base URL.
TRANSCRIPTIONS_PATH = "/audio/transcriptions"

#: Config keys read by :func:`provider_from_config`.
CONFIG_PROVIDER_KEY = "byok_provider"
CONFIG_MODEL_KEY = "byok_model"

#: Provider used when the config says nothing.
DEFAULT_PROVIDER = "mistral"

#: Languages both providers understand, for the picker. ``auto`` first: neither
#: provider needs a language, and both detect it when the field is omitted.
LANGUAGES: tuple[str, ...] = (
    LANGUAGE_AUTO,
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "ko",
    "nl",
    "pt",
    "ru",
    "zh",
)

_TRANSCRIBE_TIMEOUT_S = 300.0

#: One retry, and only for a failure raised *before* the body was sent. A
#: rejected key or a rate limit is a decision, not a hiccup; a timeout may be a
#: transcription already under way, and re-sending it would bill it twice.
_RETRIES = 1

#: Failures that prove the provider never received the audio.
_PRE_SEND_FAILURES = (ConnectionRefusedError, socket.gaierror)


def _never_reached_the_provider(exc: BaseException) -> bool:
    """True when ``exc`` proves the upload was not sent, so a retry is free.

    ``URLError`` wraps the underlying socket error in ``reason``; a timeout is
    deliberately excluded, since a request that timed out may well have been
    received, transcribed and metered.
    """
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, TimeoutError):
        return False
    return isinstance(reason, _PRE_SEND_FAILURES)


class ByokAuthError(EngineError):
    """The provider rejected the stored key (HTTP 401 or 403)."""


class ByokRateLimited(EngineError):
    """The provider rate-limited the request (HTTP 429).

    :attr:`retry_after_s` carries the ``Retry-After`` seconds when the provider
    sent a numeric one, and is None otherwise.
    """

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class Provider:
    """Everything that differs between two otherwise identical upload endpoints."""

    id: str
    name: str
    base_url: str
    default_model: str
    #: True when the endpoint accepts a ``prompt`` field, i.e. when Murmur's
    #: :class:`~engines.base.Hints` can actually bias the decode.
    supports_prompt: bool
    #: ``(header name, value template)`` pairs; ``{key}`` is the API key.
    auth_headers: tuple[tuple[str, str], ...]
    #: Models that can return segments. None means "every model can".
    segment_models: tuple[str, ...] | None
    #: Extra form fields asking for segments, and the plain alternative.
    segment_fields: tuple[tuple[str, str], ...]
    plain_fields: tuple[tuple[str, str], ...]

    def supports_segments(self, model: str) -> bool:
        """Whether ``model`` can return per-segment timings."""
        return self.segment_models is None or model in self.segment_models

    def headers_for(self, key: str) -> dict[str, str]:
        """Auth headers carrying ``key``. Never logged, never reused."""
        return {name: template.format(key=key) for name, template in self.auth_headers}

    def format_fields(self, model: str) -> dict[str, str]:
        """The response-format fields to send for ``model``."""
        pairs = self.segment_fields if self.supports_segments(model) else self.plain_fields
        return dict(pairs)


PROVIDERS: dict[str, Provider] = {
    "mistral": Provider(
        id="mistral",
        name="Mistral",
        base_url="https://api.mistral.ai/v1",
        default_model="voxtral-mini-latest",
        supports_prompt=False,
        auth_headers=(("Authorization", "Bearer {key}"), ("x-api-key", "{key}")),
        segment_models=None,
        segment_fields=(("timestamp_granularities", "segment"),),
        plain_fields=(),
    ),
    "openai": Provider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-transcribe",
        supports_prompt=True,
        auth_headers=(("Authorization", "Bearer {key}"),),
        segment_models=("whisper-1",),
        segment_fields=(("response_format", "verbose_json"),),
        plain_fields=(("response_format", "json"),),
    ),
}


def resolve_provider(provider_id: str) -> Provider:
    """Look up a provider by id, or raise :class:`ValueError` naming the known ones."""
    assert provider_id, "provider_id is required"
    provider = PROVIDERS.get(provider_id)
    if provider is None:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown BYOK provider {provider_id!r}; known providers: {known}")
    return provider


def provider_from_config(config: Mapping) -> tuple[str, str | None]:
    """Read ``(provider_id, model_override)`` out of the app config.

    ``byok_provider`` defaults to :data:`DEFAULT_PROVIDER`; an unknown one raises
    rather than silently transcribing somewhere the user did not choose.
    ``byok_model`` is optional and blank means "the provider's default".
    """
    assert config is not None, "config is required"
    raw_provider = config.get(CONFIG_PROVIDER_KEY) or DEFAULT_PROVIDER
    provider = resolve_provider(str(raw_provider).strip())
    raw_model = config.get(CONFIG_MODEL_KEY)
    model = str(raw_model).strip() if raw_model is not None else ""
    return provider.id, (model or None)


def _encode_multipart(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    """Encode ``fields`` plus one file part; return ``(body, content_type)``."""
    boundary = f"----MurmurByok{uuid.uuid4().hex}"
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


def _retry_after_seconds(headers) -> float | None:
    """``Retry-After`` in seconds, or None when absent or not a number.

    The header may also carry an HTTP date; that form is ignored rather than
    guessed at, and the caller simply has no hint to show.
    """
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


class ByokEngine(Engine):
    """Uploads a WAV to the user's own Mistral or OpenAI transcription endpoint.

    ``key_provider`` is the seam to the Keychain: it is called for every request
    and its return value never outlives the call. ``http_open`` and ``base_url``
    are the test seams, pointing the client at a loopback server.
    """

    supports_streaming = False

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        key_provider: Callable[[], str | None] | None = None,
        model: str | None = None,
        http_open=urllib.request.urlopen,
        base_url: str | None = None,
    ) -> None:
        assert key_provider is not None, "key_provider is required"
        assert http_open is not None, "http_open is required"
        self._provider = resolve_provider(str(provider).strip())
        self._key_provider = key_provider
        self._model = (model or "").strip() or self._provider.default_model
        self._http_open = http_open
        # Refused here rather than at load(): the key must not be readable by
        # an engine whose endpoint could carry it in clear.
        self._base_url = require_https_base_url(
            base_url or self._provider.base_url,
            what=f"{self._provider.name} base_url",
        )
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def supports_hints(self) -> bool:
        """Only where the endpoint has a ``prompt`` field; see the module docstring."""
        return self._provider.supports_prompt

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def url(self) -> str:
        """Full transcription URL. Carries no credential; it may be logged."""
        return f"{self._base_url}{TRANSCRIPTIONS_PATH}"

    def _key(self) -> str:
        """The stored key, or raise. The value is returned, never retained."""
        key = self._key_provider()
        if not key or not str(key).strip():
            raise EngineUnavailableError(
                f"No API key stored for {self._provider.id}. "
                f"Add one in Settings → Account before using your own key."
            )
        return str(key).strip()

    def load(self) -> None:
        """Check the provider is known and a key is stored. No network call."""
        if self._loaded:
            return
        self._key()
        self._loaded = True

    def unload(self) -> None:
        """Nothing is held open; this only clears the loaded flag. Idempotent."""
        self._loaded = False

    def info(self) -> EngineInfo:
        """Static description. ``size_bytes`` is 0: nothing is stored on disk."""
        return EngineInfo(
            id=ENGINE_ID,
            name=f"Own key ({self._provider.name})",
            model_id=self._model,
            size_bytes=0,
            languages=LANGUAGES,
            supports_streaming=self.supports_streaming,
            supports_hints=self.supports_hints,
            requires_apple_silicon=False,
        )

    def runtime_summary(self) -> str:
        """One safe line for the log: provider, model and host, never the key."""
        return f"BYOK {self._provider.name} · model {self._model} · {self.url}"

    # -- transcription -----------------------------------------------------

    def _transcribe(
        self,
        wav_path: Path,
        language: str | None,
        hints: Hints | None,
        long_form: bool,
    ) -> Transcript:
        """POST the WAV to the provider and parse its JSON.

        ``long_form`` is accepted and ignored: neither endpoint exposes a switch
        for conditioning on previously decoded text.
        """
        wav_path = Path(wav_path)
        try:
            audio = wav_path.read_bytes()
        except OSError as exc:
            raise EngineError(f"cannot read {wav_path}: {exc}") from exc

        provider = self._provider
        fields: dict[str, str] = {"model": self._model}
        fields.update(provider.format_fields(self._model))
        if language is not None and language != LANGUAGE_AUTO:
            fields["language"] = language

        prompt = hints.as_prompt_text() if hints is not None else None
        hints_applied: bool | None = None
        if prompt:
            hints_applied = provider.supports_prompt
            if provider.supports_prompt:
                fields["prompt"] = prompt

        body, content_type = _encode_multipart(fields, "file", wav_path.name, audio)
        headers = {"Content-Type": content_type}
        headers.update(provider.headers_for(self._key()))

        payload = self._post(body, headers)
        return self._parse_response(payload, hints_applied=hints_applied)

    def _post(self, body: bytes, headers: dict[str, str]) -> bytes:
        """Send the upload, retrying once and only when nothing was sent.

        Every error raised here is built from the status code alone. The
        provider's body is read and dropped: it may quote the rejected key.
        A redirect off this origin raises instead of re-sending the key.
        """
        provider = self._provider
        last_error: Exception | None = None
        for _attempt in range(_RETRIES + 1):
            request = urllib.request.Request(
                self.url, data=body, headers=dict(headers), method="POST"
            )
            try:
                with open_no_cross_host_redirect(
                    request, _TRANSCRIBE_TIMEOUT_S, self._http_open
                ) as response:
                    status = int(getattr(response, "status", 200) or 200)
                    payload = response.read()
            except urllib.error.HTTPError as exc:
                status, error_headers = int(exc.code), getattr(exc, "headers", None)
                # The body is closed unread on purpose: it may quote the key.
                try:
                    exc.close()
                except Exception:
                    pass
                self._raise_for_status(status, error_headers)
                raise EngineError(
                    f"{provider.name} {TRANSCRIPTIONS_PATH} returned HTTP {status}"
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if _never_reached_the_provider(exc):
                    # Connection refused or DNS: the audio was never sent.
                    continue
                # A timeout or a reset mid-upload: the provider may already be
                # transcribing this clip, and the user pays for every attempt.
                raise EngineError(
                    f"{provider.name} {TRANSCRIPTIONS_PATH} did not answer; "
                    f"not retried so the clip cannot be billed twice "
                    f"({type(exc).__name__})"
                ) from None

            if not 200 <= status < 300:
                self._raise_for_status(status, None)
                raise EngineError(
                    f"{provider.name} {TRANSCRIPTIONS_PATH} returned HTTP {status}"
                )
            return payload

        raise EngineError(
            f"{provider.name} {TRANSCRIPTIONS_PATH} is unreachable "
            f"after {_RETRIES + 1} attempts ({type(last_error).__name__})"
        ) from None

    def _raise_for_status(self, status: int, headers) -> None:
        """Raise the specific error for a status that has one; else return."""
        provider = self._provider
        if status in (401, 403):
            raise ByokAuthError(
                f"{provider.name} rejected the stored API key (HTTP {status}). "
                f"Check the key in Settings → Account."
            ) from None
        if status == 429:
            retry_after = _retry_after_seconds(headers)
            suffix = f" Retry after {retry_after:g}s." if retry_after is not None else ""
            raise ByokRateLimited(
                f"{provider.name} rate-limited this request (HTTP 429).{suffix}",
                retry_after_s=retry_after,
            ) from None

    def _parse_response(self, payload: bytes, hints_applied: bool | None = None) -> Transcript:
        """Turn a provider JSON body into a :class:`Transcript`."""
        provider = self._provider
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EngineError(f"{provider.name} returned a body that is not JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise EngineError(
                f"{provider.name} returned a JSON {type(data).__name__}, expected an object"
            )
        if "text" not in data:
            raise EngineError(f"{provider.name} returned a JSON object without a 'text' field")

        segments: list[Segment] = []
        for raw in data.get("segments") or ():
            if not isinstance(raw, dict):
                continue
            segments.append(
                Segment(
                    start=float(raw.get("start") or 0.0),
                    end=float(raw.get("end") or 0.0),
                    text=str(raw.get("text", "")).strip(),
                )
            )

        language = data.get("language")
        return Transcript(
            text=str(data.get("text", "")).strip(),
            language=str(language) if language else None,
            duration_s=_duration_seconds(data),
            segments=tuple(segments),
            engine_id=ENGINE_ID,
            hints_applied=hints_applied,
        )


def _duration_seconds(data: dict) -> float | None:
    """Clip length: OpenAI's ``duration``, else Mistral's ``usage`` audio seconds."""
    duration = data.get("duration")
    if duration is None:
        usage = data.get("usage")
        if isinstance(usage, dict):
            duration = usage.get("prompt_audio_seconds")
    try:
        return float(duration) if duration is not None else None
    except (TypeError, ValueError):
        return None


#: Consumed by :func:`engines.create_engine`.
ENGINE_CLASS = ByokEngine
