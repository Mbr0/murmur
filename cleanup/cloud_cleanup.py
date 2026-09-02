#!/usr/bin/env python3
"""Cleanup through the Boske proxy, for when Murmur Cloud is the live route.

Decision D6 of the Murmur v2 plan: the user never pastes a key. Murmur sends
the Ed25519 lease JWT that ``services/license_service.py`` produces as a bearer
token; this module is the HTTP client only, and the lease arrives through
``lease_provider``.

**The contract is the local client's, deliberately.**
:class:`CloudCleanupClient.cleanup` takes the same arguments as
:meth:`cleanup.llama_server.CleanupClient.cleanup` and returns the same
:class:`~cleanup.llama_server.CleanupResult`, so the wiring picks one object or
the other and the rest of the pipeline never learns which. The prompts are the
same Wave 2 templates rendered by :func:`cleanup.modes.render_system_prompt`;
nothing about the mode or tone changes because the model is hosted. The two
budgets — :func:`~cleanup.llama_server.timeout_for_text` and
:func:`~cleanup.llama_server.max_tokens_for_text` — are imported rather than
re-derived, so a change to the policy moves both clients at once.

Proxy contract assumed (the OpenAI-compatible shape Boske already speaks; keep
this in step with ``decisions.md`` D6 and with :mod:`engines.cloud`):

* ``base`` must be an ``https://`` URL — the lease is a bearer token, so an
  ``http://`` endpoint is refused at construction. Loopback is the one
  exception, for the test servers.
* ``POST {base}/v1/chat/completions`` — the same route
  :class:`cleanup.llama_server.CleanupClient` speaks to locally:
  ``{"model", "messages": [system, user], "stream": false, "temperature",
  "max_tokens"}``, answering ``choices[0].message.content``.
* ``model`` is :data:`MODEL_ID`, a constant rather than a picker: cleanup is
  sold as part of Murmur Cloud and per-modality cost is pinned on the Boske
  side against exactly this id. **To confirm with Boske** — the id below is
  what the proxy is expected to expose for the small instruct model, and a
  mismatch is a 4xx, which skips visibly rather than losing text.
* ``Idempotency-Key``: a fresh UUID per :meth:`CloudCleanupClient.cleanup`
  call. There is no retry here (see below), but a cleanup that times out may
  well have been received and metered, and the key lets the proxy return the
  first result rather than bill a second generation if the user re-dictates.
* Errors are JSON, with the code at the top level or under ``error``, exactly
  as :mod:`engines.cloud` documents.

**One attempt, never a retry.** ``engines.cloud`` retries a transcription
because losing it would lose the user's words. Cleanup is different: the words
are already safe, and the whole feature lives inside a 3–20 s budget on the
paste path. A second attempt would double the wait to improve punctuation, so a
rate limit or a 5xx becomes a visible skip instead.

**What raises and what skips.** Everything transient — a timeout, a 5xx, a
blank or unreadable reply, a refused redirect — comes back as
:class:`~cleanup.llama_server.CleanupResult` with ``skipped=True`` and the
caller's original string, byte for byte, which the UI turns into the visible
"cleanup skipped" notice. Two cases raise, because the wiring must *do*
something about them:

* :class:`~engines.cloud.CloudAuthError` — no lease, or the proxy rejected it.
  The user re-links this Mac.
* :class:`~engines.cloud.CloudAllowanceExhausted` — the metered allowance is
  spent. This is the one deliberate product fallback in the whole app: the
  wiring drops to the local cleanup client and shows the one-time notice. Only
  the ``allowance_exhausted`` code triggers it, never a bare status; a plain
  ``429`` is a rate limit and skips as :data:`RATE_LIMITED_REASON`.

Metering is the wiring's job, not this module's. ``services/usage_service.py``
is deliberately not imported: this stays a pure HTTP client. After a successful
call the wiring records the words the proxy returned::

    result = client.cleanup(text, prompt)
    if not result.skipped:
        usage.record("cloud", 0, words_in(result.text))

``seconds`` is 0 because no audio was sent — the minutes allowance is spent by
transcription alone. A *skipped* result must not be metered: its text is the
original transcript, which the transcription call already counted.

Privacy: transcript text, the cleaned reply, the system prompt, the proxy's
error bodies and the lease are never logged, and neither is any exception
message from the lease provider — only its type. Only lengths, timings, status
codes and fixed reasons are. ``reason`` is shown to the user, so it never
carries model output or a proxy message either.
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable

# Same package, same route, same reply shape: reusing the local client's
# budgets and its reader is what keeps the two clients swappable. A second
# reading of ``choices[0].message.content`` here would be free to drift.
from cleanup.llama_server import (
    CHAT_PATH,
    MIN_OUTPUT_TOKENS,
    CleanupResult,
    _content_of,
    max_tokens_for_text,
    timeout_for_text,
)
from engines._http import open_no_cross_host_redirect, require_https_base_url
from engines.base import EngineError
from engines.cloud import (
    ALLOWANCE_EXHAUSTED_CODE,
    ALLOWANCE_MESSAGE,
    ENGINE_NAME,
    CloudAllowanceExhausted,
    CloudAuthError,
)

logger = logging.getLogger(__name__)

#: The single model Murmur Cloud cleans with. Constant on purpose, and pinned
#: against the Boske per-modality cost table. **Confirm with Boske** before the
#: proxy goes live; a wrong id answers 4xx, which skips rather than breaking.
MODEL_ID = "ministral-3b-latest"

#: Config key: "clean up through Murmur Cloud rather than the local model".
CONFIG_CLOUD_CLEANUP_KEY = "cleanup_cloud"

#: The Pro feature this asks the gate about. One name, one call site.
CLEANUP_FEATURE = "cleanup"

#: Reason for a plain ``429``. Deliberately free of any HTTP status: the user
#: is told the service is busy, never that they got a 429.
RATE_LIMITED_REASON = "rate limited"

#: Window assumed for the hosted model, used only to bound the reply budget
#: through :func:`~cleanup.llama_server.max_tokens_for_text`. Conservative on
#: purpose — the hosted model's real window is larger, and this number can only
#: ever make the requested reply *shorter*, never overflow anything.
CONTEXT_TOKENS = 32768

_CONTENT_TYPE = "application/json"

#: Re-exported so :data:`CHAT_PATH` reads as part of this module's contract too.
__all__ = [
    "CHAT_PATH",
    "CLEANUP_FEATURE",
    "CONFIG_CLOUD_CLEANUP_KEY",
    "CONTEXT_TOKENS",
    "MODEL_ID",
    "RATE_LIMITED_REASON",
    "CleanupResult",
    "CloudCleanupClient",
    "should_use_cloud_cleanup",
    "words_in",
]


def words_in(text: str | None) -> int:
    """Whitespace-separated word count — the unit ``UsageService`` meters.

    The wiring calls ``usage.record("cloud", 0, words_in(result.text))`` after a
    cleanup that actually ran; see the module docstring for why a skip is not
    metered and why the seconds are zero.
    """
    return len(text.split()) if text else 0


def should_use_cloud_cleanup(
    config: dict | None,
    *,
    pro_gate: Callable[[str], bool],
    cloud_engine_active: bool,
) -> bool:
    """Whether this dictation's cleanup should go to the proxy.

    Three conditions, all required:

    * the user turned cloud cleanup on (``config["cleanup_cloud"]``);
    * Murmur Cloud is the route this dictation was actually transcribed on
      (``cloud_engine_active``) — sending text to the proxy for cleanup while
      transcription runs on this Mac would break the promise the privacy tab
      makes, so the two travel together;
    * the Pro gate allows cleanup at all.

    ``pro_gate`` is :func:`services.license_service.is_pro_feature_enabled`,
    injected rather than imported so this stays pure and testable. It is asked
    last, and only when the other two hold.
    """
    assert callable(pro_gate), "pro_gate must be callable"
    if not (config or {}).get(CONFIG_CLOUD_CLEANUP_KEY):
        return False
    if not cloud_engine_active:
        return False
    return bool(pro_gate(CLEANUP_FEATURE))


def _error_code(payload: bytes) -> str | None:
    """Read ``code`` from a JSON error body, top level or under ``error``.

    Mirrors the envelope :mod:`engines.cloud` documents. Only the *code* is
    read: the accompanying message is proxy-authored text that would end up in
    a user-visible reason, so it is never looked at here.
    """
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    code = data.get("code")
    error = data.get("error")
    if code is None and isinstance(error, dict):
        code = error.get("code")
    return str(code) if code else None


class CloudCleanupClient:
    """Cleans transcript text through the proxy's chat route.

    ``http_open`` and ``clock`` are seams: the tests point a real ``urlopen`` at
    a loopback server and inject a clock, so timings are asserted without
    waiting. Requests go through
    :func:`engines._http.open_no_cross_host_redirect`, so a 30x to another host
    is refused rather than handing that host the lease.
    """

    def __init__(
        self,
        base_url: str,
        lease_provider: Callable[[], str | None],
        http_open: Callable = urllib.request.urlopen,
        clock: Callable[[], float] = time.monotonic,
        model: str = MODEL_ID,
    ) -> None:
        assert base_url is not None, "base_url is required"
        assert callable(lease_provider), "lease_provider must be callable"
        assert callable(http_open), "http_open must be callable"
        assert callable(clock), "clock must be callable"
        assert model, "model is required"
        # Checked here, not on first use: a mistyped or plain-http base URL must
        # fail before an object exists that could send a lease over it.
        self._base_url = require_https_base_url(base_url, what=f"{ENGINE_NAME} base_url")
        self._lease_provider = lease_provider
        self._http_open = http_open
        self._clock = clock
        self._model = str(model)

    @property
    def model(self) -> str:
        """Model id sent with every request."""
        return self._model

    @property
    def chat_url(self) -> str:
        """Absolute URL of the chat route. Safe to log; carries no credential."""
        return self._base_url + CHAT_PATH

    def runtime_summary(self) -> str:
        """What the app may log about this client. Never the lease."""
        return f"endpoint={self._base_url} model={self._model}"

    # -- cleanup -----------------------------------------------------------

    def _lease(self) -> str:
        """The current lease, or :class:`CloudAuthError` when there is none.

        Any provider failure (a locked Keychain, a refresh that could not
        complete) is an auth problem from here. Only the exception's *type* is
        repeated: a provider is free to put a token, or a proxy's response body,
        into the message it raises.
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

    def cleanup(
        self,
        text: str,
        system_prompt: str,
        temperature: float = 0.1,
    ) -> CleanupResult:
        """Rewrite ``text`` under ``system_prompt``; never raise on a bad reply.

        Raises only :class:`~engines.cloud.CloudAuthError` and
        :class:`~engines.cloud.CloudAllowanceExhausted`, both of which the
        wiring acts on. Everything else is a visible skip carrying ``text``
        back unchanged.
        """
        assert text is not None, "text is required"
        assert system_prompt, "system_prompt is required"

        started_at = self._clock()
        # Before the request, so a missing lease costs nothing and reaches
        # nobody. This is also the only exception raised without a round trip.
        lease = self._lease()

        max_tokens = max_tokens_for_text(text, CONTEXT_TOKENS)
        if max_tokens < MIN_OUTPUT_TOKENS:
            # The transcript alone fills the window. Sending it would only earn
            # an error and a bill, so skip and hand the original text back.
            return self._skip(text, "too long for cleanup", started_at)

        timeout_s = float(timeout_for_text(text))
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "temperature": float(temperature),
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.chat_url,
            data=payload,
            headers={
                "Content-Type": _CONTENT_TYPE,
                # One key per call. Nothing is retried here, but a call that
                # times out may already have been generated and metered.
                "Idempotency-Key": str(uuid.uuid4()),
                "Authorization": f"Bearer {lease}",
            },
            method="POST",
        )

        try:
            with open_no_cross_host_redirect(request, timeout_s, self._http_open) as response:
                status = int(getattr(response, "status", 200) or 200)
                body = response.read()
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            body = b""
            with contextlib.suppress(Exception):
                body = exc.read() or b""
            exc.close()
        except (TimeoutError, socket.timeout):
            return self._skip(text, f"timed out after {timeout_s:g}s", started_at)
        except urllib.error.URLError as exc:
            # A read that runs out of time surfaces as URLError(timeout) once
            # urllib has wrapped the socket error.
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return self._skip(text, f"timed out after {timeout_s:g}s", started_at)
            return self._skip(text, f"{ENGINE_NAME} is unreachable", started_at)
        except EngineError:
            # The redirect guard. The lease was not handed on, which is the
            # point; losing the transcript over it would not be.
            return self._skip(text, "the proxy redirected the request", started_at)
        except OSError:
            return self._skip(text, f"{ENGINE_NAME} is unreachable", started_at)

        # The allowance is a product state the body declares, at any status —
        # including a 200 from a proxy that queues rather than rejects — and it
        # is the one thing here the wiring must act on rather than skip.
        if _error_code(body) == ALLOWANCE_EXHAUSTED_CODE:
            logger.info("cleanup: %s allowance exhausted; falling back to local", ENGINE_NAME)
            raise CloudAllowanceExhausted(ALLOWANCE_MESSAGE)
        if status == 401:
            raise CloudAuthError(f"{ENGINE_NAME} rejected the lease; link this Mac again")
        if status == 429:
            # A rate limit, not the allowance. One attempt only: cleanup lives
            # inside the paste path's budget and a wait would cost more than the
            # punctuation is worth.
            return self._skip(text, RATE_LIMITED_REASON, started_at)
        if not 200 <= status < 300:
            # The status, never the body: an error message is proxy-authored
            # text and this reason is shown to the user.
            return self._skip(text, f"{ENGINE_NAME} returned HTTP {status}", started_at)

        cleaned = _content_of(body)
        if cleaned is None:
            return self._skip(text, f"{ENGINE_NAME} returned an unreadable reply", started_at)
        if not cleaned.strip():
            return self._skip(text, "the model returned an empty reply", started_at)

        elapsed_s = self._clock() - started_at
        # Lengths and timings only: no transcript, no reply, no lease.
        logger.debug(
            "cloud cleanup ok: %d chars in, %d chars out, %.2fs (budget %.1fs)",
            len(text),
            len(cleaned.strip()),
            elapsed_s,
            timeout_s,
        )
        return CleanupResult(text=cleaned.strip(), skipped=False, elapsed_s=elapsed_s)

    def _skip(self, text: str, reason: str, started_at: float) -> CleanupResult:
        """Hand the original text back with a reason the UI can show."""
        elapsed_s = self._clock() - started_at
        logger.info(
            "cloud cleanup skipped after %.2fs on %d chars: %s",
            elapsed_s,
            len(text),
            reason,
        )
        return CleanupResult(text=text, skipped=True, reason=reason, elapsed_s=elapsed_s)
