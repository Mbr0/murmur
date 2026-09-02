#!/usr/bin/env python3
"""Bundled ``llama-server`` child process and the cleanup chat client.

Decision D3 of the Murmur v2 plan runs cleanup locally on a ~3B GGUF served by
a bundled ``llama-server``, spoken to over the OpenAI-compatible chat route.
The child-process pattern (free-port probe, health poll, bounded stderr drain,
``DEVNULL`` stdout) is the one :mod:`engines.whispercpp` established for D2.

Contract, VERIFIED against the upstream ``tools/server/README.md`` at the
pinned tag :data:`LLAMA_CPP_TAG`:

* ``GET /health`` answers **503** with ``{"error": {"code": 503, "message":
  "Loading model", ...}}`` while the model loads and **200** with
  ``{"status": "ok"}`` once it is ready. So a 503 means "not yet", not
  "broken", and the readiness poll simply keeps asking.
* ``POST /v1/chat/completions`` is the OpenAI-compatible chat route and
  returns ``choices[0].message.content``.
* The flags used below are all in the pinned tag's option table:
  ``-m, --model``, ``--host``, ``--port``, ``-c, --ctx-size``,
  ``--log-disable`` and ``--no-webui``. ``--jinja`` is on by default at this
  tag, so the GGUF's own chat template is applied and none is passed here.

Two rules from the work folder shape the error handling:

* Cleanup is a **best-effort improvement**, never a way to lose a transcript.
  A timeout, an HTTP error or an empty reply comes back as
  :class:`CleanupResult` with ``skipped=True`` and the original text, and the
  caller shows a visible "cleanup skipped" notice. It is not a silent drop.
* Asking a stopped server to clean text is a programming error, so that one
  raises :class:`LlamaServerError` immediately.

No transcript text is ever logged: only lengths, timings and reasons.
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from engines.base import EngineError, EngineUnavailableError
from engines.model_store import HF_RESOLVE_URL, ModelFile, ModelSpec
from services.model_profile_service import detect_ram_gb

logger = logging.getLogger(__name__)

#: Upstream tag the bundled binary is built from. Kept in step with
#: ``scripts/tools/fetch_llama.sh``.
LLAMA_CPP_TAG = "v0.3.0"

#: Name of the binary produced by the fetch script and bundled by PyInstaller.
BINARY_NAME = "llama-server"

#: Environment override, checked first by :func:`resolve_llama_server_binary`.
BINARY_ENV_VAR = "MURMUR_LLAMA_SERVER"

#: Path of the helper that builds the binary, named in the "missing" error.
FETCH_SCRIPT = "scripts/tools/fetch_llama.sh"

#: Routes, verified against the pinned tag (see the module docstring).
HEALTH_PATH = "/health"
CHAT_PATH = "/v1/chat/completions"

#: Engine field of :data:`CLEANUP_MODEL_SPEC`. The cleanup model is not a
#: speech engine, so it carries its own id and the app composes it into the
#: store catalog rather than :mod:`engines.model_store` hard-coding it.
CLEANUP_ENGINE = "llama_cleanup"

#: Catalog id of the cleanup model.
CLEANUP_MODEL_ID = "ministral-3-3b-instruct-2512-q4_k_m"

_CLEANUP_REPO = "mistralai/Ministral-3-3B-Instruct-2512-GGUF"
_CLEANUP_FILE = "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"

#: The cleanup model, to be composed into the store catalog by the app as
#: ``ModelStore(catalog=CATALOG + (CLEANUP_MODEL_SPEC,))``.
#:
#: Mistral's own GGUF repository, Apache-2.0 and ungated as decision D3
#: requires. Size and digest were read from Hugging Face blob metadata
#: (``/api/models/<repo>?blobs=true``) on 2026-09-02 and pin the ``main``
#: revision as it stood then; the digest is the LFS ``sha256`` the API
#: publishes, cross-checked against the ``X-Linked-Etag`` of the resolve URL.
CLEANUP_MODEL_SPEC: ModelSpec = ModelSpec(
    id=CLEANUP_MODEL_ID,
    engine=CLEANUP_ENGINE,
    display_name="Ministral 3 3B Instruct (Q4_K_M, 2.1 GB)",
    files=(
        ModelFile(
            name=_CLEANUP_FILE,
            size_bytes=2147023008,
            sha256="9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8",
            url=HF_RESOLVE_URL.format(repo=_CLEANUP_REPO, name=_CLEANUP_FILE),
        ),
    ),
    source=f"https://huggingface.co/{_CLEANUP_REPO}",
    license="Apache-2.0",
)

#: Risk register: "Cleanup latency on 8 GB Macs → cleanup off by default below
#: 16 GB; user can enable with a warning."
CLEANUP_MIN_RAM_GB = 16

#: Timeout policy from the plan: 2 s per 100 words, floored and capped.
TIMEOUT_SECONDS_PER_100_WORDS = 2.0
TIMEOUT_MIN_S = 3.0
TIMEOUT_MAX_S = 20.0

#: Output budget: cleanup rewrites, it does not expand. 1.5 tokens per input
#: word covers a typical token-per-word ratio plus punctuation, and the
#: headroom absorbs a short trailing sentence.
MAX_TOKENS_PER_WORD = 1.5
MAX_TOKENS_HEADROOM = 64

#: Size of the ``-c`` window the child is started with.
DEFAULT_CONTEXT_TOKENS = 4096

#: Tokens per word used to estimate how much of the window the prompt eats.
#: Higher than :data:`MAX_TOKENS_PER_WORD` on purpose: over-estimating the
#: prompt shrinks the reply, while under-estimating it overflows the window and
#: the whole call fails.
TOKENS_PER_WORD = 1.4

#: Kept free at the end of the window for the chat template's own tokens.
CONTEXT_RESERVE_TOKENS = 64

#: Below this many output tokens there is no point asking: whatever came back
#: would be a truncated fragment of the user's text, which is worse than the
#: text itself. The caller skips instead.
MIN_OUTPUT_TOKENS = 32

_HOST = "127.0.0.1"
_HEALTH_POLL_INTERVAL_S = 0.05
_HEALTH_REQUEST_TIMEOUT_S = 2.0
_TERMINATE_GRACE_S = 3.0
_STDERR_TAIL_CHARS = 2000
_STDERR_TAIL_LINES = 50
_DRAIN_JOIN_S = 1.0


class LlamaServerError(EngineError):
    """The cleanup server could not be started, or was asked to work stopped."""


def _repo_root() -> Path:
    """Directory holding the ``cleanup`` package, i.e. the repository root."""
    return Path(__file__).resolve().parent.parent


def resolve_llama_server_binary() -> Path:
    """Locate the bundled ``llama-server``.

    Order: ``MURMUR_LLAMA_SERVER``, then ``<_MEIPASS>/bin/llama-server`` in a
    frozen bundle, then ``<repo>/vendor/llamacpp/llama-server``.

    Raises :class:`~engines.base.EngineUnavailableError` naming the fetch
    script when absent, exactly as the whisper.cpp resolver does.
    """
    candidates: list[Path] = []

    override = os.environ.get(BINARY_ENV_VAR)
    if override:
        candidates.append(Path(override).expanduser())

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "bin" / BINARY_NAME)

    candidates.append(_repo_root() / "vendor" / "llamacpp" / BINARY_NAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    looked_in = ", ".join(str(candidate) for candidate in candidates)
    raise EngineUnavailableError(
        f"{BINARY_NAME} not found (looked in: {looked_in}). "
        f"Build it with `bash {FETCH_SCRIPT}` or set {BINARY_ENV_VAR}."
    )


def _free_port() -> int:
    """Ask the kernel for a free loopback port and release it immediately."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((_HOST, 0))
        return int(probe.getsockname()[1])


def _word_count(text: str) -> int:
    """Whitespace-separated word count; the unit both budgets are built on."""
    return len(text.split())


def timeout_for_text(text: str) -> float:
    """Seconds to allow cleanup of ``text``: 2 s per 100 words, 3 s to 20 s.

    The floor keeps a one-line message from failing on model warm-up alone;
    the cap keeps a pathological paste from holding the paste path hostage.
    """
    budget = _word_count(text) * TIMEOUT_SECONDS_PER_100_WORDS / 100.0
    return min(max(budget, TIMEOUT_MIN_S), TIMEOUT_MAX_S)


def estimate_prompt_tokens(text: str, system_prompt: str = "") -> int:
    """Rough size of the chat prompt in tokens: words x 1.4, rounded up.

    Deliberately crude and deliberately generous. The number exists only to
    keep :func:`max_tokens_for_text` inside the context window, and an
    over-estimate costs a slightly shorter reply while an under-estimate costs
    the whole request.
    """
    words = _word_count(text) + _word_count(system_prompt)
    return math.ceil(words * TOKENS_PER_WORD)


def max_tokens_for_text(
    text: str,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    prompt_tokens_estimate: int | None = None,
) -> int:
    """Generation budget for cleaning ``text``, capped by the context window.

    The uncapped budget is input words x 1.5 plus 64 — cleanup rewrites, it
    does not expand. The cap is what keeps a long paste honest: the child is
    started with ``-c 4096``, so a budget that ignores the prompt already
    sitting in that window makes ``llama-server`` reject the request and every
    call for a long transcript skips.

    The result can be smaller than :data:`MIN_OUTPUT_TOKENS`, or negative: that
    is the caller's signal that the input alone leaves no room to answer.
    """
    budget = int(_word_count(text) * MAX_TOKENS_PER_WORD) + MAX_TOKENS_HEADROOM
    prompt_tokens = (
        estimate_prompt_tokens(text)
        if prompt_tokens_estimate is None
        else int(prompt_tokens_estimate)
    )
    room = int(context_tokens) - CONTEXT_RESERVE_TOKENS - prompt_tokens
    return min(budget, room)


def should_enable_cleanup_by_default(ram_gb: int | None) -> bool:
    """Whether cleanup starts switched on, per the plan's 16 GB risk mitigation.

    Unknown RAM answers False: the failure mode of guessing high is a Mac that
    swaps while the user waits for their text, so the probe failing means the
    conservative answer, not an optimistic one.
    """
    if ram_gb is None:
        return False
    return ram_gb >= CLEANUP_MIN_RAM_GB


def cleanup_default_for_current_machine() -> bool:
    """:func:`should_enable_cleanup_by_default` applied to this machine's RAM."""
    return should_enable_cleanup_by_default(detect_ram_gb())


class _StderrDrain:
    """Reads a child's stderr on a daemon thread into a bounded line buffer.

    Same reason as :mod:`engines.whispercpp`: a ``PIPE`` nobody reads fills its
    ~64 KiB kernel buffer and then blocks the child on its next write, which
    for a server means the *next request* hangs. Draining keeps the pipe empty
    for the life of the process and keeps the last
    :data:`_STDERR_TAIL_LINES` lines for :meth:`tail`.

    Those lines are never logged; they only ever reach a
    :class:`LlamaServerError` message the caller already chose to surface.
    """

    def __init__(self, stream) -> None:
        assert stream is not None, "stream is required"
        self._stream = stream
        self._lines: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="llama-server-stderr", daemon=True
        )
        self._thread.start()

    @classmethod
    def attach(cls, process) -> "_StderrDrain | None":
        """Drain ``process.stderr``, or return None when there is no pipe."""
        stream = getattr(process, "stderr", None)
        return cls(stream) if stream is not None else None

    def _run(self) -> None:
        """Read until EOF. A closed or broken stream just ends the thread."""
        try:
            while True:
                raw = self._stream.readline()
                if not raw:
                    return
                line = (
                    raw.decode("utf-8", errors="replace")
                    if isinstance(raw, bytes)
                    else str(raw)
                )
                with self._lock:
                    self._lines.append(line)
        except Exception:
            return

    def tail(self) -> str:
        """Bounded tail of what the child said. Never raises."""
        with self._lock:
            text = "".join(self._lines)
        return text[-_STDERR_TAIL_CHARS:]

    def join(self, timeout: float = _DRAIN_JOIN_S) -> None:
        """Wait briefly for the reader to see EOF. Never raises."""
        try:
            self._thread.join(timeout)
        except Exception:
            pass


def _stderr_tail(drain: "_StderrDrain | None") -> str:
    """Bounded tail of the drained child stderr, or "" when nothing was drained."""
    return drain.tail() if drain is not None else ""


class LlamaServer:
    """Runs ``llama-server`` on loopback and owns its lifetime.

    ``spawn`` and ``http_open`` are seams: tests inject a recording spawn and a
    real ``urlopen`` pointed at a fake HTTP server. :attr:`http_open` is read
    back by :class:`CleanupClient` so both talk through the same seam.
    """

    def __init__(
        self,
        model_path: Path,
        binary: Path | None = None,
        spawn=subprocess.Popen,
        http_open=urllib.request.urlopen,
        startup_timeout_s: float = 60.0,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    ) -> None:
        assert model_path is not None, "model_path is required"
        assert spawn is not None, "spawn is required"
        assert http_open is not None, "http_open is required"
        assert startup_timeout_s > 0, "startup_timeout_s must be positive"
        assert context_tokens > 0, "context_tokens must be positive"
        self._model_path = Path(model_path)
        self._binary = Path(binary) if binary is not None else None
        self._spawn = spawn
        self._http_open = http_open
        self._startup_timeout_s = float(startup_timeout_s)
        self._context_tokens = int(context_tokens)
        self._process = None
        self._stderr: _StderrDrain | None = None
        self._port: int | None = None
        self._exit_code: int | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while the child is alive, answered ``/health`` and was not stopped.

        ``poll()`` is what catches a child that died under us — an OOM kill on
        an 8 GB Mac is the realistic case. Without it the port stays set, every
        later request fails to connect, and the client skips cleanup as
        "unreachable" for the rest of the session. Finding the corpse here also
        reaps it, so the next :meth:`start` spawns a fresh child instead of
        returning early.
        """
        process = self._process
        if process is None or self._port is None:
            return False
        exit_code = process.poll()
        if exit_code is None:
            return True
        # Read before stop(): terminating a corpse overwrites its returncode.
        self._exit_code = int(exit_code)
        logger.warning("%s exited on its own with code %s", BINARY_NAME, exit_code)
        self.stop()
        return False

    @property
    def exit_code(self) -> int | None:
        """Status of a child that died on its own, or None if none has.

        Only a crash sets it: a deliberate :meth:`stop` is not a failure, and
        :meth:`start` clears it so it never describes an older run.
        """
        return self._exit_code

    @property
    def context_tokens(self) -> int:
        """Size of the ``-c`` window the child is started with."""
        return self._context_tokens

    @property
    def port(self) -> int | None:
        """Loopback port the child is listening on, or None when stopped."""
        return self._port

    @property
    def http_open(self):
        """The ``urlopen``-shaped callable this server (and its client) uses."""
        return self._http_open

    def url(self, path: str) -> str:
        """Absolute URL of ``path`` on the running child."""
        return f"http://{_HOST}:{self._port}{path}"

    def _resolve_binary(self) -> Path:
        """Explicit binary wins; it must exist, like a resolved one."""
        if self._binary is None:
            return resolve_llama_server_binary()
        if not self._binary.is_file():
            raise EngineUnavailableError(
                f"{BINARY_NAME} not found at {self._binary}. "
                f"Build it with `bash {FETCH_SCRIPT}` or set {BINARY_ENV_VAR}."
            )
        return self._binary

    def argv(self, binary: Path, port: int) -> list[str]:
        """Command line for the child. Every flag is in the pinned tag's table."""
        return [
            str(binary),
            "-m",
            str(self._model_path),
            "--host",
            _HOST,
            "--port",
            str(port),
            "-c",
            str(self._context_tokens),
            # A background helper has no use for the bundled Web UI, and its
            # logs would only ever hold user text.
            "--no-webui",
            "--log-disable",
        ]

    def start(self) -> None:
        """Spawn the server on a free port and wait for ``/health``. Idempotent.

        ``/health`` answers 503 for as long as the GGUF is loading, which on a
        2 GB model is most of the startup window, so a 503 is polled through
        rather than treated as a failure.
        """
        if self.is_running:
            return
        self._exit_code = None

        binary = self._resolve_binary()
        port = _free_port()
        argv = self.argv(binary, port)

        # stdout is discarded: nothing reads it, and a PIPE nobody drains is a
        # deadlock. stderr stays a PIPE only because _StderrDrain empties it.
        started_at = time.monotonic()
        process = self._spawn(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        drain = _StderrDrain.attach(process)
        deadline = started_at + self._startup_timeout_s
        while True:
            if process.poll() is not None:
                exit_code = process.returncode
                self._stop(process, drain)
                detail = _stderr_tail(drain)
                raise LlamaServerError(
                    f"{BINARY_NAME} exited with code {exit_code} during startup"
                    + (f": {detail}" if detail else "")
                )
            if self._health_ok(port):
                self._process = process
                self._stderr = drain
                self._port = port
                logger.info(
                    "%s ready on port %d after %.1fs",
                    BINARY_NAME,
                    port,
                    time.monotonic() - started_at,
                )
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(_HEALTH_POLL_INTERVAL_S)

        self._stop(process, drain)
        detail = _stderr_tail(drain)
        raise LlamaServerError(
            f"{BINARY_NAME} did not answer {HEALTH_PATH} within "
            f"{self._startup_timeout_s:g}s on port {port}"
            + (f": {detail}" if detail else "")
        )

    def _health_ok(self, port: int) -> bool:
        """True when ``GET /health`` answers 2xx. A 503 or any failure is "not yet"."""
        request = urllib.request.Request(
            f"http://{_HOST}:{port}{HEALTH_PATH}", method="GET"
        )
        try:
            with self._http_open(request, timeout=_HEALTH_REQUEST_TIMEOUT_S) as response:
                return 200 <= int(getattr(response, "status", 200) or 200) < 300
        except Exception:
            return False

    @staticmethod
    def _stop(process, drain: "_StderrDrain | None" = None) -> None:
        """Terminate, wait ``_TERMINATE_GRACE_S``, then kill. Never raises.

        The dead child's pipes are closed and the stderr reader joined, so
        neither a file descriptor nor a thread outlives the process.
        """
        if process is None:
            return
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.wait(timeout=_TERMINATE_GRACE_S)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=_TERMINATE_GRACE_S)
            except Exception:
                pass
        # Joining before closing lets the reader finish on the child's own EOF.
        if drain is not None:
            drain.join()
        for name in ("stdout", "stderr"):
            stream = getattr(process, name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the child, close its pipes and join the drain. Idempotent."""
        process, self._process = self._process, None
        drain, self._stderr = self._stderr, None
        self._port = None
        self._stop(process, drain)


@dataclass(frozen=True)
class CleanupResult:
    """What the cleanup pass produced, and whether it actually ran.

    ``text`` is always safe to insert: on a skip it is the caller's original
    string, byte for byte. ``skipped`` with a ``reason`` is what the UI turns
    into the visible "cleanup skipped" notice, so a skip is never silent.
    """

    text: str
    skipped: bool = False
    reason: str | None = None
    elapsed_s: float = 0.0


class CleanupClient:
    """Cleans transcript text through ``POST /v1/chat/completions``.

    Everything that can go wrong at request time — the model being slow, the
    server erroring, the reply coming back blank or malformed, the transcript
    being longer than the context window holds — degrades to
    :class:`CleanupResult` with ``skipped=True`` and the untouched input. The
    exceptions are a server that was never started (the caller's bug) and one
    whose child died under it (a restart, not a retry): both raise.
    """

    def __init__(self, server: LlamaServer, timeout_policy=timeout_for_text) -> None:
        assert server is not None, "server is required"
        assert callable(timeout_policy), "timeout_policy must be callable"
        self._server = server
        self._timeout_policy = timeout_policy

    def cleanup(
        self,
        text: str,
        system_prompt: str,
        temperature: float = 0.1,
    ) -> CleanupResult:
        """Rewrite ``text`` under ``system_prompt``; never raise on a bad reply."""
        assert text is not None, "text is required"
        assert system_prompt, "system_prompt is required"
        if not self._server.is_running:
            exit_code = self._server.exit_code
            if exit_code is not None:
                # A child that died is not a bad reply to degrade around: the
                # wiring restarts the server lazily, so say so and fail fast.
                raise LlamaServerError(f"{BINARY_NAME} exited (code {exit_code})")
            raise LlamaServerError(
                f"{BINARY_NAME} is not running; call start() before cleanup()"
            )

        started_at = time.monotonic()
        prompt_tokens = estimate_prompt_tokens(text, system_prompt)
        max_tokens = max_tokens_for_text(text, self._server.context_tokens, prompt_tokens)
        if max_tokens < MIN_OUTPUT_TOKENS:
            # The transcript alone fills the window. Sending it would only earn
            # an error, so skip visibly and hand the original text straight back.
            return self._skip(text, "too long for cleanup", started_at)

        timeout_s = float(self._timeout_policy(text))
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "temperature": float(temperature),
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            self._server.url(CHAT_PATH),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with self._server.http_open(request, timeout=timeout_s) as response:
                status = int(getattr(response, "status", 200) or 200)
                body = response.read()
        except (TimeoutError, socket.timeout):
            return self._skip(text, f"timed out after {timeout_s:g}s", started_at)
        except urllib.error.HTTPError as exc:
            exc.close()
            return self._skip(text, f"{BINARY_NAME} returned HTTP {exc.code}", started_at)
        except urllib.error.URLError as exc:
            # A read that runs out of time surfaces as URLError(timeout) rather
            # than TimeoutError once urllib has wrapped the socket error.
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return self._skip(text, f"timed out after {timeout_s:g}s", started_at)
            return self._skip(text, f"{BINARY_NAME} is unreachable", started_at)
        except OSError:
            return self._skip(text, f"{BINARY_NAME} is unreachable", started_at)

        if not 200 <= status < 300:
            return self._skip(text, f"{BINARY_NAME} returned HTTP {status}", started_at)

        cleaned = _content_of(body)
        if cleaned is None:
            return self._skip(text, f"{BINARY_NAME} returned an unreadable reply", started_at)
        if not cleaned.strip():
            return self._skip(text, "the model returned an empty reply", started_at)

        elapsed_s = time.monotonic() - started_at
        # Lengths and timings only: transcript text never reaches a log.
        logger.debug(
            "cleanup ok: %d chars in, %d chars out, %.2fs (budget %.1fs)",
            len(text),
            len(cleaned.strip()),
            elapsed_s,
            timeout_s,
        )
        return CleanupResult(text=cleaned.strip(), skipped=False, elapsed_s=elapsed_s)

    @staticmethod
    def _skip(text: str, reason: str, started_at: float) -> CleanupResult:
        """Hand the original text back with a reason the UI can show."""
        elapsed_s = time.monotonic() - started_at
        logger.info(
            "cleanup skipped after %.2fs on %d chars: %s", elapsed_s, len(text), reason
        )
        return CleanupResult(text=text, skipped=True, reason=reason, elapsed_s=elapsed_s)


def _content_of(body: bytes) -> str | None:
    """``choices[0].message.content`` of a chat completion, or None if absent.

    None means "this body cannot be read as a completion" and is a skip, not
    an exception: a malformed reply must not cost the user their transcript.
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


__all__ = [
    "BINARY_ENV_VAR",
    "BINARY_NAME",
    "CHAT_PATH",
    "CLEANUP_ENGINE",
    "CLEANUP_MIN_RAM_GB",
    "CLEANUP_MODEL_ID",
    "CLEANUP_MODEL_SPEC",
    "CONTEXT_RESERVE_TOKENS",
    "DEFAULT_CONTEXT_TOKENS",
    "FETCH_SCRIPT",
    "HEALTH_PATH",
    "LLAMA_CPP_TAG",
    "MAX_TOKENS_HEADROOM",
    "MAX_TOKENS_PER_WORD",
    "MIN_OUTPUT_TOKENS",
    "TIMEOUT_MAX_S",
    "TIMEOUT_MIN_S",
    "TIMEOUT_SECONDS_PER_100_WORDS",
    "TOKENS_PER_WORD",
    "CleanupClient",
    "CleanupResult",
    "LlamaServer",
    "LlamaServerError",
    "cleanup_default_for_current_machine",
    "estimate_prompt_tokens",
    "max_tokens_for_text",
    "resolve_llama_server_binary",
    "should_enable_cleanup_by_default",
    "timeout_for_text",
]
