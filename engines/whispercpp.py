#!/usr/bin/env python3
"""whisper.cpp engine: a bundled ``whisper-server`` child process spoken to over HTTP.

Decision D2 of the Murmur v2 plan bundles the binary instead of a Python binding.

Contract, VERIFIED against the upstream source at the pinned tag
``v1.7.5`` (``examples/server/server.cpp`` of ggml-org/whisper.cpp):

* There is **no** ``/v1/audio/transcriptions`` route at this tag. The transcription
  route is ``POST /inference`` (``sparams.inference_path``), so that is what this
  client uses. The other routes are ``POST /load``, ``GET /health`` and ``GET /``.
* ``GET /health`` exists and answers ``{"status":"ok"}`` once the model is loaded,
  so it is the readiness probe.
* ``POST /inference`` is ``multipart/form-data``. The audio part must be named
  ``file`` (a missing ``file`` part is the only hard-rejected case). The optional
  text parts this client sends are ``language``, ``prompt`` (mapped to
  ``whisper_full_params.initial_prompt``) and ``response_format``.
* ``response_format=verbose_json`` returns ``{"task", "language", "duration",
  "text", "segments": [{"id", "text", "start", "end", ...}]}`` with per-segment
  timings in seconds, which is what :class:`~engines.base.Transcript` needs.
* The server's own ``language`` default is ``"en"``, and per-request parameters are
  reset from the command line defaults after every request. So the server is
  spawned with ``-l auto``: omitting the ``language`` part then means
  auto-detection rather than a silent switch to English.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from pathlib import Path

from engines.base import (
    LANGUAGE_AUTO,
    WHISPER_LANGUAGES,
    Engine,
    EngineError,
    EngineInfo,
    EngineUnavailableError,
    Hints,
    Segment,
    Transcript,
    normalize_language_code,
)

#: Upstream tag the bundled binary is built from. Kept in step with
#: ``scripts/tools/fetch_whispercpp.sh``.
WHISPER_CPP_TAG = "v1.7.5"

#: Engine id, also the value of :attr:`engines.base.Transcript.engine_id`.
ENGINE_ID = "whispercpp"

#: Name of the binary produced by the fetch script and bundled by PyInstaller.
BINARY_NAME = "whisper-server"

#: Environment override, checked first by :func:`resolve_whisper_server_binary`.
BINARY_ENV_VAR = "MURMUR_WHISPER_SERVER"

#: Path of the helper that builds the binary, named in the "missing" error.
FETCH_SCRIPT = "scripts/tools/fetch_whispercpp.sh"

#: Routes, verified against the pinned tag (see the module docstring).
HEALTH_PATH = "/health"
INFERENCE_PATH = "/inference"

#: ``verbose_json`` is the only format carrying segments and duration.
RESPONSE_FORMAT = "verbose_json"

_HOST = "127.0.0.1"
_HEALTH_POLL_INTERVAL_S = 0.05
_HEALTH_REQUEST_TIMEOUT_S = 2.0
_TRANSCRIBE_TIMEOUT_S = 600.0
_TERMINATE_GRACE_S = 3.0
_STDERR_TAIL_CHARS = 2000
_STDERR_TAIL_LINES = 50
_DRAIN_JOIN_S = 1.0


def _repo_root() -> Path:
    """Directory holding the ``engines`` package, i.e. the repository root."""
    return Path(__file__).resolve().parent.parent


def resolve_whisper_server_binary() -> Path:
    """Locate the bundled ``whisper-server``.

    Order: ``MURMUR_WHISPER_SERVER``, then ``<_MEIPASS>/bin/whisper-server`` in a
    frozen bundle, then ``<repo>/vendor/whispercpp/whisper-server``.

    Raises :class:`EngineUnavailableError` naming the fetch script when absent.
    """
    candidates: list[Path] = []

    override = os.environ.get(BINARY_ENV_VAR)
    if override:
        candidates.append(Path(override).expanduser())

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "bin" / BINARY_NAME)

    candidates.append(_repo_root() / "vendor" / "whispercpp" / BINARY_NAME)

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


def _encode_multipart(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    """Encode ``fields`` plus one file part; return ``(body, content_type)``."""
    boundary = f"----MurmurWhisperCpp{uuid.uuid4().hex}"
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


class _StderrDrain:
    """Reads a child's stderr on a daemon thread into a bounded line buffer.

    ``whisper-server`` logs a few lines per request. Nobody was reading the
    pipe, so once the ~64 KiB kernel buffer filled the child blocked on its
    next write and ``POST /inference`` hung until the request timeout. Draining
    keeps the pipe empty for the whole life of the process, and the last
    :data:`_STDERR_TAIL_LINES` lines are kept for :meth:`tail`.

    The lines are never logged; they only ever reach an :class:`EngineError`
    message the caller already chose to surface.
    """

    def __init__(self, stream) -> None:
        assert stream is not None, "stream is required"
        self._stream = stream
        self._lines: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="whispercpp-stderr", daemon=True
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
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
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


class WhisperCppEngine(Engine):
    """Runs ``whisper-server`` on loopback and transcribes over ``POST /inference``.

    ``spawn`` and ``http_open`` are seams: tests inject a recording spawn and a
    real ``urlopen`` pointed at a fake HTTP server.
    """

    supports_streaming = False
    supports_hints = True

    def __init__(
        self,
        model_path: Path,
        binary: Path | None = None,
        spawn=subprocess.Popen,
        http_open=urllib.request.urlopen,
        startup_timeout_s: float = 30.0,
    ) -> None:
        assert model_path is not None, "model_path is required"
        assert spawn is not None, "spawn is required"
        assert http_open is not None, "http_open is required"
        assert startup_timeout_s > 0, "startup_timeout_s must be positive"
        self._model_path = Path(model_path)
        self._binary = Path(binary) if binary is not None else None
        self._spawn = spawn
        self._http_open = http_open
        self._startup_timeout_s = float(startup_timeout_s)
        self._process = None
        self._stderr: _StderrDrain | None = None
        self._port: int | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._process is not None and self._port is not None

    def _resolve_binary(self) -> Path:
        """Explicit binary wins; it must exist, like a resolved one."""
        if self._binary is None:
            return resolve_whisper_server_binary()
        if not self._binary.is_file():
            raise EngineUnavailableError(
                f"{BINARY_NAME} not found at {self._binary}. "
                f"Build it with `bash {FETCH_SCRIPT}` or set {BINARY_ENV_VAR}."
            )
        return self._binary

    def load(self) -> None:
        """Spawn the server on a free port and wait for ``/health``. Idempotent."""
        if self.is_loaded:
            return

        binary = self._resolve_binary()
        port = _free_port()
        argv = [
            str(binary),
            "-m",
            str(self._model_path),
            "--host",
            _HOST,
            "--port",
            str(port),
            # Server-side default; an omitted request "language" then auto-detects.
            "-l",
            LANGUAGE_AUTO,
        ]

        # stdout is discarded: nothing reads it, and a PIPE nobody drains is a
        # deadlock. stderr stays a PIPE only because _StderrDrain empties it.
        process = self._spawn(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        drain = _StderrDrain.attach(process)
        deadline = time.monotonic() + self._startup_timeout_s
        while True:
            if process.poll() is not None:
                exit_code = process.returncode
                self._stop(process, drain)
                detail = _stderr_tail(drain)
                raise EngineError(
                    f"{BINARY_NAME} exited with code {exit_code} during startup"
                    + (f": {detail}" if detail else "")
                )
            if self._health_ok(port):
                self._process = process
                self._stderr = drain
                self._port = port
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(_HEALTH_POLL_INTERVAL_S)

        self._stop(process, drain)
        detail = _stderr_tail(drain)
        raise EngineError(
            f"{BINARY_NAME} did not answer {HEALTH_PATH} within "
            f"{self._startup_timeout_s:g}s on port {port}"
            + (f": {detail}" if detail else "")
        )

    def _health_ok(self, port: int) -> bool:
        """True when ``GET /health`` answers 2xx. Any failure means "not yet"."""
        request = urllib.request.Request(f"http://{_HOST}:{port}{HEALTH_PATH}", method="GET")
        try:
            with self._http_open(request, timeout=_HEALTH_REQUEST_TIMEOUT_S) as response:
                return 200 <= int(getattr(response, "status", 200) or 200) < 300
        except Exception:
            return False

    @staticmethod
    def _stop(process, drain: "_StderrDrain | None" = None) -> None:
        """Terminate, wait ``_TERMINATE_GRACE_S``, then kill. Never raises.

        The dead child's pipes are closed and the stderr reader joined, so
        neither a file descriptor nor a thread outlives the process. The
        drained tail survives the join and stays readable afterwards.
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

    def unload(self) -> None:
        """Stop the child process, close its pipes and join the drain. Idempotent."""
        process, self._process = self._process, None
        drain, self._stderr = self._stderr, None
        self._port = None
        self._stop(process, drain)

    def info(self) -> EngineInfo:
        """Static description; the model id is the GGUF file stem.

        ``languages`` is the shared :data:`~engines.base.WHISPER_LANGUAGES` list,
        not ``("auto",)``: the server detects the language when asked to, but it
        also accepts an explicit ISO code, and a one-row list would leave the
        Settings picker with nothing to pick.
        """
        try:
            size_bytes = self._model_path.stat().st_size
        except OSError:
            size_bytes = 0
        return EngineInfo(
            id=ENGINE_ID,
            name="whisper.cpp",
            model_id=self._model_path.stem,
            size_bytes=size_bytes,
            languages=WHISPER_LANGUAGES,
            supports_streaming=self.supports_streaming,
            supports_hints=self.supports_hints,
            requires_apple_silicon=False,
        )

    # -- transcription -----------------------------------------------------

    def _transcribe(
        self,
        wav_path: Path,
        language: str | None,
        hints: Hints | None,
        long_form: bool,
    ) -> Transcript:
        """POST the WAV to ``/inference`` and parse ``verbose_json``.

        ``long_form`` is accepted and ignored: the server owns its own windowing
        and exposes no per-request switch for conditioning on previous text.
        """
        wav_path = Path(wav_path)
        try:
            audio = wav_path.read_bytes()
        except OSError as exc:
            raise EngineError(f"cannot read {wav_path}: {exc}") from exc

        fields: dict[str, str] = {"response_format": RESPONSE_FORMAT}
        if language is not None and language != LANGUAGE_AUTO:
            fields["language"] = language
        prompt = hints.as_prompt_text() if hints is not None else None
        if prompt:
            fields["prompt"] = prompt

        body, content_type = _encode_multipart(fields, "file", wav_path.name, audio)
        request = urllib.request.Request(
            f"http://{_HOST}:{self._port}{INFERENCE_PATH}",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )

        try:
            with self._http_open(request, timeout=_TRANSCRIBE_TIMEOUT_S) as response:
                status = int(getattr(response, "status", 200) or 200)
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raise EngineError(
                f"{BINARY_NAME} {INFERENCE_PATH} returned HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise EngineError(f"{BINARY_NAME} {INFERENCE_PATH} is unreachable: {exc}") from exc

        if not 200 <= status < 300:
            raise EngineError(f"{BINARY_NAME} {INFERENCE_PATH} returned HTTP {status}")

        return self._parse_response(payload, hints_applied=True if prompt else None)

    @staticmethod
    def _parse_response(payload: bytes, hints_applied: bool | None = None) -> Transcript:
        """Turn a ``verbose_json`` body into a :class:`Transcript`."""
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EngineError(f"{BINARY_NAME} returned a body that is not JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise EngineError(f"{BINARY_NAME} returned a JSON {type(data).__name__}, expected an object")
        if "error" in data and "text" not in data:
            raise EngineError(f"{BINARY_NAME} reported: {data['error']}")

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

        duration = data.get("duration")
        language = data.get("language")
        return Transcript(
            text=str(data.get("text", "")).strip(),
            language=normalize_language_code(str(language) if language else None),
            duration_s=float(duration) if duration is not None else None,
            segments=tuple(segments),
            engine_id=ENGINE_ID,
            hints_applied=hints_applied,
        )


#: Consumed by :func:`engines.create_engine`.
ENGINE_CLASS = WhisperCppEngine
