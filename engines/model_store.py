#!/usr/bin/env python3
"""Catalog, downloader and integrity store for the local speech models.

Models live under ``~/Library/Application Support/Murmur/models/<model_id>/``.
A model is one or more files; multi-file models are directories, single-file
models are a directory holding one file, so the layout never special-cases.

Design rules carried from the work folder:

* Fail fast. A checksum mismatch is an error, never a silent re-download.
* Nothing about file *contents* is ever logged; only names, sizes and ids.
* Stdlib only, so the store works inside the PyInstaller bundle unchanged.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

#: Where models are installed when the caller does not override the root.
DEFAULT_ROOT = Path.home() / "Library" / "Application Support" / "Murmur" / "models"

#: Read/write granularity for downloads and hashing.
CHUNK_BYTES = 1024 * 1024

#: Template for a Hugging Face ``main`` revision download URL.
HF_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/main/{name}"


class ModelStoreError(Exception):
    """Base class for every error raised by this module."""


class UnknownModelError(ModelStoreError):
    """The requested model id is not in the catalog."""


class ModelIntegrityError(ModelStoreError):
    """A file is missing, truncated, unverifiable, or has the wrong sha256."""


class DownloadCancelled(ModelStoreError):
    """The caller's cancel event was set while a download was in flight."""


class DownloadIncomplete(ModelStoreError):
    """The transfer ended early; the ``.part`` is kept so it can be resumed.

    Deliberately **not** a :class:`ModelIntegrityError`: bytes that never
    arrived are not corrupt bytes. A server that closes the connection
    mid-body ends the read loop with a clean EOF and no exception, so this is
    the ordinary shape of a dropped download, and re-running
    :meth:`ModelStore.download` picks it up where it stopped.
    """


@dataclass(frozen=True)
class ModelFile:
    """One downloadable file of a model.

    ``sha256`` is the lowercase hex digest published by the source, or ``""``
    when no checksum is on record. A blank checksum is honest ignorance: the
    file still downloads (its size is checked), but :meth:`ModelStore.verify`
    refuses to certify it.
    """

    name: str
    size_bytes: int
    sha256: str
    url: str


@dataclass(frozen=True)
class ModelSpec:
    """A catalog entry: everything needed to fetch and identify one model."""

    id: str
    engine: str
    display_name: str
    files: tuple[ModelFile, ...]
    source: str
    license: str

    @property
    def size_bytes(self) -> int:
        """Total on-disk size of every file in the model."""
        return sum(item.size_bytes for item in self.files)


@dataclass(frozen=True)
class DownloadProgress:
    """One progress tick for a single file of a model.

    ``bytes_done``/``bytes_total`` describe the *file*, not the whole model.
    ``done`` is True on the last tick emitted for that file, so the final tick
    of a download always carries ``done=True``.

    Callbacks run on the thread that called :meth:`ModelStore.download`; a UI
    must hop to its own thread itself.
    """

    model_id: str
    file_name: str
    bytes_done: int
    bytes_total: int
    done: bool


def _hf_file(repo: str, name: str, size_bytes: int, sha256: str) -> ModelFile:
    """Build a :class:`ModelFile` for ``name`` in the Hugging Face ``repo``."""
    return ModelFile(
        name=name,
        size_bytes=size_bytes,
        sha256=sha256,
        url=HF_RESOLVE_URL.format(repo=repo, name=name),
    )


_WHISPER_REPO = "ggerganov/whisper.cpp"
_VOXTRAL_REPO = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"

#: The models Murmur offers.
#:
#: Sizes and digests were read from Hugging Face blob metadata
#: (``/api/models/<repo>?blobs=true``) on 2026-09-02 and pin the ``main``
#: revision as it stood then. LFS blobs publish their own sha256; the two
#: small Voxtral JSON files are not LFS, so their digests were computed from
#: the downloaded bytes (their sizes match the API metadata exactly). Nothing
#: here is guessed — a file we cannot get a digest for would carry ``""``.
CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="whispercpp-large-v3-turbo-q5_0",
        engine="whispercpp",
        display_name="Whisper large-v3-turbo (quantised)",
        files=(
            _hf_file(
                _WHISPER_REPO,
                "ggml-large-v3-turbo-q5_0.bin",
                574041195,
                "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
            ),
        ),
        source=f"https://huggingface.co/{_WHISPER_REPO}",
        license="MIT",
    ),
    ModelSpec(
        id="whispercpp-large-v3-turbo",
        engine="whispercpp",
        display_name="Whisper large-v3-turbo",
        files=(
            _hf_file(
                _WHISPER_REPO,
                "ggml-large-v3-turbo.bin",
                1624555275,
                "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
            ),
        ),
        source=f"https://huggingface.co/{_WHISPER_REPO}",
        license="MIT",
    ),
    ModelSpec(
        id="voxtral-mini-4b-realtime-4bit",
        engine="voxtral_mlx",
        display_name="Voxtral Mini 4B Realtime (4-bit MLX)",
        files=(
            _hf_file(
                _VOXTRAL_REPO,
                "config.json",
                1513,
                "02060864a4f33df5e4944684fc17f3026af4011830cac4def6e9e025315b10c5",
            ),
            _hf_file(
                _VOXTRAL_REPO,
                "model.safetensors",
                3133798126,
                "6f59b425d8a1ceb2de795454558be63937cf75b59f9c9bc77accd85aaf32af05",
            ),
            _hf_file(
                _VOXTRAL_REPO,
                "model.safetensors.index.json",
                118632,
                "80f68b80cf4b1638d864d1504061a266f59e37a8d90d7b20f2e1f30c2d034c2e",
            ),
            _hf_file(
                _VOXTRAL_REPO,
                "tekken.json",
                14910348,
                "8434af1d39eba99f0ef46cf1450bf1a63fa941a26933a1ef5dbbf4adf0d00e44",
            ),
        ),
        source=f"https://huggingface.co/{_VOXTRAL_REPO}",
        license="Apache-2.0",
    ),
)


def models_for_engine(
    engine_id: str, catalog: Iterable[ModelSpec] = CATALOG
) -> tuple[ModelSpec, ...]:
    """Every catalog entry belonging to ``engine_id``, in catalog order.

    An engine with no models yields an empty tuple rather than raising: the
    caller is a picker deciding what to offer, not code asking for a model.
    """
    return tuple(spec for spec in catalog if spec.engine == engine_id)


def human_size(num_bytes: int) -> str:
    """Format a byte count the way model sources publish it (decimal units)."""
    assert num_bytes >= 0, f"size cannot be negative: {num_bytes}"
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.1f} GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.0f} MB"
    if num_bytes >= 1_000:
        return f"{num_bytes / 1_000:.0f} KB"
    return f"{num_bytes} bytes"


class ModelStore:
    """Installs, verifies and removes catalog models under ``root``."""

    def __init__(
        self,
        root: Path | None = None,
        catalog: Iterable[ModelSpec] = CATALOG,
    ) -> None:
        self._root = Path(root) if root is not None else DEFAULT_ROOT
        specs = tuple(catalog)
        ids = [spec.id for spec in specs]
        assert len(ids) == len(set(ids)), f"duplicate model ids in catalog: {ids}"
        self._catalog = specs

    @property
    def root(self) -> Path:
        """Directory holding every installed model."""
        return self._root

    @property
    def catalog(self) -> tuple[ModelSpec, ...]:
        """The specs this store knows about."""
        return self._catalog

    def spec(self, model_id: str) -> ModelSpec:
        """Return the spec for ``model_id`` or raise :class:`UnknownModelError`."""
        for spec in self._catalog:
            if spec.id == model_id:
                return spec
        known = ", ".join(spec.id for spec in self._catalog) or "<empty catalog>"
        raise UnknownModelError(f"Unknown model id {model_id!r}; known ids: {known}")

    def path(self, model_id: str) -> Path:
        """Directory the model installs into (whether or not it exists yet)."""
        return self._root / self.spec(model_id).id

    def file_path(self, model_id: str, file_name: str) -> Path:
        """Path of one file of the model on disk."""
        return self.path(model_id) / file_name

    def engine_model_path(self, model_id: str) -> Path:
        """The path this model's engine wants for ``model_path``.

        A single-file model resolves to the file itself: whisper.cpp models go
        to :class:`~engines.whispercpp.WhisperCppEngine`, which hands its
        ``model_path`` straight to ``whisper-server -m``. A multi-file model
        resolves to the model directory: Voxtral models go to
        :class:`~engines.voxtral_mlx.VoxtralMlxEngine`, which loads a
        directory of weights and configs.

        Raises :class:`UnknownModelError` for an id outside the catalog.
        """
        spec = self.spec(model_id)
        if len(spec.files) == 1:
            return self.file_path(model_id, spec.files[0].name)
        return self.path(model_id)

    def is_installed(self, model_id: str) -> bool:
        """True when every file is present at its recorded size."""
        directory = self.path(model_id)
        for item in self.spec(model_id).files:
            target = directory / item.name
            if not target.is_file():
                return False
            if item.size_bytes and target.stat().st_size != item.size_bytes:
                return False
        return True

    def installed_models(self) -> tuple[ModelSpec, ...]:
        """Every catalog spec that is fully installed, in catalog order."""
        return tuple(spec for spec in self._catalog if self.is_installed(spec.id))

    def delete(self, model_id: str) -> None:
        """Remove the model directory, partial downloads included. Idempotent."""
        shutil.rmtree(self.path(model_id), ignore_errors=True)

    def verify(self, model_id: str) -> None:
        """Re-hash every file; raise :class:`ModelIntegrityError` on any problem.

        A file whose digest does not match is deleted, so the next download
        starts clean instead of resuming onto corrupt bytes. A file with no
        checksum on record cannot be certified and is reported as such.
        """
        directory = self.path(model_id)
        for item in self.spec(model_id).files:
            target = directory / item.name
            if not target.is_file():
                raise ModelIntegrityError(f"{item.name} is missing from {directory}")
            if not item.sha256:
                raise ModelIntegrityError(f"no checksum on record for {item.name}")
            actual = _sha256_of(target)
            if actual != item.sha256:
                target.unlink(missing_ok=True)
                raise ModelIntegrityError(
                    f"sha256 mismatch for {item.name}: "
                    f"expected {item.sha256}, got {actual}"
                )

    def download(
        self,
        model_id: str,
        progress: Callable[[DownloadProgress], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Path:
        """Fetch every missing file of ``model_id`` and return its directory.

        Downloads land in ``<file>.part`` and are renamed onto the final name
        only after the checksum matches, so a half-written file is never
        mistaken for an installed one. An existing ``.part`` is resumed with a
        ``Range`` request. ``progress`` is called on this thread; ``cancel``
        is polled between chunks and raises :class:`DownloadCancelled`,
        leaving the ``.part`` in place for a later resume. A transfer that
        ends early raises :class:`DownloadIncomplete` and likewise keeps the
        ``.part``; only a full-size file with the wrong bytes is destroyed,
        as :class:`ModelIntegrityError`.
        """
        spec = self.spec(model_id)
        directory = self.path(model_id)
        directory.mkdir(parents=True, exist_ok=True)
        for item in spec.files:
            self._download_file(spec, item, directory, progress, cancel)
        return directory

    def _download_file(
        self,
        spec: ModelSpec,
        item: ModelFile,
        directory: Path,
        progress: Callable[[DownloadProgress], None] | None,
        cancel: threading.Event | None,
    ) -> None:
        final = directory / item.name
        if final.is_file() and (
            not item.size_bytes or final.stat().st_size == item.size_bytes
        ):
            size = final.stat().st_size
            _emit(progress, DownloadProgress(spec.id, item.name, size, size, True))
            return

        part = final.with_name(f"{item.name}.part")
        _raise_if_cancelled(cancel, spec.id, item.name)
        resume_from = part.stat().st_size if part.is_file() else 0
        hasher = hashlib.sha256()
        response, resumed = _open_source(item.url, resume_from)
        bytes_done = 0
        with response:
            if resumed:
                _feed_hasher(hasher, part)
                bytes_done = resume_from
                mode = "ab"
            else:
                mode = "wb"
            total = _total_bytes(response, bytes_done, item)
            _emit(
                progress,
                DownloadProgress(spec.id, item.name, bytes_done, total, False),
            )
            with open(part, mode) as handle:
                while True:
                    _raise_if_cancelled(cancel, spec.id, item.name)
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    hasher.update(chunk)
                    bytes_done += len(chunk)
                    _emit(
                        progress,
                        DownloadProgress(
                            spec.id,
                            item.name,
                            bytes_done,
                            max(total, bytes_done),
                            False,
                        ),
                    )

        # Size before checksum. A connection dropped mid-body ends the loop on
        # a clean EOF, and hashing a short file always "mismatches"; blaming
        # the digest would both misname the cause and delete a .part that is
        # perfectly good to resume from.
        if item.size_bytes and bytes_done < item.size_bytes:
            raise DownloadIncomplete(
                f"{item.name} stopped after {bytes_done} of {item.size_bytes} "
                f"bytes; run the download again to resume from there"
            )
        if item.size_bytes and bytes_done > item.size_bytes:
            part.unlink(missing_ok=True)
            raise ModelIntegrityError(
                f"{item.name} downloaded as {bytes_done} bytes, "
                f"expected {item.size_bytes}"
            )
        digest = hasher.hexdigest()
        if item.sha256 and digest != item.sha256:
            part.unlink(missing_ok=True)
            raise ModelIntegrityError(
                f"sha256 mismatch for {item.name}: "
                f"expected {item.sha256}, got {digest}"
            )
        part.replace(final)
        _emit(
            progress,
            DownloadProgress(spec.id, item.name, bytes_done, bytes_done, True),
        )


#: Socket timeout for every network read, in seconds.
TIMEOUT_SECONDS = 60


def _emit(
    progress: Callable[[DownloadProgress], None] | None,
    event: DownloadProgress,
) -> None:
    """Call ``progress`` when the caller supplied one."""
    if progress is not None:
        progress(event)


def _raise_if_cancelled(
    cancel: threading.Event | None, model_id: str, file_name: str
) -> None:
    """Raise :class:`DownloadCancelled` when the caller asked us to stop."""
    if cancel is not None and cancel.is_set():
        raise DownloadCancelled(f"download of {model_id}/{file_name} was cancelled")


def _sha256_of(path: Path) -> str:
    """Hex digest of a file, read in chunks so large models stay off the heap."""
    hasher = hashlib.sha256()
    _feed_hasher(hasher, path)
    return hasher.hexdigest()


def _feed_hasher(hasher, path: Path) -> None:
    """Stream ``path`` into ``hasher``. Contents are never logged or returned."""
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                return
            hasher.update(chunk)


def _open_source(url: str, resume_from: int):
    """Open ``url``, asking for a byte range when resuming.

    Returns ``(response, resumed)`` where ``resumed`` is True only when the
    server actually honoured the range with a 206; a server that ignores the
    header and replays the whole body restarts the file from zero.
    """
    request = urllib.request.Request(url)
    if resume_from > 0:
        request.add_header("Range", f"bytes={resume_from}-")
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        exc.close()
        if resume_from > 0 and exc.code == 416:
            return _open_source(url, 0)
        raise ModelStoreError(f"HTTP {exc.code} while fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise ModelStoreError(f"cannot reach {url}: {exc.reason}") from exc
    status = getattr(response, "status", None) or response.getcode()
    return response, resume_from > 0 and status == 206


def _total_bytes(response, offset: int, item: ModelFile) -> int:
    """Best estimate of the file's full size, for progress reporting."""
    length = response.headers.get("Content-Length")
    if length is not None and length.isdigit():
        return offset + int(length)
    return item.size_bytes


__all__ = [
    "CATALOG",
    "DEFAULT_ROOT",
    "DownloadCancelled",
    "DownloadIncomplete",
    "DownloadProgress",
    "ModelFile",
    "ModelIntegrityError",
    "ModelSpec",
    "ModelStore",
    "ModelStoreError",
    "UnknownModelError",
    "human_size",
    "models_for_engine",
]
