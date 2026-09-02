"""Tests for engines.model_store.

No network: a loopback http.server stands in for Hugging Face, and every test
drives a test-only catalog through ``ModelStore(catalog=...)``. The handler
honours ``Range: bytes=N-`` so resume is exercised for real.
"""

import hashlib
import http.server
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from engines import ENGINE_IDS, model_store
from engines.model_store import (
    CATALOG,
    DEFAULT_ROOT,
    DownloadCancelled,
    DownloadIncomplete,
    DownloadProgress,
    ModelFile,
    ModelIntegrityError,
    ModelSpec,
    ModelStore,
    ModelStoreError,
    UnknownModelError,
)

BODY = bytes((index * 7 + 11) % 251 for index in range(512))
BODY_SHA = hashlib.sha256(BODY).hexdigest()
OTHER = bytes((index * 3 + 5) % 251 for index in range(256))
OTHER_SHA = hashlib.sha256(OTHER).hexdigest()
WRONG_SHA = "0" * 64


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves fixed bodies and records the Range headers it was asked for."""

    protocol_version = "HTTP/1.0"
    bodies: dict[str, bytes] = {"/model.bin": BODY, "/second.bin": OTHER}
    ranges: list[str] = []
    #: When set, a first (rangeless) GET promises the whole file and then closes
    #: the connection after this many bytes: a clean EOF, no exception, exactly
    #: what a dropped transfer looks like to http.client.
    truncate_bytes: int | None = None

    def log_message(self, *args):  # noqa: D102 - silence the test server
        return

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler naming
        body = self.bodies.get(self.path)
        if body is None:
            self.send_error(404)
            return
        full_length = len(body)
        range_header = self.headers.get("Range")
        truncate = type(self).truncate_bytes
        if truncate is not None and not range_header:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(full_length))
            self.end_headers()
            self.wfile.write(body[:truncate])
            self.close_connection = True
            return
        status = 200
        if range_header:
            type(self).ranges.append(range_header)
            start = int(range_header.split("=", 1)[1].split("-", 1)[0])
            if start >= full_length:
                self.send_error(416)
                return
            body = body[start:]
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header(
                "Content-Range",
                f"bytes {full_length - len(body)}-{full_length - 1}/{full_length}",
            )
        self.end_headers()
        self.wfile.write(body)


class _QuietServer(http.server.ThreadingHTTPServer):
    """A cancelled download closes the socket mid-write; do not shout about it."""

    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        return


class ModelStoreTestCase(unittest.TestCase):
    """Shared loopback server, temp root and test-only catalog."""

    @classmethod
    def setUpClass(cls):
        cls.server = _QuietServer(("127.0.0.1", 0), _Handler)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        _Handler.ranges = []
        _Handler.truncate_bytes = None
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "models"

    def make_spec(self, sha256=BODY_SHA, size_bytes=len(BODY), model_id="test-model"):
        return ModelSpec(
            id=model_id,
            engine="test",
            display_name="Test model",
            files=(
                ModelFile(
                    name="model.bin",
                    size_bytes=size_bytes,
                    sha256=sha256,
                    url=f"{self.base_url}/model.bin",
                ),
            ),
            source=self.base_url,
            license="MIT",
        )

    def make_multi_spec(self, model_id="multi"):
        return ModelSpec(
            id=model_id,
            engine="test",
            display_name="Multi file model",
            files=(
                ModelFile("model.bin", len(BODY), BODY_SHA, f"{self.base_url}/model.bin"),
                ModelFile("second.bin", len(OTHER), OTHER_SHA, f"{self.base_url}/second.bin"),
            ),
            source=self.base_url,
            license="MIT",
        )

    def make_store(self, *specs):
        return ModelStore(root=self.root, catalog=specs or (self.make_spec(),))


class DownloadTests(ModelStoreTestCase):
    def test_fresh_download_installs_file_and_reports_progress(self):
        store = self.make_store()
        events: list[DownloadProgress] = []

        with patch.object(model_store, "CHUNK_BYTES", 64):
            directory = store.download("test-model", progress=events.append)

        self.assertEqual(directory, self.root / "test-model")
        installed = directory / "model.bin"
        self.assertEqual(installed.read_bytes(), BODY)
        self.assertTrue(store.is_installed("test-model"))
        self.assertGreater(len(events), 1)
        self.assertTrue(events[-1].done)
        self.assertEqual(events[-1].bytes_done, len(BODY))
        self.assertEqual(events[-1].model_id, "test-model")
        self.assertEqual(events[-1].file_name, "model.bin")
        self.assertFalse(any(event.done for event in events[:-1]))
        self.assertFalse((directory / "model.bin.part").exists())

    def test_progress_totals_are_the_full_file_size(self):
        store = self.make_store()
        events: list[DownloadProgress] = []
        with patch.object(model_store, "CHUNK_BYTES", 64):
            store.download("test-model", progress=events.append)
        self.assertTrue(all(event.bytes_total == len(BODY) for event in events))

    def test_download_resumes_from_an_existing_part_file(self):
        store = self.make_store()
        directory = self.root / "test-model"
        directory.mkdir(parents=True)
        part = directory / "model.bin.part"
        part.write_bytes(BODY[:200])

        store.download("test-model")

        self.assertTrue(_Handler.ranges, "server never saw a Range request")
        self.assertEqual(_Handler.ranges[-1], "bytes=200-")
        installed = directory / "model.bin"
        self.assertEqual(installed.read_bytes(), BODY)
        self.assertEqual(hashlib.sha256(installed.read_bytes()).hexdigest(), BODY_SHA)
        self.assertFalse(part.exists())

    def test_already_installed_file_is_not_refetched(self):
        store = self.make_store()
        store.download("test-model")
        events: list[DownloadProgress] = []

        store.download("test-model", progress=events.append)

        self.assertEqual(_Handler.ranges, [])
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].done)

    def test_checksum_mismatch_raises_and_removes_the_file(self):
        store = self.make_store(self.make_spec(sha256=WRONG_SHA))

        with self.assertRaises(ModelIntegrityError) as caught:
            store.download("test-model")

        self.assertIn("model.bin", str(caught.exception))
        directory = self.root / "test-model"
        self.assertFalse((directory / "model.bin").exists())
        self.assertFalse((directory / "model.bin.part").exists())
        self.assertFalse(store.is_installed("test-model"))

    def test_oversized_download_raises_integrity_and_removes_the_part(self):
        store = self.make_store(self.make_spec(size_bytes=len(BODY) - 1))

        with self.assertRaises(ModelIntegrityError):
            store.download("test-model")

        self.assertFalse((self.root / "test-model" / "model.bin.part").exists())

    def test_truncated_transfer_reports_incomplete_and_keeps_the_part(self):
        store = self.make_store()
        _Handler.truncate_bytes = 200

        with self.assertRaises(DownloadIncomplete) as caught:
            store.download("test-model")

        # A clean EOF is a short transfer, not corruption: say so, and do not
        # blame the checksum for bytes that never arrived.
        self.assertIsInstance(caught.exception, ModelStoreError)
        self.assertNotIsInstance(caught.exception, ModelIntegrityError)
        message = str(caught.exception)
        self.assertIn("200", message)
        self.assertIn(str(len(BODY)), message)
        self.assertNotIn("sha256", message)
        directory = self.root / "test-model"
        part = directory / "model.bin.part"
        self.assertTrue(part.exists(), "a resumable .part must survive")
        self.assertEqual(part.stat().st_size, 200)
        self.assertFalse((directory / "model.bin").exists())
        self.assertFalse(store.is_installed("test-model"))

    def test_incomplete_download_resumes_on_the_next_call(self):
        store = self.make_store()
        _Handler.truncate_bytes = 200
        with self.assertRaises(DownloadIncomplete):
            store.download("test-model")

        store.download("test-model")

        self.assertEqual(_Handler.ranges[-1], "bytes=200-")
        installed = self.root / "test-model" / "model.bin"
        self.assertEqual(installed.read_bytes(), BODY)
        self.assertEqual(hashlib.sha256(installed.read_bytes()).hexdigest(), BODY_SHA)
        self.assertFalse((self.root / "test-model" / "model.bin.part").exists())
        self.assertTrue(store.is_installed("test-model"))

    def test_blank_checksum_still_downloads(self):
        store = self.make_store(self.make_spec(sha256=""))
        store.download("test-model")
        self.assertTrue(store.is_installed("test-model"))

    def test_cancel_raises_and_keeps_the_part_file_for_resume(self):
        store = self.make_store()
        cancel = threading.Event()

        def on_progress(event: DownloadProgress) -> None:
            if event.bytes_done >= 64:
                cancel.set()

        with patch.object(model_store, "CHUNK_BYTES", 64):
            with self.assertRaises(DownloadCancelled):
                store.download("test-model", progress=on_progress, cancel=cancel)

        directory = self.root / "test-model"
        part = directory / "model.bin.part"
        self.assertTrue(part.exists())
        self.assertGreater(part.stat().st_size, 0)
        self.assertLess(part.stat().st_size, len(BODY))
        self.assertFalse((directory / "model.bin").exists())
        self.assertFalse(store.is_installed("test-model"))

    def test_cancelled_download_resumes_on_the_next_call(self):
        store = self.make_store()
        cancel = threading.Event()

        def on_progress(event: DownloadProgress) -> None:
            if event.bytes_done >= 64:
                cancel.set()

        with patch.object(model_store, "CHUNK_BYTES", 64):
            with self.assertRaises(DownloadCancelled):
                store.download("test-model", progress=on_progress, cancel=cancel)

        store.download("test-model")

        self.assertTrue(_Handler.ranges)
        self.assertEqual((self.root / "test-model" / "model.bin").read_bytes(), BODY)

    def test_multi_file_model_downloads_every_file(self):
        spec = self.make_multi_spec()
        store = self.make_store(spec)

        directory = store.download("multi")

        self.assertEqual((directory / "model.bin").read_bytes(), BODY)
        self.assertEqual((directory / "second.bin").read_bytes(), OTHER)
        self.assertTrue(store.is_installed("multi"))
        self.assertEqual(spec.size_bytes, len(BODY) + len(OTHER))
        self.assertEqual(len(spec.files), 2)


class VerifyDeleteAndLookupTests(ModelStoreTestCase):
    def test_verify_passes_for_a_good_download(self):
        store = self.make_store()
        store.download("test-model")
        store.verify("test-model")  # must not raise

    def test_verify_deletes_a_corrupt_file_and_raises(self):
        store = self.make_store()
        store.download("test-model")
        installed = self.root / "test-model" / "model.bin"
        installed.write_bytes(b"x" * len(BODY))

        with self.assertRaises(ModelIntegrityError) as caught:
            store.verify("test-model")

        self.assertIn("sha256 mismatch", str(caught.exception))
        self.assertFalse(installed.exists())

    def test_verify_reports_a_missing_file(self):
        store = self.make_store()
        with self.assertRaises(ModelIntegrityError) as caught:
            store.verify("test-model")
        self.assertIn("missing", str(caught.exception))

    def test_verify_refuses_to_certify_a_blank_checksum(self):
        store = self.make_store(self.make_spec(sha256=""))
        store.download("test-model")

        with self.assertRaises(ModelIntegrityError) as caught:
            store.verify("test-model")

        self.assertEqual(
            str(caught.exception), "no checksum on record for model.bin"
        )

    def test_delete_removes_the_model_directory(self):
        store = self.make_store()
        store.download("test-model")
        self.assertTrue(store.is_installed("test-model"))

        store.delete("test-model")

        self.assertFalse(store.is_installed("test-model"))
        self.assertFalse((self.root / "test-model").exists())
        store.delete("test-model")  # idempotent

    def test_is_installed_is_false_for_a_truncated_file(self):
        store = self.make_store()
        store.download("test-model")
        (self.root / "test-model" / "model.bin").write_bytes(BODY[:10])
        self.assertFalse(store.is_installed("test-model"))

    def test_installed_models_lists_only_complete_models(self):
        first = self.make_spec(model_id="test-model")
        second = self.make_spec(model_id="other-model")
        store = self.make_store(first, second)
        self.assertEqual(store.installed_models(), ())

        store.download("other-model")

        self.assertEqual([spec.id for spec in store.installed_models()], ["other-model"])

    def test_unknown_model_id_raises(self):
        store = self.make_store()
        for call in (store.spec, store.path, store.is_installed, store.download,
                     store.verify, store.delete, store.engine_model_path):
            with self.subTest(call=call.__name__):
                with self.assertRaises(UnknownModelError):
                    call("nope")

    def test_paths_follow_the_documented_layout(self):
        store = self.make_store()
        self.assertEqual(store.root, self.root)
        self.assertEqual(store.path("test-model"), self.root / "test-model")
        self.assertEqual(
            store.file_path("test-model", "model.bin"),
            self.root / "test-model" / "model.bin",
        )

    def test_engine_model_path_is_the_file_for_a_single_file_model(self):
        # whisper-server takes `-m <file>`, so a one-file model resolves to it.
        store = self.make_store()
        self.assertEqual(
            store.engine_model_path("test-model"),
            self.root / "test-model" / "model.bin",
        )
        self.assertNotEqual(
            store.engine_model_path("test-model"), store.path("test-model")
        )

    def test_engine_model_path_is_the_directory_for_a_multi_file_model(self):
        # VoxtralMlxEngine loads a directory of weights and configs.
        store = self.make_store(self.make_multi_spec())
        self.assertEqual(store.engine_model_path("multi"), self.root / "multi")
        self.assertEqual(store.engine_model_path("multi"), store.path("multi"))

    def test_duplicate_catalog_ids_are_rejected(self):
        spec = self.make_spec()
        with self.assertRaises(AssertionError):
            ModelStore(root=self.root, catalog=(spec, spec))


class CatalogTests(unittest.TestCase):
    def test_default_root_is_application_support(self):
        self.assertEqual(
            DEFAULT_ROOT,
            Path.home() / "Library" / "Application Support" / "Murmur" / "models",
        )

    def test_default_store_uses_the_shipped_catalog(self):
        store = ModelStore()
        self.assertEqual(store.root, DEFAULT_ROOT)
        self.assertEqual(store.catalog, CATALOG)

    def test_catalog_ids_are_unique(self):
        ids = [spec.id for spec in CATALOG]
        self.assertEqual(len(ids), len(set(ids)))

    def test_catalog_entries_are_well_formed(self):
        self.assertTrue(CATALOG)
        for spec in CATALOG:
            with self.subTest(model=spec.id):
                self.assertTrue(spec.id and spec.display_name and spec.license)
                self.assertIn(spec.engine, ENGINE_IDS)
                self.assertTrue(spec.source.startswith("https://huggingface.co/"))
                self.assertTrue(spec.files)
                self.assertEqual(spec.size_bytes, sum(f.size_bytes for f in spec.files))
                names = [item.name for item in spec.files]
                self.assertEqual(len(names), len(set(names)))
                for item in spec.files:
                    self.assertTrue(item.url.startswith("https://huggingface.co/"))
                    self.assertTrue(item.url.endswith(f"/{item.name}"))
                    self.assertGreaterEqual(item.size_bytes, 0)
                    self.assertTrue(
                        item.sha256 == "" or _is_sha256(item.sha256),
                        f"{item.name} has a malformed sha256",
                    )

    def test_catalog_covers_both_local_engines(self):
        engines_covered = {spec.engine for spec in CATALOG}
        self.assertIn("whispercpp", engines_covered)
        self.assertIn("voxtral_mlx", engines_covered)

    def test_multi_file_catalog_entries_are_directories(self):
        voxtral = [spec for spec in CATALOG if spec.engine == "voxtral_mlx"]
        self.assertTrue(voxtral)
        for spec in voxtral:
            with self.subTest(model=spec.id):
                self.assertGreater(len(spec.files), 1)
                self.assertGreater(spec.size_bytes, 0)

    def test_specs_are_frozen(self):
        spec = CATALOG[0]
        with self.assertRaises(Exception):
            spec.id = "mutated"
        with self.assertRaises(Exception):
            spec.files[0].sha256 = "mutated"


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


if __name__ == "__main__":
    unittest.main()
