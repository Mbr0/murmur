"""Tests for the download sheet state and its controller.

Nothing here touches AppKit or the network: the controller is driven with a
fake store, an immediate main-thread dispatcher and a synchronous spawner, so
every assertion is deterministic.
"""

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cleanup.llama_server import CLEANUP_MODEL_SPEC
from engines import ENGINE_IDS
from engines.model_store import (
    CATALOG,
    DownloadCancelled,
    DownloadProgress,
    ModelStore,
    ModelStoreError,
    human_size,
    models_for_engine,
)
from services.model_profile_service import CHIP_APPLE_SILICON, VOXTRAL_MIN_RAM_GB
from ui.download_sheet import (
    PHASE_CANCELLED,
    PHASE_DONE,
    PHASE_DOWNLOADING,
    PHASE_FAILED,
    PHASE_IDLE,
    PHASE_VERIFYING,
    DownloadController,
    DownloadSheetState,
    EngineSectionModel,
)

GB = 1_000_000_000


def _tick(name, done, total, finished=False):
    return DownloadProgress("m", name, done, total, finished)


class HumanSizeTests(unittest.TestCase):
    def test_uses_decimal_units_like_the_model_source(self):
        self.assertEqual(human_size(574041195), "574 MB")
        self.assertEqual(human_size(1624555275), "1.6 GB")
        self.assertEqual(human_size(3133798126), "3.1 GB")

    def test_small_sizes_stay_readable(self):
        self.assertEqual(human_size(0), "0 bytes")
        self.assertEqual(human_size(512), "512 bytes")
        self.assertEqual(human_size(14910348), "15 MB")

    def test_negative_size_is_a_programming_error(self):
        with self.assertRaises(AssertionError):
            human_size(-1)


class ModelsForEngineTests(unittest.TestCase):
    def test_filters_the_catalog_in_order(self):
        specs = models_for_engine("whispercpp")
        self.assertTrue(specs)
        self.assertTrue(all(spec.engine == "whispercpp" for spec in specs))

    def test_unknown_engine_yields_nothing(self):
        self.assertEqual(models_for_engine("nope"), ())


class DownloadSheetStateTests(unittest.TestCase):
    def test_starts_idle_with_no_progress(self):
        state = DownloadSheetState("voxtral", total_bytes=3 * GB)
        self.assertEqual(state.phase, PHASE_IDLE)
        self.assertEqual(state.bytes_done, 0)
        self.assertEqual(state.bytes_total, 3 * GB)
        self.assertEqual(state.percent, 0.0)
        self.assertIsNone(state.error)
        self.assertEqual(state.status_line(), "Ready to download")

    def test_progress_moves_the_phase_and_the_bytes(self):
        state = DownloadSheetState("voxtral", total_bytes=3 * GB)
        state.update(_tick("model.safetensors", 1_200_000_000, 3 * GB))
        self.assertEqual(state.phase, PHASE_DOWNLOADING)
        self.assertEqual(state.bytes_done, 1_200_000_000)
        self.assertEqual(state.status_line(), "1.2 GB of 3.0 GB")
        self.assertAlmostEqual(state.percent, 40.0, places=3)

    def test_bytes_from_several_files_add_up(self):
        state = DownloadSheetState("voxtral")
        state.update(_tick("config.json", 1000, 1000, True))
        state.update(_tick("weights.bin", 500, 2000))
        self.assertEqual(state.bytes_done, 1500)
        self.assertEqual(state.bytes_total, 3000)

    def test_a_seeded_total_is_never_shrunk_by_a_single_file(self):
        state = DownloadSheetState("voxtral", total_bytes=3000)
        state.update(_tick("config.json", 400, 1000))
        self.assertEqual(state.bytes_total, 3000)

    def test_verifying_done_failed_and_cancelled_read_plainly(self):
        state = DownloadSheetState("voxtral", total_bytes=1000)
        state.update(_tick("weights.bin", 1000, 1000, True))
        state.mark_verifying()
        self.assertEqual(state.phase, PHASE_VERIFYING)
        self.assertEqual(state.status_line(), "Verifying…")

        state.mark_done()
        self.assertEqual(state.phase, PHASE_DONE)
        self.assertEqual(state.percent, 100.0)
        self.assertEqual(state.status_line(), "Installed")

        state.mark_failed("sha256 mismatch for weights.bin")
        self.assertEqual(state.phase, PHASE_FAILED)
        self.assertEqual(state.error, "sha256 mismatch for weights.bin")
        self.assertEqual(
            state.status_line(), "Failed: sha256 mismatch for weights.bin"
        )

        state.mark_cancelled()
        self.assertEqual(state.phase, PHASE_CANCELLED)
        self.assertIsNone(state.error)
        self.assertEqual(state.status_line(), "Cancelled")

    def test_is_active_only_while_work_is_in_flight(self):
        state = DownloadSheetState("voxtral")
        self.assertFalse(state.is_active)
        state.update(_tick("weights.bin", 1, 10))
        self.assertTrue(state.is_active)
        state.mark_verifying()
        self.assertTrue(state.is_active)
        state.mark_done()
        self.assertFalse(state.is_active)

    def test_reset_forgets_the_previous_model(self):
        state = DownloadSheetState("voxtral", total_bytes=1000)
        state.update(_tick("weights.bin", 400, 1000))
        state.mark_failed("boom")
        state.reset("whisper", total_bytes=500)
        self.assertEqual(state.model_id, "whisper")
        self.assertEqual(state.phase, PHASE_IDLE)
        self.assertEqual(state.bytes_done, 0)
        self.assertEqual(state.bytes_total, 500)
        self.assertIsNone(state.error)

    def test_percent_never_exceeds_a_hundred(self):
        state = DownloadSheetState("voxtral", total_bytes=1000)
        state.update(_tick("weights.bin", 1500, 1000))
        self.assertEqual(state.percent, 100.0)

    def test_failed_without_a_message_is_refused(self):
        state = DownloadSheetState("voxtral")
        with self.assertRaises(AssertionError):
            state.mark_failed("")


class _FakeStore:
    """Stands in for :class:`~engines.model_store.ModelStore`."""

    def __init__(self, ticks=(), download_error=None, verify_error=None):
        self._ticks = tuple(ticks)
        self._download_error = download_error
        self._verify_error = verify_error
        self.downloaded = []
        self.verified = []
        self.cancel_events = []

    def download(self, model_id, progress=None, cancel=None):
        self.downloaded.append(model_id)
        self.cancel_events.append(cancel)
        for tick in self._ticks:
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled(f"{model_id} cancelled")
            if progress is not None:
                progress(tick)
        if self._download_error is not None:
            raise self._download_error

    def verify(self, model_id):
        self.verified.append(model_id)
        if self._verify_error is not None:
            raise self._verify_error


def _controller(store, **kwargs):
    """A controller whose work and UI hops both run on the calling thread."""
    kwargs.setdefault("dispatch", lambda func: func())
    kwargs.setdefault("spawn", lambda func: func())
    return DownloadController(store, **kwargs)


class DownloadControllerTests(unittest.TestCase):
    def test_a_clean_run_downloads_then_verifies_then_finishes(self):
        store = _FakeStore(ticks=[_tick("weights.bin", 1000, 1000, True)])
        seen = []
        controller = _controller(store, on_change=lambda state: seen.append(state.phase))

        controller.start("whisper", total_bytes=1000)

        self.assertEqual(store.downloaded, ["whisper"])
        self.assertEqual(store.verified, ["whisper"])
        self.assertEqual(controller.state.phase, PHASE_DONE)
        self.assertEqual(controller.state.model_id, "whisper")
        self.assertIn(PHASE_DOWNLOADING, seen)
        self.assertIn(PHASE_VERIFYING, seen)
        self.assertEqual(seen[-1], PHASE_DONE)

    def test_progress_reaches_the_state_through_the_dispatcher(self):
        store = _FakeStore(ticks=[_tick("weights.bin", 400, 1000)])
        hops = []
        controller = _controller(
            store, dispatch=lambda func: (hops.append(func), func())[1]
        )

        controller.start("whisper", total_bytes=1000)

        self.assertTrue(hops, "every state change must go through the dispatcher")
        self.assertEqual(controller.state.bytes_done, 400)

    def test_a_store_error_lands_as_a_failed_phase(self):
        store = _FakeStore(download_error=ModelStoreError("cannot reach host"))
        controller = _controller(store)

        controller.start("whisper")

        self.assertEqual(controller.state.phase, PHASE_FAILED)
        self.assertEqual(controller.state.error, "cannot reach host")
        self.assertEqual(store.verified, [], "a failed download is never verified")

    def test_a_verify_error_lands_as_a_failed_phase(self):
        store = _FakeStore(verify_error=ModelStoreError("sha256 mismatch"))
        controller = _controller(store)

        controller.start("whisper")

        self.assertEqual(controller.state.phase, PHASE_FAILED)
        self.assertEqual(controller.state.error, "sha256 mismatch")

    def test_cancel_sets_the_event_and_the_cancelled_phase(self):
        store = _FakeStore(ticks=[_tick("weights.bin", 1, 10)])
        controller = _controller(store)
        controller.cancel()  # before any run: harmless
        self.assertEqual(controller.state.phase, PHASE_IDLE)

        cancelled = _FakeStore(ticks=[_tick("weights.bin", 1, 10)])

        def spawn_after_cancel(func):
            controller_ref[0].cancel()
            func()

        controller_ref = [None]
        controller_ref[0] = DownloadController(
            cancelled, dispatch=lambda func: func(), spawn=spawn_after_cancel
        )
        controller_ref[0].start("whisper")

        self.assertEqual(controller_ref[0].state.phase, PHASE_CANCELLED)
        self.assertEqual(cancelled.verified, [])

    def test_a_second_start_while_running_is_refused(self):
        store = _FakeStore()
        controller = DownloadController(
            store, dispatch=lambda func: func(), spawn=lambda func: None
        )
        controller.start("whisper")
        with self.assertRaises(AssertionError):
            controller.start("voxtral")

    def test_the_real_thread_path_completes(self):
        store = _FakeStore(ticks=[_tick("weights.bin", 10, 10, True)])
        finished = threading.Event()
        controller = DownloadController(
            store,
            dispatch=lambda func: func(),
            on_change=lambda state: finished.set() if state.phase == PHASE_DONE else None,
        )
        controller.start("whisper", total_bytes=10)
        self.assertTrue(finished.wait(timeout=5), "download thread never finished")
        self.assertEqual(controller.state.phase, PHASE_DONE)
        self.assertFalse(controller.is_running)


class SpeechEngineFilterTests(unittest.TestCase):
    """The section lists speech models only, never the cleanup GGUF.

    The app composes one store out of ``engines.model_store.CATALOG`` plus
    ``cleanup.llama_server.CLEANUP_MODEL_SPEC`` so both downloads share the same
    resume, checksum and delete code. That store is also what Settings holds, so
    without a filter the "Speech engine" popup would offer a chat model as a
    transcriber.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _section(self, catalog):
        store = ModelStore(root=Path(self._tmp.name), catalog=catalog)
        return EngineSectionModel(
            {},
            store,
            chip=CHIP_APPLE_SILICON,
            ram_gb=VOXTRAL_MIN_RAM_GB,
            default_engine="whispercpp",
        )

    def test_the_cleanup_model_is_not_a_speech_engine_choice(self):
        section = self._section(CATALOG + (CLEANUP_MODEL_SPEC,))
        offered = [choice.model_id for choice in section.choices]

        self.assertNotIn(CLEANUP_MODEL_SPEC.id, offered)
        self.assertEqual(offered, [spec.id for spec in CATALOG])
        with self.assertRaises(AssertionError):
            section.spec(CLEANUP_MODEL_SPEC.id)

    def test_the_cleanup_model_engine_is_not_a_registered_engine(self):
        # The filter is "is this a speech engine?", not a hard-coded model id.
        self.assertNotIn(CLEANUP_MODEL_SPEC.engine, ENGINE_IDS)

    def test_speech_models_are_unaffected(self):
        self.assertEqual(
            [choice.model_id for choice in self._section(CATALOG).choices],
            [spec.id for spec in CATALOG],
        )


if __name__ == "__main__":
    unittest.main()
