"""Tests for the capture service's batch buffer and its live PCM feed.

No microphone and no PortAudio: ``sounddevice.InputStream`` is replaced with a
fake whose ``start`` simply hands the service's own callback a few blocks, so
every assertion is deterministic and the suite runs headless.
"""

import threading
import unittest
from unittest.mock import patch

import numpy as np

from services.audio_capture_service import (
    PCM_DTYPE,
    PCM_SCALE,
    AudioCaptureService,
    pcm_bytes,
)


class TestLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


class FakeStream:
    """Stands in for ``sd.InputStream``; feeds blocks on demand."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.stopped = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def feed(self, block, status=None):
        self.callback(block, len(block), None, status)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _block(*values):
    return np.array([[value] for value in values], dtype=np.float32)


class PcmBytesTests(unittest.TestCase):
    def test_float32_frames_become_int16_little_endian(self):
        raw = pcm_bytes(_block(0.0, 1.0, -1.0))

        self.assertEqual(len(raw), 3 * 2)  # three frames, two bytes each
        self.assertEqual(
            list(np.frombuffer(raw, dtype=PCM_DTYPE)), [0, PCM_SCALE, -PCM_SCALE]
        )

    def test_byte_order_is_pinned_not_inherited(self):
        # "<i2" and not np.int16: the format is part of the contract with the
        # engine, and a big-endian host must not quietly change it.
        # ``dtype.str`` always spells the byte order out, even where numpy has
        # folded "<" into the native "=".
        self.assertEqual(np.dtype(PCM_DTYPE).str, "<i2")
        self.assertEqual(np.dtype(PCM_DTYPE).itemsize, 2)

    def test_an_empty_block_produces_no_bytes(self):
        self.assertEqual(pcm_bytes(np.zeros((0, 1), dtype=np.float32)), b"")


class _ServiceCase(unittest.TestCase):
    def setUp(self):
        FakeStream.instances = []
        patcher = patch(
            "services.audio_capture_service.sd.InputStream", new=FakeStream
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.logger = TestLogger()
        self.service = AudioCaptureService(sample_rate=16000, logger=self.logger)

    @property
    def stream(self):
        return FakeStream.instances[-1]


class BatchCaptureTests(_ServiceCase):
    def test_blocks_are_kept_for_the_wav(self):
        self.service.start()
        self.stream.feed(_block(0.5))
        self.stream.feed(_block(-0.5))

        self.assertEqual(len(self.service.chunks), 2)
        self.service.stop()
        self.assertTrue(self.stream.stopped and self.stream.closed)

    def test_each_recording_starts_from_an_empty_buffer(self):
        self.service.start()
        self.stream.feed(_block(0.5))
        self.service.stop()

        self.service.start()
        self.assertEqual(self.service.chunks, [])

    def test_a_callback_status_is_logged_not_raised(self):
        self.service.start()
        self.stream.feed(_block(0.1), status="input overflow")

        self.assertEqual(len(self.logger.warnings), 1)
        self.assertEqual(len(self.service.chunks), 1)

    def test_starting_twice_is_a_programming_error(self):
        self.service.start()
        with self.assertRaises(AssertionError):
            self.service.start()

    def test_stopping_twice_is_harmless(self):
        self.service.start()
        self.service.stop()
        self.service.stop()


class StreamingToggleTests(_ServiceCase):
    def test_streaming_is_off_until_asked_for(self):
        self.assertFalse(self.service.streaming_enabled)
        self.service.start()
        self.stream.feed(_block(0.25))

        # No queue at all, so nothing to consume and nothing to leak.
        with self.assertRaises(AssertionError):
            self.service.pcm_chunks()

    def test_streaming_cannot_be_switched_mid_recording(self):
        self.service.enable_streaming(True)
        self.service.start()
        with self.assertRaises(AssertionError):
            self.service.enable_streaming(False)

    def test_it_can_be_switched_between_recordings(self):
        self.service.enable_streaming(True)
        self.service.start()
        self.service.stop()

        self.service.enable_streaming(False)
        self.assertFalse(self.service.streaming_enabled)
        self.service.start()
        with self.assertRaises(AssertionError):
            self.service.pcm_chunks()


class PcmFeedTests(_ServiceCase):
    def setUp(self):
        super().setUp()
        self.service.enable_streaming(True)
        self.service.start()

    def test_blocks_arrive_as_pcm_and_the_sentinel_ends_the_generator(self):
        chunks = self.service.pcm_chunks()
        self.stream.feed(_block(1.0))
        self.stream.feed(_block(-1.0))
        self.service.stop()

        self.assertEqual(
            list(chunks), [pcm_bytes(_block(1.0)), pcm_bytes(_block(-1.0))]
        )

    def test_the_wav_buffer_is_filled_as_well(self):
        self.service.pcm_chunks()
        self.stream.feed(_block(0.5))

        self.assertEqual(len(self.service.chunks), 1)

    def test_the_generator_blocks_until_a_block_or_the_sentinel_arrives(self):
        chunks = self.service.pcm_chunks()
        collected = []
        started = threading.Event()

        def consume():
            started.set()
            collected.extend(chunks)

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        self.assertTrue(started.wait(timeout=2))
        worker.join(0.05)
        self.assertTrue(worker.is_alive(), "the generator must wait for audio")

        self.stream.feed(_block(0.25))
        self.service.stop()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(collected, [pcm_bytes(_block(0.25))])

    def test_a_device_that_fails_to_close_still_releases_the_consumer(self):
        chunks = self.service.pcm_chunks()

        def explode():
            raise OSError("device went away")

        self.stream.stop = explode
        with self.assertRaises(OSError):
            self.service.stop()

        # The sentinel went in regardless: a consumer must never be stranded.
        self.assertEqual(list(chunks), [])

    def test_a_consumer_of_the_previous_utterance_is_not_disturbed(self):
        first = self.service.pcm_chunks()
        self.stream.feed(_block(0.5))
        self.service.stop()

        self.service.enable_streaming(True)
        self.service.start()
        self.stream.feed(_block(0.75))

        # The first generator drains its own queue and ends on its own sentinel.
        self.assertEqual(list(first), [pcm_bytes(_block(0.5))])


if __name__ == "__main__":
    unittest.main()
