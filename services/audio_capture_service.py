#!/usr/bin/env python3
"""Audio capture service wrapping sounddevice stream lifecycle.

Two consumers, one stream:

* the **batch** path keeps every block in :attr:`AudioCaptureService.chunks`,
  which the app concatenates into the WAV it hands to ``Engine.transcribe``;
* the **live** path, switched on with :meth:`enable_streaming`, additionally
  pushes each block onto a queue as 16-bit PCM so a streaming engine can decode
  the utterance while it is still being spoken (``Engine.stream``).

The live feed is deliberately additive: the WAV is written either way, so a
stream that fails mid-utterance costs nothing — the caller falls back to the
batch transcription of the audio it already has.

Audio samples never reach a log; only counts and statuses do.
"""

from __future__ import annotations

import queue
from collections.abc import Iterator
from typing import Any

import numpy as np
import sounddevice as sd

#: Sample format of the live feed: signed 16-bit, little-endian, mono. It is
#: what ``whisper.cpp`` and ``mlx-audio`` both want, and it halves the bytes
#: crossing the queue compared with float32.
PCM_DTYPE = "<i2"

#: Scale from sounddevice's float32 [-1.0, 1.0] to int16.
PCM_SCALE = 32767

#: Pushed onto the queue by :meth:`AudioCaptureService.stop` so the blocking
#: generator ends instead of waiting for a block that will never come. A unique
#: object, not None or b"": an empty block is a legitimate (if odd) value.
_END_OF_STREAM = object()


def pcm_bytes(indata: Any) -> bytes:
    """One capture block as int16 little-endian bytes.

    ``indata`` is sounddevice's float32 frame buffer in [-1.0, 1.0]; the same
    conversion the WAV writer uses, so the live feed and the file the engine
    later reads carry the same samples.
    """
    return (np.asarray(indata) * PCM_SCALE).astype(PCM_DTYPE).tobytes()


class AudioCaptureService:
    def __init__(self, sample_rate: int, logger: Any):
        self.sample_rate = sample_rate
        self._logger = logger
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._streaming = False
        self._queue: queue.Queue | None = None

    @property
    def chunks(self) -> list[np.ndarray]:
        return self._chunks

    @property
    def streaming_enabled(self) -> bool:
        """Whether the next :meth:`start` will also feed :meth:`pcm_chunks`."""
        return self._streaming

    def enable_streaming(self, enabled: bool) -> None:
        """Turn the live PCM feed on or off for the *next* recording.

        Changing it mid-recording would either strand a consumer on a queue
        nothing fills any more or start one halfway through an utterance, so it
        is refused while a stream is open.
        """
        assert self._stream is None, "Streaming must be switched before starting a recording"
        self._streaming = bool(enabled)

    def pcm_chunks(self) -> Iterator[bytes]:
        """Blocking generator over the live feed; ends when :meth:`stop` is called.

        The queue is captured up front, so a consumer that is still draining the
        tail of one utterance is unaffected by the next recording starting.
        """
        source = self._queue
        assert source is not None, "Call enable_streaming(True) and start() before pcm_chunks()"

        def generate() -> Iterator[bytes]:
            while True:
                block = source.get()
                if block is _END_OF_STREAM:
                    return
                yield block

        return generate()

    def start(self) -> None:
        assert self._stream is None, "Audio stream must be stopped before starting a new one"
        self._chunks = []
        self._queue = queue.Queue() if self._streaming else None
        live = self._queue

        def audio_callback(indata, frames, time_info, status):
            if status:
                self._logger.warning(f"Audio callback status: {status}")
            self._chunks.append(indata.copy())
            if live is not None:
                # Runs on PortAudio's callback thread: put() on an unbounded
                # queue never blocks, so the conversion is the only work done
                # here and the audio thread is not held up.
                live.put(pcm_bytes(indata))

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        live, self._queue = self._queue, None
        try:
            if stream is not None:
                stream.stop()
                stream.close()
        finally:
            # Always, even if closing the device raised: a consumer blocked on
            # the queue would otherwise wait for the rest of the process's life.
            if live is not None:
                live.put(_END_OF_STREAM)
