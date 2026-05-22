#!/usr/bin/env python3
"""Audio capture service wrapping sounddevice stream lifecycle."""

from __future__ import annotations

from typing import Any

import numpy as np
import sounddevice as sd


class AudioCaptureService:
    def __init__(self, sample_rate: int, logger: Any):
        self.sample_rate = sample_rate
        self._logger = logger
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    @property
    def chunks(self) -> list[np.ndarray]:
        return self._chunks

    def start(self) -> None:
        assert self._stream is None, "Audio stream must be stopped before starting a new one"
        self._chunks = []

        def audio_callback(indata, frames, time_info, status):
            if status:
                self._logger.warning(f"Audio callback status: {status}")
            self._chunks.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
