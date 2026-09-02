#!/usr/bin/env python3
"""Hardware-aware model and engine recommendation helpers."""

from __future__ import annotations

import platform
import subprocess

CHIP_APPLE_SILICON = "apple_silicon"
CHIP_INTEL = "intel"

ENGINE_VOXTRAL_MLX = "voxtral_mlx"
ENGINE_WHISPERCPP = "whispercpp"

#: Decision D1: Voxtral needs Apple Silicon and this much unified memory.
VOXTRAL_MIN_RAM_GB = 16


def detect_chip() -> str:
    """Return the chip family of this machine."""
    return CHIP_APPLE_SILICON if platform.machine() == "arm64" else CHIP_INTEL


def detect_ram_gb() -> int | None:
    """Return installed RAM in whole gigabytes, or None when the probe fails."""
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        ram_bytes = int(result.stdout.strip())
    except (ValueError, OSError, subprocess.CalledProcessError):
        return None
    return ram_bytes // (1024**3)


def voxtral_eligible(chip: str, ram_gb: int | None) -> bool:
    """Return whether this machine is eligible to opt into Voxtral Mini 4B Realtime."""
    return chip == CHIP_APPLE_SILICON and ram_gb is not None and ram_gb >= VOXTRAL_MIN_RAM_GB


def select_engine_id(chip: str, ram_gb: int | None) -> str:
    """Decision D1 (provisional, 2026-09-02): whisper.cpp large-v3-turbo q5_0 is the
    default engine on all Macs. Voxtral Mini 4B Realtime is opt-in on eligible Apple
    Silicon machines; see ``voxtral_eligible``."""
    assert chip in (CHIP_APPLE_SILICON, CHIP_INTEL), f"unknown chip: {chip!r}"
    return ENGINE_WHISPERCPP


def default_engine_for_current_machine() -> str:
    """Return the engine id this machine should default to."""
    return select_engine_id(detect_chip(), detect_ram_gb())


def default_model_for_current_machine() -> str:
    """Return a conservative default Whisper model based on RAM tiers."""
    ram_gb = detect_ram_gb()
    if ram_gb is None:
        return "base"

    if ram_gb >= 32:
        return "large"
    if ram_gb >= 16:
        return "medium"
    if ram_gb >= 8:
        return "small"
    return "base"
