#!/usr/bin/env python3
"""Hardware-aware model recommendation helpers."""

from __future__ import annotations

import subprocess


def default_model_for_current_machine() -> str:
    """Return a conservative default model based on RAM tiers."""
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        ram_bytes = int(result.stdout.strip())
        ram_gb = ram_bytes // (1024 ** 3)
    except (ValueError, OSError, subprocess.CalledProcessError):
        return "base"

    if ram_gb >= 32:
        return "large"
    if ram_gb >= 16:
        return "medium"
    if ram_gb >= 8:
        return "small"
    return "base"
