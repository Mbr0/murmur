#!/usr/bin/env python3
"""Persistence service for local config/history data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
import shutil
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "save_audio": False,
    "save_history": False,
    "privacy_mode": True,
    "appearance_mode": "system",
    "hotkey_keycode": 49,
    "hotkey_command": False,
    "hotkey_option": True,
    "hotkey_control": False,
    "hotkey_shift": False,
    "hotkey_fn": False,
}

DEBUG_LOG_PATHS: tuple[str, ...] = (
    os.path.expanduser("~/Library/Logs/Murmur/murmur.log"),
    "/tmp/murmur_debug.log",
)


def should_log_sensitive(config: dict[str, Any]) -> bool:
    """Whether detailed logs that may reveal user content are permitted."""
    if config.get("privacy_mode") is True:
        return False
    return bool(config.get("save_history", DEFAULT_CONFIG["save_history"]))


@dataclass(frozen=True)
class PersistencePaths:
    config_file: str
    history_file: str


class PersistenceService:
    def __init__(self, paths: PersistencePaths, logger: Any):
        self._paths = paths
        self._logger = logger

    def load_config(self, default: dict[str, Any]) -> dict[str, Any]:
        return self._load_json_with_default(self._paths.config_file, default)

    def save_config(self, config: dict[str, Any]) -> None:
        self._save_json_file(self._paths.config_file, config)

    def load_history(self) -> list[dict[str, Any]]:
        return self._load_json_with_default(self._paths.history_file, [])

    def save_history(self, history: list[dict[str, Any]]) -> None:
        self._save_json_file(self._paths.history_file, history)

    def add_history_entry(
        self,
        history: list[dict[str, Any]],
        *,
        text: str,
        source_type: str,
        filename: str | None = None,
        audio_path: str | None = None,
    ) -> list[dict[str, Any]]:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source_type,
            "text": text,
            "filename": filename,
            "audio_path": audio_path,
        }
        updated = [entry, *history]
        return updated[:100]

    def clear_debug_log(self) -> None:
        """Remove local Murmur debug log files."""
        for path in DEBUG_LOG_PATHS:
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
            except OSError as error:
                self._logger.error(f"Failed to delete debug log {path}: {error}")

    def clear_all_local_data(self, audio_dir: str) -> None:
        """Delete transcription history, stored audio files, and debug logs."""
        if os.path.exists(self._paths.history_file):
            try:
                os.remove(self._paths.history_file)
            except OSError as error:
                self._logger.error(f"Failed to delete history file: {error}")

        if os.path.isdir(audio_dir):
            try:
                shutil.rmtree(audio_dir)
                self.ensure_audio_dir(audio_dir)
            except OSError as error:
                self._logger.error(f"Failed to clear audio directory: {error}")

        self.clear_debug_log()

    def ensure_audio_dir(self, audio_dir: str) -> None:
        """Create the audio directory with owner-only permissions."""
        try:
            os.makedirs(audio_dir, mode=0o700, exist_ok=True)
            os.chmod(audio_dir, 0o700)
        except OSError as error:
            self._logger.error(f"Failed to secure audio directory {audio_dir}: {error}")

    def _load_json_with_default(self, path: str, default: Any) -> Any:
        try:
            if os.path.exists(path):
                with open(path, "r") as file:
                    payload = json.load(file)
                    if isinstance(default, dict) and isinstance(payload, dict):
                        return {**default, **payload}
                    return payload
        except (json.JSONDecodeError, OSError) as error:
            self._logger.error(f"Failed to load JSON data from {path}: {error}")
        return default

    def _save_json_file(self, path: str, data: Any) -> None:
        try:
            with open(path, "w") as file:
                json.dump(data, file, indent=2)
            os.chmod(path, 0o600)
        except OSError as error:
            self._logger.error(f"Failed to save JSON data to {path}: {error}")
