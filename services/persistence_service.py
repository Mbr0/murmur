#!/usr/bin/env python3
"""Persistence service for local config/history data.

Config keys (defaults live in ``DEFAULT_CONFIG`` below):

- ``save_audio``, ``save_history``, ``privacy_mode``: user data retention toggles.
- ``appearance_mode``: ``"system" | "light" | "dark"``.
- ``hotkey_keycode``, ``hotkey_command``, ``hotkey_option``, ``hotkey_control``,
  ``hotkey_shift``, ``hotkey_fn``: the push-to-talk key combo.
- ``hotkey_mode``: ``"auto" | "toggle" | "hold"`` — how a hotkey press behaves
  (``services/hotkey_service.py``, Wave 1a).
- ``mic_device_index``, ``mic_device_name``: the selected input device.
- ``language``: ``"auto"`` or an ISO code, the default transcription language
  (``services/language_service.py``).
- ``language_by_app``: ``{bundle_id: language_code}``, overriding ``language``
  per front app (``services/language_service.py``).
- ``vocabulary_terms``: list of terms biasing transcription
  (``cleanup/vocabulary.py``).
- ``vocabulary_replacements``: list of ``{"from", "to", "match_case"}`` text
  replacements applied to transcripts (``cleanup/vocabulary.py``).
- ``engine_id``, ``model_id``: the chosen speech engine and model id; ``None``
  until chosen at first run (Wave 1c).
"""

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
    "hotkey_mode": "auto",
    "mic_device_index": None,
    "mic_device_name": None,
    "language": "auto",
    "language_by_app": {},
    "vocabulary_terms": [],
    "vocabulary_replacements": [],
    "engine_id": None,
    "model_id": None,
}

DEBUG_LOG_PATHS: tuple[str, ...] = (
    os.path.expanduser("~/Library/Logs/Murmur/murmur.log"),
    "/tmp/murmur_debug.log",
)

LEGACY_DATA_PATHS: tuple[str, ...] = (
    os.path.expanduser("~/.mywhisper_config.json"),
    os.path.expanduser("~/.mywhisper_history.json"),
    os.path.expanduser("~/.mywhisper_audio"),
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

    def clear_all_local_data(
        self,
        audio_dir: str,
        *,
        legacy_paths: tuple[str, ...] | None = None,
    ) -> None:
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

        paths = LEGACY_DATA_PATHS if legacy_paths is None else legacy_paths
        for path in paths:
            self._remove_path(path)

        self.clear_debug_log()

    def _remove_path(self, path: str) -> None:
        """Remove a file or directory if it exists."""
        if not os.path.exists(path):
            return
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError as error:
            self._logger.error(f"Failed to delete legacy path {path}: {error}")

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
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fd = os.open(path, flags, 0o600)
            # Create uses 0o600; fchmod covers rewrites of looser existing files.
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as file:
                json.dump(data, file, indent=2)
        except OSError as error:
            self._logger.error(f"Failed to save JSON data to {path}: {error}")
