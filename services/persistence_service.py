#!/usr/bin/env python3
"""Persistence service for local config/history data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from typing import Any


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
        except OSError as error:
            self._logger.error(f"Failed to save JSON data to {path}: {error}")
