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
- ``hotkey_label``: what to print for the shortcut's key, when the captured
  character says it better than the key code does (``services/hotkey_service.py``).
- ``hints_notice_shown``: ``{engine_id: True}`` for engines whose "this engine
  ignores your vocabulary" notice has already been shown once (``murmur.py``).
- ``onboarding_completed``, ``onboarding_version``: whether the first-run wizard
  has been finished, and which version of it (``ui/onboarding_window.py``).

Wave 2 (the smart layer) adds:

- ``cleanup_enabled``: whether the local cleanup pass runs at all. ``None`` in
  the defaults means "not decided yet": the first load resolves it from
  :func:`cleanup.llama_server.cleanup_default_for_current_machine` (off below
  16 GB of RAM, per the plan's latency risk) and writes the answer back, so the
  probe runs once per install and the file always states the real setting. See
  :func:`resolve_cleanup_enabled`.
- ``cleanup_mode``: ``"dictation" | "message" | "mail" | "notes" | "code"`` —
  the fallback mode when the front app is not in the table (``cleanup/modes.py``).
- ``cleanup_tone``: ``"neutral" | "warm" | "formal" | "terse"``.
- ``mode_by_app``: ``{bundle_id: mode}`` user overrides, which always win over
  the built-in table (``cleanup/context.py``).
- ``context_awareness``: whether the built-in bundle-id → mode table applies.
  Off means only ``mode_by_app`` and ``cleanup_mode`` decide.
- ``include_selection``: whether the Accessibility selected-text probe runs when
  capturing context. Off by default — no prompt consumes the selection yet, and
  it is the most sensitive thing this app can read.
- ``cleanup_model_id``: catalog id of the GGUF the cleanup server loads.
- ``pill_enabled``: whether the floating pill is shown while dictating
- ``cleanup_prewarm``: start the local cleanup server in the background at launch when cleanup is enabled and its model is installed
  (``ui/pill_window.py``).

Wave 3 (cloud and the settings tabs) adds:

- ``cloud_mode``: ``"off" | "murmur_cloud" | "own_key"`` — where transcription
  happens. ``"off"`` means the local engine and nothing leaving the Mac
  (read by :func:`what_leaves_the_mac`).
- ``byok_provider``: ``"mistral" | "openai"`` — which provider the own-key
  engine talks to. Defaults to ``"mistral"``, matching the Engine tab's own
  default (``ui/settings/engine_tab.py``).
- ``cleanup_cloud``: whether that cleanup runs in the cloud rather than on this
  Mac. Only meaningful when ``cleanup_enabled`` is true.
- ``update_channel``: ``"stable" | "beta"``, default ``"stable"``
  (``ui/settings/account_tab.py``).
- ``launch_at_login``: whether Murmur starts itself when the user logs in.
  ``False`` until the user turns it on (``ui/settings/general_tab.py``).
- ``settings_last_tab``: the identifier of the Settings tab last shown, so the
  window reopens where the user left it (``ui/settings/window.py``).

Writers that own only a few keys must go through :meth:`PersistenceService.update_config`
rather than saving a whole config they loaded earlier: a snapshot save silently
reverts every key another part of the app wrote in the meantime.

Two keys keep the names Murmur has always used rather than gaining synonyms:
``save_history`` is the history toggle and ``save_audio`` the keep-audio
toggle. Use :data:`CONFIG_HISTORY_ENABLED` and :data:`CONFIG_KEEP_AUDIO`.

History entries (``history.json``) are dicts with ``timestamp``, ``source``,
``text``, ``filename``, ``audio_path`` and, since Wave 3, ``origin``
(``"local" | "cloud" | "byok"``), ``engine_id`` and ``duration_s``. Entries
written before Wave 3 read back as ``origin="local"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
import shutil
import threading
from typing import Any

from cleanup.llama_server import CLEANUP_MODEL_ID


#: Serialises read-modify-write cycles on the config file. Module level rather
#: than per instance: the app, the Settings window and the wizard each build
#: their own :class:`PersistenceService` over the same file in the same process.
_CONFIG_LOCK = threading.RLock()


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
    "hotkey_label": None,
    "hints_notice_shown": {},
    "onboarding_completed": False,
    "onboarding_version": None,
    # -- Wave 2: the smart layer ------------------------------------------
    # None, not False: "nobody has decided yet". resolve_cleanup_enabled()
    # turns it into a real bool on first load and the app stores that.
    "cleanup_enabled": None,
    "cleanup_mode": "dictation",
    "cleanup_tone": "neutral",
    "mode_by_app": {},
    "context_awareness": True,
    "include_selection": False,
    "cleanup_model_id": CLEANUP_MODEL_ID,
    "pill_enabled": True,
    "cleanup_prewarm": True,
    # -- Wave 3: cloud and the settings tabs ------------------------------
    "cloud_mode": "off",
    "byok_provider": "mistral",
    "cleanup_cloud": False,
    "update_channel": "stable",
    "launch_at_login": False,
    "settings_last_tab": "general",
}

#: Config keys the privacy surface reads and writes.
CONFIG_HISTORY_ENABLED = "save_history"
CONFIG_KEEP_AUDIO = "save_audio"
CONFIG_PRIVACY_MODE = "privacy_mode"
CONFIG_CLOUD_MODE = "cloud_mode"
CONFIG_BYOK_PROVIDER = "byok_provider"
CONFIG_CLEANUP_ENABLED = "cleanup_enabled"
CONFIG_CLEANUP_CLOUD = "cleanup_cloud"

CLOUD_MODE_OFF = "off"
CLOUD_MODE_MURMUR = "murmur_cloud"
CLOUD_MODE_OWN_KEY = "own_key"

#: Where a transcription was produced, recorded on every history entry.
ORIGIN_LOCAL = "local"
ORIGIN_CLOUD = "cloud"
ORIGIN_BYOK = "byok"
HISTORY_ORIGINS: tuple[str, ...] = (ORIGIN_LOCAL, ORIGIN_CLOUD, ORIGIN_BYOK)

#: Config keys holding what the user said or typed, as opposed to preferences.
#: :meth:`PersistenceService.delete_all_data` clears these and keeps the rest.
#:
#: ``hints_notice_shown`` is deliberately *not* here: it records which
#: once-per-engine notices the user has already dismissed, which is a choice
#: they made, not something they said. Clearing it would replay a notice the
#: dialog promises to leave alone.
USER_CONTENT_CONFIG_KEYS: tuple[str, ...] = (
    "vocabulary_terms",
    "vocabulary_replacements",
    "language_by_app",
    "mode_by_app",
)

#: Own-key providers, in the words a person would recognise.
BYOK_PROVIDER_LABELS: dict[str, str] = {
    "mistral": "Mistral",
    "openai": "OpenAI",
}

#: Key whose ``None`` means "ask the machine once"; see :func:`resolve_cleanup_enabled`.
CLEANUP_ENABLED_KEY = "cleanup_enabled"


def resolve_cleanup_enabled(config: dict[str, Any], *, probe: Any = None) -> bool:
    """Whether cleanup is on, deciding it from this machine the first time.

    A stored bool is the user's answer and is returned untouched, including an
    explicit ``False``. Anything else (the ``None`` default, or a value some
    older config never carried) means the question has not been asked yet, so
    the RAM probe answers it — and the caller is expected to write the result
    back, which is why this never guesses twice for the same install.

    ``probe`` exists for the tests; the default is imported lazily so that
    reading config on a machine without the cleanup runtime costs nothing.
    """
    assert config is not None, "config is required"
    stored = config.get(CLEANUP_ENABLED_KEY)
    if isinstance(stored, bool):
        return stored
    if probe is None:
        from cleanup.llama_server import cleanup_default_for_current_machine

        probe = cleanup_default_for_current_machine
    return bool(probe())

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


def _fresh_default(key: str) -> Any:
    """A copy of ``DEFAULT_CONFIG[key]``, so resetting cannot alias the default."""
    default = DEFAULT_CONFIG[key]
    if isinstance(default, dict):
        return dict(default)
    if isinstance(default, list):
        return list(default)
    return default


def validate_history_origin(origin: str) -> str:
    """Return ``origin`` when it is one Murmur knows, else raise ``ValueError``."""
    if origin not in HISTORY_ORIGINS:
        allowed = ", ".join(HISTORY_ORIGINS)
        raise ValueError(f"Unknown history origin {origin!r}; expected one of {allowed}.")
    return origin


def normalize_history(entries: Any) -> list[dict[str, Any]]:
    """Read history entries, giving pre-Wave-3 ones the fields they lack.

    Legacy entries carry no ``origin``: they can only have come from a local
    engine, so they read back as ``"local"``. An entry with an origin Murmur
    does not know is read as local too — history is a display surface, and a
    corrupt file should not stop the window from opening. Non-dict junk is
    dropped.
    """
    if not isinstance(entries, list):
        return []
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        if item.get("origin") not in HISTORY_ORIGINS:
            item["origin"] = ORIGIN_LOCAL
        item.setdefault("engine_id", None)
        item.setdefault("duration_s", None)
        normalized.append(item)
    return normalized


def _engine_name(engine_info: Any) -> str | None:
    """The display name of the selected engine, from an ``EngineInfo`` or dict."""
    if engine_info is None:
        return None
    if isinstance(engine_info, dict):
        name = engine_info.get("name")
    else:
        name = getattr(engine_info, "name", None)
    return name or None


def what_leaves_the_mac(
    config: dict[str, Any],
    *,
    engine_info: Any = None,
) -> list[str]:
    """Plain-language lines describing what this configuration sends away.

    The order is fixed so the Privacy tab never reshuffles under the reader:
    transcription, cleanup, model downloads, update checks, then what is kept
    on this Mac. ``engine_info`` is only used to name the local engine, and
    only when transcription actually runs here.
    """
    cloud_mode = config.get(CONFIG_CLOUD_MODE, DEFAULT_CONFIG[CONFIG_CLOUD_MODE])
    lines: list[str] = []

    if cloud_mode == CLOUD_MODE_MURMUR:
        lines.append(
            "Audio is sent to Murmur Cloud (the Boske proxy, hosted in the EU) "
            "to be turned into text."
        )
    elif cloud_mode == CLOUD_MODE_OWN_KEY:
        provider = config.get(CONFIG_BYOK_PROVIDER)
        label = BYOK_PROVIDER_LABELS.get(provider) if provider else None
        if label:
            lines.append(
                f"Audio is sent to {label} with your own API key, to be turned into text."
            )
        else:
            lines.append(
                "Audio is sent to the provider you choose, with your own API key, "
                "to be turned into text."
            )
    else:
        # Anything unrecognised is the local engine: never claim less privacy
        # than the app actually gives, and never claim more.
        lines.append("Nothing. Audio and text stay on this Mac.")
        name = _engine_name(engine_info)
        if name:
            lines.append(f"Transcription runs here, using {name}.")

    cleanup_on = bool(config.get(CONFIG_CLEANUP_ENABLED, DEFAULT_CONFIG[CONFIG_CLEANUP_ENABLED]))
    cleanup_cloud = bool(config.get(CONFIG_CLEANUP_CLOUD, DEFAULT_CONFIG[CONFIG_CLEANUP_CLOUD]))
    if cleanup_on and cleanup_cloud:
        lines.append(
            "Transcribed text is sent to Murmur Cloud (the Boske proxy, hosted in the EU) "
            "to be cleaned up."
        )

    lines.append("Model files are downloaded from Hugging Face when you choose an engine.")
    lines.append("The app checks GitHub for updates when you ask it to.")

    if config.get(CONFIG_HISTORY_ENABLED, DEFAULT_CONFIG[CONFIG_HISTORY_ENABLED]):
        lines.append("Transcriptions are kept on this Mac, in your history.")
    else:
        lines.append("Transcriptions are not kept after they are typed.")

    if config.get(CONFIG_KEEP_AUDIO, DEFAULT_CONFIG[CONFIG_KEEP_AUDIO]):
        lines.append("Recordings are kept on this Mac.")
    else:
        lines.append("Recordings are deleted as soon as they are transcribed.")

    return lines


@dataclass(frozen=True)
class DeletionSummary:
    """What :meth:`PersistenceService.delete_all_data` actually removed."""

    history_entries: int = 0
    audio_files: int = 0
    debug_logs: int = 0
    config_keys: tuple[str, ...] = ()

    def describe(self) -> str:
        """One plain-language sentence for the confirmation alert."""
        parts = [
            f"{self.history_entries} history entries",
            f"{self.audio_files} saved recordings",
        ]
        if self.config_keys:
            parts.append("your vocabulary and per-app choices")
        return "Removed " + ", ".join(parts[:-1]) + f" and {parts[-1]}."


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
        """Write the whole config. Prefer :meth:`update_config` from any caller
        that owns only some of the keys."""
        with _CONFIG_LOCK:
            self._save_json_file(self._paths.config_file, config)

    def update_config(
        self,
        changes: dict[str, Any],
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge ``changes`` into the config on disk and return the merged result.

        Load, merge and save happen under one lock, so a caller writes only the
        keys it owns and leaves the rest exactly as it found them. This is the
        difference between the Settings window saving the two switches the user
        touched and it writing back the snapshot it loaded when the window
        opened — which used to revert ``engine_id``, ``onboarding_completed``
        and anything else the app had written while the window was up.
        """
        assert changes is not None, "changes is required"
        base = DEFAULT_CONFIG if default is None else default
        with _CONFIG_LOCK:
            config = self.load_config(dict(base))
            config.update(changes)
            self._save_json_file(self._paths.config_file, config)
            return config

    def load_history(self) -> list[dict[str, Any]]:
        return normalize_history(self._load_json_with_default(self._paths.history_file, []))

    def save_history(self, history: list[dict[str, Any]]) -> None:
        self._save_json_file(self._paths.history_file, history)

    def add_history_entry(
        self,
        history: list[dict[str, Any]],
        *,
        text: str,
        source_type: str,
        origin: str = ORIGIN_LOCAL,
        engine_id: str | None = None,
        duration_s: float | None = None,
        filename: str | None = None,
        audio_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Prepend one entry, capped at 100, and return the new list.

        ``origin`` says where the transcription happened and is validated here:
        a typo must not quietly label a cloud transcription as local.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source_type,
            "text": text,
            "filename": filename,
            "audio_path": audio_path,
            "origin": validate_history_origin(origin),
            "engine_id": engine_id,
            "duration_s": duration_s,
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

    def delete_all_data(
        self,
        audio_dir: str,
        config: dict[str, Any] | None = None,
        *,
        legacy_paths: tuple[str, ...] | None = None,
    ) -> DeletionSummary:
        """Remove everything the user said, keep everything they chose.

        History, saved recordings, debug logs and the legacy ``~/.mywhisper_*``
        files go. So do the config keys holding user content
        (:data:`USER_CONTENT_CONFIG_KEYS`) when ``config`` is given: keys with a
        documented default are reset to it, keys owned by other features are
        dropped. Preferences — the hotkey, the engine and model, appearance,
        onboarding flags — are untouched, and the trimmed config is saved.
        """
        history_entries = len(self.load_history())
        audio_files = 0
        if os.path.isdir(audio_dir):
            audio_files = sum(1 for _ in os.scandir(audio_dir))
        debug_logs = sum(1 for path in DEBUG_LOG_PATHS if os.path.exists(path))

        self.clear_all_local_data(audio_dir, legacy_paths=legacy_paths)

        removed_keys: list[str] = []
        if config is not None:
            for key in USER_CONTENT_CONFIG_KEYS:
                if key not in config:
                    continue
                if key in DEFAULT_CONFIG:
                    config[key] = _fresh_default(key)
                else:
                    # Owned by a feature that documents no default of its own.
                    del config[key]
                removed_keys.append(key)
            self.save_config(config)

        return DeletionSummary(
            history_entries=history_entries,
            audio_files=audio_files,
            debug_logs=debug_logs,
            config_keys=tuple(removed_keys),
        )

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
