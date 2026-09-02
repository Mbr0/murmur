import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cleanup.llama_server import CLEANUP_MODEL_ID
from services.persistence_service import (
    CLEANUP_ENABLED_KEY,
    CLOUD_MODE_MURMUR,
    CLOUD_MODE_OFF,
    CLOUD_MODE_OWN_KEY,
    CONFIG_CLEANUP_CLOUD,
    CONFIG_CLEANUP_ENABLED,
    CONFIG_CLOUD_MODE,
    CONFIG_KEEP_AUDIO,
    CONFIG_HISTORY_ENABLED,
    DEBUG_LOG_PATHS,
    DEFAULT_CONFIG,
    ORIGIN_BYOK,
    ORIGIN_CLOUD,
    ORIGIN_LOCAL,
    USER_CONTENT_CONFIG_KEYS,
    PersistencePaths,
    PersistenceService,
    normalize_history,
    resolve_cleanup_enabled,
    should_log_sensitive,
    what_leaves_the_mac,
)


class TestLogger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)


class PersistenceServiceTests(unittest.TestCase):
    def test_load_config_returns_default_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "config.json"
            history = Path(tmp_dir) / "history.json"
            service = PersistenceService(
                PersistencePaths(str(config), str(history)),
                logger=TestLogger(),
            )
            self.assertEqual(service.load_config({"model": "medium"}), {"model": "medium"})

    def test_load_config_returns_default_when_corrupted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "config.json"
            history = Path(tmp_dir) / "history.json"
            config.write_text("{not valid json", encoding="utf-8")
            logger = TestLogger()
            service = PersistenceService(
                PersistencePaths(str(config), str(history)),
                logger=logger,
            )
            self.assertEqual(service.load_config({"model": "medium"}), {"model": "medium"})
            self.assertEqual(len(logger.errors), 1)

    def test_clear_all_local_data_removes_history_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "config.json"
            history = Path(tmp_dir) / "history.json"
            audio_dir = Path(tmp_dir) / "audio"
            audio_dir.mkdir()
            history.write_text('[{"text":"hello"}]', encoding="utf-8")
            (audio_dir / "sample.wav").write_text("audio", encoding="utf-8")
            service = PersistenceService(
                PersistencePaths(str(config), str(history)),
                logger=TestLogger(),
            )

            service.clear_all_local_data(str(audio_dir))

            self.assertFalse(history.exists())
            self.assertFalse(any(audio_dir.iterdir()))

    def test_clear_all_local_data_removes_legacy_mywhisper_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "config.json"
            history = Path(tmp_dir) / "history.json"
            audio_dir = Path(tmp_dir) / "audio"
            audio_dir.mkdir()
            legacy_config = Path(tmp_dir) / "legacy_config.json"
            legacy_history = Path(tmp_dir) / "legacy_history.json"
            legacy_audio = Path(tmp_dir) / "legacy_audio"
            legacy_audio.mkdir()
            legacy_config.write_text('{"model":"tiny"}', encoding="utf-8")
            legacy_history.write_text('[{"text":"old"}]', encoding="utf-8")
            (legacy_audio / "clip.wav").write_text("audio", encoding="utf-8")
            service = PersistenceService(
                PersistencePaths(str(config), str(history)),
                logger=TestLogger(),
            )

            service.clear_all_local_data(
                str(audio_dir),
                legacy_paths=(
                    str(legacy_config),
                    str(legacy_history),
                    str(legacy_audio),
                ),
            )

            self.assertFalse(legacy_config.exists())
            self.assertFalse(legacy_history.exists())
            self.assertFalse(legacy_audio.exists())

    def test_save_config_sets_restrictive_file_permissions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "config.json"
            history = Path(tmp_dir) / "history.json"
            service = PersistenceService(
                PersistencePaths(str(config), str(history)),
                logger=TestLogger(),
            )

            service.save_config({"model": "small", "save_audio": False})

            mode = stat.S_IMODE(config.stat().st_mode)
            self.assertEqual(mode, 0o600)

    def test_save_history_sets_restrictive_file_permissions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "config.json"
            history = Path(tmp_dir) / "history.json"
            service = PersistenceService(
                PersistencePaths(str(config), str(history)),
                logger=TestLogger(),
            )

            service.save_history([{"text": "hello"}])

            mode = stat.S_IMODE(history.stat().st_mode)
            self.assertEqual(mode, 0o600)

    def test_save_config_creates_file_with_owner_only_mode_atomically(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "config.json"
            history = Path(tmp_dir) / "history.json"
            service = PersistenceService(
                PersistencePaths(str(config), str(history)),
                logger=TestLogger(),
            )

            with patch(
                "services.persistence_service.os.open",
                wraps=os.open,
            ) as mock_open:
                service.save_config({"model": "small", "save_audio": False})

            mock_open.assert_called()
            _path, flags, mode = mock_open.call_args.args[:3]
            self.assertEqual(mode, 0o600)
            self.assertTrue(flags & os.O_CREAT)
            self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o600)

    def test_ensure_audio_dir_sets_restrictive_permissions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_dir = Path(tmp_dir) / "audio"
            service = PersistenceService(
                PersistencePaths(str(Path(tmp_dir) / "config.json"), str(Path(tmp_dir) / "history.json")),
                logger=TestLogger(),
            )

            service.ensure_audio_dir(str(audio_dir))

            mode = stat.S_IMODE(audio_dir.stat().st_mode)
            self.assertEqual(mode, 0o700)

    def test_clear_debug_log_removes_known_log_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_paths = [
                Path(tmp_dir) / "murmur.log",
                Path(tmp_dir) / "murmur_debug.log",
            ]
            for path in log_paths:
                path.write_text("debug", encoding="utf-8")

            service = PersistenceService(
                PersistencePaths(str(Path(tmp_dir) / "config.json"), str(Path(tmp_dir) / "history.json")),
                logger=TestLogger(),
            )
            original_paths = DEBUG_LOG_PATHS
            try:
                import services.persistence_service as persistence_module

                persistence_module.DEBUG_LOG_PATHS = tuple(str(path) for path in log_paths)
                service.clear_debug_log()
            finally:
                persistence_module.DEBUG_LOG_PATHS = original_paths

            for path in log_paths:
                self.assertFalse(path.exists())

    def test_clear_all_local_data_also_clears_debug_logs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "config.json"
            history = Path(tmp_dir) / "history.json"
            audio_dir = Path(tmp_dir) / "audio"
            audio_dir.mkdir()
            history.write_text('[{"text":"hello"}]', encoding="utf-8")
            debug_log = Path(tmp_dir) / "murmur_debug.log"
            debug_log.write_text("debug", encoding="utf-8")
            service = PersistenceService(
                PersistencePaths(str(config), str(history)),
                logger=TestLogger(),
            )
            original_paths = DEBUG_LOG_PATHS
            try:
                import services.persistence_service as persistence_module

                persistence_module.DEBUG_LOG_PATHS = (str(debug_log),)
                service.clear_all_local_data(str(audio_dir))
            finally:
                persistence_module.DEBUG_LOG_PATHS = original_paths

            self.assertFalse(history.exists())
            self.assertFalse(debug_log.exists())

    def test_should_log_sensitive_respects_privacy_defaults(self):
        self.assertFalse(should_log_sensitive(DEFAULT_CONFIG))
        self.assertFalse(
            should_log_sensitive(
                {"privacy_mode": True, "save_history": True, "save_audio": False}
            )
        )
        self.assertTrue(
            should_log_sensitive(
                {"privacy_mode": False, "save_history": True, "save_audio": False}
            )
        )
        self.assertTrue(should_log_sensitive({"save_history": True}))

    def test_default_config_is_privacy_first(self):
        self.assertFalse(DEFAULT_CONFIG["save_audio"])
        self.assertFalse(DEFAULT_CONFIG["save_history"])
        self.assertTrue(DEFAULT_CONFIG["privacy_mode"])
        self.assertIsNone(DEFAULT_CONFIG["mic_device_index"])
        self.assertIsNone(DEFAULT_CONFIG["mic_device_name"])

    def test_default_config_declares_every_key_the_app_writes(self):
        """A key written but never declared has no default, so a fresh install
        reads None where the code expects a value."""
        self.assertIsNone(DEFAULT_CONFIG["hotkey_label"])
        self.assertEqual(DEFAULT_CONFIG["hints_notice_shown"], {})
        self.assertFalse(DEFAULT_CONFIG["onboarding_completed"])
        self.assertIsNone(DEFAULT_CONFIG["onboarding_version"])

    def _service(self, tmp_dir):
        return PersistenceService(
            PersistencePaths(
                str(Path(tmp_dir) / "config.json"),
                str(Path(tmp_dir) / "history.json"),
            ),
            logger=TestLogger(),
        )

    def test_update_config_writes_only_the_given_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self._service(tmp_dir)
            service.save_config({**DEFAULT_CONFIG, "language": "fr", "engine_id": "whispercpp"})

            merged = service.update_config({"language": "nl"})

            self.assertEqual(merged["language"], "nl")
            self.assertEqual(merged["engine_id"], "whispercpp")
            self.assertEqual(service.load_config(dict(DEFAULT_CONFIG))["engine_id"], "whispercpp")

    def test_update_config_keeps_a_write_that_landed_after_the_caller_loaded(self):
        """The Settings-window bug: a snapshot taken at open time and saved on
        Save reverted whatever the app wrote while the window was up."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self._service(tmp_dir)
            service.save_config({**DEFAULT_CONFIG, "engine_id": None, "model_id": None})

            snapshot = service.load_config(dict(DEFAULT_CONFIG))  # window opens
            service.update_config({"engine_id": "whispercpp", "model_id": "turbo"})  # app writes

            self.assertIsNone(snapshot["engine_id"])  # the stale view
            merged = service.update_config({"language": "de"})  # window saves its own key

            self.assertEqual(merged["engine_id"], "whispercpp")
            self.assertEqual(merged["model_id"], "turbo")
            self.assertEqual(merged["language"], "de")

    def test_update_config_fills_missing_keys_from_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self._service(tmp_dir)

            merged = service.update_config({"onboarding_completed": True})

            self.assertTrue(merged["onboarding_completed"])
            self.assertEqual(merged["language"], DEFAULT_CONFIG["language"])
            self.assertEqual(set(merged), set(DEFAULT_CONFIG))

    def test_add_history_entry_keeps_latest_100(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "config.json"
            history = Path(tmp_dir) / "history.json"
            service = PersistenceService(
                PersistencePaths(str(config), str(history)),
                logger=TestLogger(),
            )

            items = []
            for i in range(120):
                items = service.add_history_entry(
                    items,
                    text=f"entry-{i}",
                    source_type="live",
                    audio_path=None,
                )
            self.assertEqual(len(items), 100)
            self.assertEqual(items[0]["text"], "entry-119")
            self.assertEqual(items[-1]["text"], "entry-20")

    def test_save_and_load_config_round_trips_all_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "config.json"
            history = Path(tmp_dir) / "history.json"
            service = PersistenceService(
                PersistencePaths(str(config), str(history)),
                logger=TestLogger(),
            )

            full_config = {
                "save_audio": True,
                "save_history": True,
                "privacy_mode": False,
                "appearance_mode": "dark",
                "hotkey_keycode": 12,
                "hotkey_command": True,
                "hotkey_option": False,
                "hotkey_control": True,
                "hotkey_shift": True,
                "hotkey_fn": True,
                "hotkey_mode": "toggle",
                "mic_device_index": 2,
                "mic_device_name": "USB Mic",
                "language": "fr",
                "language_by_app": {"com.apple.Terminal": "en"},
                "vocabulary_terms": ["Murmur", "Voxtral"],
                "vocabulary_replacements": [
                    {"from": "teh", "to": "the", "match_case": False}
                ],
                "engine_id": "voxtral_mlx",
                "model_id": "voxtral-mini-4b-realtime-4bit",
                "hotkey_label": "Space",
                "hints_notice_shown": {"voxtral_mlx": True},
                "onboarding_completed": True,
                "onboarding_version": 1,
                "cleanup_enabled": True,
                "cleanup_mode": "message",
                "cleanup_tone": "warm",
                "mode_by_app": {"com.apple.mail": "mail"},
                "context_awareness": False,
                "include_selection": True,
                "cleanup_model_id": "ministral-3-3b-instruct-2512-q4_k_m",
                "pill_enabled": False,
                "cleanup_prewarm": False,
                "cloud_mode": "murmur_cloud",
                "byok_provider": "openai",
                "cleanup_cloud": True,
                "update_channel": "beta",
                "launch_at_login": True,
                "settings_last_tab": "account",
            }
            # Fails loudly if DEFAULT_CONFIG gains a key this fixture forgot.
            self.assertEqual(set(full_config.keys()), set(DEFAULT_CONFIG.keys()))

            service.save_config(full_config)
            loaded = service.load_config(DEFAULT_CONFIG)

            self.assertEqual(loaded, full_config)


class CleanupDefaultsTests(unittest.TestCase):
    """The Wave 2 keys and the one default that has to ask the machine."""

    def test_smart_layer_defaults(self):
        self.assertIsNone(DEFAULT_CONFIG["cleanup_enabled"])
        self.assertEqual(DEFAULT_CONFIG["cleanup_mode"], "dictation")
        self.assertEqual(DEFAULT_CONFIG["cleanup_tone"], "neutral")
        self.assertEqual(DEFAULT_CONFIG["mode_by_app"], {})
        self.assertTrue(DEFAULT_CONFIG["context_awareness"])
        self.assertFalse(DEFAULT_CONFIG["include_selection"])
        self.assertEqual(DEFAULT_CONFIG["cleanup_model_id"], CLEANUP_MODEL_ID)
        self.assertTrue(DEFAULT_CONFIG["pill_enabled"])

    def test_undecided_asks_the_machine(self):
        calls = []

        def probe():
            calls.append(1)
            return True

        self.assertTrue(resolve_cleanup_enabled({}, probe=probe))
        self.assertTrue(
            resolve_cleanup_enabled({CLEANUP_ENABLED_KEY: None}, probe=probe)
        )
        self.assertEqual(len(calls), 2)

    def test_a_stored_answer_is_never_second_guessed(self):
        def probe():
            raise AssertionError("the probe must not run once the key is a bool")

        self.assertTrue(
            resolve_cleanup_enabled({CLEANUP_ENABLED_KEY: True}, probe=probe)
        )
        # False is an answer, not an absence: a 16 GB Mac must not switch
        # cleanup back on for a user who turned it off.
        self.assertFalse(
            resolve_cleanup_enabled({CLEANUP_ENABLED_KEY: False}, probe=probe)
        )

    def test_the_probe_result_is_coerced_to_a_bool(self):
        self.assertIs(resolve_cleanup_enabled({}, probe=lambda: 0), False)
        self.assertIs(resolve_cleanup_enabled({}, probe=lambda: 1), True)

    def test_update_config_stores_the_resolved_answer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = PersistenceService(
                PersistencePaths(
                    str(Path(tmp_dir) / "config.json"),
                    str(Path(tmp_dir) / "history.json"),
                ),
                logger=TestLogger(),
            )
            loaded = service.load_config(dict(DEFAULT_CONFIG))
            enabled = resolve_cleanup_enabled(loaded, probe=lambda: True)
            merged = service.update_config({CLEANUP_ENABLED_KEY: enabled})

            self.assertTrue(merged[CLEANUP_ENABLED_KEY])
            reloaded = service.load_config(dict(DEFAULT_CONFIG))
            self.assertIs(reloaded[CLEANUP_ENABLED_KEY], True)


def _service(tmp_dir, logger=None):
    return PersistenceService(
        PersistencePaths(
            str(Path(tmp_dir) / "config.json"),
            str(Path(tmp_dir) / "history.json"),
        ),
        logger=logger or TestLogger(),
    )


class HistoryOriginTests(unittest.TestCase):
    def test_add_history_entry_defaults_to_local_origin(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)

            items = service.add_history_entry([], text="hello", source_type="live")

            self.assertEqual(items[0]["origin"], ORIGIN_LOCAL)
            self.assertIsNone(items[0]["engine_id"])
            self.assertIsNone(items[0]["duration_s"])

    def test_add_history_entry_records_origin_engine_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)

            items = service.add_history_entry(
                [],
                text="bonjour",
                source_type="live",
                origin=ORIGIN_CLOUD,
                engine_id="murmur_cloud",
                duration_s=4.25,
            )

            entry = items[0]
            self.assertEqual(entry["origin"], ORIGIN_CLOUD)
            self.assertEqual(entry["engine_id"], "murmur_cloud")
            self.assertEqual(entry["duration_s"], 4.25)
            # The pre-Wave-3 shape stays readable by history_window.py.
            self.assertEqual(entry["text"], "bonjour")
            self.assertEqual(entry["source"], "live")
            self.assertIsNone(entry["filename"])
            self.assertIsNone(entry["audio_path"])
            self.assertIn("timestamp", entry)

    def test_add_history_entry_accepts_byok_origin(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)

            items = service.add_history_entry(
                [], text="hi", source_type="file", origin=ORIGIN_BYOK, engine_id="byok_openai"
            )

            self.assertEqual(items[0]["origin"], ORIGIN_BYOK)

    def test_add_history_entry_rejects_unknown_origin(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = _service(tmp_dir)

            with self.assertRaises(ValueError):
                service.add_history_entry([], text="hi", source_type="live", origin="server")

    def test_load_history_reads_legacy_entries_as_local(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = Path(tmp_dir) / "history.json"
            history.write_text(
                '[{"timestamp": "2026-01-01T00:00:00", "source": "live", "text": "old"}]',
                encoding="utf-8",
            )
            service = _service(tmp_dir)

            entries = service.load_history()

            self.assertEqual(entries[0]["origin"], ORIGIN_LOCAL)
            self.assertIsNone(entries[0]["engine_id"])
            self.assertEqual(entries[0]["text"], "old")

    def test_normalize_history_keeps_a_known_origin_and_drops_junk(self):
        entries = normalize_history(
            [
                {"text": "a", "origin": ORIGIN_CLOUD, "engine_id": "cloud"},
                {"text": "b", "origin": "nonsense"},
                "not-a-dict",
            ]
        )

        self.assertEqual(entries[0]["origin"], ORIGIN_CLOUD)
        self.assertEqual(entries[1]["origin"], ORIGIN_LOCAL)
        self.assertEqual(len(entries), 2)


class WhatLeavesTheMacTests(unittest.TestCase):
    """One row per configuration: config in, exact lines out, in order."""

    LOCAL = "Nothing. Audio and text stay on this Mac."
    DOWNLOADS = "Model files are downloaded from Hugging Face when you choose an engine."
    UPDATES = "The app checks GitHub for updates when you ask it to."
    CLOUD_AUDIO = (
        "Audio is sent to Murmur Cloud (the Boske proxy, hosted in the EU) "
        "to be turned into text."
    )
    CLOUD_CLEANUP = (
        "Transcribed text is sent to Murmur Cloud (the Boske proxy, hosted in the EU) "
        "to be cleaned up."
    )
    MISTRAL_AUDIO = (
        "Audio is sent to Mistral with your own API key, to be turned into text."
    )
    OPENAI_AUDIO = (
        "Audio is sent to OpenAI with your own API key, to be turned into text."
    )
    UNKNOWN_PROVIDER_AUDIO = (
        "Audio is sent to the provider you choose, with your own API key, "
        "to be turned into text."
    )
    HISTORY_ON = "Transcriptions are kept on this Mac, in your history."
    HISTORY_OFF = "Transcriptions are not kept after they are typed."
    AUDIO_ON = "Recordings are kept on this Mac."
    AUDIO_OFF = "Recordings are deleted as soon as they are transcribed."

    def test_lines_per_configuration(self):
        table = [
            (
                "local engine, nothing kept",
                {},
                [self.LOCAL, self.DOWNLOADS, self.UPDATES, self.HISTORY_OFF, self.AUDIO_OFF],
            ),
            (
                "local engine, history and audio kept",
                {
                    CONFIG_CLOUD_MODE: CLOUD_MODE_OFF,
                    CONFIG_HISTORY_ENABLED: True,
                    CONFIG_KEEP_AUDIO: True,
                },
                [self.LOCAL, self.DOWNLOADS, self.UPDATES, self.HISTORY_ON, self.AUDIO_ON],
            ),
            (
                "local engine, local cleanup on",
                {CONFIG_CLEANUP_ENABLED: True, CONFIG_CLEANUP_CLOUD: False},
                [self.LOCAL, self.DOWNLOADS, self.UPDATES, self.HISTORY_OFF, self.AUDIO_OFF],
            ),
            (
                "Murmur Cloud, cleanup off",
                {CONFIG_CLOUD_MODE: CLOUD_MODE_MURMUR},
                [
                    self.CLOUD_AUDIO,
                    self.DOWNLOADS,
                    self.UPDATES,
                    self.HISTORY_OFF,
                    self.AUDIO_OFF,
                ],
            ),
            (
                "Murmur Cloud with cloud cleanup and history",
                {
                    CONFIG_CLOUD_MODE: CLOUD_MODE_MURMUR,
                    CONFIG_CLEANUP_ENABLED: True,
                    CONFIG_CLEANUP_CLOUD: True,
                    CONFIG_HISTORY_ENABLED: True,
                },
                [
                    self.CLOUD_AUDIO,
                    self.CLOUD_CLEANUP,
                    self.DOWNLOADS,
                    self.UPDATES,
                    self.HISTORY_ON,
                    self.AUDIO_OFF,
                ],
            ),
            (
                "Murmur Cloud, cleanup enabled but running locally",
                {
                    CONFIG_CLOUD_MODE: CLOUD_MODE_MURMUR,
                    CONFIG_CLEANUP_ENABLED: True,
                    CONFIG_CLEANUP_CLOUD: False,
                },
                [
                    self.CLOUD_AUDIO,
                    self.DOWNLOADS,
                    self.UPDATES,
                    self.HISTORY_OFF,
                    self.AUDIO_OFF,
                ],
            ),
            (
                "own key, Mistral",
                {CONFIG_CLOUD_MODE: CLOUD_MODE_OWN_KEY, "byok_provider": "mistral"},
                [
                    self.MISTRAL_AUDIO,
                    self.DOWNLOADS,
                    self.UPDATES,
                    self.HISTORY_OFF,
                    self.AUDIO_OFF,
                ],
            ),
            (
                "own key, OpenAI, audio kept",
                {
                    CONFIG_CLOUD_MODE: CLOUD_MODE_OWN_KEY,
                    "byok_provider": "openai",
                    CONFIG_KEEP_AUDIO: True,
                },
                [
                    self.OPENAI_AUDIO,
                    self.DOWNLOADS,
                    self.UPDATES,
                    self.HISTORY_OFF,
                    self.AUDIO_ON,
                ],
            ),
            (
                "own key, provider not chosen yet",
                {CONFIG_CLOUD_MODE: CLOUD_MODE_OWN_KEY, "byok_provider": None},
                [
                    self.UNKNOWN_PROVIDER_AUDIO,
                    self.DOWNLOADS,
                    self.UPDATES,
                    self.HISTORY_OFF,
                    self.AUDIO_OFF,
                ],
            ),
            (
                "cloud cleanup on but cloud cleanup needs cleanup enabled",
                {CONFIG_CLEANUP_ENABLED: False, CONFIG_CLEANUP_CLOUD: True},
                [self.LOCAL, self.DOWNLOADS, self.UPDATES, self.HISTORY_OFF, self.AUDIO_OFF],
            ),
            (
                "own key, provider not yet overridden defaults to Mistral",
                {CONFIG_CLOUD_MODE: CLOUD_MODE_OWN_KEY},
                [
                    self.MISTRAL_AUDIO,
                    self.DOWNLOADS,
                    self.UPDATES,
                    self.HISTORY_OFF,
                    self.AUDIO_OFF,
                ],
            ),
        ]

        for name, overrides, expected in table:
            with self.subTest(configuration=name):
                config = {**DEFAULT_CONFIG, **overrides}
                self.assertEqual(what_leaves_the_mac(config), expected)

    def test_local_line_names_the_engine_actually_selected(self):
        class FakeInfo:
            name = "whisper.cpp large-v3-turbo"

        lines = what_leaves_the_mac(dict(DEFAULT_CONFIG), engine_info=FakeInfo())

        self.assertEqual(lines[0], self.LOCAL)
        self.assertEqual(lines[1], "Transcription runs here, using whisper.cpp large-v3-turbo.")

    def test_engine_name_is_not_claimed_when_transcription_is_remote(self):
        class FakeInfo:
            name = "whisper.cpp large-v3-turbo"

        config = {**DEFAULT_CONFIG, CONFIG_CLOUD_MODE: CLOUD_MODE_MURMUR}

        lines = what_leaves_the_mac(config, engine_info=FakeInfo())

        self.assertEqual(lines[0], self.CLOUD_AUDIO)
        self.assertFalse(any("runs here" in line for line in lines))

    def test_engine_info_accepts_a_mapping(self):
        lines = what_leaves_the_mac(dict(DEFAULT_CONFIG), engine_info={"name": "Voxtral Mini"})

        self.assertEqual(lines[1], "Transcription runs here, using Voxtral Mini.")

    def test_unknown_cloud_mode_is_treated_as_local(self):
        config = {**DEFAULT_CONFIG, CONFIG_CLOUD_MODE: "something-else"}

        self.assertEqual(what_leaves_the_mac(config)[0], self.LOCAL)

    def test_default_config_documents_the_privacy_keys(self):
        self.assertEqual(DEFAULT_CONFIG[CONFIG_CLOUD_MODE], CLOUD_MODE_OFF)
        self.assertEqual(DEFAULT_CONFIG["byok_provider"], "mistral")
        # None, not False: Wave 2 probes the machine once on first load and
        # stores the answer. Until then the privacy text must read it as off.
        self.assertIsNone(DEFAULT_CONFIG[CONFIG_CLEANUP_ENABLED])
        self.assertFalse(DEFAULT_CONFIG[CONFIG_CLEANUP_CLOUD])
        # History and audio reuse the keys Murmur already shipped.
        self.assertEqual(CONFIG_HISTORY_ENABLED, "save_history")
        self.assertEqual(CONFIG_KEEP_AUDIO, "save_audio")


class DeleteAllDataTests(unittest.TestCase):
    def _seeded_config(self):
        return {
            **DEFAULT_CONFIG,
            "hotkey_keycode": 12,
            "hotkey_mode": "hold",
            "engine_id": "whispercpp",
            "model_id": "large-v3-turbo",
            "appearance_mode": "dark",
            "onboarding_completed": True,
            "vocabulary_terms": ["Murmur"],
            "vocabulary_replacements": [{"from": "teh", "to": "the", "match_case": False}],
            "language_by_app": {"com.apple.Terminal": "en"},
            "mode_by_app": {"com.apple.mail": "mail"},
            "hints_notice_shown": {"whispercpp": True},
        }

    def test_delete_all_data_removes_history_audio_and_user_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = Path(tmp_dir) / "history.json"
            audio_dir = Path(tmp_dir) / "audio"
            audio_dir.mkdir()
            history.write_text('[{"text": "a"}, {"text": "b"}]', encoding="utf-8")
            (audio_dir / "one.wav").write_text("audio", encoding="utf-8")
            (audio_dir / "two.wav").write_text("audio", encoding="utf-8")
            service = _service(tmp_dir)
            config = self._seeded_config()

            summary = service.delete_all_data(str(audio_dir), config, legacy_paths=())

            self.assertFalse(history.exists())
            self.assertTrue(audio_dir.is_dir())
            self.assertEqual(list(audio_dir.iterdir()), [])
            self.assertEqual(summary.history_entries, 2)
            self.assertEqual(summary.audio_files, 2)
            self.assertEqual(sorted(summary.config_keys), sorted(USER_CONTENT_CONFIG_KEYS))
            self.assertIn("2 history entries", summary.describe())

    def test_delete_all_data_keeps_preferences(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_dir = Path(tmp_dir) / "audio"
            audio_dir.mkdir()
            service = _service(tmp_dir)
            config = self._seeded_config()

            service.delete_all_data(str(audio_dir), config, legacy_paths=())

            self.assertEqual(config["hotkey_keycode"], 12)
            self.assertEqual(config["hotkey_mode"], "hold")
            self.assertEqual(config["engine_id"], "whispercpp")
            self.assertEqual(config["model_id"], "large-v3-turbo")
            self.assertEqual(config["appearance_mode"], "dark")
            self.assertTrue(config["onboarding_completed"])
            # User content is reset, not left behind.
            self.assertEqual(config["vocabulary_terms"], [])
            self.assertEqual(config["vocabulary_replacements"], [])
            self.assertEqual(config["language_by_app"], {})
            self.assertNotIn("mode_by_app", config)

    def test_delete_all_data_does_not_re_arm_a_notice_already_shown(self):
        """"Delete all data" promises to keep what the user chose. Clearing
        ``hints_notice_shown`` would replay a once-per-engine notice they have
        already dismissed — that is a preference, not something they said."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_dir = Path(tmp_dir) / "audio"
            audio_dir.mkdir()
            service = _service(tmp_dir)
            config = self._seeded_config()

            summary = service.delete_all_data(str(audio_dir), config, legacy_paths=())

            self.assertEqual(config["hints_notice_shown"], {"whispercpp": True})
            self.assertNotIn("hints_notice_shown", summary.config_keys)
            self.assertNotIn("hints_notice_shown", USER_CONTENT_CONFIG_KEYS)

    def test_delete_all_data_writes_the_trimmed_config_back(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_dir = Path(tmp_dir) / "audio"
            audio_dir.mkdir()
            service = _service(tmp_dir)
            config = self._seeded_config()

            service.delete_all_data(str(audio_dir), config, legacy_paths=())
            reloaded = service.load_config(dict(DEFAULT_CONFIG))

            self.assertEqual(reloaded["vocabulary_terms"], [])
            self.assertEqual(reloaded["engine_id"], "whispercpp")

    def test_delete_all_data_without_config_only_touches_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_dir = Path(tmp_dir) / "audio"
            audio_dir.mkdir()
            service = _service(tmp_dir)

            summary = service.delete_all_data(str(audio_dir), legacy_paths=())

            self.assertEqual(summary.config_keys, ())
            self.assertEqual(summary.history_entries, 0)


if __name__ == "__main__":
    unittest.main()
