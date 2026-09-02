import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.persistence_service import (
    DEBUG_LOG_PATHS,
    DEFAULT_CONFIG,
    PersistencePaths,
    PersistenceService,
    should_log_sensitive,
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
            }
            # Fails loudly if DEFAULT_CONFIG gains a key this fixture forgot.
            self.assertEqual(set(full_config.keys()), set(DEFAULT_CONFIG.keys()))

            service.save_config(full_config)
            loaded = service.load_config(DEFAULT_CONFIG)

            self.assertEqual(loaded, full_config)


if __name__ == "__main__":
    unittest.main()
