import tempfile
import unittest
from pathlib import Path

from services.persistence_service import PersistencePaths, PersistenceService


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


if __name__ == "__main__":
    unittest.main()
