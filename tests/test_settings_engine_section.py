"""Tests for the Settings "Speech engine" section state.

A real :class:`~engines.model_store.ModelStore` over a temporary root backs
every test, so "installed" means files actually on disk. No AppKit, no network.

The language and vocabulary section tests that used to live here went with
``settings_window.py`` into ``_archive/``; those rules are now owned by
``tests/test_settings_general_tab.py`` and ``tests/test_settings_smart_tab.py``.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from engines.model_store import ModelFile, ModelSpec, ModelStore
from services.model_profile_service import CHIP_APPLE_SILICON, CHIP_INTEL, VOXTRAL_MIN_RAM_GB
from ui.download_sheet import CONFIG_ENGINE_ID, CONFIG_MODEL_ID, EngineSectionModel

TURBO_Q5 = ModelSpec(
    id="whispercpp-turbo-q5",
    engine="whispercpp",
    display_name="Whisper large-v3-turbo (quantised)",
    files=(ModelFile("turbo-q5.bin", 574_000_000, "a" * 64, "http://x/turbo-q5.bin"),),
    source="http://x",
    license="MIT",
)
TURBO = ModelSpec(
    id="whispercpp-turbo",
    engine="whispercpp",
    display_name="Whisper large-v3-turbo",
    files=(ModelFile("turbo.bin", 1_600_000_000, "b" * 64, "http://x/turbo.bin"),),
    source="http://x",
    license="MIT",
)
VOXTRAL = ModelSpec(
    id="voxtral-4bit",
    engine="voxtral_mlx",
    display_name="Voxtral Mini 4B Realtime (4-bit MLX)",
    files=(
        ModelFile("config.json", 1_000, "c" * 64, "http://x/config.json"),
        ModelFile("weights.bin", 3_100_000_000, "d" * 64, "http://x/weights.bin"),
    ),
    source="http://x",
    license="Apache-2.0",
)
CATALOG = (TURBO_Q5, TURBO, VOXTRAL)


class _Fixture:
    """A store over a throwaway root plus the config dict the section writes."""

    def __init__(self, tmp: str):
        self.store = ModelStore(root=Path(tmp), catalog=CATALOG)
        self.config: dict = {}
        self.engine_changes: list[tuple[str, str]] = []
        self.saved_changes: list[dict] = []
        self.saves = 0

    def install(self, spec: ModelSpec) -> None:
        directory = self.store.path(spec.id)
        directory.mkdir(parents=True, exist_ok=True)
        for item in spec.files:
            # Sparse: is_installed only checks st_size, and a real 3.1 GB of
            # zeroes in a unit test would be absurd.
            with open(directory / item.name, "wb") as handle:
                handle.truncate(item.size_bytes)

    def section(
        self,
        chip=CHIP_APPLE_SILICON,
        ram_gb=VOXTRAL_MIN_RAM_GB,
        default_engine="whispercpp",
        refusal=None,
    ):
        def on_engine_change(engine, model):
            self.engine_changes.append((engine, model))
            return refusal

        return EngineSectionModel(
            self.config,
            self.store,
            chip=chip,
            ram_gb=ram_gb,
            default_engine=default_engine,
            on_engine_change=on_engine_change,
            save_changes=self._save_changes,
        )

    def _save_changes(self, changes):
        # Only the keys the section owns; never a whole-config snapshot.
        assert set(changes) == {CONFIG_ENGINE_ID, CONFIG_MODEL_ID}, changes
        self.saved_changes.append(dict(changes))
        self.saves += 1


class EngineSectionModelTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = _Fixture(self._tmp.name)

    def test_apple_silicon_sees_voxtral(self):
        section = self.fx.section(chip=CHIP_APPLE_SILICON)
        self.assertEqual(
            [choice.model_id for choice in section.choices],
            ["whispercpp-turbo-q5", "whispercpp-turbo", "voxtral-4bit"],
        )

    def test_intel_never_sees_voxtral(self):
        section = self.fx.section(chip=CHIP_INTEL)
        self.assertEqual(
            [choice.model_id for choice in section.choices],
            ["whispercpp-turbo-q5", "whispercpp-turbo"],
        )

    def test_low_ram_apple_silicon_does_not_see_voxtral(self):
        section = self.fx.section(chip=CHIP_APPLE_SILICON, ram_gb=8)
        self.assertEqual(
            [choice.model_id for choice in section.choices],
            ["whispercpp-turbo-q5", "whispercpp-turbo"],
        )

    def test_titles_carry_name_size_and_install_state(self):
        self.fx.install(TURBO_Q5)
        section = self.fx.section()
        titles = [choice.title for choice in section.choices]
        self.assertEqual(
            titles[0], "Whisper large-v3-turbo (quantised) · 574 MB · Installed"
        )
        self.assertEqual(titles[1], "Whisper large-v3-turbo · 1.6 GB · Not downloaded")
        self.assertEqual(
            titles[2],
            "Voxtral Mini 4B Realtime (4-bit MLX) · 3.1 GB · Not downloaded",
        )

    def test_default_selection_is_the_default_engines_first_model(self):
        section = self.fx.section(default_engine="voxtral_mlx")
        self.assertEqual(section.selected_model_id, "voxtral-4bit")
        self.assertEqual(section.selected_index, 2)
        self.assertTrue(section.selected_choice.recommended)
        self.assertIsNone(section.active_engine_id)
        self.assertIsNone(section.active_model_id)

    def test_a_default_engine_with_no_available_model_falls_to_the_first_choice(self):
        section = self.fx.section(chip=CHIP_INTEL, default_engine="voxtral_mlx")
        self.assertEqual(section.selected_model_id, "whispercpp-turbo-q5")

    def test_config_selection_wins_over_the_default(self):
        self.fx.config.update({CONFIG_ENGINE_ID: "whispercpp", CONFIG_MODEL_ID: "whispercpp-turbo"})
        section = self.fx.section()
        self.assertEqual(section.selected_model_id, "whispercpp-turbo")
        self.assertEqual(section.active_model_id, "whispercpp-turbo")

    def test_a_config_model_this_machine_cannot_run_is_not_offered(self):
        self.fx.config.update({CONFIG_ENGINE_ID: "voxtral_mlx", CONFIG_MODEL_ID: "voxtral-4bit"})
        section = self.fx.section(chip=CHIP_INTEL)
        self.assertEqual(section.selected_model_id, "whispercpp-turbo-q5")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = _Fixture(self._tmp.name)

    def test_selecting_an_installed_model_writes_config_and_notifies(self):
        self.fx.install(TURBO)
        section = self.fx.section()

        self.assertTrue(section.select("whispercpp-turbo"))

        self.assertEqual(self.fx.config[CONFIG_ENGINE_ID], "whispercpp")
        self.assertEqual(self.fx.config[CONFIG_MODEL_ID], "whispercpp-turbo")
        self.assertEqual(self.fx.engine_changes, [("whispercpp", "whispercpp-turbo")])
        self.assertEqual(self.fx.saves, 1)
        self.assertEqual(
            self.fx.saved_changes,
            [{CONFIG_ENGINE_ID: "whispercpp", CONFIG_MODEL_ID: "whispercpp-turbo"}],
        )

    def test_selecting_a_missing_model_only_moves_the_highlight(self):
        section = self.fx.section()

        self.assertFalse(section.select("voxtral-4bit"))

        self.assertEqual(section.selected_model_id, "voxtral-4bit")
        self.assertNotIn(CONFIG_MODEL_ID, self.fx.config)
        self.assertEqual(self.fx.engine_changes, [])
        self.assertTrue(section.can_download)
        self.assertFalse(section.can_delete)

    def test_selecting_by_index_matches_the_popup_order(self):
        self.fx.install(VOXTRAL)
        section = self.fx.section()
        self.assertTrue(section.select_index(2))
        self.assertEqual(self.fx.engine_changes, [("voxtral_mlx", "voxtral-4bit")])

    def test_selecting_an_unknown_model_is_a_programming_error(self):
        section = self.fx.section()
        with self.assertRaises(AssertionError):
            section.select("not-in-catalog")

    def test_reselecting_the_active_model_does_not_reload_the_engine(self):
        self.fx.install(TURBO_Q5)
        section = self.fx.section()
        section.select("whispercpp-turbo-q5")
        section.select("whispercpp-turbo-q5")
        self.assertEqual(len(self.fx.engine_changes), 1)

    def test_a_finished_download_installs_and_activates_the_model(self):
        section = self.fx.section()
        section.select("voxtral-4bit")
        self.assertFalse(section.selected_choice.installed)

        self.fx.install(VOXTRAL)
        section.on_download_finished("voxtral-4bit")

        self.assertTrue(section.selected_choice.installed)
        self.assertEqual(self.fx.config[CONFIG_MODEL_ID], "voxtral-4bit")
        self.assertEqual(self.fx.engine_changes, [("voxtral_mlx", "voxtral-4bit")])

    def test_a_refused_swap_writes_nothing_and_reverts_the_highlight(self):
        """Config must never claim an engine the app declined to load."""
        self.fx.install(TURBO_Q5)
        self.fx.install(TURBO)
        self.fx.config.update(
            {CONFIG_ENGINE_ID: "whispercpp", CONFIG_MODEL_ID: "whispercpp-turbo-q5"}
        )
        section = self.fx.section(refusal="Stop recording before switching.")

        self.assertFalse(section.select("whispercpp-turbo"))

        self.assertEqual(section.refusal, "Stop recording before switching.")
        self.assertEqual(section.selected_model_id, "whispercpp-turbo-q5")
        self.assertEqual(self.fx.config[CONFIG_MODEL_ID], "whispercpp-turbo-q5")
        self.assertEqual(self.fx.saves, 0)
        self.assertEqual(self.fx.saved_changes, [])

    def test_the_app_is_asked_before_anything_is_written(self):
        self.fx.install(TURBO)
        seen = []
        section = EngineSectionModel(
            self.fx.config,
            self.fx.store,
            chip=CHIP_APPLE_SILICON,
            ram_gb=VOXTRAL_MIN_RAM_GB,
            default_engine="whispercpp",
            on_engine_change=lambda engine, model: seen.append(
                ("asked", self.fx.config.get(CONFIG_MODEL_ID))
            ),
            save_changes=lambda changes: seen.append(("saved", changes)),
        )

        self.assertTrue(section.select("whispercpp-turbo"))

        self.assertEqual(seen[0], ("asked", None))  # nothing written yet
        self.assertEqual(seen[1][0], "saved")
        self.assertIsNone(section.refusal)

    def test_an_accepted_swap_clears_an_earlier_refusal(self):
        self.fx.install(TURBO)
        self.fx.install(VOXTRAL)
        refusals = ["busy", None]
        section = EngineSectionModel(
            self.fx.config,
            self.fx.store,
            chip=CHIP_APPLE_SILICON,
            ram_gb=VOXTRAL_MIN_RAM_GB,
            default_engine="whispercpp",
            on_engine_change=lambda engine, model: refusals.pop(0),
        )

        self.assertFalse(section.select("whispercpp-turbo"))
        self.assertEqual(section.refusal, "busy")

        self.assertTrue(section.select("whispercpp-turbo"))
        self.assertIsNone(section.refusal)


class DeleteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = _Fixture(self._tmp.name)

    def test_deleting_an_unused_model_removes_it_from_disk(self):
        self.fx.install(TURBO_Q5)
        self.fx.install(TURBO)
        self.fx.config.update(
            {CONFIG_ENGINE_ID: "whispercpp", CONFIG_MODEL_ID: "whispercpp-turbo-q5"}
        )
        section = self.fx.section()

        self.assertIsNone(section.delete("whispercpp-turbo"))

        self.assertFalse(self.fx.store.is_installed("whispercpp-turbo"))
        self.assertTrue(self.fx.store.is_installed("whispercpp-turbo-q5"))
        self.assertEqual(section.choices[1].installed, False)

    def test_deleting_the_model_in_use_is_refused_with_a_message(self):
        self.fx.install(TURBO_Q5)
        section = self.fx.section()
        section.select("whispercpp-turbo-q5")

        message = section.delete()

        self.assertIsNotNone(message)
        self.assertIn("Whisper large-v3-turbo (quantised)", message)
        self.assertIn("Murmur is using", message)
        self.assertTrue(self.fx.store.is_installed("whispercpp-turbo-q5"))

    def test_deleting_a_model_that_is_not_downloaded_is_refused(self):
        section = self.fx.section()
        section.select("whispercpp-turbo")
        message = section.delete()
        self.assertIsNotNone(message)
        self.assertIn("not downloaded", message)
        self.assertFalse(section.can_delete)

    def test_deleting_an_unknown_model_is_a_programming_error(self):
        section = self.fx.section()
        with self.assertRaises(AssertionError):
            section.delete("not-in-catalog")


class DetailLineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = _Fixture(self._tmp.name)

    def test_reads_as_a_size_a_licence_and_a_recommendation(self):
        section = self.fx.section(default_engine="whispercpp")
        self.assertEqual(
            section.detail_line,
            "574 MB download · MIT · Recommended for this Mac",
        )

    def test_an_installed_model_says_so(self):
        self.fx.install(VOXTRAL)
        section = self.fx.section()
        section.select("voxtral-4bit")
        self.assertEqual(section.detail_line, "3.1 GB on this Mac · Apache-2.0")


if __name__ == "__main__":
    unittest.main()
