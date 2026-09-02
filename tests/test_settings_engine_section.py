"""Tests for the Settings "Speech engine" section state.

A real :class:`~engines.model_store.ModelStore` over a temporary root backs
every test, so "installed" means files actually on disk. No AppKit, no network.
"""

import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cleanup.vocabulary import VocabularyError
from engines.model_store import ModelFile, ModelSpec, ModelStore
from services.model_profile_service import CHIP_APPLE_SILICON, CHIP_INTEL, VOXTRAL_MIN_RAM_GB
from engines.base import LANGUAGE_AUTO, WHISPER_LANGUAGES
from services.language_service import available_languages
from settings_window import CONFIG_LANGUAGE, LanguageSectionModel, VocabularySectionModel
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




class VocabularySectionModelTests(unittest.TestCase):
    """The Settings "Vocabulary" group's editing rules, without AppKit."""

    CONFIG = {
        "vocabulary_terms": ["Boske", "Murmur"],
        "vocabulary_replacements": [
            {"from": "canopy", "to": "Canopy Studio", "match_case": False}
        ],
    }

    def test_loads_terms_and_replacements_from_config(self):
        model = VocabularySectionModel(self.CONFIG)
        self.assertEqual(model.terms_text, "Boske\nMurmur")
        self.assertEqual(model.row_count, 1)
        self.assertEqual(model.value_for(0, "from"), "canopy")
        self.assertEqual(model.value_for(0, "to"), "Canopy Studio")
        self.assertIs(model.value_for(0, "match_case"), False)

    def test_empty_config_is_an_empty_section(self):
        model = VocabularySectionModel({})
        self.assertEqual(model.terms_text, "")
        self.assertEqual(model.row_count, 0)

    def test_terms_box_drops_blanks_and_repeats_and_trims(self):
        model = VocabularySectionModel({})
        model.set_terms_text("  Boske \n\nMurmur\nBoske\n   \n")
        self.assertEqual(model.terms, ["Boske", "Murmur"])

    def test_added_row_starts_blank_and_is_returned_for_editing(self):
        model = VocabularySectionModel(self.CONFIG)
        row = model.add_replacement()
        self.assertEqual(row, 1)
        self.assertEqual(model.value_for(row, "from"), "")
        self.assertIs(model.value_for(row, "match_case"), False)

    def test_editing_a_cell_writes_it_back(self):
        model = VocabularySectionModel(self.CONFIG)
        model.set_value(0, "from", "boske")
        model.set_value(0, "to", "Boske")
        model.set_value(0, "match_case", True)
        self.assertEqual(model.value_for(0, "from"), "boske")
        self.assertEqual(model.value_for(0, "to"), "Boske")
        self.assertIs(model.value_for(0, "match_case"), True)

    def test_editing_rejects_the_wrong_type_rather_than_coercing(self):
        model = VocabularySectionModel(self.CONFIG)
        with self.assertRaises(AssertionError):
            model.set_value(0, "match_case", 1)
        with self.assertRaises(AssertionError):
            model.set_value(0, "from", 7)

    def test_unknown_column_and_row_fail_fast(self):
        model = VocabularySectionModel(self.CONFIG)
        with self.assertRaises(AssertionError):
            model.value_for(0, "nope")
        with self.assertRaises(AssertionError):
            model.value_for(5, "from")

    def test_removing_selects_the_neighbour_and_minus_one_when_empty(self):
        model = VocabularySectionModel({})
        model.add_replacement()
        model.add_replacement()
        model.set_value(0, "from", "a")
        model.set_value(1, "from", "b")
        self.assertEqual(model.remove_replacement(1), 0)
        self.assertEqual(model.value_for(0, "from"), "a")
        self.assertEqual(model.remove_replacement(0), -1)
        self.assertEqual(model.row_count, 0)

    def test_blank_rows_are_never_saved(self):
        model = VocabularySectionModel(self.CONFIG)
        model.add_replacement()
        saved = model.to_config()
        self.assertEqual(len(saved["vocabulary_replacements"]), 1)
        self.assertEqual(saved["vocabulary_terms"], ["Boske", "Murmur"])

    def test_config_round_trips_through_the_section(self):
        model = VocabularySectionModel(self.CONFIG)
        self.assertEqual(
            VocabularySectionModel(model.to_config()).to_config(), model.to_config()
        )

    def test_csv_round_trips(self):
        model = VocabularySectionModel(self.CONFIG)
        exported = model.export_text()
        empty = VocabularySectionModel({})
        empty.import_text(exported)
        self.assertEqual(empty.to_config(), model.to_config())

    def test_a_bad_csv_names_the_line_and_leaves_the_section_alone(self):
        model = VocabularySectionModel(self.CONFIG)
        with self.assertRaises(VocabularyError) as caught:
            model.import_text("kind,from,to,match_case\nterm,Boske,,\nnope,x,y,false\n")
        self.assertIn("Line 3", str(caught.exception))
        self.assertEqual(model.terms, ["Boske", "Murmur"])
        self.assertEqual(model.row_count, 1)


class LanguageSectionModelTests(unittest.TestCase):
    """The Language popup's rows, and the one key it is allowed to write."""

    WHISPER_CODES = available_languages(
        types.SimpleNamespace(languages=WHISPER_LANGUAGES)
    )

    def test_an_untouched_popup_writes_nothing(self):
        model = LanguageSectionModel({CONFIG_LANGUAGE: "fr"}, self.WHISPER_CODES)
        self.assertEqual(model.changes_for_index(model.selected_index), {})

    def test_saving_a_new_choice_writes_the_code(self):
        model = LanguageSectionModel({CONFIG_LANGUAGE: "fr"}, self.WHISPER_CODES)
        index = model.codes.index("nl")
        self.assertEqual(model.changes_for_index(index), {CONFIG_LANGUAGE: "nl"})

    def test_a_one_row_engine_cannot_overwrite_the_users_language(self):
        """whisper.cpp reported only "auto"; every Save then wrote language=auto."""
        model = LanguageSectionModel({CONFIG_LANGUAGE: "fr"}, (LANGUAGE_AUTO,))

        self.assertIn("fr", model.codes)
        self.assertEqual(model.selected_index, model.codes.index("fr"))
        self.assertEqual(model.changes_for_index(model.selected_index), {})

    def test_a_missing_language_gets_its_own_row_and_a_title(self):
        model = LanguageSectionModel({CONFIG_LANGUAGE: "sv"}, (LANGUAGE_AUTO, "en"))
        self.assertEqual(model.codes, (LANGUAGE_AUTO, "en", "sv"))
        self.assertEqual(model.titles, ("Automatic", "English", "SV"))

    def test_auto_is_the_default_when_config_says_nothing(self):
        model = LanguageSectionModel({}, self.WHISPER_CODES)
        self.assertEqual(model.initial, LANGUAGE_AUTO)
        self.assertEqual(model.code_at(model.selected_index), LANGUAGE_AUTO)

    def test_switching_engines_relists_languages_and_keeps_the_choice(self):
        """The popup used to keep the old engine's list, and save by index."""
        model = LanguageSectionModel({CONFIG_LANGUAGE: "nl"}, self.WHISPER_CODES)
        voxtral_codes = (LANGUAGE_AUTO, "en", "es", "fr", "nl", "pt")

        rebuilt = model.rebuilt(voxtral_codes, model.selected_index)

        self.assertEqual(rebuilt.codes, voxtral_codes)
        self.assertEqual(rebuilt.code_at(rebuilt.selected_index), "nl")
        self.assertEqual(rebuilt.changes_for_index(rebuilt.selected_index), {})

    def test_a_language_the_new_engine_lacks_survives_the_switch(self):
        model = LanguageSectionModel({CONFIG_LANGUAGE: "ja"}, self.WHISPER_CODES)

        rebuilt = model.rebuilt((LANGUAGE_AUTO, "en", "fr"), model.selected_index)

        self.assertIn("ja", rebuilt.codes)
        self.assertEqual(rebuilt.code_at(rebuilt.selected_index), "ja")

    def test_an_out_of_range_row_is_a_programming_error(self):
        model = LanguageSectionModel({}, (LANGUAGE_AUTO, "en"))
        with self.assertRaises(AssertionError):
            model.code_at(2)


if __name__ == "__main__":
    unittest.main()
