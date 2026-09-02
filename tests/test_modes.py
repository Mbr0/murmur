import os
import unittest
from pathlib import Path

from cleanup.modes import (
    MODE_IDS,
    MODES,
    TONE_IDS,
    TONES,
    Mode,
    PromptFileMissingError,
    Tone,
    UnknownModeError,
    UnknownToneError,
    default_tone_for,
    load_prompt,
    mode_from_config,
    render_system_prompt,
    tone_from_config,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "modes"
FIXED_LANGUAGE = "French"
FIXED_VOCABULARY = ("Canopy Studio", "Murmur", "Voxtral")
UPDATE_GOLDEN = os.environ.get("UPDATE_GOLDEN") == "1"


class GoldenPromptTests(unittest.TestCase):
    """One golden file per (mode, tone) pair: 5 modes x 4 tones = 20 combos."""

    def test_golden_prompts_for_every_mode_tone_pair(self) -> None:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for mode_id in MODE_IDS:
            for tone_id in TONE_IDS:
                with self.subTest(mode=mode_id, tone=tone_id):
                    rendered = render_system_prompt(
                        mode_id, tone_id, FIXED_LANGUAGE, FIXED_VOCABULARY
                    )
                    golden_path = GOLDEN_DIR / f"{mode_id}-{tone_id}.txt"
                    if UPDATE_GOLDEN or not golden_path.exists():
                        golden_path.write_text(rendered, encoding="utf-8")
                        continue
                    expected = golden_path.read_text(encoding="utf-8")
                    self.assertEqual(rendered, expected)


class ModeRegistryTests(unittest.TestCase):
    def test_registries_cover_every_declared_id(self) -> None:
        self.assertEqual(set(MODES.keys()), set(MODE_IDS))
        self.assertEqual(set(TONES.keys()), set(TONE_IDS))

    def test_dictation_is_the_only_passthrough_mode(self) -> None:
        self.assertTrue(MODES["dictation"].is_passthrough)
        for mode_id in MODE_IDS:
            if mode_id == "dictation":
                continue
            with self.subTest(mode=mode_id):
                self.assertFalse(MODES[mode_id].is_passthrough)

    def test_modes_and_tones_carry_ui_copy_as_data(self) -> None:
        for mode in MODES.values():
            with self.subTest(mode=mode.id):
                self.assertIsInstance(mode, Mode)
                self.assertTrue(mode.display_name.strip())
                self.assertTrue(mode.description.strip())
        for tone in TONES.values():
            with self.subTest(tone=tone.id):
                self.assertIsInstance(tone, Tone)
                self.assertTrue(tone.display_name.strip())
                self.assertTrue(tone.description.strip())
                self.assertTrue(tone.instruction.strip())


class PromptFileTests(unittest.TestCase):
    def test_prompt_files_exist_for_every_mode(self) -> None:
        for mode_id in MODE_IDS:
            with self.subTest(mode=mode_id):
                template = load_prompt(mode_id)
                self.assertTrue(template.strip())
                self.assertIn("{tone_instruction}", template)
                self.assertIn("{language}", template)
                self.assertIn("{vocabulary}", template)

    def test_load_prompt_is_cached(self) -> None:
        first = load_prompt("message")
        second = load_prompt(MODES["message"])
        self.assertEqual(first, second)

    def test_load_prompt_missing_file_raises_loudly(self) -> None:
        with self.assertRaises(PromptFileMissingError):
            load_prompt("does-not-exist")


class RenderSystemPromptTests(unittest.TestCase):
    def test_placeholders_all_substituted(self) -> None:
        for mode_id in MODE_IDS:
            for tone_id in TONE_IDS:
                with self.subTest(mode=mode_id, tone=tone_id):
                    rendered = render_system_prompt(mode_id, tone_id, "English", ("Term",))
                    self.assertNotIn("{", rendered)
                    self.assertNotIn("}", rendered)

    def test_vocabulary_empty_renders_cleanly(self) -> None:
        for mode_id in MODE_IDS:
            with self.subTest(mode=mode_id):
                rendered = render_system_prompt(mode_id, "neutral", "English", ())
                self.assertNotIn("{", rendered)
                self.assertNotIn("None", rendered)
                self.assertTrue(rendered.strip())

    def test_language_none_renders_cleanly(self) -> None:
        rendered = render_system_prompt("message", "neutral", None, ())
        self.assertNotIn("{", rendered)
        self.assertNotIn("None", rendered)

    def test_language_none_empty_and_auto_all_mean_same_as_dictation(self) -> None:
        for missing_language in (None, "", "auto"):
            with self.subTest(language=missing_language):
                rendered = render_system_prompt(
                    "message", "neutral", missing_language, ()
                )
                self.assertIn("the same language as the dictation", rendered)

    def test_language_explicit_value_is_used_verbatim(self) -> None:
        rendered = render_system_prompt("message", "neutral", "fr", ())
        self.assertIn("fr", rendered)
        self.assertNotIn("the same language as the dictation", rendered)

    def test_code_mode_appends_llm_hint_as_final_paragraph(self) -> None:
        from cleanup.coding_mode import code_mode_llm_hint

        rendered = render_system_prompt("code", "neutral", "English", ())
        self.assertTrue(rendered.rstrip("\n").endswith(code_mode_llm_hint()))

    def test_non_code_modes_do_not_get_the_code_hint(self) -> None:
        from cleanup.coding_mode import code_mode_llm_hint

        for mode_id in MODE_IDS:
            if mode_id == "code":
                continue
            with self.subTest(mode=mode_id):
                rendered = render_system_prompt(mode_id, "neutral", "English", ())
                self.assertNotIn(code_mode_llm_hint(), rendered)

    def test_vocabulary_terms_appear_verbatim_when_present(self) -> None:
        rendered = render_system_prompt("notes", "neutral", "English", ("Canopy Studio",))
        self.assertIn("Canopy Studio", rendered)

    def test_accepts_mode_and_tone_objects_as_well_as_ids(self) -> None:
        by_id = render_system_prompt("mail", "formal", "English", ())
        by_object = render_system_prompt(MODES["mail"], TONES["formal"], "English", ())
        self.assertEqual(by_id, by_object)

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(UnknownModeError):
            render_system_prompt("bogus-mode", "neutral", "English", ())

    def test_unknown_tone_raises(self) -> None:
        with self.assertRaises(UnknownToneError):
            render_system_prompt("dictation", "bogus-tone", "English", ())


class ConfigResolutionTests(unittest.TestCase):
    def test_mode_from_config_defaults_to_dictation(self) -> None:
        self.assertEqual(mode_from_config({}).id, "dictation")

    def test_tone_from_config_defaults_to_neutral(self) -> None:
        self.assertEqual(tone_from_config({}).id, "neutral")

    def test_mode_from_config_reads_configured_value(self) -> None:
        self.assertEqual(mode_from_config({"cleanup_mode": "mail"}).id, "mail")

    def test_tone_from_config_reads_configured_value(self) -> None:
        self.assertEqual(tone_from_config({"cleanup_tone": "terse"}).id, "terse")

    def test_mode_from_config_raises_on_unknown_value(self) -> None:
        with self.assertRaises(UnknownModeError):
            mode_from_config({"cleanup_mode": "bogus-mode"})

    def test_tone_from_config_raises_on_unknown_value(self) -> None:
        with self.assertRaises(UnknownToneError):
            tone_from_config({"cleanup_tone": "bogus-tone"})


class DefaultToneTests(unittest.TestCase):
    def test_default_tone_for_returns_a_tone_object(self) -> None:
        for mode_id in MODE_IDS:
            with self.subTest(mode=mode_id):
                tone = default_tone_for(mode_id)
                self.assertIsInstance(tone, Tone)

    def test_default_tone_for_accepts_mode_object(self) -> None:
        self.assertEqual(default_tone_for(MODES["dictation"]).id, default_tone_for("dictation").id)


if __name__ == "__main__":
    unittest.main()
