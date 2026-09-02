"""Tests for Settings → General.

The tab is a rendering of :class:`~ui.settings.general_tab.GeneralTabModel`,
so everything worth asserting — which keys it owns, which ones it reports as
changed, and above all which ones it leaves alone — is asserted here without
AppKit.
"""

import unittest

from engines import LANGUAGE_AUTO
from engines.base import EngineInfo
from services.hotkey_service import (
    DEFAULT_HOTKEY,
    HOTKEY_MODE_CONFIG_KEY,
    HOTKEY_MODE_HOLD,
    HOTKEY_MODE_TOGGLE,
    HotkeyBinding,
)
from services.persistence_service import DEFAULT_CONFIG
from ui.settings.general_tab import (
    APPEARANCE_MODES,
    CONFIG_APPEARANCE,
    CONFIG_LANGUAGE,
    CONFIG_LAUNCH_AT_LOGIN,
    FALLBACK_LANGUAGES,
    GeneralTabModel,
    language_codes,
    needs_hotkey_reload,
)

#: The config keys Murmur shipped before Wave 3. The redesign may add keys;
#: it may not quietly rewrite one of these.
TWELVE_EXISTING_KEYS = (
    "save_audio",
    "save_history",
    "privacy_mode",
    "appearance_mode",
    "hotkey_keycode",
    "hotkey_command",
    "hotkey_option",
    "hotkey_control",
    "hotkey_shift",
    "hotkey_fn",
    "mic_device_index",
    "mic_device_name",
)

#: Every key the General tab is allowed to write.
OWNED_KEYS = frozenset(
    {
        "hotkey_keycode",
        "hotkey_command",
        "hotkey_option",
        "hotkey_control",
        "hotkey_shift",
        "hotkey_fn",
        "hotkey_label",
        HOTKEY_MODE_CONFIG_KEY,
        CONFIG_LANGUAGE,
        CONFIG_APPEARANCE,
        CONFIG_LAUNCH_AT_LOGIN,
    }
)


class ConfigRoundTripTests(unittest.TestCase):
    """Wave 3's "Done when": all twelve existing config keys still round-trip."""

    def test_the_twelve_existing_keys_survive_the_model_untouched(self):
        config = dict(DEFAULT_CONFIG)
        before = {key: config[key] for key in TWELVE_EXISTING_KEYS}

        model = GeneralTabModel(config)
        merged = {**config, **model.apply()}

        for key in TWELVE_EXISTING_KEYS:
            self.assertEqual(merged[key], before[key], key)

    def test_opening_and_closing_general_writes_nothing(self):
        config = dict(DEFAULT_CONFIG)

        self.assertEqual(GeneralTabModel(config).apply(), {})

    def test_a_populated_config_also_round_trips(self):
        config = {
            **DEFAULT_CONFIG,
            "hotkey_keycode": 8,
            "hotkey_command": True,
            "hotkey_option": False,
            "hotkey_label": "C",
            HOTKEY_MODE_CONFIG_KEY: HOTKEY_MODE_HOLD,
            CONFIG_LANGUAGE: "fr",
            CONFIG_APPEARANCE: "dark",
            CONFIG_LAUNCH_AT_LOGIN: True,
        }

        model = GeneralTabModel(config)

        self.assertEqual(model.apply(), {})
        self.assertEqual(model.as_config()[HOTKEY_MODE_CONFIG_KEY], HOTKEY_MODE_HOLD)

    def test_the_model_never_mutates_the_config_it_was_built_from(self):
        config = dict(DEFAULT_CONFIG)
        snapshot = dict(config)

        model = GeneralTabModel(config)
        model.set_language("nl")
        model.set_appearance("dark")
        model.apply()

        self.assertEqual(config, snapshot)

    def test_apply_never_reports_a_key_the_tab_does_not_own(self):
        model = GeneralTabModel(dict(DEFAULT_CONFIG))
        model.set_binding(HotkeyBinding(keycode=8, command=True), label="C")
        model.set_hotkey_mode(HOTKEY_MODE_TOGGLE)
        model.set_language("de")
        model.set_appearance("light")
        model.set_launch_at_login(True)

        self.assertLessEqual(set(model.apply()), OWNED_KEYS)


class ChangedKeyTests(unittest.TestCase):
    def model(self, **overrides):
        return GeneralTabModel({**DEFAULT_CONFIG, **overrides})

    def test_changing_the_language_reports_only_the_language(self):
        model = self.model()

        model.set_language("fr")

        self.assertEqual(model.apply(), {CONFIG_LANGUAGE: "fr"})

    def test_changing_the_appearance_reports_only_the_appearance(self):
        model = self.model()

        model.set_appearance("dark")

        self.assertEqual(model.apply(), {CONFIG_APPEARANCE: "dark"})

    def test_changing_the_shortcut_behaviour_reports_only_the_mode(self):
        model = self.model()

        model.set_hotkey_mode(HOTKEY_MODE_HOLD)

        self.assertEqual(model.apply(), {HOTKEY_MODE_CONFIG_KEY: HOTKEY_MODE_HOLD})

    def test_turning_launch_at_login_on_reports_only_that_key(self):
        model = self.model()

        model.set_launch_at_login(True)

        self.assertEqual(model.apply(), {CONFIG_LAUNCH_AT_LOGIN: True})

    def test_recording_a_shortcut_reports_the_binding_and_its_label(self):
        model = self.model()

        model.set_binding(HotkeyBinding(keycode=8, command=True, shift=True), label="C")

        self.assertEqual(
            model.apply(),
            {
                "hotkey_keycode": 8,
                "hotkey_command": True,
                "hotkey_option": False,
                "hotkey_shift": True,
                "hotkey_label": "C",
            },
        )

    def test_resetting_the_shortcut_from_a_custom_one_restores_the_default(self):
        model = self.model(hotkey_keycode=8, hotkey_command=True, hotkey_option=False)

        model.reset_shortcut()

        changed = model.apply()
        self.assertEqual(changed["hotkey_keycode"], DEFAULT_HOTKEY.keycode)
        self.assertTrue(changed["hotkey_option"])
        self.assertEqual(changed["hotkey_label"], "Space")

    def test_resetting_an_already_default_shortcut_changes_nothing(self):
        model = self.model(hotkey_label="Space")

        model.reset_shortcut()

        self.assertEqual(model.apply(), {})

    def test_marking_saved_makes_the_next_apply_empty(self):
        model = self.model()
        model.set_language("it")
        model.apply()

        model.mark_saved()

        self.assertEqual(model.apply(), {})

    def test_setting_a_value_back_to_where_it_started_reports_nothing(self):
        model = self.model(language="fr")

        model.set_language("de")
        model.set_language("fr")

        self.assertEqual(model.apply(), {})

    def test_an_unknown_appearance_or_mode_is_refused(self):
        model = self.model()

        with self.assertRaises(AssertionError):
            model.set_appearance("sepia")
        with self.assertRaises(AssertionError):
            model.set_hotkey_mode("push")

    def test_the_shortcut_label_is_what_the_recorder_button_shows(self):
        self.assertEqual(self.model(hotkey_label="Space").shortcut_label, "⌥ Space")


class HotkeyReloadTests(unittest.TestCase):
    def test_a_shortcut_or_mode_change_asks_the_app_to_re_register(self):
        self.assertTrue(needs_hotkey_reload({"hotkey_keycode": 8}))
        self.assertTrue(needs_hotkey_reload({"hotkey_label": "C"}))
        self.assertTrue(needs_hotkey_reload({HOTKEY_MODE_CONFIG_KEY: HOTKEY_MODE_HOLD}))

    def test_a_language_or_appearance_change_does_not(self):
        self.assertFalse(needs_hotkey_reload({CONFIG_LANGUAGE: "fr"}))
        self.assertFalse(needs_hotkey_reload({CONFIG_APPEARANCE: "dark"}))
        self.assertFalse(needs_hotkey_reload({}))


class LanguageChoiceTests(unittest.TestCase):
    def engine_info(self, *languages):
        return EngineInfo(
            id="whispercpp",
            name="Whisper large-v3-turbo",
            model_id="whispercpp-turbo",
            size_bytes=1,
            languages=tuple(languages),
            supports_streaming=False,
            supports_hints=True,
            requires_apple_silicon=False,
        )

    def test_without_an_engine_the_known_set_is_offered(self):
        self.assertEqual(language_codes(None), FALLBACK_LANGUAGES)

    def test_the_engines_own_languages_are_offered_when_one_is_loaded(self):
        codes = language_codes(self.engine_info("fr", "en"))

        self.assertEqual(codes, (LANGUAGE_AUTO, "en", "fr"))

    def test_an_engine_that_cannot_answer_falls_back_and_says_so(self):
        class Broken:
            @property
            def languages(self):
                raise RuntimeError("engine is gone")

        with self.assertLogs("ui.settings.general_tab", level="WARNING"):
            self.assertEqual(language_codes(Broken()), FALLBACK_LANGUAGES)

    def test_a_stored_language_the_engine_does_not_claim_still_shows(self):
        model = GeneralTabModel(
            {**DEFAULT_CONFIG, CONFIG_LANGUAGE: "ja"},
            engine_info=self.engine_info("en", "fr"),
        )

        self.assertIn("ja", model.language_choices)
        self.assertEqual(model.language_choices[model.language_index], "ja")
        self.assertEqual(model.apply(), {})

    def test_automatic_is_the_first_title_and_the_default_selection(self):
        model = GeneralTabModel(dict(DEFAULT_CONFIG))

        self.assertEqual(model.language_titles[0], "Automatic")
        self.assertEqual(model.language_index, 0)
        self.assertEqual(len(model.language_titles), len(model.language_choices))


class AppearanceConstantTests(unittest.TestCase):
    def test_the_models_appearance_modes_match_the_themes(self):
        import ui_theme

        self.assertEqual(APPEARANCE_MODES, tuple(ui_theme.APPEARANCE_MODES))


if __name__ == "__main__":
    unittest.main()
