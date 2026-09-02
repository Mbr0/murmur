"""Tests for Settings → General.

The tab is a rendering of :class:`~ui.settings.general_tab.GeneralTabModel`,
so everything worth asserting — which keys it owns, which ones it reports as
changed, and above all which ones it leaves alone — is asserted here without
AppKit.
"""

import unittest
from types import SimpleNamespace

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
from ui.settings.base import TabContext
from ui.settings.general_tab import (
    APPEARANCE_MODES,
    CONFIG_APPEARANCE,
    CONFIG_LANGUAGE,
    CONFIG_LAUNCH_AT_LOGIN,
    FALLBACK_LANGUAGES,
    LAUNCH_AT_LOGIN_NEEDS_APPROVAL,
    LAUNCH_AT_LOGIN_UNSUPPORTED,
    GeneralTab,
    GeneralTabModel,
    language_codes,
    needs_hotkey_reload,
    supports_launch_at_login,
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
        model = GeneralTabModel(dict(DEFAULT_CONFIG), launch_at_login_supported=True)
        model.set_binding(HotkeyBinding(keycode=8, command=True), label="C")
        model.set_hotkey_mode(HOTKEY_MODE_TOGGLE)
        model.set_language("de")
        model.set_appearance("light")
        model.set_launch_at_login(True)

        self.assertLessEqual(set(model.apply()), OWNED_KEYS)


class ChangedKeyTests(unittest.TestCase):
    def model(self, *, launch_at_login_supported=False, **overrides):
        return GeneralTabModel(
            {**DEFAULT_CONFIG, **overrides},
            launch_at_login_supported=launch_at_login_supported,
        )

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
        model = self.model(launch_at_login_supported=True)

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

    def test_set_engine_info_changes_the_choices_and_keeps_the_selection(self):
        model = GeneralTabModel(
            {**DEFAULT_CONFIG, CONFIG_LANGUAGE: "en"},
            engine_info=self.engine_info("en", "fr"),
        )
        self.assertEqual(model.language_choices, (LANGUAGE_AUTO, "en", "fr"))

        model.set_engine_info(self.engine_info("en", "de"))

        self.assertEqual(model.language_choices, (LANGUAGE_AUTO, "de", "en"))
        self.assertEqual(model.language_choices[model.language_index], "en")

    def test_set_engine_info_keeps_a_configured_language_the_new_engine_lacks(self):
        model = GeneralTabModel(
            {**DEFAULT_CONFIG, CONFIG_LANGUAGE: "ja"},
            engine_info=self.engine_info("en", "fr"),
        )

        model.set_engine_info(self.engine_info("en", "de"))

        self.assertIn("ja", model.language_choices)
        self.assertEqual(model.language_choices[model.language_index], "ja")


class LaunchAtLoginTests(unittest.TestCase):
    """The checkbox may only be offered where something implements it.

    Writing ``launch_at_login`` into the config when nothing registers a login
    item is a switch that lies: it stays on across restarts and Murmur still
    does not start itself.
    """

    def test_a_build_without_the_hook_cannot_be_switched_on(self):
        model = GeneralTabModel(dict(DEFAULT_CONFIG))

        self.assertFalse(model.launch_at_login_supported)
        with self.assertLogs("ui.settings.general_tab", level="INFO"):
            model.set_launch_at_login(True)

        self.assertFalse(model.launch_at_login)
        self.assertEqual(model.apply(), {})
        self.assertNotIn(CONFIG_LAUNCH_AT_LOGIN, model.as_config())

    def test_an_unsupported_build_never_writes_the_key_even_when_it_is_on_disk(self):
        model = GeneralTabModel({**DEFAULT_CONFIG, CONFIG_LAUNCH_AT_LOGIN: True})

        self.assertEqual(model.apply(), {})
        self.assertNotIn(CONFIG_LAUNCH_AT_LOGIN, model.as_config())

    def test_a_supported_build_owns_the_key(self):
        model = GeneralTabModel(dict(DEFAULT_CONFIG), launch_at_login_supported=True)

        self.assertTrue(model.launch_at_login_supported)
        model.set_launch_at_login(True)

        self.assertTrue(model.launch_at_login)
        self.assertEqual(model.apply(), {CONFIG_LAUNCH_AT_LOGIN: True})

    def test_the_hint_says_why_the_checkbox_is_dead(self):
        unsupported = GeneralTabModel(dict(DEFAULT_CONFIG))
        supported = GeneralTabModel(dict(DEFAULT_CONFIG), launch_at_login_supported=True)

        self.assertEqual(unsupported.launch_at_login_hint, LAUNCH_AT_LOGIN_UNSUPPORTED)
        self.assertIsNone(supported.launch_at_login_hint)

    def test_support_follows_the_running_app(self):
        class WithHook:
            def set_launch_at_login(self, enabled):
                return None

        self.assertFalse(supports_launch_at_login(None))
        self.assertFalse(supports_launch_at_login(object()))
        self.assertTrue(supports_launch_at_login(WithHook()))


class LaunchAtLoginSwitchTests(unittest.TestCase):
    """What the switch does, and what it says afterwards.

    macOS may accept a registration and still refuse to act on it until the
    user allows the login item in System Settings. The app reports the state it
    read back, and the tab shows that state — never the state that was asked
    for.
    """

    class FakeLabel:
        def __init__(self):
            self.text = None

        def setStringValue_(self, value):
            self.text = value

    class SwitchTab(GeneralTab):
        """The tab with its one AppKit call — setting the box — recorded."""

        def __init__(self):
            super().__init__()
            self.checkbox_states = []

        def _set_checkbox(self, on):
            self.checkbox_states.append(on)

    def setUp(self):
        self.saved = []

    def tab(self, app, *, supported=True, stored=False):
        tab = self.SwitchTab()
        config = dict(DEFAULT_CONFIG)
        config[CONFIG_LAUNCH_AT_LOGIN] = stored
        tab.context = TabContext(
            config=config, save=self.saved.append, app=app, theme=object()
        )
        tab.model = GeneralTabModel(config, launch_at_login_supported=supported)
        tab._launch_checkbox = object()
        tab._launch_hint = self.FakeLabel()
        return tab

    @staticmethod
    def app_returning(state):
        class App:
            def set_launch_at_login(self, enabled):
                return state

        return App()

    def test_a_registration_that_took_is_written_and_shown_as_on(self):
        tab = self.tab(self.app_returning(True))

        tab.set_launch_at_login(True)

        self.assertEqual(self.saved, [{CONFIG_LAUNCH_AT_LOGIN: True}])
        self.assertEqual(tab.checkbox_states, [True])
        self.assertEqual(tab._launch_hint.text, " ")

    def test_a_registration_awaiting_approval_puts_the_box_back_and_explains(self):
        tab = self.tab(self.app_returning(False))

        tab.set_launch_at_login(True)

        self.assertEqual(self.saved, [])  # nothing changed, so nothing written
        self.assertEqual(tab.checkbox_states, [False])
        self.assertEqual(tab._launch_hint.text, LAUNCH_AT_LOGIN_NEEDS_APPROVAL)
        self.assertFalse(tab.model.launch_at_login)

    def test_turning_it_off_writes_the_state_the_app_read_back(self):
        tab = self.tab(self.app_returning(False), stored=True)

        tab.set_launch_at_login(False)

        self.assertEqual(self.saved, [{CONFIG_LAUNCH_AT_LOGIN: False}])
        self.assertEqual(tab.checkbox_states, [False])
        self.assertEqual(tab._launch_hint.text, " ")

    def test_an_app_that_refuses_leaves_the_box_where_the_system_is(self):
        class Refusing:
            def set_launch_at_login(self, enabled):
                raise RuntimeError("ServiceManagement is not available in this build")

        tab = self.tab(Refusing())

        with self.assertLogs("ui.settings.general_tab", level="WARNING"):
            tab.set_launch_at_login(True)

        self.assertEqual(tab.checkbox_states, [False])
        self.assertEqual(tab._launch_hint.text, LAUNCH_AT_LOGIN_NEEDS_APPROVAL)
        self.assertEqual(self.saved, [])

    def test_an_app_that_answers_nothing_is_taken_at_its_word(self):
        # ``app_call`` returns None for a method the app does not have; the
        # checkbox is only offered where it does, so this is the honest read.
        tab = self.tab(SimpleNamespace())

        tab.set_launch_at_login(True)

        self.assertEqual(self.saved, [{CONFIG_LAUNCH_AT_LOGIN: True}])
        self.assertEqual(tab.checkbox_states, [True])

    def test_without_a_running_app_the_request_is_logged_not_applied(self):
        tab = self.tab(None)

        with self.assertLogs("ui.settings.general_tab", level="INFO"):
            tab.set_launch_at_login(True)

        self.assertEqual(self.saved, [{CONFIG_LAUNCH_AT_LOGIN: True}])

    def test_a_build_that_cannot_register_writes_nothing_at_all(self):
        tab = self.tab(self.app_returning(True), supported=False)

        tab.set_launch_at_login(True)

        self.assertEqual(self.saved, [])
        self.assertEqual(tab._launch_hint.text, LAUNCH_AT_LOGIN_UNSUPPORTED)


class CaptureTeardownTests(unittest.TestCase):
    """Closing Settings mid-capture must give the event monitor back.

    A local monitor left installed keeps swallowing key events for the life of
    the process, long after the window that asked for it is gone.
    """

    class RecordingTab(GeneralTab):
        """A tab whose only AppKit call — removing the monitor — is recorded."""

        def __init__(self):
            super().__init__()
            self.removed = []

        def _remove_monitor(self, monitor):
            self.removed.append(monitor)

    def test_closing_removes_a_live_monitor(self):
        tab = self.RecordingTab()
        monitor = object()
        tab._monitor = monitor
        tab._capture_modifiers = 7

        tab.close()

        self.assertEqual(tab.removed, [monitor])
        self.assertIsNone(tab._monitor)
        self.assertEqual(tab._capture_modifiers, 0)

    def test_closing_without_a_capture_touches_nothing(self):
        tab = self.RecordingTab()

        tab.close()

        self.assertEqual(tab.removed, [])

    def test_closing_twice_removes_the_monitor_once(self):
        tab = self.RecordingTab()
        tab._monitor = object()

        tab.close()
        tab.close()

        self.assertEqual(len(tab.removed), 1)


class AppearanceConstantTests(unittest.TestCase):
    def test_the_models_appearance_modes_match_the_themes(self):
        import ui_theme

        self.assertEqual(APPEARANCE_MODES, tuple(ui_theme.APPEARANCE_MODES))


class PermissionStatusTests(unittest.TestCase):
    """The Accessibility permission line ported from the old single-page window."""

    def test_permission_status_uses_the_real_message_by_default(self):
        from services.hotkey_service import permission_status_message

        model = GeneralTabModel(dict(DEFAULT_CONFIG))

        self.assertEqual(model.permission_status, permission_status_message())

    def test_permission_status_uses_the_injected_provider(self):
        model = GeneralTabModel(
            dict(DEFAULT_CONFIG), permission_status=lambda: "custom status line"
        )

        self.assertEqual(model.permission_status, "custom status line")


if __name__ == "__main__":
    unittest.main()
