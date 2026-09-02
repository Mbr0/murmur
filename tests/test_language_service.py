import unittest

from engines import LANGUAGE_AUTO, EngineInfo

from services.language_service import (
    DEFAULT_LANGUAGE,
    available_languages,
    forget_language,
    language_display_name,
    remember_language,
    resolve_language,
)


def make_engine_info(languages: tuple[str, ...]) -> EngineInfo:
    return EngineInfo(
        id="fake",
        name="Fake",
        model_id="fake-model",
        size_bytes=1,
        languages=languages,
        supports_streaming=False,
        supports_hints=True,
        requires_apple_silicon=False,
    )


class ResolveLanguageTests(unittest.TestCase):
    def test_defaults_to_auto_when_config_empty(self):
        self.assertEqual(resolve_language({}, None), DEFAULT_LANGUAGE)
        self.assertEqual(DEFAULT_LANGUAGE, LANGUAGE_AUTO)

    def test_uses_configured_default_language(self):
        self.assertEqual(resolve_language({"language": "fr"}, None), "fr")

    def test_no_bundle_id_uses_default(self):
        config = {"language": "fr", "language_by_app": {"com.apple.Terminal": "en"}}
        self.assertEqual(resolve_language(config, None), "fr")

    def test_per_app_override_wins_over_default(self):
        config = {"language": "fr", "language_by_app": {"com.apple.Terminal": "en"}}
        self.assertEqual(resolve_language(config, "com.apple.Terminal"), "en")

    def test_unmatched_bundle_id_falls_back_to_default(self):
        config = {"language": "fr", "language_by_app": {"com.apple.Terminal": "en"}}
        self.assertEqual(resolve_language(config, "com.apple.Mail"), "fr")

    def test_missing_language_by_app_key_is_tolerated(self):
        self.assertEqual(resolve_language({"language": "nl"}, "com.apple.Mail"), "nl")


class RememberForgetLanguageTests(unittest.TestCase):
    def test_remember_language_adds_override_without_mutating_input(self):
        config = {"language": "auto", "language_by_app": {}}
        updated = remember_language(config, "com.apple.Terminal", "en")

        self.assertEqual(updated["language_by_app"], {"com.apple.Terminal": "en"})
        self.assertEqual(config["language_by_app"], {})

    def test_remember_language_overwrites_existing_override(self):
        config = {"language_by_app": {"com.apple.Terminal": "en"}}
        updated = remember_language(config, "com.apple.Terminal", "fr")
        self.assertEqual(updated["language_by_app"], {"com.apple.Terminal": "fr"})

    def test_remember_language_tolerates_missing_key(self):
        updated = remember_language({}, "com.apple.Terminal", "en")
        self.assertEqual(updated["language_by_app"], {"com.apple.Terminal": "en"})

    def test_forget_language_removes_override_without_mutating_input(self):
        config = {"language_by_app": {"com.apple.Terminal": "en", "com.apple.Mail": "fr"}}
        updated = forget_language(config, "com.apple.Terminal")

        self.assertEqual(updated["language_by_app"], {"com.apple.Mail": "fr"})
        self.assertEqual(
            config["language_by_app"],
            {"com.apple.Terminal": "en", "com.apple.Mail": "fr"},
        )

    def test_forget_language_is_a_noop_when_absent(self):
        config = {"language_by_app": {"com.apple.Mail": "fr"}}
        updated = forget_language(config, "com.apple.Terminal")
        self.assertEqual(updated["language_by_app"], {"com.apple.Mail": "fr"})


class AvailableLanguagesTests(unittest.TestCase):
    def test_auto_comes_first(self):
        info = make_engine_info((LANGUAGE_AUTO, "fr", "de", "en"))
        result = available_languages(info)
        self.assertEqual(result[0], LANGUAGE_AUTO)

    def test_dedups_and_sorts_remaining_codes(self):
        info = make_engine_info((LANGUAGE_AUTO, "fr", "de", "en", "fr"))
        self.assertEqual(available_languages(info), (LANGUAGE_AUTO, "de", "en", "fr"))

    def test_works_when_engine_advertises_only_auto(self):
        info = make_engine_info((LANGUAGE_AUTO,))
        self.assertEqual(available_languages(info), (LANGUAGE_AUTO,))

    def test_adds_auto_even_if_engine_omits_it(self):
        info = make_engine_info(("en", "fr"))
        self.assertEqual(available_languages(info), (LANGUAGE_AUTO, "en", "fr"))


class LanguageDisplayNameTests(unittest.TestCase):
    def test_known_codes(self):
        self.assertEqual(language_display_name("en"), "English")
        self.assertEqual(language_display_name("fr"), "Français")
        self.assertEqual(language_display_name("nl"), "Nederlands")
        self.assertEqual(language_display_name("de"), "Deutsch")
        self.assertEqual(language_display_name("es"), "Español")
        self.assertEqual(language_display_name("it"), "Italiano")
        self.assertEqual(language_display_name("pt"), "Português")

    def test_unknown_code_falls_back_to_upper_cased(self):
        self.assertEqual(language_display_name("ja"), "JA")
        self.assertEqual(language_display_name("zh"), "ZH")

    def test_auto_has_a_friendly_name(self):
        self.assertEqual(language_display_name(LANGUAGE_AUTO), "Auto")


if __name__ == "__main__":
    unittest.main()
