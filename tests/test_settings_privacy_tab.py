"""Settings → Privacy: the model, the generated text, and the delete action.

Everything here is headless: ``ui.settings.privacy_tab`` imports no AppKit at
module scope, so the rules can be tested without a window server.
"""

import unittest

from services.persistence_service import (
    CLOUD_MODE_MURMUR,
    CLOUD_MODE_OWN_KEY,
    CONFIG_CLEANUP_CLOUD,
    CONFIG_CLEANUP_ENABLED,
    CONFIG_CLOUD_MODE,
    CONFIG_HISTORY_ENABLED,
    CONFIG_KEEP_AUDIO,
    CONFIG_PRIVACY_MODE,
    DEFAULT_CONFIG,
    DeletionSummary,
    what_leaves_the_mac,
)
from ui.settings import registered_tabs
from ui.settings.base import TAB_PRIVACY
from ui.settings.privacy_tab import (
    DEFAULT_AUDIO_DIR,
    PrivacyTab,
    PrivacyTabModel,
)


class FakePersistence:
    """Records the delete call and reports a summary."""

    def __init__(self, summary=None):
        self.calls = []
        self.summary = summary or DeletionSummary(
            history_entries=3, audio_files=1, config_keys=("vocabulary_terms",)
        )

    def delete_all_data(self, audio_dir, config=None, **kwargs):
        self.calls.append((audio_dir, config, kwargs))
        return self.summary


class PrivacyTabModelTests(unittest.TestCase):
    def test_reads_the_toggles_from_config(self):
        model = PrivacyTabModel(
            {**DEFAULT_CONFIG, CONFIG_HISTORY_ENABLED: True, CONFIG_KEEP_AUDIO: True}
        )

        self.assertTrue(model.history_enabled)
        self.assertTrue(model.keep_audio)

    def test_falls_back_to_the_privacy_first_defaults(self):
        model = PrivacyTabModel({})

        self.assertFalse(model.history_enabled)
        self.assertFalse(model.keep_audio)

    def test_lines_match_what_leaves_the_mac_for_the_stored_config(self):
        config = {**DEFAULT_CONFIG, CONFIG_CLOUD_MODE: CLOUD_MODE_MURMUR}
        model = PrivacyTabModel(config)

        self.assertEqual(model.lines, what_leaves_the_mac(config))

    def test_lines_follow_the_toggles_before_they_are_saved(self):
        model = PrivacyTabModel(dict(DEFAULT_CONFIG))
        self.assertIn("Recordings are deleted as soon as they are transcribed.", model.lines)

        model.set_keep_audio(True)

        self.assertIn("Recordings are kept on this Mac.", model.lines)
        self.assertNotIn(
            "Recordings are deleted as soon as they are transcribed.", model.lines
        )

    def test_lines_name_the_local_engine_from_the_context(self):
        model = PrivacyTabModel(dict(DEFAULT_CONFIG), engine_info={"name": "Voxtral Mini"})

        self.assertEqual(model.lines[1], "Transcription runs here, using Voxtral Mini.")

    def test_lines_report_the_own_key_provider(self):
        model = PrivacyTabModel(
            {
                **DEFAULT_CONFIG,
                CONFIG_CLOUD_MODE: CLOUD_MODE_OWN_KEY,
                "byok_provider": "openai",
            }
        )

        self.assertEqual(
            model.lines[0],
            "Audio is sent to OpenAI with your own API key, to be turned into text.",
        )

    def test_lines_report_cloud_cleanup(self):
        model = PrivacyTabModel(
            {
                **DEFAULT_CONFIG,
                CONFIG_CLOUD_MODE: CLOUD_MODE_MURMUR,
                CONFIG_CLEANUP_ENABLED: True,
                CONFIG_CLEANUP_CLOUD: True,
            }
        )

        self.assertTrue(any("to be cleaned up." in line for line in model.lines))

    def test_summary_text_is_one_bullet_per_line(self):
        model = PrivacyTabModel(dict(DEFAULT_CONFIG))

        rendered = model.summary_text.splitlines()

        self.assertEqual(len(rendered), len(model.lines))
        self.assertTrue(all(line.startswith("• ") for line in rendered))

    def test_apply_returns_only_what_changed(self):
        model = PrivacyTabModel(dict(DEFAULT_CONFIG))

        model.set_history_enabled(True)

        self.assertEqual(
            model.apply(),
            {CONFIG_HISTORY_ENABLED: True, CONFIG_PRIVACY_MODE: False},
        )

    def test_apply_is_empty_when_nothing_moved(self):
        model = PrivacyTabModel(dict(DEFAULT_CONFIG))

        self.assertEqual(model.apply(), {})

    def test_turning_everything_off_again_restores_privacy_mode(self):
        model = PrivacyTabModel(
            {**DEFAULT_CONFIG, CONFIG_HISTORY_ENABLED: True, CONFIG_PRIVACY_MODE: False}
        )

        model.set_history_enabled(False)

        self.assertEqual(
            model.apply(),
            {CONFIG_HISTORY_ENABLED: False, CONFIG_PRIVACY_MODE: True},
        )

    def test_privacy_mode_stays_off_while_audio_is_kept(self):
        model = PrivacyTabModel(
            {
                **DEFAULT_CONFIG,
                CONFIG_HISTORY_ENABLED: True,
                CONFIG_KEEP_AUDIO: True,
                CONFIG_PRIVACY_MODE: False,
            }
        )

        model.set_history_enabled(False)

        self.assertEqual(model.apply(), {CONFIG_HISTORY_ENABLED: False})

    def test_mark_saved_clears_the_pending_diff(self):
        model = PrivacyTabModel(dict(DEFAULT_CONFIG))
        model.set_keep_audio(True)
        changed = model.apply()

        model.mark_saved()

        self.assertTrue(changed)
        self.assertEqual(model.apply(), {})


class DeleteAllDataActionTests(unittest.TestCase):
    def test_delete_all_data_passes_the_live_config_and_audio_dir(self):
        config = {**DEFAULT_CONFIG, "vocabulary_terms": ["Murmur"]}
        model = PrivacyTabModel(config)
        persistence = FakePersistence()

        summary = model.delete_all_data(persistence, "/tmp/murmur-audio")

        self.assertEqual(persistence.calls[0][0], "/tmp/murmur-audio")
        self.assertIs(persistence.calls[0][1], config)
        self.assertEqual(summary.history_entries, 3)

    def test_delete_all_data_leaves_the_toggles_alone(self):
        model = PrivacyTabModel(
            {**DEFAULT_CONFIG, CONFIG_HISTORY_ENABLED: True, CONFIG_KEEP_AUDIO: True}
        )

        model.delete_all_data(FakePersistence(), DEFAULT_AUDIO_DIR)

        self.assertTrue(model.history_enabled)
        self.assertTrue(model.keep_audio)
        self.assertEqual(model.apply(), {})

    def test_default_audio_dir_matches_the_app(self):
        self.assertTrue(DEFAULT_AUDIO_DIR.endswith("/.murmur_audio"))


class PrivacyTabRegistrationTests(unittest.TestCase):
    def test_tab_identity(self):
        self.assertEqual(PrivacyTab.identifier, TAB_PRIVACY)
        self.assertEqual(PrivacyTab.title, "Privacy")

    def test_tab_registers_itself_on_import(self):
        self.assertIn(PrivacyTab, registered_tabs())

    def test_a_fresh_tab_has_nothing_built_yet(self):
        tab = PrivacyTab()

        self.assertIsNone(tab.context)
        self.assertIsNone(tab.model)
        # refresh before build must be a no-op, not a crash.
        tab.refresh()


if __name__ == "__main__":
    unittest.main()
