"""Settings → Smart: cleanup, modes, context, vocabulary and snippets.

Everything here is headless: ``ui.settings.smart_tab`` imports no AppKit at
module scope, so every rule the tab enforces is tested without a window server.
"""

import unittest

from cleanup.modes import CONFIG_MODE_KEY, CONFIG_TONE_KEY
from cleanup.context import MODE_OVERRIDES_CONFIG_KEY
from cleanup.snippets import CONFIG_SNIPPETS_KEY, FREE_SNIPPET_LIMIT, Snippet
from cleanup.vocabulary import FREE_TERM_LIMIT, VocabularyError
from services.persistence_service import (
    CLOUD_MODE_MURMUR,
    CONFIG_CLEANUP_CLOUD,
    CONFIG_CLEANUP_ENABLED,
    CONFIG_CLOUD_MODE,
    DEFAULT_CONFIG,
)
from ui.settings import registered_tabs
from ui.settings.base import TAB_SMART, SettingsTab
from ui.settings.smart_tab import (
    CLEANUP_DEFAULT_HINT,
    CONFIG_CONTEXT_AWARENESS,
    CONFIG_INCLUDE_SELECTION,
    FEATURE_CLEANUP,
    FEATURE_SNIPPETS,
    FEATURE_VOCABULARY_BEYOND_FREE,
    OWNED_CONFIG_KEYS,
    PRO_CLEANUP_HINT,
    SNIPPETS_LIMIT_HINT,
    VOCABULARY_LIMIT_HINT,
    OverridesSectionModel,
    SmartTab,
    SmartTabModel,
    built_in_mode_for,
    spoken_symbol_rows,
)

ALL_FEATURES = (FEATURE_CLEANUP, FEATURE_VOCABULARY_BEYOND_FREE, FEATURE_SNIPPETS)


def make_model(config=None, *, allowed=ALL_FEATURES, machine_default=False):
    """A model over ``DEFAULT_CONFIG`` plus ``config``, with a fake Pro gate."""
    base = dict(DEFAULT_CONFIG)
    base.update(config or {})
    return SmartTabModel(
        base,
        pro_gate=lambda feature: feature in allowed,
        cleanup_probe=lambda: machine_default,
    )


class ProGateTests(unittest.TestCase):
    """The tab asks ``is_pro_feature_enabled(feature)`` and nothing else."""

    def test_no_gate_means_no_pro_feature(self):
        model = SmartTabModel(dict(DEFAULT_CONFIG), pro_gate=None, cleanup_probe=lambda: True)

        self.assertFalse(model.cleanup_entitled)
        self.assertEqual(model.cleanup_pro_hint, PRO_CLEANUP_HINT)

    def test_a_gate_that_raises_is_a_closed_gate(self):
        def broken(feature):
            raise RuntimeError("no licence service")

        model = SmartTabModel(
            dict(DEFAULT_CONFIG), pro_gate=broken, cleanup_probe=lambda: True
        )

        self.assertFalse(model.cleanup_entitled)

    def test_the_gate_is_asked_by_feature_name(self):
        asked = []

        def gate(feature):
            asked.append(feature)
            return True

        model = SmartTabModel(dict(DEFAULT_CONFIG), pro_gate=gate, cleanup_probe=lambda: True)
        model.cleanup_entitled

        self.assertEqual(asked, [FEATURE_CLEANUP])

    def test_the_gating_table(self):
        """One row per plan: what is locked, and what the hints say."""
        terms = [f"term{index}" for index in range(FREE_TERM_LIMIT + 1)]
        snippets = [
            {"trigger": f"t{index}", "text": "x"} for index in range(FREE_SNIPPET_LIMIT + 1)
        ]
        config = {"vocabulary_terms": terms, CONFIG_SNIPPETS_KEY: snippets}

        cases = (
            ((), False, VOCABULARY_LIMIT_HINT, SNIPPETS_LIMIT_HINT),
            (ALL_FEATURES, True, None, None),
            ((FEATURE_CLEANUP,), True, VOCABULARY_LIMIT_HINT, SNIPPETS_LIMIT_HINT),
            ((FEATURE_VOCABULARY_BEYOND_FREE,), False, None, SNIPPETS_LIMIT_HINT),
            ((FEATURE_SNIPPETS,), False, VOCABULARY_LIMIT_HINT, None),
        )
        for allowed, cleanup, vocabulary_hint, snippet_hint in cases:
            with self.subTest(allowed=allowed):
                model = make_model(config, allowed=allowed)

                self.assertEqual(model.cleanup_entitled, cleanup)
                self.assertEqual(model.vocabulary_limit_hint, vocabulary_hint)
                self.assertEqual(model.snippets_limit_hint, snippet_hint)

    def test_locked_cleanup_controls_refuse_to_move(self):
        model = make_model({CONFIG_MODE_KEY: "notes"}, allowed=())

        model.set_cleanup_enabled(True)
        model.set_mode("mail")
        model.set_tone("terse")
        model.set_context_awareness(False)

        self.assertTrue(model.cleanup_is_machine_default)
        self.assertEqual(model.mode, "notes")
        self.assertEqual(model.tone, "neutral")
        self.assertTrue(model.context_awareness)
        self.assertEqual(model.apply(), {})

    def test_the_selection_probe_is_never_behind_the_pro_gate(self):
        """A capture permission a user cannot switch off would be indefensible."""
        model = make_model(allowed=())

        model.set_include_selection(True)

        self.assertTrue(model.include_selection)
        self.assertEqual(model.apply(), {CONFIG_INCLUDE_SELECTION: True})


class FreeLimitTests(unittest.TestCase):
    def test_the_hint_names_the_free_term_limit(self):
        self.assertIn(str(FREE_TERM_LIMIT), VOCABULARY_LIMIT_HINT)
        self.assertIn("free plan", VOCABULARY_LIMIT_HINT)

    def test_the_hint_names_the_free_snippet_limit(self):
        self.assertIn(str(FREE_SNIPPET_LIMIT), SNIPPETS_LIMIT_HINT)

    def test_exactly_the_free_limit_is_not_over_it(self):
        terms = [f"term{index}" for index in range(FREE_TERM_LIMIT)]
        model = make_model({"vocabulary_terms": terms}, allowed=())

        self.assertIsNone(model.vocabulary_limit_hint)

    def test_a_snippet_with_no_trigger_does_not_count_towards_the_limit(self):
        snippets = [
            {"trigger": f"t{index}", "text": "x"} for index in range(FREE_SNIPPET_LIMIT)
        ]
        model = make_model({CONFIG_SNIPPETS_KEY: snippets}, allowed=())
        model.snippets.add_snippet()

        self.assertIsNone(model.snippets_limit_hint)

    def test_terms_over_the_limit_are_kept_not_truncated(self):
        """The app truncates at use; losing terms on save would destroy data."""
        terms = [f"term{index}" for index in range(FREE_TERM_LIMIT + 5)]
        model = make_model({"vocabulary_terms": terms}, allowed=())

        self.assertEqual(model.vocabulary_limit_hint, VOCABULARY_LIMIT_HINT)
        self.assertEqual(model.as_config()["vocabulary_terms"], terms)

    def test_snippets_over_the_limit_are_kept_not_truncated(self):
        snippets = [
            {"trigger": f"t{index}", "text": "x"} for index in range(FREE_SNIPPET_LIMIT + 3)
        ]
        model = make_model({CONFIG_SNIPPETS_KEY: snippets}, allowed=())

        self.assertEqual(model.snippets_limit_hint, SNIPPETS_LIMIT_HINT)
        self.assertEqual(len(model.as_config()[CONFIG_SNIPPETS_KEY]), len(snippets))


class CleanupEnabledTests(unittest.TestCase):
    """``cleanup_enabled`` is the one tri-state on the tab."""

    def test_none_shows_what_this_machine_would_do(self):
        for machine_default in (True, False):
            with self.subTest(machine_default=machine_default):
                model = make_model(
                    {CONFIG_CLEANUP_ENABLED: None}, machine_default=machine_default
                )

                self.assertEqual(model.cleanup_enabled, machine_default)
                self.assertTrue(model.cleanup_is_machine_default)
                self.assertEqual(model.cleanup_default_hint, CLEANUP_DEFAULT_HINT)

    def test_the_hint_says_where_the_default_comes_from(self):
        self.assertIn("16 GB", CLEANUP_DEFAULT_HINT)

    def test_an_undecided_value_is_never_written_back_on_its_own(self):
        model = make_model({CONFIG_CLEANUP_ENABLED: None}, machine_default=True)

        self.assertEqual(model.apply(), {})
        self.assertIsNone(model.as_config()[CONFIG_CLEANUP_ENABLED])

    def test_a_stored_bool_is_the_users_answer(self):
        for stored in (True, False):
            with self.subTest(stored=stored):
                model = make_model(
                    {CONFIG_CLEANUP_ENABLED: stored}, machine_default=not stored
                )

                self.assertEqual(model.cleanup_enabled, stored)
                self.assertFalse(model.cleanup_is_machine_default)
                self.assertIsNone(model.cleanup_default_hint)

    def test_a_value_that_is_not_a_bool_counts_as_undecided(self):
        model = make_model({CONFIG_CLEANUP_ENABLED: "yes"}, machine_default=False)

        self.assertFalse(model.cleanup_enabled)
        self.assertTrue(model.cleanup_is_machine_default)

    def test_the_checkbox_writes_a_real_bool(self):
        model = make_model({CONFIG_CLEANUP_ENABLED: None}, machine_default=False)

        model.set_cleanup_enabled(True)

        self.assertEqual(model.apply(), {CONFIG_CLEANUP_ENABLED: True})

    def test_choosing_the_machine_default_is_still_a_choice(self):
        model = make_model({CONFIG_CLEANUP_ENABLED: None}, machine_default=False)

        model.set_cleanup_enabled(False)

        self.assertEqual(model.apply(), {CONFIG_CLEANUP_ENABLED: False})
        self.assertFalse(model.cleanup_is_machine_default)

    def test_the_checkbox_only_takes_a_bool(self):
        model = make_model()

        with self.assertRaises(AssertionError):
            model.set_cleanup_enabled(1)


class CloudCleanupTests(unittest.TestCase):
    def test_available_only_on_murmur_cloud_and_only_for_pro(self):
        cases = (
            (CLOUD_MODE_MURMUR, ALL_FEATURES, True),
            (CLOUD_MODE_MURMUR, (), False),
            ("off", ALL_FEATURES, False),
            ("own_key", ALL_FEATURES, False),
        )
        for cloud_mode, allowed, available in cases:
            with self.subTest(cloud_mode=cloud_mode, allowed=allowed):
                model = make_model({CONFIG_CLOUD_MODE: cloud_mode}, allowed=allowed)

                self.assertEqual(model.cloud_cleanup_available, available)

    def test_the_checkbox_is_refused_when_it_is_not_available(self):
        model = make_model({CONFIG_CLOUD_MODE: "off"})

        model.set_cleanup_cloud(True)

        self.assertFalse(model.cleanup_cloud)
        self.assertEqual(model.apply(), {})

    def test_the_checkbox_writes_when_murmur_cloud_is_on(self):
        model = make_model({CONFIG_CLOUD_MODE: CLOUD_MODE_MURMUR})

        model.set_cleanup_cloud(True)

        self.assertEqual(model.apply(), {CONFIG_CLEANUP_CLOUD: True})


class ModeAndToneTests(unittest.TestCase):
    def test_the_popups_list_every_mode_and_tone(self):
        model = make_model()

        self.assertIn("Dictation", model.mode_titles)
        self.assertIn("Code", model.mode_titles)
        self.assertEqual(len(model.tone_titles), 4)

    def test_the_description_follows_the_chosen_mode(self):
        model = make_model()
        dictation = model.mode_description

        model.set_mode("code")

        self.assertNotEqual(model.mode_description, dictation)
        self.assertTrue(model.mode_description)

    def test_a_mode_is_chosen_by_popup_row(self):
        model = make_model()

        model.set_mode_index(model.mode_titles.index("Mail"))

        self.assertEqual(model.mode, "mail")
        self.assertEqual(model.apply(), {CONFIG_MODE_KEY: "mail"})

    def test_an_unknown_stored_mode_reads_as_the_default(self):
        model = make_model({CONFIG_MODE_KEY: "haiku"})

        self.assertEqual(model.mode, "dictation")

    def test_an_unknown_stored_tone_reads_as_the_default(self):
        model = make_model({CONFIG_TONE_KEY: "sarcastic"})

        self.assertEqual(model.tone, "neutral")

    def test_an_unknown_mode_cannot_be_chosen(self):
        model = make_model()

        with self.assertRaises(AssertionError):
            model.set_mode("haiku")


class OverridesTests(unittest.TestCase):
    OVERRIDES = {"com.apple.Notes": "notes", "com.apple.Terminal": "code"}

    def test_reads_the_stored_overrides_in_a_stable_order(self):
        section = OverridesSectionModel({MODE_OVERRIDES_CONFIG_KEY: self.OVERRIDES})

        self.assertEqual(
            [row.bundle_id for row in section.rows],
            ["com.apple.Notes", "com.apple.Terminal"],
        )
        self.assertEqual(section.row_count, 2)

    def test_the_mode_cell_holds_the_popup_row(self):
        section = OverridesSectionModel({MODE_OVERRIDES_CONFIG_KEY: {"a": "code"}})

        self.assertEqual(section.value_for(0, "bundle_id"), "a")
        self.assertEqual(section.value_for(0, "mode"), 4)  # "code" is last in MODE_IDS
        self.assertEqual(section.rows[0].mode_display_name, "Code")

    def test_adds_a_row_and_returns_its_index(self):
        section = OverridesSectionModel({})

        self.assertEqual(section.add_override(), 0)
        self.assertEqual(section.add_override("com.apple.Notes", "notes"), 1)
        self.assertEqual(section.row_count, 2)

    def test_a_new_row_is_edited_through_the_table(self):
        section = OverridesSectionModel({})
        row = section.add_override()

        section.set_value(row, "bundle_id", "  com.apple.Notes ")
        section.set_value(row, "mode", 3)

        self.assertEqual(
            section.to_config(), {MODE_OVERRIDES_CONFIG_KEY: {"com.apple.Notes": "notes"}}
        )

    def test_a_blank_row_is_never_saved(self):
        section = OverridesSectionModel({MODE_OVERRIDES_CONFIG_KEY: {"a": "code"}})
        section.add_override()

        self.assertEqual(section.to_config(), {MODE_OVERRIDES_CONFIG_KEY: {"a": "code"}})

    def test_removing_returns_the_row_to_select_next(self):
        section = OverridesSectionModel({MODE_OVERRIDES_CONFIG_KEY: self.OVERRIDES})

        self.assertEqual(section.remove_override(0), 0)
        self.assertEqual(section.remove_override(0), -1)
        self.assertEqual(section.to_config(), {MODE_OVERRIDES_CONFIG_KEY: {}})

    def test_removing_a_row_that_is_not_there_is_a_programming_error(self):
        section = OverridesSectionModel({})

        with self.assertRaises(AssertionError):
            section.remove_override(0)

    def test_an_unknown_stored_mode_reads_as_the_default(self):
        section = OverridesSectionModel({MODE_OVERRIDES_CONFIG_KEY: {"a": "haiku"}})

        self.assertEqual(section.to_config(), {MODE_OVERRIDES_CONFIG_KEY: {"a": "dictation"}})

    def test_an_unreadable_override_map_is_ignored_rather_than_fatal(self):
        section = OverridesSectionModel({MODE_OVERRIDES_CONFIG_KEY: ["com.apple.Notes"]})

        self.assertEqual(section.row_count, 0)

    def test_the_built_in_table_is_only_ever_read(self):
        self.assertEqual(built_in_mode_for("com.apple.Terminal"), "code")
        self.assertIsNone(built_in_mode_for("com.example.unknown"))


class VocabularyPortTests(unittest.TestCase):
    """The Wave 1 editor, ported whole; the rules must not have drifted."""

    CONFIG = {
        "vocabulary_terms": ["Boske", "Murmur"],
        "vocabulary_replacements": [{"from": "teh", "to": "the", "match_case": False}],
    }

    def test_the_terms_box_holds_one_term_per_line(self):
        model = make_model(self.CONFIG)

        self.assertEqual(model.vocabulary.terms_text, "Boske\nMurmur")

    def test_reading_the_box_back_drops_blanks_and_repeats(self):
        model = make_model()

        model.vocabulary.set_terms_text(" Boske \n\nBoske\nMurmur\n")

        self.assertEqual(model.vocabulary.terms, ["Boske", "Murmur"])

    def test_a_replacement_round_trips_through_the_table(self):
        model = make_model(self.CONFIG)

        self.assertEqual(model.vocabulary.value_for(0, "from"), "teh")
        self.assertEqual(model.vocabulary.value_for(0, "to"), "the")
        self.assertIs(model.vocabulary.value_for(0, "match_case"), False)

        model.vocabulary.set_value(0, "match_case", True)

        self.assertIs(model.vocabulary.value_for(0, "match_case"), True)

    def test_a_half_typed_replacement_is_never_saved(self):
        model = make_model(self.CONFIG)
        model.vocabulary.add_replacement()

        self.assertEqual(len(model.as_config()["vocabulary_replacements"]), 1)

    def test_removing_returns_the_row_to_select_next(self):
        model = make_model(self.CONFIG)

        self.assertEqual(model.vocabulary.remove_replacement(0), -1)

    def test_an_unreadable_vocabulary_is_ignored_rather_than_fatal(self):
        model = make_model({"vocabulary_terms": "Boske"})

        self.assertEqual(model.vocabulary.terms, [])

    def test_an_unreadable_vocabulary_is_never_written_over(self):
        model = make_model({"vocabulary_terms": "Boske"})

        self.assertEqual(model.apply(), {})

    def test_csv_round_trips(self):
        model = make_model(self.CONFIG)
        exported = model.vocabulary.export_text()

        empty = make_model()
        empty.vocabulary.import_text(exported)

        self.assertEqual(empty.as_config()["vocabulary_terms"], ["Boske", "Murmur"])
        self.assertEqual(empty.as_config()["vocabulary_replacements"], self.CONFIG["vocabulary_replacements"])

    def test_a_bad_csv_names_the_line_it_failed_on(self):
        model = make_model(self.CONFIG)

        with self.assertRaises(VocabularyError) as caught:
            model.vocabulary.import_text("kind,from,to,match_case\nterm,Boske,,\nnonsense,a,b,false\n")

        self.assertIn("Line 3", str(caught.exception))

    def test_a_bad_csv_leaves_the_current_lists_alone(self):
        model = make_model(self.CONFIG)

        with self.assertRaises(VocabularyError):
            model.vocabulary.import_text("wrong,header\n")

        self.assertEqual(model.vocabulary.terms, ["Boske", "Murmur"])
        self.assertEqual(model.apply(), {})


class SnippetsSectionTests(unittest.TestCase):
    CONFIG = {CONFIG_SNIPPETS_KEY: [{"trigger": "my address", "text": "12 Rue Oberkampf"}]}

    def test_reads_the_stored_snippets(self):
        model = make_model(self.CONFIG)

        self.assertEqual(model.snippets.row_count, 1)
        self.assertEqual(model.snippets.value_for(0, "trigger"), "my address")
        self.assertEqual(model.snippets.value_for(0, "text"), "12 Rue Oberkampf")

    def test_a_cell_is_edited_through_the_table(self):
        model = make_model(self.CONFIG)

        model.snippets.set_value(0, "text", "12 Rue Oberkampf, Paris")

        self.assertEqual(
            model.as_config()[CONFIG_SNIPPETS_KEY],
            [{"trigger": "my address", "text": "12 Rue Oberkampf, Paris"}],
        )

    def test_adds_a_row_and_returns_its_index(self):
        model = make_model(self.CONFIG)

        self.assertEqual(model.snippets.add_snippet(), 1)

    def test_a_snippet_with_no_trigger_is_never_saved(self):
        model = make_model(self.CONFIG)
        model.snippets.add_snippet()

        self.assertEqual(len(model.as_config()[CONFIG_SNIPPETS_KEY]), 1)
        self.assertEqual(model.apply(), {})

    def test_removing_returns_the_row_to_select_next(self):
        model = make_model(self.CONFIG)
        model.snippets.add_snippet()

        self.assertEqual(model.snippets.remove_snippet(0), 0)
        self.assertEqual(model.snippets.remove_snippet(0), -1)

    def test_an_unreadable_snippet_list_is_ignored_rather_than_fatal(self):
        model = make_model({CONFIG_SNIPPETS_KEY: ["my address"]})

        self.assertEqual(model.snippets.row_count, 0)
        self.assertEqual(model.apply(), {})

    def test_the_saved_list_is_what_the_expander_reads(self):
        model = make_model(self.CONFIG)

        self.assertEqual(
            model.snippets.saved, (Snippet("my address", "12 Rue Oberkampf"),)
        )


class PersistenceTests(unittest.TestCase):
    def test_the_tab_owns_exactly_these_keys(self):
        model = make_model()

        self.assertEqual(set(model.as_config()), set(OWNED_CONFIG_KEYS))

    def test_every_owned_key_has_a_documented_default(self):
        for key in OWNED_CONFIG_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, DEFAULT_CONFIG)

    def test_an_untouched_tab_writes_nothing(self):
        self.assertEqual(make_model().apply(), {})

    def test_apply_reports_only_what_moved(self):
        model = make_model()

        model.set_tone("terse")

        self.assertEqual(model.apply(), {CONFIG_TONE_KEY: "terse"})

    def test_marking_saved_makes_the_next_apply_empty(self):
        model = make_model()
        model.set_tone("terse")
        model.apply()

        model.mark_saved()

        self.assertEqual(model.apply(), {})

    def test_every_control_round_trips_its_key(self):
        config = {**DEFAULT_CONFIG, CONFIG_CLOUD_MODE: CLOUD_MODE_MURMUR}
        model = make_model(config)

        model.set_cleanup_enabled(True)
        model.set_cleanup_cloud(True)
        model.set_mode("mail")
        model.set_tone("terse")
        model.set_context_awareness(False)
        model.set_include_selection(True)
        model.overrides.add_override("com.example.app", "code")
        model.vocabulary.set_terms_text("Boske")
        row = model.vocabulary.add_replacement()
        model.vocabulary.set_value(row, "from", "teh")
        model.vocabulary.set_value(row, "to", "the")
        row = model.snippets.add_snippet()
        model.snippets.set_value(row, "trigger", "my address")
        model.snippets.set_value(row, "text", "12 Rue Oberkampf")
        changed = model.apply()

        self.assertEqual(set(changed), set(OWNED_CONFIG_KEYS))

        again = make_model({**config, **changed})

        self.assertEqual(again.apply(), {})
        self.assertTrue(again.cleanup_enabled)
        self.assertFalse(again.cleanup_is_machine_default)
        self.assertTrue(again.cleanup_cloud)
        self.assertEqual(again.mode, "mail")
        self.assertEqual(again.tone, "terse")
        self.assertFalse(again.context_awareness)
        self.assertTrue(again.include_selection)
        self.assertEqual(again.overrides.value_for(0, "bundle_id"), "com.example.app")
        self.assertEqual(again.vocabulary.terms, ["Boske"])
        self.assertEqual(again.vocabulary.value_for(0, "to"), "the")
        self.assertEqual(again.snippets.value_for(0, "trigger"), "my address")

    def test_refresh_re_reads_the_live_config(self):
        config = dict(DEFAULT_CONFIG)
        model = SmartTabModel(
            config, pro_gate=lambda feature: True, cleanup_probe=lambda: False
        )

        config[CONFIG_MODE_KEY] = "notes"
        config[CONFIG_CLOUD_MODE] = CLOUD_MODE_MURMUR
        config["vocabulary_terms"] = ["Boske"]
        model.refresh()

        self.assertEqual(model.mode, "notes")
        self.assertTrue(model.cloud_cleanup_available)
        self.assertEqual(model.vocabulary.terms, ["Boske"])
        self.assertEqual(model.apply(), {})

    def test_the_model_never_mutates_the_config_it_was_given(self):
        config = dict(DEFAULT_CONFIG)
        model = make_model(config)

        model.set_mode("mail")
        model.apply()

        self.assertEqual(config[CONFIG_MODE_KEY], DEFAULT_CONFIG[CONFIG_MODE_KEY])


class SpokenSymbolTests(unittest.TestCase):
    def test_lists_one_row_per_symbol(self):
        rows = spoken_symbol_rows()
        symbols = [symbol for symbol, _ in rows]

        self.assertEqual(len(symbols), len(set(symbols)))

    def test_names_every_way_of_saying_a_symbol(self):
        by_symbol = dict(spoken_symbol_rows())

        self.assertIn("open paren", by_symbol["("])
        self.assertIn("star", by_symbol["*"])
        self.assertIn("asterisk", by_symbol["*"])


class TabContractTests(unittest.TestCase):
    def test_it_satisfies_the_settings_tab_protocol(self):
        self.assertIsInstance(SmartTab(), SettingsTab)

    def test_it_is_the_smart_tab(self):
        self.assertEqual(SmartTab.identifier, TAB_SMART)
        self.assertEqual(SmartTab.title, "Smart")

    def test_it_registers_itself_on_import(self):
        self.assertIn(SmartTab, registered_tabs())

    def test_refresh_before_build_does_nothing(self):
        SmartTab().refresh()

    def test_close_is_safe_to_call_twice_on_a_tab_that_never_built(self):
        tab = SmartTab()

        tab.close()
        tab.close()


if __name__ == "__main__":
    unittest.main()
