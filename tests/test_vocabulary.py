import unittest

from engines import Hints

from cleanup.vocabulary import (
    CSV_HEADER,
    FREE_TERM_LIMIT,
    Replacement,
    Vocabulary,
    VocabularyError,
    apply_replacements,
    export_csv,
    hints_from_vocabulary,
    import_csv,
    vocabulary_from_config,
    vocabulary_to_config,
)


class DataclassTests(unittest.TestCase):
    def test_replacement_is_frozen(self):
        replacement = Replacement(from_text="teh", to_text="the", match_case=False)
        with self.assertRaises(Exception):
            replacement.from_text = "other"  # type: ignore[misc]

    def test_vocabulary_is_frozen(self):
        vocabulary = Vocabulary(terms=("Murmur",), replacements=())
        with self.assertRaises(Exception):
            vocabulary.terms = ("Other",)  # type: ignore[misc]

    def test_vocabulary_defaults_to_empty(self):
        vocabulary = Vocabulary()
        self.assertEqual(vocabulary.terms, ())
        self.assertEqual(vocabulary.replacements, ())

    def test_free_term_limit_is_twenty(self):
        self.assertEqual(FREE_TERM_LIMIT, 20)


class ApplyReplacementsTests(unittest.TestCase):
    def test_replaces_whole_word_only(self):
        vocabulary = Vocabulary(
            replacements=(Replacement(from_text="cat", to_text="dog", match_case=True),)
        )
        self.assertEqual(apply_replacements("cat category cats cat", vocabulary), "dog category cats dog")

    def test_case_insensitive_when_match_case_false(self):
        vocabulary = Vocabulary(
            replacements=(Replacement(from_text="teh", to_text="the", match_case=False),)
        )
        self.assertEqual(apply_replacements("Teh cat sat TEH mat", vocabulary), "the cat sat the mat")

    def test_case_sensitive_when_match_case_true(self):
        vocabulary = Vocabulary(
            replacements=(Replacement(from_text="Boske", to_text="Boské", match_case=True),)
        )
        self.assertEqual(apply_replacements("Boske and boske", vocabulary), "Boské and boske")

    def test_replacement_text_inserted_as_given_regardless_of_match_case(self):
        vocabulary = Vocabulary(
            replacements=(Replacement(from_text="voxtral", to_text="Voxtral", match_case=False),)
        )
        self.assertEqual(apply_replacements("VOXTRAL is fast", vocabulary), "Voxtral is fast")

    def test_longer_from_text_wins_over_shorter_contained_word(self):
        vocabulary = Vocabulary(
            replacements=(
                Replacement(from_text="New York", to_text="NYC", match_case=False),
                Replacement(from_text="York", to_text="Yorkshire", match_case=False),
            )
        )
        self.assertEqual(apply_replacements("I live in New York", vocabulary), "I live in NYC")

    def test_equal_length_replacements_run_in_listed_order(self):
        # Both from_text values are length 1, a tie broken only by list order.
        # "a" -> "b" runs first, then "b" -> "c" runs on that result, so the
        # single "a" chains all the way to "c" — proving the pass is
        # sequential in listed order, not a single simultaneous substitution.
        vocabulary = Vocabulary(
            replacements=(
                Replacement(from_text="a", to_text="b", match_case=True),
                Replacement(from_text="b", to_text="c", match_case=True),
            )
        )
        self.assertEqual(apply_replacements("a", vocabulary), "c")

        reversed_vocabulary = Vocabulary(
            replacements=(
                Replacement(from_text="b", to_text="c", match_case=True),
                Replacement(from_text="a", to_text="b", match_case=True),
            )
        )
        self.assertEqual(apply_replacements("a", reversed_vocabulary), "b")

    def test_no_replacements_returns_text_unchanged(self):
        self.assertEqual(apply_replacements("hello world", Vocabulary()), "hello world")

    def test_idempotent_on_fixture(self):
        vocabulary = Vocabulary(
            replacements=(
                Replacement(from_text="teh", to_text="the", match_case=False),
                Replacement(from_text="recieve", to_text="receive", match_case=False),
            )
        )
        once = apply_replacements("teh cat will recieve teh mail", vocabulary)
        twice = apply_replacements(once, vocabulary)
        self.assertEqual(once, twice)
        self.assertEqual(once, "the cat will receive the mail")

    def test_empty_from_text_is_skipped(self):
        vocabulary = Vocabulary(replacements=(Replacement(from_text="", to_text="x", match_case=False),))
        self.assertEqual(apply_replacements("unchanged", vocabulary), "unchanged")


class HintsFromVocabularyTests(unittest.TestCase):
    def test_terms_only_no_replacements(self):
        vocabulary = Vocabulary(
            terms=("Murmur", "Voxtral"),
            replacements=(Replacement(from_text="teh", to_text="the", match_case=False),),
        )
        hints = hints_from_vocabulary(vocabulary)
        self.assertIsInstance(hints, Hints)
        self.assertEqual(hints.vocabulary, ("Murmur", "Voxtral"))
        self.assertIsNone(hints.initial_prompt)

    def test_strips_and_drops_empty_terms(self):
        vocabulary = Vocabulary(terms=("  Murmur  ", "", "   ", "Voxtral"))
        hints = hints_from_vocabulary(vocabulary)
        self.assertEqual(hints.vocabulary, ("Murmur", "Voxtral"))

    def test_deduplicates_keeping_first_occurrence(self):
        vocabulary = Vocabulary(terms=("Murmur", "Voxtral", "Murmur", "murmur"))
        hints = hints_from_vocabulary(vocabulary)
        self.assertEqual(hints.vocabulary, ("Murmur", "Voxtral", "murmur"))


class CsvRoundTripTests(unittest.TestCase):
    def test_export_header_matches_csv_header_constant(self):
        csv_text = export_csv(Vocabulary())
        first_line = csv_text.splitlines()[0]
        self.assertEqual(first_line, ",".join(CSV_HEADER))

    def test_round_trips_terms_and_replacements(self):
        vocabulary = Vocabulary(
            terms=("Murmur", "Voxtral"),
            replacements=(
                Replacement(from_text="teh", to_text="the", match_case=False),
                Replacement(from_text="Boske", to_text="Boské", match_case=True),
            ),
        )
        csv_text = export_csv(vocabulary)
        round_tripped = import_csv(csv_text)
        self.assertEqual(round_tripped, vocabulary)

    def test_round_trips_commas_and_quotes_and_unicode(self):
        vocabulary = Vocabulary(
            terms=('Say "hello", say "goodbye"', "Café Müller, LLC"),
            replacements=(
                Replacement(from_text='"quoted, term"', to_text="plain", match_case=False),
                Replacement(from_text="naïve", to_text="naive", match_case=True),
            ),
        )
        csv_text = export_csv(vocabulary)
        round_tripped = import_csv(csv_text)
        self.assertEqual(round_tripped, vocabulary)

    def test_import_rejects_wrong_header(self):
        with self.assertRaises(VocabularyError) as ctx:
            import_csv("kind,from,to\nterm,Murmur,,\n")
        self.assertIn("Line 1", str(ctx.exception))

    def test_import_rejects_empty_input(self):
        with self.assertRaises(VocabularyError) as ctx:
            import_csv("")
        self.assertIn("Line 1", str(ctx.exception))

    def test_import_rejects_unknown_kind(self):
        csv_text = "kind,from,to,match_case\nphrase,Murmur,,\n"
        with self.assertRaises(VocabularyError) as ctx:
            import_csv(csv_text)
        self.assertIn("Line 2", str(ctx.exception))

    def test_import_rejects_wrong_field_count(self):
        csv_text = "kind,from,to,match_case\nterm,Murmur,extra,,\n"
        with self.assertRaises(VocabularyError) as ctx:
            import_csv(csv_text)
        self.assertIn("Line 2", str(ctx.exception))

    def test_import_rejects_missing_from_value(self):
        csv_text = "kind,from,to,match_case\nterm,,,\n"
        with self.assertRaises(VocabularyError) as ctx:
            import_csv(csv_text)
        self.assertIn("Line 2", str(ctx.exception))

    def test_import_rejects_invalid_match_case(self):
        csv_text = "kind,from,to,match_case\nreplacement,teh,the,maybe\n"
        with self.assertRaises(VocabularyError) as ctx:
            import_csv(csv_text)
        self.assertIn("Line 2", str(ctx.exception))

    def test_import_reports_correct_line_number_for_second_bad_row(self):
        csv_text = "kind,from,to,match_case\nterm,Murmur,,\nphrase,Bad,,\n"
        with self.assertRaises(VocabularyError) as ctx:
            import_csv(csv_text)
        self.assertIn("Line 3", str(ctx.exception))

    def test_import_skips_trailing_blank_line(self):
        csv_text = "kind,from,to,match_case\nterm,Murmur,,\n\n"
        vocabulary = import_csv(csv_text)
        self.assertEqual(vocabulary.terms, ("Murmur",))


class ConfigRoundTripTests(unittest.TestCase):
    def test_missing_keys_mean_empty(self):
        vocabulary = vocabulary_from_config({})
        self.assertEqual(vocabulary, Vocabulary())

    def test_round_trips_through_config(self):
        vocabulary = Vocabulary(
            terms=("Murmur", "Voxtral"),
            replacements=(Replacement(from_text="teh", to_text="the", match_case=False),),
        )
        config = vocabulary_to_config(vocabulary)
        self.assertEqual(
            config,
            {
                "vocabulary_terms": ["Murmur", "Voxtral"],
                "vocabulary_replacements": [
                    {"from": "teh", "to": "the", "match_case": False}
                ],
            },
        )
        self.assertEqual(vocabulary_from_config(config), vocabulary)

    def test_wrong_type_for_terms_raises(self):
        with self.assertRaises(VocabularyError):
            vocabulary_from_config({"vocabulary_terms": "not-a-list"})

    def test_non_string_term_raises(self):
        with self.assertRaises(VocabularyError):
            vocabulary_from_config({"vocabulary_terms": ["ok", 5]})

    def test_wrong_type_for_replacements_raises(self):
        with self.assertRaises(VocabularyError):
            vocabulary_from_config({"vocabulary_replacements": "not-a-list"})

    def test_replacement_missing_field_raises(self):
        with self.assertRaises(VocabularyError):
            vocabulary_from_config(
                {"vocabulary_replacements": [{"from": "teh", "to": "the"}]}
            )

    def test_replacement_wrong_match_case_type_raises(self):
        with self.assertRaises(VocabularyError):
            vocabulary_from_config(
                {
                    "vocabulary_replacements": [
                        {"from": "teh", "to": "the", "match_case": "false"}
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
