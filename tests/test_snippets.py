"""Snippets: spoken triggers that expand into stored text.

Plain Python, no AppKit: the expansion rules are the ones the transcription
pipeline will run, so they are tested on their own rather than through the tab
that edits them.
"""

import unittest

from cleanup.snippets import (
    CONFIG_SNIPPETS_KEY,
    FREE_SNIPPET_LIMIT,
    Snippet,
    SnippetError,
    expand_snippets,
    snippets_from_config,
    snippets_to_config,
)


class SnippetConfigTests(unittest.TestCase):
    def test_reads_an_empty_list_when_the_key_is_absent(self):
        self.assertEqual(snippets_from_config({}), ())

    def test_reads_trigger_and_text(self):
        config = {CONFIG_SNIPPETS_KEY: [{"trigger": "my address", "text": "12 Rue Oberkampf"}]}

        self.assertEqual(
            snippets_from_config(config),
            (Snippet(trigger="my address", text="12 Rue Oberkampf"),),
        )

    def test_round_trips_through_config(self):
        snippets = (Snippet("sign off", "Best,\nMatthieu"), Snippet("my email", "m@example.com"))

        self.assertEqual(snippets_from_config(snippets_to_config(snippets)), snippets)

    def test_rejects_a_non_list(self):
        with self.assertRaises(SnippetError):
            snippets_from_config({CONFIG_SNIPPETS_KEY: {"trigger": "x"}})

    def test_rejects_a_row_that_is_not_an_object(self):
        with self.assertRaises(SnippetError):
            snippets_from_config({CONFIG_SNIPPETS_KEY: ["my address"]})

    def test_rejects_wrong_field_types(self):
        with self.assertRaises(SnippetError):
            snippets_from_config({CONFIG_SNIPPETS_KEY: [{"trigger": "x", "text": 3}]})

    def test_names_the_offending_row(self):
        with self.assertRaises(SnippetError) as caught:
            snippets_from_config(
                {CONFIG_SNIPPETS_KEY: [{"trigger": "a", "text": "b"}, {"trigger": None}]}
            )

        self.assertIn("1", str(caught.exception))


class ExpandSnippetsTests(unittest.TestCase):
    ADDRESS = Snippet("my address", "12 Rue Oberkampf, Paris")

    def test_returns_the_text_untouched_when_there_are_no_snippets(self):
        self.assertEqual(expand_snippets("hello there", ()), "hello there")

    def test_replaces_a_trigger_phrase(self):
        self.assertEqual(
            expand_snippets("send it to my address please", (self.ADDRESS,)),
            "send it to 12 Rue Oberkampf, Paris please",
        )

    def test_matches_regardless_of_case(self):
        self.assertEqual(
            expand_snippets("My Address is here", (self.ADDRESS,)),
            "12 Rue Oberkampf, Paris is here",
        )

    def test_matches_whole_words_only(self):
        snippets = (Snippet("cat", "feline"),)

        self.assertEqual(expand_snippets("category of cat", snippets), "category of feline")

    def test_tolerates_extra_whitespace_inside_the_phrase(self):
        self.assertEqual(
            expand_snippets("my  address", (self.ADDRESS,)), "12 Rue Oberkampf, Paris"
        )

    def test_the_longest_trigger_wins(self):
        snippets = (
            Snippet("address", "SHORT"),
            Snippet("my address", "LONG"),
        )

        self.assertEqual(expand_snippets("my address", snippets), "LONG")

    def test_a_blank_trigger_is_ignored(self):
        self.assertEqual(expand_snippets("hello", (Snippet("  ", "boom"),)), "hello")

    def test_expands_every_occurrence(self):
        snippets = (Snippet("sig", "Matthieu"),)

        self.assertEqual(expand_snippets("sig and sig", snippets), "Matthieu and Matthieu")

    def test_does_not_rescan_its_own_output(self):
        """A snippet whose text contains another trigger expands once, not twice."""
        snippets = (Snippet("greeting", "hello my address"), self.ADDRESS)

        self.assertEqual(expand_snippets("greeting", snippets), "hello my address")

    def test_is_idempotent(self):
        snippets = (Snippet("sig", "sig: Matthieu"), self.ADDRESS)
        once = expand_snippets("my sig, my address", snippets)

        self.assertEqual(expand_snippets(once, snippets), once)

    def test_is_idempotent_for_ordinary_snippets(self):
        once = expand_snippets("post it to my address", (self.ADDRESS,))

        self.assertEqual(expand_snippets(once, (self.ADDRESS,)), once)


class FreeLimitTests(unittest.TestCase):
    def test_the_free_limit_is_five(self):
        self.assertEqual(FREE_SNIPPET_LIMIT, 5)


if __name__ == "__main__":
    unittest.main()
