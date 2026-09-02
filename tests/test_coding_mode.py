"""Tests for cleanup.coding_mode: rule-based spoken-to-code transforms."""

import unittest

from cleanup.coding_mode import (
    SPOKEN_SYMBOLS,
    code_mode_llm_hint,
    transform_spoken_code,
)


# Each entry: (spoken text, expected code text, language).
# Grouped by the rule it exercises; see cleanup/coding_mode.py's module
# docstring for the documented transform order and spacing rules.
FIXTURES: tuple[tuple[str, str, str], ...] = (
    # -- Punctuation and brackets -------------------------------------
    ("open paren x close paren", "(x)", "en"),
    ("open bracket 1 comma 2 close bracket", "[1, 2]", "en"),
    ("open brace close brace", "{}", "en"),
    ("call open paren x close paren", "call (x)", "en"),
    ("wait comma then go", "wait, then go", "en"),
    ("done period", "done.", "en"),
    ("key colon value", "key: value", "en"),
    ("print open paren x close paren semicolon", "print (x);", "en"),
    # -- Operators (spaced on both sides) ------------------------------
    ("x double equals y", "x == y", "en"),
    ("x not equals y", "x != y", "en"),
    ("x equals y", "x = y", "en"),
    ("a plus b", "a + b", "en"),
    ("a minus b", "a - b", "en"),
    ("5 star 3", "5 * 3", "en"),
    ("5 asterisk 3", "5 * 3", "en"),
    ("10 slash 2", "10 / 2", "en"),
    ("one backslash two", "one \\ two", "en"),
    ("a pipe b", "a | b", "en"),
    ("a ampersand b", "a & b", "en"),
    ("value percent d", "value % d", "en"),
    ("x caret 2", "x ^ 2", "en"),
    ("tilde home", "~ home", "en"),
    ("hash comment here", "# comment here", "en"),
    ("price dollar 5", "price $ 5", "en"),
    ("is it true question mark", "is it true ?", "en"),
    ("wow exclamation mark", "wow !", "en"),
    ("x arrow y", "x -> y", "en"),
    ("x fat arrow y", "x => y", "en"),
    ("x less than y", "x < y", "en"),
    ("x greater than y", "x > y", "en"),
    # -- Glue-both-sides tokens -----------------------------------------
    ("reply at sign user", "reply@user", "en"),
    ("max underscore retries", "max_retries", "en"),
    ("line one new line line two", "line one\nline two", "en"),
    ("indent tab body", "indent\tbody", "en"),
    # -- Toggling quotes --------------------------------------------------
    ("say quote hello quote", 'say "hello"', "en"),
    ("say double quote hello quote", 'say "hello"', "en"),
    ("print single quote hi single quote", "print 'hi'", "en"),
    ("print apostrophe hi apostrophe", "print 'hi'", "en"),
    ("wrap backtick code backtick", "wrap `code`", "en"),
    # -- CLI dash-flags ---------------------------------------------------
    ("dash dash force", "--force", "en"),
    ("dash v", "-v", "en"),
    ("dash dash verbose", "--verbose", "en"),
    ("git commit dash m", "git commit -m", "en"),
    ("pip install", "pip install", "en"),
    ("dash dash", "dash dash", "en"),
    ("a dash of salt", "a dash of salt", "en"),
    ("dash open paren", "dash (", "en"),
    ("dash Force", "-Force", "en"),
    ("dash dash force", "--force", "en"),
    ("ls dash l a", "ls -l a", "en"),
    # -- Spelled-out tool names --------------------------------------------
    ("n p m install", "npm install", "en"),
    ("s s h server", "ssh server", "en"),
    ("g i t status", "git status", "en"),
    ("h t m l file", "html file", "en"),
    ("h t t p request", "http request", "en"),
    ("j s o n data", "json data", "en"),
    # -- Case commands ------------------------------------------------------
    ("camel case foo bar baz", "fooBarBaz", "en"),
    ("pascal case foo bar baz", "FooBarBaz", "en"),
    ("snake case foo bar baz", "foo_bar_baz", "en"),
    ("kebab case foo bar baz", "foo-bar-baz", "en"),
    ("screaming snake foo bar baz", "FOO_BAR_BAZ", "en"),
    ("constant case foo bar baz", "FOO_BAR_BAZ", "en"),
    ("all caps foo", "FOO", "en"),
    ("camel case foo bar end case next word", "fooBar next word", "en"),
    ("camel case foo bar comma next", "fooBar, next", "en"),
    ("camel case stop next", "next", "en"),
    ("snake case max retries 3", "max_retries_3", "en"),
    # -- French --------------------------------------------------------------
    ("parenthèse ouvrante x parenthèse fermante", "(x)", "fr"),
    ("un virgule deux", "un, deux", "fr"),
    ("tiret tiret force", "--force", "fr"),
    ("fin point", "fin.", "fr"),
)


class TransformSpokenCodeTests(unittest.TestCase):
    def test_fixtures(self) -> None:
        for text, expected, language in FIXTURES:
            with self.subTest(text=text, language=language):
                self.assertEqual(
                    transform_spoken_code(text, language=language), expected
                )

    def test_idempotent_over_all_fixtures(self) -> None:
        for text, _expected, language in FIXTURES:
            with self.subTest(text=text, language=language):
                once = transform_spoken_code(text, language=language)
                twice = transform_spoken_code(once, language=language)
                self.assertEqual(once, twice)

    def test_noop_on_plain_prose(self) -> None:
        prose = "hello world this is a normal sentence with no commands"
        self.assertEqual(transform_spoken_code(prose), prose)

    def test_noop_on_prose_with_mixed_case(self) -> None:
        prose = "Please Review This Pull Request Today"
        self.assertEqual(transform_spoken_code(prose), prose)

    def test_empty_string_is_unchanged(self) -> None:
        self.assertEqual(transform_spoken_code(""), "")

    def test_unsupported_language_raises(self) -> None:
        with self.assertRaises(ValueError):
            transform_spoken_code("open paren x close paren", language="de")

    def test_default_language_is_english(self) -> None:
        self.assertEqual(transform_spoken_code("open paren x close paren"), "(x)")

    def test_none_and_auto_language_are_treated_as_english(self) -> None:
        for language in (None, "auto"):
            with self.subTest(language=language):
                self.assertEqual(
                    transform_spoken_code(
                        "open paren x close paren", language=language
                    ),
                    "(x)",
                )

    def test_unsupported_explicit_language_still_raises(self) -> None:
        with self.assertRaises(ValueError):
            transform_spoken_code("open paren x close paren", language="xx")

    def test_matching_is_case_insensitive(self) -> None:
        self.assertEqual(transform_spoken_code("OPEN PAREN x CLOSE PAREN"), "(x)")
        self.assertEqual(transform_spoken_code("Dash Dash Force"), "--Force")
        self.assertEqual(
            transform_spoken_code("Camel Case foo bar"), "fooBar"
        )

    def test_spelled_letters_require_known_tool_name(self) -> None:
        # "x y z" spells no known tool, so the letters stay separate words.
        self.assertEqual(transform_spoken_code("x y z"), "x y z")

    def test_single_letter_words_in_prose_are_not_joined(self) -> None:
        self.assertEqual(transform_spoken_code("i have a plan"), "i have a plan")


class SpokenSymbolsTests(unittest.TestCase):
    def test_exposes_non_empty_display_table(self) -> None:
        self.assertGreater(len(SPOKEN_SYMBOLS), 0)
        for phrase, symbol in SPOKEN_SYMBOLS:
            self.assertIsInstance(phrase, str)
            self.assertIsInstance(symbol, str)
            self.assertTrue(phrase)
            self.assertTrue(symbol)

    def test_known_punctuation_entries_present(self) -> None:
        table = dict(SPOKEN_SYMBOLS)
        self.assertEqual(table["open paren"], "(")
        self.assertEqual(table["close paren"], ")")
        self.assertEqual(table["arrow"], "->")
        self.assertEqual(table["fat arrow"], "=>")


class CodeModeLlmHintTests(unittest.TestCase):
    def test_returns_single_sentence_string(self) -> None:
        hint = code_mode_llm_hint()
        self.assertIsInstance(hint, str)
        self.assertTrue(hint.strip())
        self.assertEqual(hint.count("."), 1)

    def test_mentions_leaving_tokens_untouched(self) -> None:
        hint = code_mode_llm_hint().lower()
        self.assertIn("flag", hint)
        self.assertIn("identifier", hint)


if __name__ == "__main__":
    unittest.main()
