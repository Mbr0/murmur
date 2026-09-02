"""Tests for cleanup.context — front-app capture and mode mapping.

These run on any machine: every AppKit / Accessibility call is injected.
"""

import ast
import re
import unittest
from pathlib import Path

from cleanup.context import (
    DEFAULT_MODE,
    DEFAULT_MODE_BY_BUNDLE,
    DEFAULT_MODE_BY_BUNDLE_PREFIX,
    KNOWN_MODES,
    MODE_CONFIG_KEY,
    MODE_OVERRIDES_CONFIG_KEY,
    UNVERIFIED_BUNDLE_IDS,
    VERIFIED_BUNDLE_IDS,
    AppContext,
    capture_context,
    default_mode_for_bundle,
    forget_mode,
    front_app_bundle_id,
    is_terminal_or_editor,
    remember_mode,
    resolve_mode,
)

CONTEXT_SOURCE = Path(__file__).resolve().parents[1] / "cleanup" / "context.py"

# Reverse-DNS: dot-separated segments, each starting alphanumeric. Cursor's
# "com.todesktop.230313mzl4w4u92" is why segments may start with a digit.
REVERSE_DNS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*(\.[A-Za-z0-9][A-Za-z0-9-]*)+$")


class FakeWorkspace:
    """Stands in for NSWorkspace.sharedWorkspace().frontmostApplication()."""

    def __init__(self, bundle_id=None, app_name=None, pid=None):
        self._result = (bundle_id, app_name, pid)
        self.calls = 0

    def frontmost_application(self):
        self.calls += 1
        return self._result


class FakeAccessibility:
    """Stands in for the AXUIElement reads; counts each getter."""

    def __init__(self, trusted=True, title=None, selection=None):
        self._trusted = trusted
        self._title = title
        self._selection = selection
        self.trusted_calls = 0
        self.title_calls = 0
        self.selection_calls = 0

    def is_trusted(self):
        self.trusted_calls += 1
        return self._trusted

    def window_title(self, pid):
        self.title_calls += 1
        return self._title

    def selected_text(self, pid):
        self.selection_calls += 1
        return self._selection


class CaptureContextTests(unittest.TestCase):
    def test_captures_all_fields_when_accessibility_granted(self):
        workspace = FakeWorkspace("com.apple.mail", "Mail", 501)
        ax = FakeAccessibility(trusted=True, title="Inbox", selection="quoted line")

        context = capture_context(include_selection=True, workspace=workspace, ax=ax)

        self.assertEqual(
            context,
            AppContext(
                bundle_id="com.apple.mail",
                app_name="Mail",
                window_title="Inbox",
                selected_text="quoted line",
            ),
        )
        self.assertEqual(ax.title_calls, 1)
        self.assertEqual(ax.selection_calls, 1)

    def test_include_selection_false_never_calls_the_selection_getter(self):
        workspace = FakeWorkspace("com.apple.Notes", "Notes", 7)
        ax = FakeAccessibility(trusted=True, title="Scratch", selection="secret")

        context = capture_context(include_selection=False, workspace=workspace, ax=ax)

        self.assertIsNone(context.selected_text)
        self.assertEqual(context.window_title, "Scratch")
        self.assertEqual(ax.selection_calls, 0)

    def test_default_never_calls_the_selection_getter(self):
        workspace = FakeWorkspace("com.apple.Notes", "Notes", 7)
        ax = FakeAccessibility(trusted=True, title="Scratch", selection="secret")

        context = capture_context(workspace=workspace, ax=ax)

        self.assertIsNone(context.selected_text)
        self.assertEqual(ax.selection_calls, 0)

    def test_no_accessibility_permission_yields_none_and_reads_nothing(self):
        workspace = FakeWorkspace("com.apple.Terminal", "Terminal", 12)
        ax = FakeAccessibility(trusted=False, title="zsh", selection="ls -la")

        context = capture_context(include_selection=True, workspace=workspace, ax=ax)

        self.assertEqual(context.bundle_id, "com.apple.Terminal")
        self.assertEqual(context.app_name, "Terminal")
        self.assertIsNone(context.window_title)
        self.assertIsNone(context.selected_text)
        self.assertEqual(ax.title_calls, 0)
        self.assertEqual(ax.selection_calls, 0)

    def test_no_frontmost_application_skips_accessibility_entirely(self):
        workspace = FakeWorkspace()
        ax = FakeAccessibility(trusted=True, title="x", selection="y")

        context = capture_context(include_selection=True, workspace=workspace, ax=ax)

        self.assertEqual(context, AppContext(None, None, None, None))
        self.assertEqual(ax.trusted_calls, 0)

    def test_app_context_is_frozen(self):
        context = AppContext("com.apple.mail", "Mail", None, None)
        with self.assertRaises(Exception):
            context.bundle_id = "other"  # type: ignore[misc]

    def test_front_app_bundle_id_is_a_one_call_helper(self):
        workspace = FakeWorkspace("md.obsidian", "Obsidian", 99)

        self.assertEqual(front_app_bundle_id(workspace=workspace), "md.obsidian")
        self.assertEqual(workspace.calls, 1)

    def test_front_app_bundle_id_is_none_without_a_frontmost_app(self):
        self.assertIsNone(front_app_bundle_id(workspace=FakeWorkspace()))

    def test_module_imports_pyobjc_lazily(self):
        """No AppKit/ApplicationServices import at module scope: imports anywhere."""
        tree = ast.parse(CONTEXT_SOURCE.read_text(encoding="utf-8"))
        top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = set()
        for node in top_level:
            if isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
            else:
                names.update(alias.name for alias in node.names)
        self.assertFalse(
            names & {"AppKit", "ApplicationServices", "Foundation", "Quartz"},
            f"PyObjC must be imported inside functions, found top-level: {names}",
        )


class ModeMappingTests(unittest.TestCase):
    def test_exact_matches_cover_each_mode_family(self):
        cases = {
            "com.apple.mail": "mail",
            "com.readdle.smartemail-Mac": "mail",
            "com.apple.MobileSMS": "message",
            "com.tinyspeck.slackmacgap": "message",
            "net.whatsapp.WhatsApp": "message",
            "com.apple.Notes": "notes",
            "md.obsidian": "notes",
            "com.apple.Terminal": "code",
            "com.mitchellh.ghostty": "code",
            "com.googlecode.iterm2": "code",
            "com.todesktop.230313mzl4w4u92": "code",
            "com.apple.dt.Xcode": "code",
        }
        for bundle_id, expected in cases.items():
            with self.subTest(bundle_id=bundle_id):
                self.assertEqual(default_mode_for_bundle(bundle_id), expected)

    def test_prefix_rule_matches_jetbrains_ides(self):
        self.assertEqual(default_mode_for_bundle("com.jetbrains.pycharm"), "code")
        self.assertEqual(default_mode_for_bundle("com.jetbrains.intellij.ce"), "code")

    def test_unknown_and_empty_bundle_ids_have_no_table_mode(self):
        self.assertIsNone(default_mode_for_bundle("com.example.unknown"))
        self.assertIsNone(default_mode_for_bundle(None))
        self.assertIsNone(default_mode_for_bundle(""))

    def test_exact_match_wins_over_prefix(self):
        # Sanity: a prefix must never shadow a more specific exact entry.
        for bundle_id, mode in DEFAULT_MODE_BY_BUNDLE.items():
            with self.subTest(bundle_id=bundle_id):
                self.assertEqual(default_mode_for_bundle(bundle_id), mode)

    def test_table_values_are_known_modes(self):
        for mode in list(DEFAULT_MODE_BY_BUNDLE.values()) + list(
            DEFAULT_MODE_BY_BUNDLE_PREFIX.values()
        ):
            self.assertIn(mode, KNOWN_MODES)

    def test_is_terminal_or_editor(self):
        self.assertTrue(is_terminal_or_editor("com.apple.Terminal"))
        self.assertTrue(is_terminal_or_editor("com.jetbrains.goland"))
        self.assertFalse(is_terminal_or_editor("com.apple.mail"))
        self.assertFalse(is_terminal_or_editor(None))


class BundleIdHygieneTests(unittest.TestCase):
    def test_every_id_is_classified_as_verified_or_unverified(self):
        table_ids = set(DEFAULT_MODE_BY_BUNDLE) | {
            prefix.rstrip(".") for prefix in DEFAULT_MODE_BY_BUNDLE_PREFIX
        }
        self.assertEqual(table_ids, VERIFIED_BUNDLE_IDS | UNVERIFIED_BUNDLE_IDS)
        self.assertFalse(VERIFIED_BUNDLE_IDS & UNVERIFIED_BUNDLE_IDS)

    def test_unverified_ids_are_still_valid_reverse_dns(self):
        self.assertTrue(UNVERIFIED_BUNDLE_IDS)
        for bundle_id in sorted(UNVERIFIED_BUNDLE_IDS):
            with self.subTest(bundle_id=bundle_id):
                self.assertRegex(bundle_id, REVERSE_DNS)

    def test_verified_ids_are_valid_reverse_dns(self):
        for bundle_id in sorted(VERIFIED_BUNDLE_IDS):
            with self.subTest(bundle_id=bundle_id):
                self.assertRegex(bundle_id, REVERSE_DNS)

    def test_prefix_rules_end_with_a_dot(self):
        for prefix in DEFAULT_MODE_BY_BUNDLE_PREFIX:
            self.assertTrue(prefix.endswith("."), prefix)


class ResolveModeTests(unittest.TestCase):
    def test_table_wins_over_the_configured_default(self):
        context = AppContext("com.apple.mail", "Mail", None, None)
        self.assertEqual(resolve_mode(context, {MODE_CONFIG_KEY: "dictation"}), "mail")

    def test_unknown_app_falls_back_to_the_configured_default(self):
        context = AppContext("com.example.unknown", "Unknown", None, None)
        self.assertEqual(resolve_mode(context, {MODE_CONFIG_KEY: "notes"}), "notes")

    def test_unknown_app_without_config_uses_dictation(self):
        context = AppContext("com.example.unknown", "Unknown", None, None)
        self.assertEqual(resolve_mode(context, {}), DEFAULT_MODE)
        self.assertEqual(DEFAULT_MODE, "dictation")

    def test_no_bundle_id_uses_the_configured_default(self):
        self.assertEqual(resolve_mode(AppContext(None, None, None, None), {}), DEFAULT_MODE)

    def test_user_override_beats_the_table(self):
        context = AppContext("com.apple.mail", "Mail", None, None)
        config = {
            MODE_CONFIG_KEY: "dictation",
            MODE_OVERRIDES_CONFIG_KEY: {"com.apple.mail": "dictation"},
        }
        self.assertEqual(resolve_mode(context, config), "dictation")

    def test_override_applies_to_apps_absent_from_the_table(self):
        context = AppContext("com.example.unknown", "Unknown", None, None)
        config = {MODE_OVERRIDES_CONFIG_KEY: {"com.example.unknown": "code"}}
        self.assertEqual(resolve_mode(context, config), "code")

    def test_prefix_rule_is_reachable_through_resolve_mode(self):
        context = AppContext("com.jetbrains.rider", "Rider", None, None)
        self.assertEqual(resolve_mode(context, {MODE_CONFIG_KEY: "message"}), "code")

    def test_unknown_configured_default_is_rejected(self):
        context = AppContext("com.example.unknown", None, None, None)
        with self.assertRaises(ValueError):
            resolve_mode(context, {MODE_CONFIG_KEY: "shouty"})

    def test_unknown_configured_default_raises_unknown_mode_error(self):
        from cleanup.modes import UnknownModeError

        context = AppContext("com.example.unknown", None, None, None)
        with self.assertRaises(UnknownModeError):
            resolve_mode(context, {MODE_CONFIG_KEY: "shouty"})

    def test_unknown_override_raises_unknown_mode_error(self):
        from cleanup.modes import UnknownModeError

        context = AppContext("com.apple.mail", "Mail", None, None)
        config = {MODE_OVERRIDES_CONFIG_KEY: {"com.apple.mail": "shouty"}}
        with self.assertRaises(UnknownModeError):
            resolve_mode(context, config)

    def test_context_awareness_false_skips_the_table(self):
        context = AppContext("com.apple.mail", "Mail", None, None)
        config = {MODE_CONFIG_KEY: "notes", "context_awareness": False}
        self.assertEqual(resolve_mode(context, config), "notes")

    def test_context_awareness_false_still_honours_user_override(self):
        context = AppContext("com.apple.mail", "Mail", None, None)
        config = {
            MODE_CONFIG_KEY: "notes",
            "context_awareness": False,
            MODE_OVERRIDES_CONFIG_KEY: {"com.apple.mail": "code"},
        }
        self.assertEqual(resolve_mode(context, config), "code")

    def test_context_awareness_true_is_the_default(self):
        context = AppContext("com.apple.mail", "Mail", None, None)
        config = {MODE_CONFIG_KEY: "notes", "context_awareness": True}
        self.assertEqual(resolve_mode(context, config), "mail")


class RememberForgetTests(unittest.TestCase):
    def test_remember_mode_returns_a_copy_and_does_not_mutate(self):
        config = {MODE_CONFIG_KEY: "dictation"}
        updated = remember_mode(config, "com.apple.mail", "notes")

        self.assertEqual(updated[MODE_OVERRIDES_CONFIG_KEY], {"com.apple.mail": "notes"})
        self.assertEqual(config, {MODE_CONFIG_KEY: "dictation"})
        self.assertIsNot(updated, config)

    def test_remember_mode_does_not_mutate_the_existing_override_dict(self):
        overrides = {"com.apple.Notes": "notes"}
        config = {MODE_OVERRIDES_CONFIG_KEY: overrides}

        updated = remember_mode(config, "com.apple.mail", "mail")

        self.assertEqual(overrides, {"com.apple.Notes": "notes"})
        self.assertEqual(
            updated[MODE_OVERRIDES_CONFIG_KEY],
            {"com.apple.Notes": "notes", "com.apple.mail": "mail"},
        )

    def test_remember_mode_replaces_an_existing_override(self):
        config = remember_mode({}, "com.apple.mail", "mail")
        config = remember_mode(config, "com.apple.mail", "dictation")
        self.assertEqual(config[MODE_OVERRIDES_CONFIG_KEY], {"com.apple.mail": "dictation"})

    def test_remember_mode_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            remember_mode({}, "", "mail")
        with self.assertRaises(ValueError):
            remember_mode({}, "com.apple.mail", "shouty")

    def test_forget_mode_returns_a_copy_and_does_not_mutate(self):
        overrides = {"com.apple.mail": "notes", "com.apple.Notes": "notes"}
        config = {MODE_OVERRIDES_CONFIG_KEY: overrides}

        updated = forget_mode(config, "com.apple.mail")

        self.assertEqual(updated[MODE_OVERRIDES_CONFIG_KEY], {"com.apple.Notes": "notes"})
        self.assertEqual(overrides, {"com.apple.mail": "notes", "com.apple.Notes": "notes"})

    def test_forget_mode_on_an_unknown_app_is_a_no_op(self):
        updated = forget_mode({MODE_OVERRIDES_CONFIG_KEY: {}}, "com.example.unknown")
        self.assertEqual(updated[MODE_OVERRIDES_CONFIG_KEY], {})

    def test_forget_mode_restores_the_table_default(self):
        context = AppContext("com.apple.mail", "Mail", None, None)
        config = remember_mode({}, "com.apple.mail", "dictation")
        self.assertEqual(resolve_mode(context, config), "dictation")
        self.assertEqual(resolve_mode(context, forget_mode(config, "com.apple.mail")), "mail")


if __name__ == "__main__":
    unittest.main()
