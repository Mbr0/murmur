"""The Wave 5 layout, asserted rather than described.

``murmur.py`` was 4,684 lines and imported AppKit, the menu bar, the engines and
every decision the app takes into one namespace. Wave 5 split it into ``app/``.
These tests are what stops it growing back: they are about *where* code lives,
not what it does, and every one of them names the failure it prevents.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
ENTRY = ROOT / "murmur.py"

#: The entry point's ceiling. Counted over *code* lines — blank lines and
#: comments are free, because a docstring that says where everything went is
#: worth more than the four lines it costs.
MAX_ENTRY_CODE_LINES = 100

#: Importing any of these pulls in PyObjC and, with it, a connection to the
#: window server. A module that does it at import scope cannot be exercised
#: headlessly, so the ones that must are marked and the rest are held to it.
GUI_MODULES = frozenset(
    {
        "AppKit",
        "ApplicationServices",
        "Cocoa",
        "Foundation",
        "PyObjCTools",
        "Quartz",
        "ServiceManagement",
        "objc",
        "rumps",
    }
)

#: The exact phrase a module's docstring must carry to be allowed one.
GUI_DECLARATION = "AppKit at import"

#: Moved into packages in Wave 5. Left at the repo root they were invisible to
#: the spec's first-party sweep and had to be listed by hand as both data files
#: and hidden imports.
MOVED_AWAY = (
    "history_window.py",
    "ui_alerts.py",
    "ui_theme.py",
    "transcription_filters.py",
)


def app_modules():
    """Every module in the app package, by path."""
    return sorted(APP_DIR.glob("*.py"))


def code_lines(path):
    """Lines that are neither blank nor a whole-line comment."""
    kept = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        kept.append(line)
    return kept


def module_scope_imports(path):
    """Top-level packages imported at module scope, with their line numbers."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module.split(".")[0], node.lineno))
    return found


class EntryPointTests(unittest.TestCase):
    """``murmur.py`` is a launcher, not a place to put things."""

    def test_the_entry_point_stays_small(self):
        lines = code_lines(ENTRY)
        self.assertLess(
            len(lines),
            MAX_ENTRY_CODE_LINES,
            f"murmur.py has {len(lines)} code lines; it launches the app and "
            "nothing else — put the change in app/ instead",
        )

    def test_the_entry_point_defines_no_app_logic(self):
        # One function (main) and nothing else. A class or a second function
        # here is the first line of the file growing back to 4,684.
        tree = ast.parse(ENTRY.read_text(encoding="utf-8"))
        defined = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        self.assertEqual(defined, ["main"])

    def test_the_app_is_reached_through_the_package(self):
        source = ENTRY.read_text(encoding="utf-8")
        self.assertIn("from app.lifecycle import MurmurApp", source)


class PackageBoundaryTests(unittest.TestCase):
    """Which way the imports point."""

    def test_no_ui_or_cleanup_module_imports_the_entry_point(self):
        # The dependency runs one way: app → ui, app → cleanup. A window that
        # imports ``murmur`` re-runs the entry point's module body under a
        # second name, which is how ``sys.modules.setdefault("murmur", ...)``
        # came to exist in the first place.
        offenders = []
        for package in ("ui", "cleanup"):
            for path in sorted((ROOT / package).rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name.split(".")[0] for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [(node.module or "").split(".")[0]]
                    else:
                        continue
                    if "murmur" in names:
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_the_old_root_modules_are_gone(self):
        still_there = [name for name in MOVED_AWAY if (ROOT / name).exists()]
        self.assertEqual(
            still_there,
            [],
            "these moved into ui/ and cleanup/ in Wave 5; a copy left at the "
            "root is the one PyInstaller would ship",
        )

    def test_what_replaced_them_is_where_it_should_be(self):
        for relative in (
            "ui/history_window.py",
            "ui/alerts.py",
            "ui/theme.py",
            "cleanup/transcription_filters.py",
        ):
            with self.subTest(relative):
                self.assertTrue((ROOT / relative).is_file())


class HeadlessImportTests(unittest.TestCase):
    """Which app modules are allowed to reach the window server at import."""

    def test_gui_imports_are_declared_in_the_docstring(self):
        for path in app_modules():
            gui = sorted(
                {
                    f"{name} (line {lineno})"
                    for name, lineno in module_scope_imports(path)
                    if name in GUI_MODULES
                }
            )
            if not gui:
                continue
            with self.subTest(path.name):
                docstring = ast.get_docstring(ast.parse(path.read_text("utf-8"))) or ""
                self.assertIn(
                    GUI_DECLARATION,
                    docstring,
                    f"app/{path.name} imports {', '.join(gui)} at module scope; "
                    f'say so with "{GUI_DECLARATION}" in its docstring, or move '
                    "the import inside the function that needs it",
                )

    def test_the_decision_layer_stays_headless(self):
        # The named two are the ones every test imports. If either grows a menu
        # bar import, importing a pure function starts a connection to the
        # window server and the suite stops running on a headless machine.
        for name in ("config.py", "decisions.py"):
            with self.subTest(name):
                gui = [
                    f"{module} (line {lineno})"
                    for module, lineno in module_scope_imports(APP_DIR / name)
                    if module in GUI_MODULES
                ]
                self.assertEqual(gui, [])


class SplitShapeTests(unittest.TestCase):
    """The package exists and carries the modules Wave 5 promised."""

    def test_every_promised_module_is_there(self):
        expected = {
            "__init__.py",
            "config.py",
            "decisions.py",
            "lifecycle.py",
            "menu.py",
            "pipeline.py",
            "services.py",
            "windows.py",
        }
        self.assertTrue(expected.issubset({path.name for path in app_modules()}))

    def test_the_package_imports_nothing_on_its_own(self):
        # ``app.config`` migrates the pre-Murmur data files at import. A package
        # __init__ that pulled it in would run that for anyone who touched the
        # name ``app`` — a test collector included.
        tree = ast.parse((APP_DIR / "__init__.py").read_text(encoding="utf-8"))
        self.assertEqual(
            [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))],
            [],
        )


if __name__ == "__main__":
    unittest.main()
