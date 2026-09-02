"""One writer owns the config file, and it merges.

The same bug shipped twice. Wave 1 had the Settings window load the whole
config when it opened and save that snapshot back when the user flipped a
switch, reverting the ``engine_id`` the app had written in between. Wave 3's
new tabs did it again, because the API still offered a whole-config writer and
it was the obvious thing to call.

So the whole-config writer is private now
(:meth:`services.persistence_service.PersistenceService._save_config_snapshot`)
and everyone else goes through ``update_config``, which reads, merges and
writes under one lock. This test is what keeps it that way: it is a structural
check, not a behavioural one, and it fails the moment a third wave reaches for
the snapshot again.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The one module allowed to write a whole config: it owns the file, and its
#: own first-run/default path is the single legitimate snapshot write.
OWNER = ROOT / "services" / "persistence_service.py"

#: Directories that are not this app's source.
SKIPPED_DIRS = frozenset({"__pycache__", "_archive", ".git", ".worktrees", "venv", "build", "dist"})

#: Function names that write a whole config in one go.
SNAPSHOT_WRITERS = frozenset({"save_config", "_save_config_snapshot"})


def python_files():
    """Every first-party module, the config file's owner excluded."""
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in SKIPPED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path == OWNER:
            continue
        yield path


def called_names(tree):
    """``(name, lineno)`` for every call in ``tree``, however it is spelled."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            yield func.attr, node.lineno
        elif isinstance(func, ast.Name):
            yield func.id, node.lineno


class SnapshotWriteTests(unittest.TestCase):
    def test_no_module_outside_persistence_saves_a_config_snapshot(self):
        offenders = []
        for path in python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name, lineno in called_names(tree):
                if name in SNAPSHOT_WRITERS:
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno} calls {name}()")
        self.assertEqual(
            offenders,
            [],
            "these write a whole config back and silently revert every key "
            "another part of the app changed in the meantime; call "
            "persistence.update_config({...}) with the keys you actually own",
        )

    def test_the_snapshot_writer_is_private_and_the_merge_writer_is_public(self):
        source = OWNER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        service = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PersistenceService"
        )
        methods = {
            node.name
            for node in service.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("_save_config_snapshot", methods)
        self.assertIn("update_config", methods)
        self.assertIn("load_config", methods)
        self.assertNotIn(
            "save_config",
            methods,
            "a public whole-config writer is the bug this test exists for",
        )


if __name__ == "__main__":
    unittest.main()
