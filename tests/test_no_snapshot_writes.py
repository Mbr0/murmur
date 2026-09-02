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


#: The one method that puts bytes on disk for any of Murmur's JSON files. Called
#: with the config path outside its owner, it is a whole-config write wearing
#: another name — the merge never happens and the lock is never taken.
JSON_WRITER = "_save_json_file"


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


def _call_name(node):
    """The plain name a call is spelled with, or None."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def getattr_snapshot_names(tree):
    """``(name, lineno)`` for every ``getattr(x, "save_config")`` spelling.

    ``called_names`` sees ``getattr`` and nothing else, so a snapshot writer
    reached by name — the shape ``close``-style optional protocols already use
    elsewhere in this app — walked straight past the guard.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) < 2:
            continue
        name = node.args[1]
        if isinstance(name, ast.Constant) and name.value in SNAPSHOT_WRITERS:
            yield name.value, node.lineno


def config_file_json_writes(tree):
    """``lineno`` for every ``_save_json_file(<something config_file>, …)``.

    The layer below the snapshot writer. Handed the config path it writes the
    whole file with no merge and no lock, which is the very bug the private
    snapshot writer exists to contain — just spelled one level down.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) != JSON_WRITER or not node.args:
            continue
        if "config_file" in ast.dump(node.args[0]):
            yield node.lineno


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

    def test_no_module_reaches_a_snapshot_writer_by_name(self):
        offenders = []
        for path in python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name, lineno in getattr_snapshot_names(tree):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno} looks up {name}")
        self.assertEqual(
            offenders,
            [],
            "spelling the snapshot writer as a string is still calling it; "
            "call persistence.update_config({...}) with the keys you own",
        )

    def test_no_module_outside_persistence_writes_the_config_file_itself(self):
        offenders = []
        for path in python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for lineno in config_file_json_writes(tree):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno} writes the config file")
        self.assertEqual(
            offenders,
            [],
            f"{JSON_WRITER}() on the config path writes the whole file with no "
            "merge and no lock; call persistence.update_config({...}) instead",
        )

    def test_the_guard_actually_catches_the_shapes_it_names(self):
        """A structural guard that matches nothing is a test that passes for
        the wrong reason, so each detector is shown a source it must reject."""
        by_name = ast.parse(
            "handler = getattr(persistence, '_save_config_snapshot', None)\n"
            "public = getattr(service, 'save_config')\n"
        )
        self.assertEqual(
            [name for name, _ in getattr_snapshot_names(by_name)],
            ["_save_config_snapshot", "save_config"],
        )
        direct = ast.parse(
            "service._save_json_file(self._paths.config_file, config)\n"
            "service._save_json_file(paths.history_file, history)\n"
        )
        self.assertEqual(list(config_file_json_writes(direct)), [1])
        # And the ordinary spellings are still caught by the original walk.
        self.assertIn(
            "_save_config_snapshot",
            [name for name, _ in called_names(ast.parse("app.persistence._save_config_snapshot(c)"))],
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
