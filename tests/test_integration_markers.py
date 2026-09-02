#!/usr/bin/env python3
"""Guards on the integration suite's contract, cheap enough for the unit run.

The integration tests must stay *out* of the unit suite and must never fail
merely because a model or a binary is missing. Both of those are easy to break
by accident — one ``__init__.py``, one forgotten ``skipUnless`` — and neither
break is visible until CI is slow or red for the wrong reason. These checks
read the files rather than importing them, so the unit suite still needs no
runtime.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INTEGRATION_DIR = REPO_ROOT / "tests" / "integration"
RUNNER = REPO_ROOT / "scripts" / "tools" / "run_integration.sh"

#: The suite the plan asks for. Named, not counted, so a deletion is caught.
EXPECTED_MODULES = {
    "test_cleanup_llama_e2e.py",
    "test_engine_voxtral_e2e.py",
    "test_engine_whispercpp_e2e.py",
    "test_paste_flow.py",
}


class IntegrationSuiteLayoutTests(unittest.TestCase):
    def test_the_expected_modules_are_present(self) -> None:
        found = {path.name for path in INTEGRATION_DIR.glob("test_*.py")}
        self.assertEqual(found, EXPECTED_MODULES)

    def test_the_directory_is_not_a_package(self) -> None:
        """No ``__init__.py``: that is what keeps `discover -s tests` out of it.

        ``unittest`` only recurses into a subdirectory that is an importable
        package, so adding one here would silently pull every model-hungry
        test into the unit run.
        """
        self.assertFalse(
            (INTEGRATION_DIR / "__init__.py").exists(),
            "tests/integration must not be a package, or the unit suite "
            "would discover the integration tests too",
        )

    def test_every_module_skips_instead_of_failing_without_its_runtime(self) -> None:
        for path in sorted(INTEGRATION_DIR.glob("test_*.py")):
            with self.subTest(module=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn(
                    "skipUnless",
                    source,
                    f"{path.name} has no skipUnless: a machine without the "
                    "runtime would fail rather than skip",
                )
                self.assertIn(
                    "_SKIP_REASON",
                    source,
                    f"{path.name} does not build an explicit skip reason",
                )

    def test_the_runner_discovers_the_integration_directory(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("discover -s tests/integration", source)


if __name__ == "__main__":
    unittest.main()
