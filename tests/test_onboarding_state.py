"""Pure-state tests for the first-run onboarding wizard.

Nothing here touches AppKit: `ui.onboarding_window` must import on any machine,
and every permission probe, download and test recording is an injected
callable.
"""

import subprocess
import sys
import types
import unittest
from pathlib import Path

from ui.onboarding_window import (
    ONBOARDING_VERSION,
    STATUS_DENIED,
    STATUS_DOWNLOADING,
    STATUS_GRANTED,
    STATUS_PENDING,
    STATUS_READY,
    STATUS_SKIPPED,
    STEP_ACCESSIBILITY,
    STEP_DONE,
    STEP_ENGINE,
    STEP_MICROPHONE,
    STEP_TEST,
    STEP_WELCOME,
    STEPS,
    STRINGS,
    EngineChoice,
    OnboardingCallbacks,
    OnboardingState,
    format_size,
    should_show,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

ENGINE = EngineChoice(
    engine_id="whispercpp",
    model_id="whispercpp-large-v3-turbo-q5_0",
    display_name="Whisper large-v3-turbo (quantised)",
    size_bytes=574041195,
)


def _tick(bytes_done, bytes_total, *, done=False, name="model.bin"):
    """A stand-in for engines.model_store.DownloadProgress."""
    return types.SimpleNamespace(
        model_id=ENGINE.model_id,
        file_name=name,
        bytes_done=bytes_done,
        bytes_total=bytes_total,
        done=done,
    )


def _state(**kwargs):
    kwargs.setdefault("engine_choice", ENGINE)
    kwargs.setdefault("is_installed", lambda model_id: False)
    return OnboardingState(**kwargs)


class ModuleImportTests(unittest.TestCase):
    def test_module_imports_without_appkit(self):
        """A clean interpreter must import the module with no PyObjC loaded."""
        code = (
            "import sys, ui;"
            "before = set(sys.modules);"
            "import ui.onboarding_window;"
            "added = set(sys.modules) - before;"
            "print(sorted(n for n in added if n.split('.')[0] "
            "in ('AppKit', 'Cocoa', 'objc', 'Quartz', 'Foundation')))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]", result.stdout)


class StepOrderTests(unittest.TestCase):
    def test_steps_are_in_the_documented_order(self):
        self.assertEqual(
            STEPS,
            (
                STEP_WELCOME,
                STEP_MICROPHONE,
                STEP_ACCESSIBILITY,
                STEP_ENGINE,
                STEP_TEST,
                STEP_DONE,
            ),
        )

    def test_every_step_has_copy(self):
        for step in STEPS:
            self.assertIn(step, STRINGS)
            self.assertTrue(STRINGS[step]["title"])
            self.assertTrue(STRINGS[step]["body"])

    def test_fresh_state_starts_on_welcome_with_everything_pending(self):
        state = _state()
        self.assertEqual(state.current_step, STEP_WELCOME)
        self.assertEqual(state.step_number, 1)
        self.assertEqual(state.step_count, len(STEPS))
        for step in STEPS:
            self.assertEqual(state.status(step), STATUS_PENDING)

    def test_installed_model_starts_the_engine_step_ready(self):
        state = _state(is_installed=lambda model_id: True)
        self.assertEqual(state.status(STEP_ENGINE), STATUS_READY)

    def test_advance_walks_forward_and_stops_at_done(self):
        state = _state()
        seen = [state.current_step]
        for _ in range(len(STEPS) + 3):
            seen.append(state.advance())
        self.assertEqual(seen[: len(STEPS)], list(STEPS))
        self.assertEqual(state.current_step, STEP_DONE)

    def test_advance_marks_welcome_ready_and_pending_work_skipped(self):
        state = _state()
        state.advance()
        self.assertEqual(state.status(STEP_WELCOME), STATUS_READY)
        state.advance()
        self.assertEqual(state.status(STEP_MICROPHONE), STATUS_SKIPPED)

    def test_advance_does_not_overwrite_a_resolved_status(self):
        state = _state(request_microphone=lambda: True)
        state.advance()
        state.request_microphone()
        state.advance()
        self.assertEqual(state.status(STEP_MICROPHONE), STATUS_GRANTED)

    def test_skip_marks_the_current_step_and_moves_on(self):
        state = _state()
        self.assertEqual(state.skip(), STEP_MICROPHONE)
        self.assertEqual(state.status(STEP_WELCOME), STATUS_SKIPPED)

    def test_back_returns_without_changing_status(self):
        state = _state()
        state.advance()
        state.skip()
        self.assertEqual(state.current_step, STEP_ACCESSIBILITY)
        self.assertEqual(state.back(), STEP_MICROPHONE)
        self.assertEqual(state.status(STEP_MICROPHONE), STATUS_SKIPPED)
        self.assertEqual(state.back(), STEP_WELCOME)
        self.assertEqual(state.back(), STEP_WELCOME)
        self.assertFalse(state.can_go_back)


class PermissionTests(unittest.TestCase):
    def test_granted_microphone_records_granted(self):
        calls = []
        state = _state(request_microphone=lambda: calls.append("mic") or True)
        self.assertTrue(state.request_microphone())
        self.assertEqual(calls, ["mic"])
        self.assertEqual(state.status(STEP_MICROPHONE), STATUS_GRANTED)

    def test_denied_microphone_records_denied(self):
        state = _state(request_microphone=lambda: False)
        self.assertFalse(state.request_microphone())
        self.assertEqual(state.status(STEP_MICROPHONE), STATUS_DENIED)

    def test_accessibility_probe_is_injected(self):
        calls = []
        state = _state(request_accessibility=lambda: calls.append("ax") or False)
        self.assertFalse(state.request_accessibility())
        self.assertEqual(calls, ["ax"])
        self.assertEqual(state.status(STEP_ACCESSIBILITY), STATUS_DENIED)

    def test_a_raising_probe_is_denied_and_recorded(self):
        def boom():
            raise OSError("TCC is unavailable")

        state = _state(request_microphone=boom)
        self.assertFalse(state.request_microphone())
        self.assertEqual(state.status(STEP_MICROPHONE), STATUS_DENIED)
        self.assertIn("TCC", state.error(STEP_MICROPHONE))

    def test_denied_permissions_still_allow_finishing(self):
        state = _state(
            request_microphone=lambda: False,
            request_accessibility=lambda: False,
        )
        state.advance()
        state.request_microphone()
        state.advance()
        state.request_accessibility()
        state.advance()
        state.skip()
        state.skip()
        self.assertEqual(state.current_step, STEP_DONE)
        self.assertTrue(state.can_finish)
        self.assertEqual(state.status(STEP_MICROPHONE), STATUS_DENIED)
        self.assertEqual(state.status(STEP_ENGINE), STATUS_SKIPPED)

    def test_a_fresh_state_cannot_finish(self):
        self.assertFalse(_state().can_finish)

    def test_the_install_probe_decides_the_engine_step_without_a_download(self):
        asked = []
        state = OnboardingState(
            engine_choice=ENGINE,
            is_installed=lambda model_id: asked.append(model_id) or True,
        )
        self.assertEqual(asked, [ENGINE.model_id])
        self.assertEqual(state.status(STEP_ENGINE), STATUS_READY)


class DownloadTests(unittest.TestCase):
    def test_progress_updates_status_and_fraction(self):
        observed = []

        def fake_download(model_id, progress, cancel):
            self.assertEqual(model_id, ENGINE.model_id)
            for done in (10, 50, 100):
                progress(_tick(done, 100, done=done == 100))
            return "/models/" + model_id

        state = _state(download=fake_download)
        state.on_download_progress = lambda tick: observed.append(
            (state.status(STEP_ENGINE), state.download_fraction)
        )
        self.assertTrue(state.start_download())
        self.assertEqual([status for status, _ in observed], [STATUS_DOWNLOADING] * 3)
        self.assertEqual([round(f, 2) for _, f in observed], [0.1, 0.5, 1.0])
        self.assertEqual(state.status(STEP_ENGINE), STATUS_READY)

    def test_failed_download_leaves_the_step_pending_with_an_error(self):
        def fake_download(model_id, progress, cancel):
            progress(_tick(5, 100))
            raise OSError("connection reset")

        state = _state(download=fake_download)
        self.assertFalse(state.start_download())
        self.assertEqual(state.status(STEP_ENGINE), STATUS_PENDING)
        self.assertIn("connection reset", state.error(STEP_ENGINE))

    def test_cancel_sets_the_event_handed_to_the_downloader(self):
        seen = {}

        def fake_download(model_id, progress, cancel):
            seen["cancel"] = cancel
            raise RuntimeError("download cancelled")

        state = _state(download=fake_download)
        state.cancel_download()
        state.start_download()
        self.assertTrue(seen["cancel"].is_set())

    def test_engine_summary_uses_the_injected_choice(self):
        state = _state()
        self.assertEqual(state.engine_choice, ENGINE)
        self.assertIn(ENGINE.display_name, state.engine_summary)
        self.assertIn("547", state.engine_summary)

    def test_format_size(self):
        self.assertEqual(format_size(0), "unknown size")
        self.assertEqual(format_size(1024 * 1024), "1 MB")
        self.assertEqual(format_size(3 * 1024**3), "3.0 GB")


class TestStepTests(unittest.TestCase):
    def test_try_it_fills_the_field_from_the_injected_recorder(self):
        state = _state(record_and_transcribe=lambda: "  hello Murmur  ")
        self.assertEqual(state.run_test(), "hello Murmur")
        self.assertEqual(state.test_text, "hello Murmur")
        self.assertEqual(state.status(STEP_TEST), STATUS_READY)

    def test_empty_transcript_leaves_the_step_pending(self):
        state = _state(record_and_transcribe=lambda: "")
        self.assertEqual(state.run_test(), "")
        self.assertEqual(state.status(STEP_TEST), STATUS_PENDING)

    def test_a_raising_recorder_is_reported_not_raised(self):
        def boom():
            raise RuntimeError("no engine loaded")

        state = _state(record_and_transcribe=boom)
        self.assertEqual(state.run_test(), "")
        self.assertIn("no engine loaded", state.error(STEP_TEST))
        self.assertEqual(state.status(STEP_TEST), STATUS_PENDING)


class ConfigTests(unittest.TestCase):
    def test_to_config_keys(self):
        state = _state()
        self.assertEqual(
            state.to_config(),
            {"onboarding_completed": True, "onboarding_version": ONBOARDING_VERSION},
        )

    def test_should_show_when_never_completed(self):
        self.assertTrue(should_show({}))
        self.assertTrue(should_show({"onboarding_completed": False}))

    def test_should_show_for_an_older_version(self):
        self.assertTrue(
            should_show(
                {
                    "onboarding_completed": True,
                    "onboarding_version": ONBOARDING_VERSION - 1,
                }
            )
        )

    def test_should_not_show_for_the_current_version(self):
        self.assertFalse(
            should_show(
                {
                    "onboarding_completed": True,
                    "onboarding_version": ONBOARDING_VERSION,
                }
            )
        )

    def test_should_show_for_a_corrupt_version(self):
        self.assertTrue(
            should_show({"onboarding_completed": True, "onboarding_version": "one"})
        )
        self.assertTrue(should_show({"onboarding_completed": True}))

    def test_summary_reports_every_step_but_done(self):
        state = _state(request_microphone=lambda: True)
        state.request_microphone()
        state.skip()
        summary = state.summary()
        self.assertEqual(len(summary), len(STEPS) - 1)
        titles = [title for title, _ in summary]
        self.assertIn(STRINGS[STEP_MICROPHONE]["title"], titles)
        self.assertIn(STRINGS["status"][STATUS_GRANTED], [label for _, label in summary])


class CallbackWiringTests(unittest.TestCase):
    def test_state_from_callbacks_uses_every_injected_callable(self):
        calls = []
        callbacks = OnboardingCallbacks(
            request_microphone=lambda: calls.append("mic") or True,
            request_accessibility=lambda: calls.append("ax") or True,
            download=lambda model_id, progress, cancel: calls.append("download"),
            record_and_transcribe=lambda: calls.append("record") or "hi",
            open_settings=lambda: calls.append("settings"),
            on_finished=lambda updates: calls.append(updates),
            is_installed=lambda model_id: False,
        )
        state = OnboardingState.from_callbacks(callbacks, engine_choice=ENGINE)
        state.request_microphone()
        state.request_accessibility()
        state.start_download()
        state.run_test()
        callbacks.open_settings()
        callbacks.on_finished(state.to_config())

        self.assertEqual(calls[:4], ["mic", "ax", "download", "record"])
        self.assertEqual(calls[4], "settings")
        self.assertEqual(calls[5]["onboarding_version"], ONBOARDING_VERSION)

    def test_callbacks_expose_the_documented_fields(self):
        fields = set(OnboardingCallbacks.__dataclass_fields__)
        self.assertTrue(
            {
                "request_microphone",
                "request_accessibility",
                "download",
                "record_and_transcribe",
                "open_settings",
                "on_finished",
            }
            <= fields
        )


if __name__ == "__main__":
    unittest.main()
