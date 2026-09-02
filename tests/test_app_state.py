import unittest
from types import SimpleNamespace

from murmur import (
    MISSING_MODEL_ONBOARDING,
    MISSING_MODEL_SETTINGS,
    NO_MODEL_STATUS,
    RELOAD_BUSY,
    RELOAD_RECORDING,
    RELOAD_START,
    RELOAD_UNCHANGED,
    about_menu_title,
    clear_mic_device_selection,
    download_progress_status,
    engine_is_ready,
    finalize_transcript,
    hints_notice_message,
    missing_model_action,
    model_integrity_message,
    model_status_title,
    model_unavailable_message,
    push_to_talk_degraded_message,
    reload_engine_decision,
    remember_hints_notice,
    resolve_engine_selection,
    resolve_mic_device,
    resolve_mic_device_index,
    should_apply_ready_on_reset,
    should_reject_toggle,
    should_reject_upload,
    should_relaunch_after_install,
    should_show_hints_notice,
    skip_audio_user_message,
    should_toggle_for_press_action,
    update_available_message,
    update_installed_message,
    update_relaunch_failed_message,
    verify_model_before_load,
)
from engines.model_store import ModelIntegrityError
from services.hotkey_service import (
    ACTION_START,
    ACTION_STOP,
    HOTKEY_MODE_AUTO,
    HOTKEY_MODE_HOLD,
    HOTKEY_MODE_TOGGLE,
)
from ui.onboarding_window import CONFIG_KEY_COMPLETED, CONFIG_KEY_VERSION, ONBOARDING_VERSION


class AppStateGuardTests(unittest.TestCase):
    def test_should_reject_toggle_while_loading_or_processing(self):
        self.assertTrue(
            should_reject_toggle(loading=True, is_processing=False, model_ready=True)
        )
        self.assertTrue(
            should_reject_toggle(loading=False, is_processing=True, model_ready=True)
        )
        self.assertFalse(
            should_reject_toggle(loading=False, is_processing=False, model_ready=True)
        )

    def test_should_reject_toggle_when_model_unavailable(self):
        self.assertTrue(
            should_reject_toggle(loading=False, is_processing=False, model_ready=False)
        )

    def test_should_reject_upload_while_busy(self):
        self.assertTrue(
            should_reject_upload(
                loading=True, is_recording=False, is_processing=False, model_ready=True
            )
        )
        self.assertTrue(
            should_reject_upload(
                loading=False, is_recording=True, is_processing=False, model_ready=True
            )
        )
        self.assertTrue(
            should_reject_upload(
                loading=False, is_recording=False, is_processing=True, model_ready=True
            )
        )
        self.assertFalse(
            should_reject_upload(
                loading=False, is_recording=False, is_processing=False, model_ready=True
            )
        )

    def test_should_reject_upload_when_model_unavailable(self):
        self.assertTrue(
            should_reject_upload(
                loading=False, is_recording=False, is_processing=False, model_ready=False
            )
        )

    def test_should_not_force_ready_while_recording(self):
        self.assertFalse(should_apply_ready_on_reset(is_recording=True))
        self.assertTrue(should_apply_ready_on_reset(is_recording=False))


class EngineReadinessTests(unittest.TestCase):
    """The engine is built inside load_model(), so it is None until that runs."""

    def test_engine_is_not_ready_before_it_is_constructed(self):
        self.assertFalse(engine_is_ready(None))

    def test_engine_is_not_ready_until_it_is_loaded(self):
        self.assertFalse(engine_is_ready(SimpleNamespace(is_loaded=False)))

    def test_engine_is_ready_once_loaded(self):
        self.assertTrue(engine_is_ready(SimpleNamespace(is_loaded=True)))


class MicDeviceConfigTests(unittest.TestCase):
    def test_resolve_mic_unset_uses_system_default(self):
        self.assertIsNone(resolve_mic_device_index(None, {0, 1, 2}))
        self.assertIsNone(resolve_mic_device(None, None, {0: "Built-in"}))

    def test_resolve_mic_valid_index(self):
        self.assertEqual(resolve_mic_device_index(1, {0, 1, 2}), 1)

    def test_resolve_mic_invalid_index_fails_fast(self):
        with self.assertRaises(ValueError):
            resolve_mic_device_index(9, {0, 1, 2})

    def test_resolve_mic_rejects_bool_and_bad_types(self):
        with self.assertRaises(ValueError):
            resolve_mic_device_index(True, {0, 1})
        with self.assertRaises(ValueError):
            resolve_mic_device_index("1", {0, 1})

    def test_resolve_mic_rejects_wrong_device_at_same_index(self):
        devices = {0: "Built-in Mic", 1: "Other Mic"}
        with self.assertRaises(ValueError):
            resolve_mic_device(1, "USB Mic", devices)

    def test_resolve_mic_index_and_name_match(self):
        devices = {0: "Built-in Mic", 1: "USB Mic"}
        self.assertEqual(resolve_mic_device(1, "USB Mic", devices), 1)

    def test_resolve_mic_index_drift_resolves_by_name(self):
        devices = {0: "USB Mic", 2: "Built-in Mic"}
        self.assertEqual(resolve_mic_device(1, "USB Mic", devices), 0)
        # Saved index points at a different device; must not keep index 1.
        devices_swapped = {0: "USB Mic", 1: "Built-in Mic"}
        self.assertEqual(resolve_mic_device(1, "USB Mic", devices_swapped), 0)

    def test_resolve_mic_without_name_uses_index(self):
        devices = {0: "Built-in Mic", 1: "USB Mic"}
        self.assertEqual(resolve_mic_device(1, None, devices), 1)

    def test_clear_mic_device_selection(self):
        cleared = clear_mic_device_selection(
            {"mic_device_index": 2, "mic_device_name": "USB Mic", "model": "base"}
        )
        self.assertIsNone(cleared["mic_device_index"])
        self.assertIsNone(cleared["mic_device_name"])
        self.assertEqual(cleared["model"], "base")


class SkipAudioMessageTests(unittest.TestCase):
    def test_short_vs_quiet_messages(self):
        self.assertIn("short", skip_audio_user_message(0.4, 0.5).lower())
        self.assertIn("quiet", skip_audio_user_message(2.0, 0.001).lower())


class EngineSelectionTests(unittest.TestCase):
    """Which engine and model load, and when config is written back."""

    CATALOG = {
        "whispercpp": ("whispercpp-turbo-q5", "whispercpp-turbo"),
        "voxtral_mlx": ("voxtral-4bit",),
        "empty_engine": (),
    }

    def _resolve(self, config, default_engine_id="whispercpp"):
        return resolve_engine_selection(
            config,
            default_engine_id=default_engine_id,
            model_ids_for_engine=self.CATALOG.__getitem__,
        )

    def test_config_with_both_keys_is_honoured_and_not_rewritten(self):
        selection = self._resolve(
            {"engine_id": "voxtral_mlx", "model_id": "voxtral-4bit"}
        )
        self.assertEqual(selection.engine_id, "voxtral_mlx")
        self.assertEqual(selection.model_id, "voxtral-4bit")
        self.assertFalse(selection.needs_persist)
        self.assertFalse(selection.from_legacy_model_key)

    def test_empty_config_falls_back_to_this_machine_default(self):
        selection = self._resolve({}, default_engine_id="voxtral_mlx")
        self.assertEqual(selection.engine_id, "voxtral_mlx")
        self.assertEqual(selection.model_id, "voxtral-4bit")
        self.assertTrue(selection.needs_persist)
        self.assertFalse(selection.from_legacy_model_key)

    def test_missing_model_keeps_the_configured_engine(self):
        selection = self._resolve(
            {"engine_id": "voxtral_mlx", "model_id": None}, default_engine_id="whispercpp"
        )
        self.assertEqual(selection.engine_id, "voxtral_mlx")
        self.assertEqual(selection.model_id, "voxtral-4bit")
        self.assertTrue(selection.needs_persist)

    def test_missing_engine_takes_the_default_engines_first_model(self):
        selection = self._resolve({"model_id": None}, default_engine_id="whispercpp")
        self.assertEqual(selection.engine_id, "whispercpp")
        self.assertEqual(selection.model_id, "whispercpp-turbo-q5")

    def test_legacy_model_key_migrates_once_and_is_flagged(self):
        selection = self._resolve({"model": "medium"}, default_engine_id="whispercpp")
        self.assertEqual(selection.engine_id, "whispercpp")
        self.assertEqual(selection.model_id, "whispercpp-turbo-q5")
        self.assertTrue(selection.needs_persist)
        self.assertTrue(selection.from_legacy_model_key)

    def test_legacy_key_alongside_a_real_choice_is_not_a_migration(self):
        selection = self._resolve(
            {"model": "medium", "engine_id": "whispercpp", "model_id": "whispercpp-turbo"}
        )
        self.assertFalse(selection.needs_persist)
        self.assertFalse(selection.from_legacy_model_key)

    def test_engine_without_a_catalog_model_fails_fast(self):
        with self.assertRaises(ValueError):
            self._resolve({}, default_engine_id="empty_engine")

    def test_blank_config_values_are_treated_as_missing(self):
        selection = self._resolve({"engine_id": "", "model_id": ""})
        self.assertEqual(selection.engine_id, "whispercpp")
        self.assertEqual(selection.model_id, "whispercpp-turbo-q5")
        self.assertTrue(selection.needs_persist)


class MissingModelTests(unittest.TestCase):
    """No model on disk sends the user somewhere useful, never to a fallback."""

    def test_first_run_gets_the_wizard(self):
        self.assertEqual(missing_model_action({}), MISSING_MODEL_ONBOARDING)

    def test_a_finished_wizard_gets_settings(self):
        config = {
            CONFIG_KEY_COMPLETED: True,
            CONFIG_KEY_VERSION: ONBOARDING_VERSION,
        }
        self.assertEqual(missing_model_action(config), MISSING_MODEL_SETTINGS)

    def test_status_title_names_the_model_or_says_there_is_none(self):
        self.assertEqual(model_status_title("Whisper large-v3-turbo"), "Model: Whisper large-v3-turbo")
        self.assertEqual(model_status_title(None), NO_MODEL_STATUS)

    def test_unavailable_message_repeats_the_reason_when_there_is_one(self):
        message = model_unavailable_message(NO_MODEL_STATUS)
        self.assertIn(NO_MODEL_STATUS, message)
        self.assertIn("Settings", message)
        self.assertIn("load", model_unavailable_message(None))


class ReloadEngineDecisionTests(unittest.TestCase):
    """The engine swap refuses rather than queues; every refusal has a reason."""

    def _decide(self, **overrides):
        kwargs = dict(
            requested=("whispercpp", "turbo"),
            active=("voxtral_mlx", "voxtral-4bit"),
            is_reloading=False,
            is_recording=False,
            is_processing=False,
            engine_ready=True,
        )
        kwargs.update(overrides)
        return reload_engine_decision(**kwargs)

    def test_idle_app_starts_the_swap(self):
        self.assertEqual(self._decide(), RELOAD_START)

    def test_recording_blocks_the_swap(self):
        self.assertEqual(self._decide(is_recording=True), RELOAD_RECORDING)

    def test_transcribing_blocks_the_swap(self):
        self.assertEqual(self._decide(is_processing=True), RELOAD_BUSY)

    def test_a_swap_already_running_blocks_another(self):
        self.assertEqual(self._decide(is_reloading=True), RELOAD_BUSY)

    def test_a_running_reload_outranks_the_recording_reason(self):
        self.assertEqual(
            self._decide(is_reloading=True, is_recording=True), RELOAD_BUSY
        )

    def test_the_same_pair_already_loaded_is_a_no_op(self):
        self.assertEqual(
            self._decide(active=("whispercpp", "turbo")), RELOAD_UNCHANGED
        )

    def test_the_same_pair_reloads_when_the_engine_never_came_up(self):
        self.assertEqual(
            self._decide(active=("whispercpp", "turbo"), engine_ready=False),
            RELOAD_START,
        )


class HintsNoticeTests(unittest.TestCase):
    """The "hints ignored" notice: once per engine, and only when it is true."""

    def test_shown_when_terms_were_given_and_the_engine_ignored_them(self):
        self.assertTrue(
            should_show_hints_notice({}, "voxtral_mlx", hints_applied=False, has_terms=True)
        )

    def test_not_shown_when_the_engine_used_the_hints(self):
        self.assertFalse(
            should_show_hints_notice({}, "whispercpp", hints_applied=True, has_terms=True)
        )

    def test_not_shown_when_the_engine_had_nothing_to_apply(self):
        self.assertFalse(
            should_show_hints_notice({}, "whispercpp", hints_applied=None, has_terms=True)
        )

    def test_not_shown_without_vocabulary_terms(self):
        self.assertFalse(
            should_show_hints_notice({}, "voxtral_mlx", hints_applied=False, has_terms=False)
        )

    def test_remembering_it_stops_the_second_showing(self):
        config = remember_hints_notice({}, "voxtral_mlx")
        self.assertFalse(
            should_show_hints_notice(config, "voxtral_mlx", hints_applied=False, has_terms=True)
        )

    def test_remembering_is_per_engine_and_does_not_mutate(self):
        original = {"vocabulary_terms": ["Boske"]}
        config = remember_hints_notice(original, "voxtral_mlx")
        self.assertNotIn("hints_notice_shown", original)
        self.assertEqual(config["vocabulary_terms"], ["Boske"])
        self.assertTrue(
            should_show_hints_notice(config, "whispercpp", hints_applied=False, has_terms=True)
        )

    def test_notice_names_the_engine_and_nothing_else(self):
        message = hints_notice_message("whisper.cpp")
        self.assertIn("whisper.cpp", message)
        self.assertIn("Vocabulary hints", message)


class AboutAndUpdateCopyTests(unittest.TestCase):
    """Menu copy for the build marker and the updater."""

    def test_source_run_has_no_build_marker(self):
        self.assertEqual(about_menu_title("1.0.0", {}), "Murmur 1.0.0")

    def test_signed_build_has_no_build_marker(self):
        self.assertEqual(about_menu_title("1.0.0", {"signed": True}), "Murmur 1.0.0")

    def test_unsigned_build_is_labelled_internal(self):
        self.assertEqual(
            about_menu_title("1.0.0", {"signed": False}), "Murmur 1.0.0 · internal build"
        )

    def test_update_offer_names_both_versions(self):
        message = update_available_message("1.1.0", "1.0.0")
        self.assertIn("1.1.0", message)
        self.assertIn("1.0.0", message)

    def test_download_status_is_a_percentage_when_the_size_is_known(self):
        self.assertIn("50%", download_progress_status(50, 100))

    def test_download_status_falls_back_to_megabytes(self):
        self.assertIn("MB", download_progress_status(3_000_000, None))

    def test_the_app_relaunches_itself_when_the_installer_did_not(self):
        """install_update leaves the new bundle unstarted on purpose: only the
        running app can shut itself down cleanly first."""
        result = SimpleNamespace(
            app_path="/Applications/Murmur.app",
            previous_path=None,
            relaunch_cmd=("open", "-n", "/Applications/Murmur.app"),
            relaunched=False,
        )
        self.assertTrue(should_relaunch_after_install(result))

    def test_no_second_launch_when_the_installer_already_started_it(self):
        result = SimpleNamespace(
            app_path="/Applications/Murmur.app",
            previous_path=None,
            relaunch_cmd=("open", "-n", "/Applications/Murmur.app"),
            relaunched=True,
        )
        self.assertFalse(should_relaunch_after_install(result))

    def test_the_handover_copy_says_it_is_restarting(self):
        message = update_installed_message("1.1.0")
        self.assertIn("1.1.0", message)
        self.assertIn("Restarting", message)
        self.assertNotIn("reopen", message.lower())

    def test_a_failed_relaunch_tells_the_user_to_reopen(self):
        message = update_relaunch_failed_message("1.1.0")
        self.assertIn("1.1.0", message)
        self.assertIn("open it again", message)


class PressActionRoutingTests(unittest.TestCase):
    """The pure seam between PressController output and toggle_recording."""

    def test_start_toggles_only_when_idle(self):
        self.assertTrue(should_toggle_for_press_action(ACTION_START, is_recording=False))
        self.assertFalse(should_toggle_for_press_action(ACTION_START, is_recording=True))

    def test_stop_toggles_only_when_recording(self):
        self.assertTrue(should_toggle_for_press_action(ACTION_STOP, is_recording=True))
        self.assertFalse(should_toggle_for_press_action(ACTION_STOP, is_recording=False))

    def test_no_action_never_toggles(self):
        self.assertFalse(should_toggle_for_press_action(None, is_recording=False))
        self.assertFalse(should_toggle_for_press_action(None, is_recording=True))

    def test_unknown_action_fails_fast(self):
        with self.assertRaises(ValueError):
            should_toggle_for_press_action("pause", is_recording=False)


class PushToTalkDegradedMessageTests(unittest.TestCase):
    def test_hold_and_auto_explain_the_fallback(self):
        for mode in (HOTKEY_MODE_HOLD, HOTKEY_MODE_AUTO):
            with self.subTest(mode=mode):
                message = push_to_talk_degraded_message(mode)
                self.assertIn(mode, message)
                self.assertIn("Accessibility", message)
                self.assertIn("toggle", message)

    def test_toggle_has_nothing_to_explain(self):
        self.assertIsNone(push_to_talk_degraded_message(HOTKEY_MODE_TOGGLE))


class FinalizeTranscriptTests(unittest.TestCase):
    """The filter must see the engine's words, not the user's rewrite of them."""

    def test_hallucination_is_detected_on_the_raw_transcript(self):
        seen = []

        def detect(value):
            seen.append(value)
            return value == "Thank you."

        text, hallucination = finalize_transcript(
            "Thank you.",
            object(),
            detect_hallucination=detect,
            replace=lambda raw, _vocab: "Dank u wel.",
        )

        self.assertEqual(seen, ["Thank you."])
        self.assertTrue(hallucination)
        self.assertEqual(text, "Dank u wel.")

    def test_a_replacement_cannot_invent_a_hallucination(self):
        text, hallucination = finalize_transcript(
            "ship the build",
            object(),
            detect_hallucination=lambda value: value == "Thank you.",
            replace=lambda _raw, _vocab: "Thank you.",
        )

        self.assertEqual(text, "Thank you.")
        self.assertFalse(hallucination)

    def test_replacements_still_run_on_a_clean_transcript(self):
        text, hallucination = finalize_transcript(
            "teh build",
            object(),
            detect_hallucination=lambda _value: False,
            replace=lambda raw, _vocab: raw.replace("teh", "the"),
        )
        self.assertEqual(text, "the build")
        self.assertFalse(hallucination)


class FakeStore:
    """Minimal ModelStore stand-in: records verify calls, optionally fails."""

    def __init__(self, error=None):
        self.error = error
        self.verified = []

    def verify(self, model_id):
        self.verified.append(model_id)
        if self.error is not None:
            raise self.error


class VerifyModelBeforeLoadTests(unittest.TestCase):
    def test_a_good_model_is_hashed_once_per_process(self):
        store = FakeStore()
        seen = set()

        verify_model_before_load(store, "turbo", seen)
        verify_model_before_load(store, "turbo", seen)

        self.assertEqual(store.verified, ["turbo"])
        self.assertIn("turbo", seen)

    def test_each_model_is_verified_on_its_own(self):
        store = FakeStore()
        seen = set()

        verify_model_before_load(store, "turbo", seen)
        verify_model_before_load(store, "voxtral", seen)

        self.assertEqual(store.verified, ["turbo", "voxtral"])

    def test_a_corrupt_model_is_refused_with_a_re_download_instruction(self):
        store = FakeStore(error=ModelIntegrityError("sha256 mismatch"))
        seen = set()

        with self.assertRaises(RuntimeError) as ctx:
            verify_model_before_load(store, "turbo", seen)

        self.assertIn("re-download", str(ctx.exception).lower())
        self.assertIn("turbo", str(ctx.exception))
        self.assertNotIn("turbo", seen)  # never cached as good

    def test_the_message_names_the_model_and_the_way_out(self):
        message = model_integrity_message("Whisper large-v3-turbo")
        self.assertIn("Whisper large-v3-turbo", message)
        self.assertIn("Settings", message)


if __name__ == "__main__":
    unittest.main()
