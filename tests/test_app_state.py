import unittest
from types import SimpleNamespace
from unittest.mock import patch

from murmur import (
    APP_CATALOG,
    CLEANUP_MODEL_MISSING_REASON,
    CLEANUP_OFF_DISABLED,
    CLEANUP_OFF_PASSTHROUGH,
    CLEANUP_OFF_PRO,
    CLEANUP_PREPARING_STATUS,
    CLEANUP_START_FAILED_REASON,
    CLEANUP_UNSTABLE_REASON,
    MODE_MENU_AUTOMATIC,
    PRO_OVERRIDE_KEY,
    CleanupPlan,
    CleanupRuntime,
    cleanup_model_missing_message,
    cleanup_plan,
    cleanup_skipped_message,
    code_transform_language,
    mode_menu_state,
    pro_enabled,
    prompt_language,
    run_cleanup,
    tone_menu_state,
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
from cleanup.context import AppContext
from cleanup.llama_server import CLEANUP_MODEL_SPEC, CleanupResult, LlamaServerError
from cleanup.modes import MODE_IDS, TONE_IDS
from cleanup.vocabulary import vocabulary_from_config
from engines.model_store import CATALOG, ModelIntegrityError
from services.persistence_service import DEFAULT_CONFIG
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


# ---------------------------------------------------------------------------
# Wave 2: the cleanup pipeline
# ---------------------------------------------------------------------------


def _context(bundle_id=None):
    return AppContext(
        bundle_id=bundle_id, app_name=None, window_title=None, selected_text=None
    )


def _config(**overrides):
    """A config that has cleanup fully switched on, minus the overrides."""
    base = {
        **DEFAULT_CONFIG,
        PRO_OVERRIDE_KEY: True,
        "cleanup_enabled": True,
        "context_awareness": False,
    }
    base.update(overrides)
    return base


class ProGateTests(unittest.TestCase):
    """One gate, one place. Wave 4 swaps the body; the call sites do not move."""

    def test_off_by_default(self):
        self.assertFalse(pro_enabled("cleanup", dict(DEFAULT_CONFIG)))

    def test_the_dev_override_unlocks_every_feature(self):
        config = {PRO_OVERRIDE_KEY: True}
        self.assertTrue(pro_enabled("cleanup", config))
        self.assertTrue(pro_enabled("coding_mode", config))

    def test_an_unnamed_feature_is_a_programming_error(self):
        with self.assertRaises(AssertionError):
            pro_enabled("", {})

    def test_the_override_is_not_a_user_facing_default(self):
        # It must never appear in a user's config file.
        self.assertNotIn(PRO_OVERRIDE_KEY, DEFAULT_CONFIG)


class LanguageNormalisationTests(unittest.TestCase):
    def test_auto_becomes_none_for_the_prompt(self):
        self.assertIsNone(prompt_language("auto"))
        self.assertIsNone(prompt_language(None))
        self.assertIsNone(prompt_language(""))
        self.assertIsNone(prompt_language("  AUTO "))

    def test_a_real_language_is_passed_through(self):
        self.assertEqual(prompt_language("fr"), "fr")
        self.assertEqual(prompt_language(" nl "), "nl")

    def test_the_code_transform_falls_back_to_english(self):
        self.assertEqual(code_transform_language("auto"), "en")
        self.assertEqual(code_transform_language(None), "en")
        self.assertEqual(code_transform_language("en"), "en")
        self.assertEqual(code_transform_language("fr"), "fr")
        self.assertEqual(code_transform_language("fr-CA"), "fr")

    def test_an_unsupported_language_never_reaches_the_transform(self):
        # transform_spoken_code raises on anything but en/fr, and losing a
        # transcript to that would be absurd.
        self.assertEqual(code_transform_language("de"), "en")
        self.assertEqual(code_transform_language("nl"), "en")


class CleanupPlanTests(unittest.TestCase):
    def test_all_three_gates_open(self):
        plan = cleanup_plan(_config(cleanup_mode="message"), _context())

        self.assertTrue(plan.enabled)
        self.assertEqual(plan.mode_id, "message")
        self.assertEqual(plan.tone_id, "neutral")
        self.assertIsNone(plan.reason)

    def test_without_pro_nothing_runs(self):
        plan = cleanup_plan(
            _config(cleanup_mode="message", **{PRO_OVERRIDE_KEY: False}), _context()
        )

        self.assertFalse(plan.enabled)
        self.assertEqual(plan.reason, CLEANUP_OFF_PRO)

    def test_the_user_switch_is_honoured(self):
        plan = cleanup_plan(
            _config(cleanup_mode="message", cleanup_enabled=False), _context()
        )

        self.assertFalse(plan.enabled)
        self.assertEqual(plan.reason, CLEANUP_OFF_DISABLED)

    def test_an_undecided_switch_asks_the_machine_and_is_not_assumed_on(self):
        config = _config(cleanup_mode="message", cleanup_enabled=None)
        with patch(
            "cleanup.llama_server.cleanup_default_for_current_machine", return_value=False
        ):
            plan = cleanup_plan(config, _context())

        self.assertFalse(plan.enabled)
        self.assertEqual(plan.reason, CLEANUP_OFF_DISABLED)

    def test_dictation_is_verbatim_by_definition(self):
        plan = cleanup_plan(_config(cleanup_mode="dictation"), _context())

        self.assertFalse(plan.enabled)
        self.assertEqual(plan.reason, CLEANUP_OFF_PASSTHROUGH)

    def test_the_front_app_picks_the_mode_when_context_is_on(self):
        plan = cleanup_plan(
            _config(cleanup_mode="dictation", context_awareness=True),
            _context("com.apple.mail"),
        )

        self.assertEqual(plan.mode_id, "mail")
        self.assertTrue(plan.enabled)

    def test_a_per_app_override_beats_the_table(self):
        plan = cleanup_plan(
            _config(
                cleanup_mode="dictation",
                context_awareness=True,
                mode_by_app={"com.apple.mail": "notes"},
            ),
            _context("com.apple.mail"),
        )

        self.assertEqual(plan.mode_id, "notes")

    def test_the_tone_comes_from_config(self):
        plan = cleanup_plan(_config(cleanup_mode="mail", cleanup_tone="formal"), _context())
        self.assertEqual(plan.tone_id, "formal")


class _RecordingCleanup:
    """Stands in for ``CleanupRuntime.cleanup``; records what it was asked."""

    def __init__(self, result=None):
        self.calls = []
        self._result = result

    def __call__(self, text, system_prompt):
        self.calls.append((text, system_prompt))
        if self._result is None:
            return CleanupResult(text=f"cleaned: {text}", elapsed_s=0.5)
        return self._result


class RunCleanupTests(unittest.TestCase):
    def test_a_disabled_plan_never_calls_the_model(self):
        call = _RecordingCleanup()
        plan = CleanupPlan("dictation", "neutral", False, CLEANUP_OFF_PASSTHROUGH)

        outcome = run_cleanup("hello there", plan, cleanup=call)

        self.assertEqual(outcome.text, "hello there")
        self.assertFalse(outcome.ran)
        self.assertIsNone(outcome.skipped_reason)
        self.assertEqual(call.calls, [])

    def test_the_cleaned_text_replaces_the_transcript(self):
        call = _RecordingCleanup()
        plan = CleanupPlan("message", "warm", True)

        outcome = run_cleanup("um so like hello", plan, cleanup=call)

        self.assertEqual(outcome.text, "cleaned: um so like hello")
        self.assertTrue(outcome.ran)
        self.assertEqual(outcome.elapsed_s, 0.5)

    def test_the_prompt_carries_mode_tone_language_and_vocabulary(self):
        rendered = []

        def render(mode, tone, language, vocabulary):
            rendered.append((mode, tone, language, vocabulary))
            return "SYSTEM"

        call = _RecordingCleanup()
        run_cleanup(
            "hello",
            CleanupPlan("mail", "formal", True),
            cleanup=call,
            language="fr",
            vocabulary_terms=("Murmur", "Boske"),
            render=render,
        )

        self.assertEqual(rendered, [("mail", "formal", "fr", ("Murmur", "Boske"))])
        self.assertEqual(call.calls[0][1], "SYSTEM")

    def test_auto_reaches_the_prompt_as_none(self):
        rendered = []
        run_cleanup(
            "hello",
            CleanupPlan("notes", "neutral", True),
            cleanup=_RecordingCleanup(),
            language="auto",
            render=lambda *args: rendered.append(args) or "SYSTEM",
        )

        self.assertIsNone(rendered[0][2])

    def test_code_mode_runs_the_rule_pass_before_the_model(self):
        order = []
        call = _RecordingCleanup()

        def transform(text, *, language):
            order.append(("transform", text, language))
            return "git commit --force"

        run_cleanup(
            "git commit dash dash force",
            CleanupPlan("code", "terse", True),
            cleanup=call,
            language="auto",
            transform_code=transform,
            render=lambda *args: "SYSTEM",
        )

        self.assertEqual(order[0][1], "git commit dash dash force")
        self.assertEqual(order[0][2], "en")
        # The model sees real code tokens, not the words for them.
        self.assertEqual(call.calls[0][0], "git commit --force")

    def test_other_modes_leave_the_words_alone(self):
        def transform(text, *, language):
            raise AssertionError("only code mode transforms spoken punctuation")

        run_cleanup(
            "open paren",
            CleanupPlan("message", "neutral", True),
            cleanup=_RecordingCleanup(),
            transform_code=transform,
            render=lambda *args: "SYSTEM",
        )

    def test_a_skipped_result_keeps_the_transcript_and_carries_the_reason(self):
        call = _RecordingCleanup(
            CleanupResult(text="original", skipped=True, reason="timed out after 3s")
        )

        outcome = run_cleanup(
            "original", CleanupPlan("message", "neutral", True), cleanup=call
        )

        self.assertEqual(outcome.text, "original")
        self.assertFalse(outcome.ran)
        self.assertEqual(outcome.skipped_reason, "timed out after 3s")

    def test_a_skip_after_the_code_pass_keeps_the_transformed_text(self):
        call = _RecordingCleanup(
            CleanupResult(text="ignored", skipped=True, reason="unreachable")
        )

        outcome = run_cleanup(
            "git commit dash dash force",
            CleanupPlan("code", "neutral", True),
            cleanup=call,
            transform_code=lambda text, *, language: "git commit --force",
            render=lambda *args: "SYSTEM",
        )

        # The rule pass is deterministic and already correct; a model that did
        # not answer must not undo it.
        self.assertEqual(outcome.text, "git commit --force")
        self.assertEqual(outcome.skipped_reason, "unreachable")

    def test_a_reasonless_skip_still_says_something(self):
        call = _RecordingCleanup(CleanupResult(text="original", skipped=True))

        outcome = run_cleanup(
            "original", CleanupPlan("mail", "neutral", True), cleanup=call
        )

        self.assertTrue(outcome.skipped_reason)

    def test_the_notice_names_the_reason_and_reassures(self):
        message = cleanup_skipped_message("timed out after 3s")
        self.assertIn("timed out after 3s", message)
        self.assertIn("unchanged", message)


class PipelineOrderTests(unittest.TestCase):
    """Hallucination filter, then replacements, then cleanup, then paste."""

    def test_the_filter_reads_the_engine_words_and_cleanup_reads_the_replacements(self):
        seen = []
        vocabulary = vocabulary_from_config(
            {
                "vocabulary_replacements": [
                    {"from": "murmer", "to": "Murmur", "match_case": False}
                ]
            }
        )

        def detect(text):
            seen.append(("filter", text))
            return False

        text, hallucination = finalize_transcript(
            "murmer is running", vocabulary, detect_hallucination=detect
        )
        self.assertEqual(seen, [("filter", "murmer is running")])
        self.assertEqual(text, "Murmur is running")
        self.assertFalse(hallucination)

        call = _RecordingCleanup()
        outcome = run_cleanup(
            text, CleanupPlan("message", "neutral", True), cleanup=call
        )

        # Cleanup receives the corrected term, so the model never "fixes" a
        # replacement the user asked for.
        self.assertEqual(call.calls[0][0], "Murmur is running")
        self.assertEqual(outcome.text, "cleaned: Murmur is running")

    def test_a_hallucination_never_reaches_cleanup(self):
        text, hallucination = finalize_transcript(
            "Thank you", vocabulary_from_config({})
        )
        self.assertTrue(hallucination)
        # The app's branch is `if text and not is_hallucination`, so cleanup is
        # simply not called; asserted here as the contract the wiring relies on.
        self.assertEqual(text, "Thank you")


class ModeAndToneMenuTests(unittest.TestCase):
    def test_the_configured_mode_is_the_ticked_one(self):
        state = mode_menu_state(_config(cleanup_mode="notes"))

        self.assertTrue(state["notes"])
        self.assertEqual([mode for mode in MODE_IDS if state[mode]], ["notes"])

    def test_every_mode_has_an_entry(self):
        state = mode_menu_state(_config())
        for mode_id in MODE_IDS:
            self.assertIn(mode_id, state)

    def test_automatic_reflects_context_awareness(self):
        self.assertTrue(
            mode_menu_state(_config(context_awareness=True))[MODE_MENU_AUTOMATIC]
        )
        self.assertFalse(
            mode_menu_state(_config(context_awareness=False))[MODE_MENU_AUTOMATIC]
        )

    def test_automatic_and_a_mode_can_both_be_ticked(self):
        # The table decides per app; the ticked mode covers everywhere else.
        state = mode_menu_state(_config(cleanup_mode="mail", context_awareness=True))
        self.assertTrue(state["mail"])
        self.assertTrue(state[MODE_MENU_AUTOMATIC])

    def test_the_default_config_ticks_dictation_and_automatic(self):
        state = mode_menu_state(dict(DEFAULT_CONFIG))
        self.assertTrue(state["dictation"])
        self.assertTrue(state[MODE_MENU_AUTOMATIC])

    def test_exactly_one_tone_is_ticked(self):
        state = tone_menu_state(_config(cleanup_tone="terse"))
        self.assertEqual([tone for tone in TONE_IDS if state[tone]], ["terse"])

    def test_an_unknown_tone_falls_back_to_the_default(self):
        state = tone_menu_state(_config(cleanup_tone="sarcastic"))
        self.assertEqual([tone for tone in TONE_IDS if state[tone]], ["neutral"])


class FakeLlamaServer:
    """A llama-server whose life is scripted by the test."""

    def __init__(self, model_path, *, start_error=None):
        self.model_path = model_path
        self.starts = 0
        self.stops = 0
        self.alive = False
        self.start_error = start_error

    def start(self):
        self.starts += 1
        if self.start_error is not None:
            raise self.start_error
        self.alive = True

    def stop(self):
        self.stops += 1
        self.alive = False

    @property
    def is_running(self):
        return self.alive

    def die(self):
        """The child was killed under us (an OOM kill on an 8 GB Mac)."""
        self.alive = False


class FakeCleanupClient:
    def __init__(self, server, replies=None):
        self.server = server
        self.calls = []
        self._replies = list(replies or [])

    def cleanup(self, text, system_prompt):
        self.calls.append(text)
        if not self.server.is_running:
            raise LlamaServerError("llama-server exited (code 137)")
        if self._replies:
            return self._replies.pop(0)
        return CleanupResult(text=f"cleaned: {text}")


class _RuntimeFixture:
    """A CleanupRuntime over fake factories, plus what they produced."""

    def __init__(self, model_path="/models/cleanup.gguf", start_errors=(), replies=None):
        self.servers = []
        self.clients = []
        self.statuses = []
        self._start_errors = list(start_errors)
        self._replies = replies
        self.model_path = model_path
        self.runtime = CleanupRuntime(
            lambda: self.model_path,
            server_factory=self._server,
            client_factory=self._client,
            on_status=self.statuses.append,
        )

    def _server(self, model_path):
        error = self._start_errors.pop(0) if self._start_errors else None
        server = FakeLlamaServer(model_path, start_error=error)
        self.servers.append(server)
        return server

    def _client(self, server):
        client = FakeCleanupClient(server, replies=self._replies)
        self.clients.append(client)
        return client


class CleanupRuntimeTests(unittest.TestCase):
    def test_nothing_starts_until_the_first_request(self):
        fixture = _RuntimeFixture()

        self.assertFalse(fixture.runtime.is_started)
        self.assertEqual(fixture.servers, [])

    def test_the_first_request_starts_the_server_and_says_so(self):
        fixture = _RuntimeFixture()

        result = fixture.runtime.cleanup("hello", "SYSTEM")

        self.assertEqual(result.text, "cleaned: hello")
        self.assertEqual(len(fixture.servers), 1)
        self.assertEqual(fixture.servers[0].starts, 1)
        self.assertEqual(fixture.servers[0].model_path, "/models/cleanup.gguf")
        self.assertEqual(fixture.statuses, [CLEANUP_PREPARING_STATUS])

    def test_the_server_is_kept_for_the_session(self):
        fixture = _RuntimeFixture()

        fixture.runtime.cleanup("one", "SYSTEM")
        fixture.runtime.cleanup("two", "SYSTEM")

        self.assertEqual(len(fixture.servers), 1)
        self.assertEqual(fixture.statuses, [CLEANUP_PREPARING_STATUS])
        self.assertEqual(fixture.clients[0].calls, ["one", "two"])

    def test_a_missing_model_skips_visibly_and_starts_nothing(self):
        fixture = _RuntimeFixture()
        fixture.model_path = None

        result = fixture.runtime.cleanup("hello", "SYSTEM")

        self.assertTrue(result.skipped)
        self.assertEqual(result.reason, CLEANUP_MODEL_MISSING_REASON)
        self.assertEqual(result.text, "hello")  # the transcript survives
        self.assertEqual(fixture.servers, [])

    def test_a_crashed_child_is_replaced_and_the_request_retried(self):
        fixture = _RuntimeFixture()
        fixture.runtime.cleanup("one", "SYSTEM")
        fixture.servers[0].die()

        result = fixture.runtime.cleanup("two", "SYSTEM")

        self.assertEqual(result.text, "cleaned: two")
        self.assertEqual(len(fixture.servers), 2)
        self.assertEqual(fixture.servers[0].stops, 1)  # the corpse was reaped
        self.assertEqual(fixture.servers[1].starts, 1)

    def test_a_server_that_will_not_start_skips_rather_than_raising(self):
        fixture = _RuntimeFixture(start_errors=[LlamaServerError("no binary")])

        result = fixture.runtime.cleanup("hello", "SYSTEM")

        self.assertTrue(result.skipped)
        self.assertEqual(result.reason, CLEANUP_START_FAILED_REASON)
        self.assertEqual(result.text, "hello")
        self.assertFalse(fixture.runtime.is_started)

    def test_a_server_that_keeps_dying_gives_up_after_one_retry(self):
        fixture = _RuntimeFixture()
        # Every client call finds a dead server: start, die, retry, die.
        original = FakeLlamaServer.start

        def start_then_die(server):
            original(server)
            server.alive = False

        with patch.object(FakeLlamaServer, "start", start_then_die):
            result = fixture.runtime.cleanup("hello", "SYSTEM")

        self.assertTrue(result.skipped)
        self.assertEqual(result.reason, CLEANUP_UNSTABLE_REASON)
        self.assertEqual(result.text, "hello")
        self.assertEqual(len(fixture.servers), 2)  # one restart, not a loop

    def test_stop_shuts_the_child_down_and_is_idempotent(self):
        fixture = _RuntimeFixture()
        fixture.runtime.cleanup("hello", "SYSTEM")

        fixture.runtime.stop()
        fixture.runtime.stop()

        self.assertEqual(fixture.servers[0].stops, 1)
        self.assertFalse(fixture.runtime.is_started)

    def test_stop_before_anything_started_is_harmless(self):
        fixture = _RuntimeFixture()
        fixture.runtime.stop()
        self.assertEqual(fixture.servers, [])

    def test_a_stop_failure_still_forgets_the_server(self):
        fixture = _RuntimeFixture()
        fixture.runtime.cleanup("hello", "SYSTEM")

        def explode():
            raise OSError("terminate failed")

        fixture.servers[0].stop = explode
        fixture.runtime.stop()

        self.assertFalse(fixture.runtime.is_started)

    def test_a_timeout_from_the_client_is_not_a_restart(self):
        fixture = _RuntimeFixture(
            replies=[CleanupResult(text="hello", skipped=True, reason="timed out after 3s")]
        )

        result = fixture.runtime.cleanup("hello", "SYSTEM")

        self.assertTrue(result.skipped)
        self.assertEqual(result.reason, "timed out after 3s")
        self.assertEqual(len(fixture.servers), 1)

    def test_the_download_offer_names_the_model_and_reassures(self):
        message = cleanup_model_missing_message("Ministral 3 3B Instruct")
        self.assertIn("Ministral 3 3B Instruct", message)
        self.assertIn("pasted", message)


class AppCatalogTests(unittest.TestCase):
    def test_the_store_carries_the_speech_models_and_the_cleanup_model(self):
        ids = [spec.id for spec in APP_CATALOG]

        self.assertIn(CLEANUP_MODEL_SPEC.id, ids)
        for spec in CATALOG:
            self.assertIn(spec.id, ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_speech_catalog_itself_stays_speech_only(self):
        self.assertNotIn(CLEANUP_MODEL_SPEC.id, [spec.id for spec in CATALOG])


if __name__ == "__main__":
    unittest.main()
