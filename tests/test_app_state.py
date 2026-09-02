import ast
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from murmur import (
    APP_CATALOG,
    APP_VERSION,
    AUDIO_DIR,
    HISTORY_ORIGIN_BY_ENGINE,
    SM_STATUS_ENABLED,
    LaunchAtLoginUnavailable,
    MurmurApp,
    apply_launch_at_login,
    history_origin_for,
    launch_at_login_enabled,
    login_item_service,
    CLEANUP_DOWNLOAD_MENU_TITLE,
    CLEANUP_MODEL_MISSING_REASON,
    CLEANUP_NOTICE_NOTIFY,
    CLEANUP_NOTICE_OFFER,
    CLEANUP_NOT_READY_REASON,
    CLEANUP_OFF_DISABLED,
    CLEANUP_OFF_PASSTHROUGH,
    CLEANUP_OFF_PRO,
    CLEANUP_PREPARING_STATUS,
    CLEANUP_PREWARM_KEY,
    CLEANUP_START_FAILED_REASON,
    CLEANUP_STOPPING_REASON,
    CLEANUP_UNSTABLE_REASON,
    MODE_MENU_AUTOMATIC,
    PRO_OVERRIDE_KEY,
    CleanupPlan,
    CleanupRuntime,
    cleanup_download_menu_enabled,
    cleanup_model_missing_message,
    cleanup_notice_kind,
    cleanup_plan,
    cleanup_skipped_message,
    code_transform_language,
    language_is_auto,
    mode_menu_state,
    paste_and_settle,
    pro_enabled,
    prompt_language,
    reapply_replacements,
    run_cleanup,
    should_offer_cleanup_download,
    should_prewarm_cleanup,
    stream_text_for_token,
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
from services.keychain import KeychainUnavailable
from services.persistence_service import (
    DEFAULT_CONFIG,
    ORIGIN_BYOK,
    ORIGIN_CLOUD,
    ORIGIN_LOCAL,
    validate_history_origin,
)
from ui.settings.general_tab import supports_launch_at_login
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
            stream_active=False,
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

    def test_an_abandoned_live_stream_still_blocks_the_swap(self):
        # The batch path gave up waiting on the live decoder, so the app looks
        # idle while a worker is still inside engine.stream(). Unloading the
        # engine there pulls the model out from under a decode in flight.
        self.assertEqual(
            self._decide(stream_active=True, is_recording=False, is_processing=False),
            RELOAD_BUSY,
        )

    def test_a_finished_stream_stops_blocking(self):
        self.assertEqual(self._decide(stream_active=False), RELOAD_START)


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

    def test_the_replacements_are_applied_again_after_the_model_answers(self):
        # The LLM rewrites sentences, and a rewrite re-cases words: "Murmur"
        # comes back as "murmur" at the start of a new clause. The user asked
        # for a spelling, so it is restored on the way to the clipboard.
        vocabulary = vocabulary_from_config(
            {
                "vocabulary_replacements": [
                    {"from": "murmer", "to": "Murmur", "match_case": False}
                ]
            }
        )
        outcome = run_cleanup(
            "Murmur is running",
            CleanupPlan("message", "neutral", True),
            cleanup=lambda text, prompt: CleanupResult(text="murmer is running"),
        )
        self.assertEqual(outcome.text, "murmer is running")
        self.assertEqual(reapply_replacements(outcome.text, vocabulary), "Murmur is running")


class ReapplyReplacementsTests(unittest.TestCase):
    """Replacements run again after cleanup; the filter does not."""

    def _vocabulary(self):
        return vocabulary_from_config(
            {
                "vocabulary_replacements": [
                    {"from": "murmer", "to": "Murmur", "match_case": False}
                ]
            }
        )

    def test_a_term_the_model_re_cased_is_put_back(self):
        self.assertEqual(
            reapply_replacements("murmer is running", self._vocabulary()),
            "Murmur is running",
        )

    def test_text_with_nothing_to_replace_is_returned_unchanged(self):
        self.assertEqual(
            reapply_replacements("nothing to fix here", self._vocabulary()),
            "nothing to fix here",
        )

    def test_the_hallucination_filter_is_not_run_again(self):
        # "Thank you" is a classic silence hallucination, but the filter reads
        # the engine's own words; re-judging a *cleaned* sentence would let the
        # model's phrasing suppress a real transcript.
        self.assertEqual(
            reapply_replacements("Thank you", vocabulary_from_config({})), "Thank you"
        )


class StreamTokenTests(unittest.TestCase):
    """An abandoned live decoder must never speak for the next utterance."""

    def test_the_text_of_the_utterance_being_collected_is_taken(self):
        self.assertEqual(stream_text_for_token((7, "  hello there  "), 7), "hello there")

    def test_a_result_from_an_abandoned_worker_is_discarded(self):
        # Utterance 7 was abandoned at the join timeout and finished later.
        # Without the token, utterance 8 pastes utterance 7's sentence.
        self.assertIsNone(stream_text_for_token((7, "the previous sentence"), 8))

    def test_no_result_at_all_is_none(self):
        self.assertIsNone(stream_text_for_token(None, 8))

    def test_a_token_already_taken_is_none(self):
        self.assertIsNone(stream_text_for_token((7, "hello"), None))

    def test_a_blank_or_failed_stream_is_none(self):
        self.assertIsNone(stream_text_for_token((7, "   "), 7))
        self.assertIsNone(stream_text_for_token((7, None), 7))


class LanguageAutoTests(unittest.TestCase):
    """Whether the live decode may stand in for the batch pass."""

    def test_blank_and_auto_leave_detection_to_the_engine(self):
        for value in (None, "", "   ", "auto", "AUTO", " Auto "):
            self.assertTrue(language_is_auto(value), value)

    def test_a_real_language_code_is_not_auto(self):
        for value in ("fr", "en-GB", "ja"):
            self.assertFalse(language_is_auto(value), value)

    def test_prompt_language_agrees_with_it(self):
        # One rule, one place: the cleanup prompt and the stream-vs-batch
        # choice must never disagree about what "auto" means.
        self.assertIsNone(prompt_language("auto"))
        self.assertEqual(prompt_language("fr"), "fr")


class CleanupNoticeKindTests(unittest.TestCase):
    """A missing model is an offer; everything else is a plain notification."""

    def test_a_missing_model_becomes_the_download_offer(self):
        self.assertEqual(
            cleanup_notice_kind(CLEANUP_MODEL_MISSING_REASON), CLEANUP_NOTICE_OFFER
        )

    def test_any_other_failure_is_a_notification(self):
        for reason in (CLEANUP_START_FAILED_REASON, CLEANUP_UNSTABLE_REASON,
                       CLEANUP_NOT_READY_REASON):
            self.assertEqual(cleanup_notice_kind(reason), CLEANUP_NOTICE_NOTIFY, reason)

    def test_a_pass_that_delivered_says_nothing(self):
        self.assertIsNone(cleanup_notice_kind(None))
        self.assertIsNone(cleanup_notice_kind(""))


class PasteAndSettleTests(unittest.TestCase):
    """The modal offer waits until the keystrokes have landed."""

    def test_the_offer_runs_after_the_paste_not_before_it(self):
        order = []
        pasted = paste_and_settle(
            "hello",
            type_text=lambda text: order.append(("paste", text)) or True,
            offer=lambda: order.append(("offer", None)),
        )
        self.assertTrue(pasted)
        self.assertEqual([step for step, _ in order], ["paste", "offer"])

    def test_the_pill_is_told_the_text_landed_before_the_offer(self):
        order = []
        pill = SimpleNamespace(
            done=lambda length: order.append(("done", length)),
            error=lambda message: order.append(("error", message)),
        )
        paste_and_settle(
            "hello",
            type_text=lambda text: True,
            pill=pill,
            offer=lambda: order.append(("offer", None)),
        )
        self.assertEqual(order, [("done", 5), ("offer", None)])

    def test_a_failed_paste_shows_the_error_and_still_offers(self):
        order = []
        pill = SimpleNamespace(
            done=lambda length: order.append(("done", length)),
            error=lambda message: order.append(("error", message)),
        )
        pasted = paste_and_settle(
            "hello",
            type_text=lambda text: False,
            pill=pill,
            offer=lambda: order.append(("offer", None)),
        )
        self.assertFalse(pasted)
        self.assertEqual([step for step, _ in order], ["error", "offer"])

    def test_nothing_pending_means_nothing_after_the_paste(self):
        self.assertTrue(paste_and_settle("hello", type_text=lambda text: True))


class CleanupDownloadOfferTests(unittest.TestCase):
    """"Not now" must not be a life sentence for the cleanup model."""

    def _offer(self, **overrides):
        kwargs = dict(declined=False, downloading=False, installed=False)
        kwargs.update(overrides)
        return should_offer_cleanup_download(**kwargs)

    def test_the_offer_is_shown_when_nothing_is_in_the_way(self):
        self.assertTrue(self._offer())

    def test_declining_stops_the_modal_for_the_session(self):
        self.assertFalse(self._offer(declined=True))

    def test_a_download_already_running_is_not_offered_again(self):
        self.assertFalse(self._offer(downloading=True))

    def test_an_installed_model_is_never_offered(self):
        self.assertFalse(self._offer(installed=True))

    def test_the_menu_entry_stays_clickable_after_a_decline(self):
        # This is what makes the decline recoverable: the alert is asked once,
        # the menu item is the way back.
        self.assertTrue(cleanup_download_menu_enabled(installed=False, downloading=False))

    def test_the_menu_entry_is_dead_while_downloading_or_installed(self):
        self.assertFalse(cleanup_download_menu_enabled(installed=False, downloading=True))
        self.assertFalse(cleanup_download_menu_enabled(installed=True, downloading=False))

    def test_the_menu_entry_says_what_it_downloads(self):
        self.assertIn("cleanup", CLEANUP_DOWNLOAD_MENU_TITLE.lower())


class PrewarmDecisionTests(unittest.TestCase):
    """Starting the 2 GB server at launch, only where it is free to do so."""

    def _decide(self, **overrides):
        kwargs = dict(
            config=_config(),
            pro=True,
            cleanup_enabled=True,
            installed=True,
            ram_gb=16,
        )
        kwargs.update(overrides)
        return should_prewarm_cleanup(**kwargs)

    def test_an_eligible_mac_pre_warms(self):
        self.assertTrue(self._decide())

    def test_a_small_mac_waits_for_the_first_utterance(self):
        self.assertFalse(self._decide(ram_gb=8))
        self.assertFalse(self._decide(ram_gb=None))

    def test_nothing_is_started_without_the_model_the_switch_or_pro(self):
        self.assertFalse(self._decide(installed=False))
        self.assertFalse(self._decide(cleanup_enabled=False))
        self.assertFalse(self._decide(pro=False))

    def test_the_config_key_switches_it_off(self):
        self.assertFalse(self._decide(config=_config(**{CLEANUP_PREWARM_KEY: False})))


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

    def __init__(self, model_path, *, start_error=None, start_gate=None):
        self.model_path = model_path
        self.starts = 0
        self.stops = 0
        self.alive = False
        self.start_error = start_error
        #: Held closed to imitate a 60 s model load. Released by the test.
        self.start_gate = start_gate
        self.entered_start = threading.Event()

    def start(self):
        self.starts += 1
        self.entered_start.set()
        if self.start_gate is not None:
            self.start_gate.wait(5.0)
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
    def __init__(self, server, replies=None, request_gate=None):
        self.server = server
        self.calls = []
        self._replies = list(replies or [])
        #: Held closed to imitate a request the server is still answering.
        self.request_gate = request_gate
        self.entered_request = threading.Event()

    def cleanup(self, text, system_prompt):
        self.calls.append(text)
        self.entered_request.set()
        if self.request_gate is not None:
            self.request_gate.wait(5.0)
        if not self.server.is_running:
            raise LlamaServerError("llama-server exited (code 137)")
        if self._replies:
            return self._replies.pop(0)
        return CleanupResult(text=f"cleaned: {text}")


class _RuntimeFixture:
    """A CleanupRuntime over fake factories, plus what they produced."""

    def __init__(
        self,
        model_path="/models/cleanup.gguf",
        start_errors=(),
        replies=None,
        start_gate=None,
        request_gate=None,
        first_use_wait_s=None,
    ):
        self.servers = []
        self.clients = []
        self.statuses = []
        self._start_errors = list(start_errors)
        self._replies = replies
        self._start_gate = start_gate
        self._request_gate = request_gate
        self.model_path = model_path
        extra = {} if first_use_wait_s is None else {"first_use_wait_s": first_use_wait_s}
        self.runtime = CleanupRuntime(
            lambda: self.model_path,
            server_factory=self._server,
            client_factory=self._client,
            on_status=self.statuses.append,
            **extra,
        )

    def _server(self, model_path):
        error = self._start_errors.pop(0) if self._start_errors else None
        server = FakeLlamaServer(
            model_path, start_error=error, start_gate=self._start_gate
        )
        self.servers.append(server)
        return server

    def _client(self, server):
        client = FakeCleanupClient(
            server, replies=self._replies, request_gate=self._request_gate
        )
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


class CleanupFirstUseTests(unittest.TestCase):
    """The first utterance waits a bounded time; nothing waits on a quit."""

    def _release(self, *gates):
        for gate in gates:
            self.addCleanup(gate.set)
        return gates[0] if len(gates) == 1 else gates

    def test_a_first_start_that_outruns_the_budget_pastes_the_raw_text(self):
        gate = self._release(threading.Event())
        fixture = _RuntimeFixture(start_gate=gate, first_use_wait_s=0.05)

        result = fixture.runtime.cleanup("hello", "SYSTEM")

        self.assertTrue(result.skipped)
        self.assertEqual(result.reason, CLEANUP_NOT_READY_REASON)
        self.assertEqual(result.text, "hello")  # the transcript is never lost
        self.assertEqual(fixture.statuses, [CLEANUP_PREPARING_STATUS])

    def test_the_server_keeps_starting_so_the_next_utterance_is_cleaned(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        fixture = _RuntimeFixture(start_gate=gate, first_use_wait_s=0.05)

        fixture.runtime.cleanup("one", "SYSTEM")
        gate.set()  # the model finished loading in the background
        result = fixture.runtime.cleanup("two", "SYSTEM")

        self.assertEqual(result.text, "cleaned: two")
        self.assertEqual(len(fixture.servers), 1)  # not started twice
        self.assertEqual(fixture.clients[0].calls, ["two"])

    def test_a_second_utterance_does_not_start_a_second_server(self):
        gate = self._release(threading.Event())
        fixture = _RuntimeFixture(start_gate=gate, first_use_wait_s=0.05)

        fixture.runtime.cleanup("one", "SYSTEM")
        fixture.runtime.cleanup("two", "SYSTEM")

        self.assertEqual(len(fixture.servers), 1)

    def test_stop_returns_while_a_request_is_still_in_flight(self):
        gate = self._release(threading.Event())
        fixture = _RuntimeFixture(request_gate=gate)
        worker = threading.Thread(
            target=fixture.runtime.cleanup, args=("hello", "SYSTEM"), daemon=True
        )
        worker.start()
        self.addCleanup(worker.join, 5.0)
        # The request lock is held by the worker for as long as the gate is shut.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not fixture.clients:
            time.sleep(0.01)
        self.assertTrue(fixture.clients)
        self.assertTrue(fixture.clients[0].entered_request.wait(2.0))

        started_at = time.monotonic()
        fixture.runtime.stop()
        elapsed = time.monotonic() - started_at

        self.assertLess(elapsed, 1.0)
        self.assertEqual(fixture.servers[0].stops, 1)

    def test_stop_does_not_wait_on_a_start_that_is_still_loading(self):
        gate = self._release(threading.Event())
        fixture = _RuntimeFixture(start_gate=gate, first_use_wait_s=0.05)
        fixture.runtime.cleanup("hello", "SYSTEM")
        self.assertTrue(fixture.servers[0].entered_start.wait(2.0))

        started_at = time.monotonic()
        fixture.runtime.stop()
        elapsed = time.monotonic() - started_at

        # Quitting must not sit behind a 60 s model load.
        self.assertLess(elapsed, 1.0)
        self.assertFalse(fixture.runtime.is_started)

    def test_a_request_after_stop_skips_rather_than_starting_again(self):
        fixture = _RuntimeFixture()
        fixture.runtime.stop()

        result = fixture.runtime.cleanup("hello", "SYSTEM")

        self.assertTrue(result.skipped)
        self.assertEqual(result.reason, CLEANUP_STOPPING_REASON)
        self.assertEqual(result.text, "hello")
        self.assertEqual(fixture.servers, [])

    def test_a_server_started_after_a_stop_is_shut_down_again(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        fixture = _RuntimeFixture(start_gate=gate, first_use_wait_s=0.05)
        fixture.runtime.cleanup("hello", "SYSTEM")
        self.assertTrue(fixture.servers[0].entered_start.wait(2.0))

        fixture.runtime.stop()
        self.assertGreaterEqual(fixture.servers[0].stops, 1)
        gate.set()  # the child finally came up, after the app asked to quit

        # It is never published, and it does not outlive the quit.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and fixture.servers[0].alive:
            time.sleep(0.01)
        self.assertFalse(fixture.servers[0].alive)
        self.assertFalse(fixture.runtime.is_started)

    def test_prewarm_starts_the_server_before_any_utterance(self):
        fixture = _RuntimeFixture()

        self.assertTrue(fixture.runtime.prewarm())
        self.assertTrue(fixture.runtime.wait_until_ready(2.0))
        self.assertEqual(len(fixture.servers), 1)
        self.assertTrue(fixture.runtime.is_started)

        # The utterance that follows pays nothing.
        self.assertEqual(fixture.runtime.cleanup("hello", "SYSTEM").text, "cleaned: hello")
        self.assertEqual(len(fixture.servers), 1)

    def test_prewarm_without_the_model_starts_nothing(self):
        fixture = _RuntimeFixture()
        fixture.model_path = None

        self.assertFalse(fixture.runtime.prewarm())
        self.assertEqual(fixture.servers, [])


class AppCatalogTests(unittest.TestCase):
    def test_the_store_carries_the_speech_models_and_the_cleanup_model(self):
        ids = [spec.id for spec in APP_CATALOG]

        self.assertIn(CLEANUP_MODEL_SPEC.id, ids)
        for spec in CATALOG:
            self.assertIn(spec.id, ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_speech_catalog_itself_stays_speech_only(self):
        self.assertNotIn(CLEANUP_MODEL_SPEC.id, [spec.id for spec in CATALOG])


# -- Settings wiring ---------------------------------------------------------
#
# ``MurmurApp`` is a ``rumps.App`` and cannot be constructed without a menu
# bar, so the methods below are called unbound against a stand-in ``self``
# carrying only the attributes each one actually reads. That is also the point
# of the tests: these methods must not reach past what they are given.


class SettingsServicesTests(unittest.TestCase):
    """The one dict Settings is handed, and what each key means."""

    KEYS = {
        "usage",
        "license",
        "pro_gate",
        "keychain",
        "scheduler",
        "version",
        "build_info",
        "persistence",
        "audio_dir",
    }

    def _services(self, keychain=None, config=None, loads=None):
        snapshot = dict(config or {})

        def runtime_config():
            if loads is not None:
                loads.append(dict(snapshot))
            return dict(snapshot)

        app = SimpleNamespace(
            persistence=object(),
            _keychain=lambda: keychain,
            runtime_config=runtime_config,
        )
        return MurmurApp._settings_services(app), app

    def test_every_key_the_window_documents_is_present(self):
        services, app = self._services()

        self.assertEqual(set(services), self.KEYS)
        self.assertIs(services["persistence"], app.persistence)
        self.assertIs(services["pro_gate"].func, pro_enabled)
        self.assertEqual(services["version"], APP_VERSION)
        self.assertEqual(services["audio_dir"], AUDIO_DIR)
        self.assertIsInstance(services["build_info"], dict)

    def test_the_wave_four_providers_are_named_but_empty(self):
        services, _ = self._services()

        self.assertIsNone(services["usage"])
        self.assertIsNone(services["license"])

    def test_the_scheduler_is_left_to_the_account_tab(self):
        # The tab's own default polls off the main thread and then redraws on
        # it; anything handed in here would replace both halves.
        services, _ = self._services()

        self.assertIsNone(services["scheduler"])

    def test_the_keychain_is_whatever_the_probe_found(self):
        store = object()
        services, _ = self._services(keychain=store)

        self.assertIs(services["keychain"], store)

    def test_the_pro_gate_reads_the_config_once_per_settings_open(self):
        # Every gated control asks on every refresh; a gate that loads the file
        # each time turns opening Settings into a burst of main-thread reads.
        loads = []
        services, _ = self._services(config={PRO_OVERRIDE_KEY: True}, loads=loads)
        gate = services["pro_gate"]

        self.assertTrue(gate("cloud_voice"))
        self.assertTrue(gate("cleanup"))
        self.assertEqual(len(loads), 1)

    def test_the_pro_gate_answers_from_the_snapshot_it_was_bound_to(self):
        off, _ = self._services(config={})
        on, _ = self._services(config={PRO_OVERRIDE_KEY: True})

        self.assertFalse(off["pro_gate"]("cloud_voice"))
        self.assertTrue(on["pro_gate"]("cloud_voice"))

    def test_an_unreachable_keychain_reaches_the_tabs_as_none(self):
        class Unavailable:
            @property
            def backend(self):
                raise KeychainUnavailable("no Security framework")

        app = SimpleNamespace(
            persistence=object(), _keychain_probed=False, _keychain_store=None
        )
        with patch("murmur.KeychainStore", Unavailable):
            services = MurmurApp._settings_services(
                SimpleNamespace(
                    persistence=app.persistence,
                    _keychain=lambda: MurmurApp._keychain(app),
                    runtime_config=dict,
                )
            )

        self.assertIsNone(services["keychain"])


class KeychainProbeTests(unittest.TestCase):
    def test_an_unreachable_keychain_becomes_none(self):
        class Unavailable:
            @property
            def backend(self):
                raise KeychainUnavailable("no Security framework")

        app = SimpleNamespace(_keychain_probed=False, _keychain_store=None)
        with patch("murmur.KeychainStore", Unavailable):
            store = MurmurApp._keychain(app)

        self.assertIsNone(store)
        self.assertTrue(app._keychain_probed)

    def test_any_other_failure_from_the_backend_becomes_none_too(self):
        # The ctypes backend raises ValueError and OSError as well as
        # KeychainError; every one of them used to escape this probe and turn
        # into "Could not open Settings".
        for error in (ValueError("bad library path"), OSError("dlopen failed")):
            with self.subTest(error=type(error).__name__):

                class Exploding:
                    @property
                    def backend(self, _error=error):
                        raise _error

                app = SimpleNamespace(_keychain_probed=False, _keychain_store=None)
                with patch("murmur.KeychainStore", Exploding):
                    with self.assertLogs("murmur", level="WARNING") as captured:
                        store = MurmurApp._keychain(app)

                self.assertIsNone(store)
                self.assertIn(type(error).__name__, "\n".join(captured.output))

    def test_the_keychain_is_asked_about_once(self):
        made = []

        class Fine:
            backend = object()

            def __init__(self):
                made.append(self)

        app = SimpleNamespace(_keychain_probed=False, _keychain_store=None)
        with patch("murmur.KeychainStore", Fine):
            first = MurmurApp._keychain(app)
            second = MurmurApp._keychain(app)

        self.assertIs(first, second)
        self.assertEqual(len(made), 1)


class HistoryOriginTests(unittest.TestCase):
    """Where a transcription happened, decided in one place."""

    def test_the_shipped_engines_all_run_on_this_mac(self):
        for engine_id in ("whispercpp", "voxtral_mlx"):
            self.assertEqual(history_origin_for(engine_id), ORIGIN_LOCAL)

    def test_the_cloud_and_own_key_engines_are_named(self):
        self.assertEqual(history_origin_for("cloud"), ORIGIN_CLOUD)
        self.assertEqual(history_origin_for("byok"), ORIGIN_BYOK)

    def test_an_unknown_or_missing_engine_reads_as_local(self):
        # Never the other way round: claiming audio left the Mac when it did
        # not is the failure that matters here.
        self.assertEqual(history_origin_for("something-new"), ORIGIN_LOCAL)
        self.assertEqual(history_origin_for(None), ORIGIN_LOCAL)
        self.assertEqual(history_origin_for(""), ORIGIN_LOCAL)

    def test_every_mapped_origin_is_one_history_accepts(self):
        for origin in HISTORY_ORIGIN_BY_ENGINE.values():
            self.assertEqual(validate_history_origin(origin), origin)


class AddToHistoryTests(unittest.TestCase):
    """Every entry says which engine wrote it, where, and how long the clip was."""

    def _app(self, engine_id="whispercpp", save_history=True):
        recorded = {}

        def add_history_entry(history, **kwargs):
            recorded.update(kwargs)
            return ["entry"]

        app = SimpleNamespace(
            history=[],
            engine=SimpleNamespace(info=lambda: SimpleNamespace(id=engine_id)),
            engine_id=engine_id,
            persistence=SimpleNamespace(add_history_entry=add_history_entry),
            runtime_config=lambda: {"save_history": save_history},
            save_history=lambda: None,
        )
        app.current_engine_id = lambda: MurmurApp.current_engine_id(app)
        return app, recorded

    def test_a_local_clip_carries_its_engine_origin_and_length(self):
        app, recorded = self._app()

        MurmurApp.add_to_history(app, "hello", "live", duration_s=3.5)

        self.assertEqual(recorded["origin"], ORIGIN_LOCAL)
        self.assertEqual(recorded["engine_id"], "whispercpp")
        self.assertEqual(recorded["duration_s"], 3.5)

    def test_a_cloud_engine_is_recorded_as_cloud(self):
        app, recorded = self._app(engine_id="cloud")

        MurmurApp.add_to_history(app, "hello", "live")

        self.assertEqual(recorded["origin"], ORIGIN_CLOUD)
        self.assertEqual(recorded["engine_id"], "cloud")

    def test_the_engine_that_is_loaded_wins_over_the_one_config_asked_for(self):
        app, recorded = self._app(engine_id="whispercpp")
        app.engine = SimpleNamespace(info=lambda: SimpleNamespace(id="cloud"))

        MurmurApp.add_to_history(app, "hello", "live")

        self.assertEqual(recorded["engine_id"], "cloud")

    def test_an_engine_that_cannot_be_asked_falls_back_to_the_configured_id(self):
        app, recorded = self._app()
        app.engine = SimpleNamespace(info=_raise_engine_error)

        MurmurApp.add_to_history(app, "hello", "live")

        self.assertEqual(recorded["engine_id"], "whispercpp")
        self.assertEqual(recorded["origin"], ORIGIN_LOCAL)

    def test_history_turned_off_writes_nothing(self):
        app, recorded = self._app(save_history=False)

        MurmurApp.add_to_history(app, "hello", "live")

        self.assertEqual(recorded, {})


def _raise_engine_error():
    raise RuntimeError("the engine is mid-swap")


class _BridgedAppService:
    """``SMAppService`` as PyObjC bridges it: ``registerAndReturnError_``."""

    def __init__(self, status=0, ok=True):
        self._status = status
        self._ok = ok
        self.calls = []

    def status(self):
        return self._status

    def registerAndReturnError_(self, _error):
        self.calls.append("register")
        if self._ok:
            self._status = SM_STATUS_ENABLED
        return (self._ok, None if self._ok else "denied")

    def unregisterAndReturnError_(self, _error):
        self.calls.append("unregister")
        if self._ok:
            self._status = 0
        return (self._ok, None if self._ok else "denied")


class _PlainAppService:
    """The same service without the error out-parameter."""

    def __init__(self, status=0):
        self._status = status
        self.calls = []

    def status(self):
        return self._status

    def register(self):
        self.calls.append("register")
        self._status = SM_STATUS_ENABLED

    def unregister(self):
        self.calls.append("unregister")
        self._status = 0


class _RequiresApprovalAppService(_BridgedAppService):
    """``register`` succeeds, but macOS waits for the user to allow the item.

    ``SMAppServiceStatusRequiresApproval``. The registration is real and the
    call reports no error, yet Murmur will not start at login until the user
    switches it on in System Settings — so the switch must not claim it is on.
    """

    SM_STATUS_REQUIRES_APPROVAL = 3

    def registerAndReturnError_(self, _error):
        self.calls.append("register")
        self._status = self.SM_STATUS_REQUIRES_APPROVAL
        return (True, None)


class LaunchAtLoginDecisionTests(unittest.TestCase):
    def test_turning_it_on_registers_the_login_item(self):
        service = _BridgedAppService()

        self.assertTrue(apply_launch_at_login(service, True))
        self.assertEqual(service.calls, ["register"])

    def test_turning_it_off_unregisters_it(self):
        service = _BridgedAppService(status=SM_STATUS_ENABLED)

        self.assertFalse(apply_launch_at_login(service, False))
        self.assertEqual(service.calls, ["unregister"])

    def test_asking_for_the_state_it_is_already_in_touches_nothing(self):
        # Re-registering can put the approval prompt back in front of a user
        # who never touched the switch.
        already_on = _BridgedAppService(status=SM_STATUS_ENABLED)
        already_off = _BridgedAppService()

        self.assertTrue(apply_launch_at_login(already_on, True))
        self.assertFalse(apply_launch_at_login(already_off, False))
        self.assertEqual(already_on.calls, [])
        self.assertEqual(already_off.calls, [])

    def test_the_plain_bridge_shape_is_driven_too(self):
        service = _PlainAppService()

        self.assertTrue(apply_launch_at_login(service, True))
        self.assertEqual(service.calls, ["register"])

    def test_no_service_at_all_is_unavailable_rather_than_silent(self):
        with self.assertRaises(LaunchAtLoginUnavailable):
            apply_launch_at_login(None, True)

    def test_a_refusal_from_the_framework_is_not_an_unavailable_build(self):
        service = _BridgedAppService(ok=False)

        with self.assertRaises(RuntimeError) as caught:
            apply_launch_at_login(service, True)
        self.assertNotIsInstance(caught.exception, LaunchAtLoginUnavailable)

    def test_a_registration_awaiting_approval_is_reported_as_not_on_yet(self):
        service = _RequiresApprovalAppService()

        self.assertFalse(apply_launch_at_login(service, True))
        self.assertEqual(service.calls, ["register"])

    def test_the_state_afterwards_is_read_back_rather_than_assumed(self):
        service = _BridgedAppService(status=SM_STATUS_ENABLED)

        self.assertFalse(apply_launch_at_login(service, False))
        self.assertEqual(launch_at_login_enabled(service), False)

    def test_reading_the_current_state(self):
        self.assertFalse(launch_at_login_enabled(None))
        self.assertFalse(launch_at_login_enabled(_BridgedAppService()))
        self.assertTrue(
            launch_at_login_enabled(_BridgedAppService(status=SM_STATUS_ENABLED))
        )

    def test_probing_for_the_framework_never_raises(self):
        # macOS 12 and every source run land here; a menu bar must still appear.
        service = login_item_service()

        self.assertTrue(service is None or hasattr(service, "status"))


class SetLaunchAtLoginTests(unittest.TestCase):
    def test_it_applies_the_decision_and_reports_the_new_state(self):
        service = _BridgedAppService()
        app = SimpleNamespace(_login_item_service=service)

        self.assertTrue(MurmurApp.set_launch_at_login(app, True))
        self.assertEqual(service.calls, ["register"])

    def test_without_the_framework_it_raises_instead_of_pretending(self):
        app = SimpleNamespace(_login_item_service=None)

        with self.assertRaises(LaunchAtLoginUnavailable):
            MurmurApp.set_launch_at_login(app, True)

    def test_a_refusal_is_told_to_the_user_not_raised_into_the_click(self):
        service = _BridgedAppService(ok=False)
        app = SimpleNamespace(_login_item_service=service)

        with patch("murmur.ui_alerts.show_alert") as alert:
            state = MurmurApp.set_launch_at_login(app, True)

        self.assertFalse(state)
        self.assertTrue(alert.called)

    def test_settings_offers_the_switch_only_where_it_works(self):
        # __init__ shadows the method with None when the probe found nothing,
        # which is exactly what the General tab's check reads.
        self.assertFalse(supports_launch_at_login(SimpleNamespace(set_launch_at_login=None)))
        self.assertFalse(supports_launch_at_login(None))
        self.assertTrue(
            supports_launch_at_login(SimpleNamespace(set_launch_at_login=lambda on: on))
        )


class ArchivedSettingsWindowTests(unittest.TestCase):
    """``settings_window.py`` is in ``_archive/``; nothing may reach for it."""

    ROOT = Path(__file__).resolve().parent.parent
    SKIP = {"_archive", ".git", "__pycache__", ".worktrees", "venv", ".venv", "build", "dist"}

    def _sources(self):
        for path in self.ROOT.rglob("*.py"):
            if any(part in self.SKIP for part in path.relative_to(self.ROOT).parts):
                continue
            yield path

    def test_nothing_imports_the_archived_module(self):
        offenders = []
        for path in self._sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(
                    name == "settings_window" or name.startswith("settings_window.")
                    for name in names
                ):
                    offenders.append(f"{path.relative_to(self.ROOT)}:{node.lineno}")

        self.assertEqual(offenders, [])

    def test_the_module_lives_in_the_archive_and_nowhere_else(self):
        self.assertFalse((self.ROOT / "settings_window.py").exists())
        self.assertTrue((self.ROOT / "_archive" / "settings_window.py").exists())

    def test_the_bundle_no_longer_ships_it(self):
        spec = (self.ROOT / "Murmur.spec").read_text(encoding="utf-8")

        self.assertNotIn("settings_window", spec)


class ServiceManagementDependencyTests(unittest.TestCase):
    """``login_item_service`` imports a framework that has to be shipped.

    Nothing else in Murmur imports ``ServiceManagement``, so a missing wheel or
    a missing hidden import shows up only as "Not available in this build" on a
    checkbox nobody can prove wrong. These two lines are that proof.
    """

    ROOT = Path(__file__).resolve().parent.parent
    PACKAGE = "pyobjc-framework-ServiceManagement"

    def test_the_framework_is_a_pinned_macos_dependency(self):
        lines = (self.ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        pinned = [line for line in lines if line.strip().startswith(self.PACKAGE)]

        self.assertEqual(len(pinned), 1, f"{self.PACKAGE} is imported but never installed")
        self.assertIn("==", pinned[0])
        self.assertIn('sys_platform == "darwin"', pinned[0])

    def test_the_bundle_names_the_framework_as_a_hidden_import(self):
        # PyInstaller cannot see an import made inside a function.
        spec = (self.ROOT / "Murmur.spec").read_text(encoding="utf-8")

        self.assertIn('"ServiceManagement"', spec)


if __name__ == "__main__":
    unittest.main()
