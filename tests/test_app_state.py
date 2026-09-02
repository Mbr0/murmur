import ast
import threading
import time
import unittest
from datetime import datetime
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
    ACCOUNT_STATUS_FREE,
    ACCOUNT_STATUS_NOT_SAVED,
    ACCOUNT_STATUS_PRO,
    ACCOUNT_STATUS_PRO_GRACE,
    ENTITLEMENT_REFRESH_INTERVAL_S,
    ENTITLEMENT_RETRY_BASE_S,
    ENTITLEMENT_RETRY_MAX_S,
    FREE_MODE_ID,
    NOTICE_KEY_FAILED,
    NOTICE_KEY_RATE_LIMITED,
    NOTICE_KEY_REJECTED,
    SIGN_IN_MENU_TITLE,
    CleanupPlan,
    CleanupRuntime,
    RemoteEngineKey,
    UsageConfigStore,
    account_menu_title,
    after_byok_failure,
    byok_provider_name,
    next_refresh_delay,
    pinned_cloud_config,
    configured_mode_id,
    expand_gated_snippets,
    gated_vocabulary,
    lease_is_present,
    notice_to_show,
    own_key_present,
    publish_entitlements,
    remote_engine_key,
    resolve_plan_mode,
    should_consume_trial,
    should_refresh_allowance,
    should_refresh_entitlements,
    cleanup_download_menu_enabled,
    cleanup_model_missing_message,
    cleanup_notice_kind,
    cleanup_plan,
    cleanup_skipped_message,
    code_transform_language,
    language_is_auto,
    mode_menu_state,
    paste_and_settle,
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
from cleanup.vocabulary import FREE_TERM_LIMIT, Vocabulary, vocabulary_from_config
from engines.base import EngineError
from engines.byok import ByokAuthError, ByokRateLimited
from engines.cloud import ALLOWANCE_MESSAGE, CloudAllowanceExhausted, CloudAuthError
from engines.factory import (
    CONFIG_CLOUD_BASE_URL,
    DEFAULT_CLOUD_BASE_URL,
    cloud_base_url,
)
from engines.model_store import CATALOG, ModelIntegrityError
from services.engine_router import (
    ENGINE_BYOK,
    ENGINE_CLOUD,
    NOTICE_ADD_KEY,
    NOTICE_CLIP_TOO_LONG,
    NOTICE_SIGN_IN,
)
from services.keychain import KeychainUnavailable
from services.license_service import (
    Entitlements,
    get_current_entitlements,
    is_pro_feature_enabled,
    set_current_entitlements,
)
from services.usage_service import USAGE_DEFAULTS, UsageService
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
        "cleanup_enabled": True,
        "context_awareness": False,
    }
    base.update(overrides)
    return base


def _entitlements(pro=True, cloud_voice=False, trial_minutes=0, in_grace=False):
    """A lease-shaped entitlement set, for publishing to the one Pro gate."""
    return Entitlements(
        pro=pro,
        cloud_voice=cloud_voice,
        msm_minutes=600 if cloud_voice else 0,
        expires_at=None,
        in_grace=in_grace,
        source="lease",
        trial_minutes=trial_minutes,
    )


class GateTestCase(unittest.TestCase):
    """Publishes entitlements to the gate for the length of one test.

    The gate is process-global by design — one writer, one reader, no feature
    check anywhere else — so a test that changes it must put it back.
    """

    ENTITLEMENTS = _entitlements()

    def setUp(self):
        super().setUp()
        self.addCleanup(set_current_entitlements, get_current_entitlements())
        set_current_entitlements(self.ENTITLEMENTS)

    def publish(self, **overrides):
        """Swap the published entitlements mid-test."""
        set_current_entitlements(_entitlements(**overrides))


class FreeTierTestCase(GateTestCase):
    """The same, with nothing entitled."""

    ENTITLEMENTS = Entitlements.none()


class ProGateCallSiteTests(GateTestCase):
    """One gate, and every place in ``murmur.py`` that asks it something.

    The gate itself is :mod:`services.license_service`'s and is tested there.
    What is tested here is that each gated feature is actually asked about at
    the point it runs, and that a free install loses the feature rather than
    the transcript.
    """

    def test_the_gate_is_the_licensed_one(self):
        # Not a config key, not a local copy: a second opinion about what "Pro"
        # means is exactly the bug the single gate exists to prevent.
        source = (Path(__file__).resolve().parent.parent / "murmur.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("pro_override_for_dev", source)
        self.assertNotIn("def pro_enabled", source)

    # -- modes and context ------------------------------------------------

    def test_a_paid_install_resolves_the_mode_from_the_front_app(self):
        mode = resolve_plan_mode(
            _config(cleanup_mode="dictation", context_awareness=True),
            _context("com.apple.mail"),
        )

        self.assertEqual(mode, "mail")

    def test_without_context_the_configured_mode_applies_everywhere(self):
        self.publish(pro=True)
        with patch("murmur.is_pro_feature_enabled", lambda f: f != "context"):
            mode = resolve_plan_mode(
                _config(cleanup_mode="notes", context_awareness=True),
                _context("com.apple.mail"),
                pro=lambda feature: feature != "context",
            )

        self.assertEqual(mode, "notes")

    def test_without_modes_everything_lands_on_dictation(self):
        mode = resolve_plan_mode(
            _config(cleanup_mode="mail"),
            _context(),
            pro=lambda feature: feature != "modes",
        )

        self.assertEqual(mode, "dictation")

    def test_a_free_install_gets_dictation_whatever_the_config_says(self):
        mode = resolve_plan_mode(
            _config(cleanup_mode="mail", context_awareness=True),
            _context("com.apple.mail"),
            pro=lambda _feature: False,
        )

        self.assertEqual(mode, "dictation")

    def test_an_unreadable_mode_is_user_data_not_a_crash(self):
        self.assertEqual(configured_mode_id({"cleanup_mode": "haiku"}), "dictation")

    # -- vocabulary -------------------------------------------------------

    def test_a_free_install_keeps_the_first_twenty_terms(self):
        terms = tuple(f"term{index}" for index in range(30))
        gated = gated_vocabulary(
            Vocabulary(terms=terms), pro=lambda _feature: False
        )

        self.assertEqual(len(gated.terms), FREE_TERM_LIMIT)
        self.assertEqual(gated.terms, terms[:FREE_TERM_LIMIT])

    def test_pro_keeps_every_term_and_the_object_itself(self):
        vocabulary = Vocabulary(terms=("Murmur", "Voxtral"))
        gated = gated_vocabulary(vocabulary, pro=lambda _feature: True)

        self.assertIs(gated, vocabulary)

    def test_truncating_never_touches_the_replacements(self):
        vocabulary = vocabulary_from_config(
            {
                "vocabulary_terms": [f"t{index}" for index in range(25)],
                "vocabulary_replacements": [
                    {"from": "teh", "to": "the", "match_case": False}
                ],
            }
        )
        gated = gated_vocabulary(vocabulary, pro=lambda _feature: False)

        self.assertEqual(gated.replacements, vocabulary.replacements)

    # -- snippets ---------------------------------------------------------

    SNIPPET_CONFIG = {"snippets": [{"trigger": "my address", "text": "12 Rue Oberkampf"}]}

    def test_snippets_expand_for_a_paid_install(self):
        text = expand_gated_snippets(
            "send it to my address", self.SNIPPET_CONFIG, pro=lambda _f: True
        )

        self.assertEqual(text, "send it to 12 Rue Oberkampf")

    def test_snippets_do_not_expand_for_a_free_one(self):
        text = expand_gated_snippets(
            "send it to my address", self.SNIPPET_CONFIG, pro=lambda _f: False
        )

        self.assertEqual(text, "send it to my address")

    def test_unreadable_snippets_cost_the_expansion_not_the_transcript(self):
        text = expand_gated_snippets(
            "hello", {"snippets": ["not an object"]}, pro=lambda _f: True
        )

        self.assertEqual(text, "hello")

    # -- coding mode ------------------------------------------------------

    def test_the_spoken_code_transform_is_its_own_feature(self):
        plan = CleanupPlan("code", "neutral", True)
        seen = []

        def transform(text, language):
            seen.append(language)
            return "--force"

        run_cleanup(
            "dash dash force",
            plan,
            cleanup=_RecordingCleanup(),
            transform_code=transform,
            pro=lambda feature: feature != "coding_mode",
        )
        self.assertEqual(seen, [])

        run_cleanup(
            "dash dash force",
            plan,
            cleanup=_RecordingCleanup(),
            transform_code=transform,
            pro=lambda _feature: True,
        )
        self.assertEqual(seen, ["en"])


class UserDataInLogsTests(GateTestCase):
    """The two config readers that used to log the exception's message.

    ``%s`` on the exception prints what it quotes, and both of these quote the
    user's own text — a cleanup mode id, a snippet trigger or its body. The log
    is not the place for either, so only the type name goes in.
    """

    SECRET = "my-private-snippet-body"

    def test_an_unreadable_cleanup_mode_logs_the_type_only(self):
        with patch("murmur.mode_from_config", side_effect=ValueError(self.SECRET)):
            with self.assertLogs("murmur", level="WARNING") as caught:
                self.assertEqual(configured_mode_id(_config()), FREE_MODE_ID)

        self.assertIn("ValueError", caught.output[0])
        self.assertNotIn(self.SECRET, caught.output[0])

    def test_unreadable_snippets_log_the_type_only(self):
        def load(_config_dict):
            raise ValueError(self.SECRET)

        with self.assertLogs("murmur", level="WARNING") as caught:
            text = expand_gated_snippets("hello", _config(), load=load)

        self.assertEqual(text, "hello")
        self.assertIn("ValueError", caught.output[0])
        self.assertNotIn(self.SECRET, caught.output[0])


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


class CleanupPlanTests(GateTestCase):
    def test_all_three_gates_open(self):
        plan = cleanup_plan(_config(cleanup_mode="message"), _context())

        self.assertTrue(plan.enabled)
        self.assertEqual(plan.mode_id, "message")
        self.assertEqual(plan.tone_id, "neutral")
        self.assertIsNone(plan.reason)

    def test_without_pro_nothing_runs(self):
        # A free install falls back to dictation as well, so the plan reports
        # the entitlement rather than the mode: the reason a user cannot clean
        # up is the licence, not a setting they could change.
        self.publish(pro=False)

        plan = cleanup_plan(_config(cleanup_mode="message"), _context())

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


class RunCleanupTests(GateTestCase):
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


class SettingsServicesTests(GateTestCase):
    """The one dict Settings is handed, and what each key means."""

    KEYS = {
        "usage",
        "license",
        "pro_gate",
        "keychain",
        "secret_store_volatile",
        "scheduler",
        "version",
        "build_info",
        "persistence",
        "audio_dir",
    }

    def _services(self, keychain=None, config=None, loads=None, usage=None, license=None):
        snapshot = dict(config or {})

        def runtime_config():
            if loads is not None:
                loads.append(dict(snapshot))
            return dict(snapshot)

        app = SimpleNamespace(
            persistence=object(),
            _keychain=lambda: keychain,
            runtime_config=runtime_config,
            usage=usage,
            license_service=license,
        )
        return MurmurApp._settings_services(app), app

    def test_every_key_the_window_documents_is_present(self):
        services, app = self._services()

        self.assertEqual(set(services), self.KEYS)
        self.assertIs(services["persistence"], app.persistence)
        self.assertIs(services["pro_gate"], is_pro_feature_enabled)
        self.assertEqual(services["version"], APP_VERSION)
        self.assertEqual(services["audio_dir"], AUDIO_DIR)
        self.assertIsInstance(services["build_info"], dict)

    def test_the_usage_provider_is_the_summary_callable_the_engine_tab_wants(self):
        usage = UsageService(config_store=_FakeConfigStore())
        services, _ = self._services(usage=usage)

        self.assertEqual(services["usage"], usage.summary)
        # And it answers: the tab calls it on every refresh.
        self.assertEqual(services["usage"]().cloud_words, 0)

    def test_the_licence_provider_is_the_service_itself(self):
        # The Account tab binds four of its methods; a summary would not do.
        service = object()
        services, _ = self._services(license=service)

        self.assertIs(services["license"], service)

    def test_a_build_without_the_two_services_still_opens_settings(self):
        services, _ = self._services(usage=None, license=None)

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

    def test_the_pro_gate_never_reads_the_config_file(self):
        # It answers from the entitlements the licence service published, so
        # the tabs may ask it once per gated control without touching disk.
        loads = []
        services, _ = self._services(loads=loads)
        gate = services["pro_gate"]

        self.publish(pro=True, cloud_voice=True)
        self.assertTrue(gate("cloud_voice"))
        self.assertTrue(gate("cleanup"))
        self.publish(pro=False)
        self.assertFalse(gate("cleanup"))
        self.assertEqual(loads, [])

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
                    usage=None,
                    license_service=None,
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


# ---------------------------------------------------------------------------
# Wave 4: engine routing, the fallback, the meter and the licence
# ---------------------------------------------------------------------------
#
# ``MurmurApp`` cannot be constructed without a menu bar, so every method below
# is called unbound against a stand-in ``self`` carrying only what that method
# actually reads. That is half the test: a routing decision that reached past
# these attributes would fail here rather than in production.


class _FakeConfigStore:
    """The ``load()``/``save()`` pair :class:`UsageService` counts through."""

    def __init__(self, config=None):
        self.config = {**USAGE_DEFAULTS, **(config or {})}
        self.saves = 0

    def load(self):
        return dict(self.config)

    def save(self, config):
        self.config = dict(config)
        self.saves += 1


class _FakeTranscript:
    def __init__(self, text="hello", duration_s=3.0, hints_applied=None):
        self.text = text
        self.duration_s = duration_s
        self.hints_applied = hints_applied


class _FakeEngine:
    """A speech engine that answers, or raises what it was told to raise."""

    def __init__(self, engine_id="cloud", raises=None, text="from the cloud"):
        self.engine_id = engine_id
        self._raises = raises
        self._text = text
        self.calls = []

    def transcribe(self, wav_path, language=None, hints=None, long_form=False):
        self.calls.append((str(wav_path), language, long_form))
        if self._raises is not None:
            raise self._raises
        return _FakeTranscript(text=self._text)

    def load(self):
        pass

    def runtime_summary(self):
        return f"fake {self.engine_id}"


def _usage(config=None, clock=None):
    store = _FakeConfigStore(config)
    kwargs = {"clock": clock} if clock is not None else {}
    return UsageService(config_store=store, **kwargs), store


def _bind_hosted_config(app):
    """Bind the one door every hosted client goes through. Returns ``app``.

    ``_hosted_config`` pins the proxy origin, so a fixture that skips it is
    testing a path the app does not have.
    """
    app._hosted_config = lambda cfg: MurmurApp._hosted_config(app, cfg)
    return app


class RoutingDecisionTests(GateTestCase):
    """The whole routing table, through the app's own decision method."""

    def _app(self, config=None, entitlements=None, lease=True, key=True, engine_id="whispercpp"):
        if entitlements is not None:
            set_current_entitlements(entitlements)
        usage, _store = _usage()
        keychain = SimpleNamespace(has=lambda _name: key)
        license_service = SimpleNamespace(
            current_lease_token=lambda: "lease-token" if lease else None
        )
        return SimpleNamespace(
            engine_id=engine_id,
            usage=usage,
            license_service=license_service,
            _keychain=lambda: keychain,
        ), dict(_config(**(config or {})))

    def route(self, clip_seconds=10.0, **kwargs):
        app, config = self._app(**kwargs)
        return MurmurApp._route_for(app, config, clip_seconds)

    def test_cloud_when_entitled_and_under_the_limit(self):
        route = self.route(
            config={"cloud_mode": "murmur_cloud"},
            entitlements=_entitlements(cloud_voice=True),
        )

        self.assertEqual(route.engine_id, ENGINE_CLOUD)
        self.assertIsNone(route.notice)

    def test_off_stays_on_the_engine_the_user_chose(self):
        route = self.route(
            config={"cloud_mode": "off"}, entitlements=_entitlements(cloud_voice=True)
        )

        self.assertEqual(route.engine_id, "whispercpp")
        self.assertIsNone(route.notice)

    def test_own_key_with_a_key_goes_to_byok(self):
        route = self.route(config={"cloud_mode": "own_key"}, key=True)

        self.assertEqual(route.engine_id, ENGINE_BYOK)

    def test_own_key_without_a_key_says_where_to_put_one(self):
        route = self.route(config={"cloud_mode": "own_key"}, key=False)

        self.assertEqual(route.engine_id, "whispercpp")
        self.assertEqual(route.notice, NOTICE_ADD_KEY)

    def test_murmur_cloud_without_a_lease_asks_for_a_sign_in(self):
        route = self.route(
            config={"cloud_mode": "murmur_cloud"},
            entitlements=_entitlements(cloud_voice=True),
            lease=False,
        )

        self.assertEqual(route.engine_id, "whispercpp")
        self.assertEqual(route.notice, NOTICE_SIGN_IN)

    def test_a_recording_over_an_hour_is_transcribed_here(self):
        route = self.route(
            config={"cloud_mode": "murmur_cloud"},
            entitlements=_entitlements(cloud_voice=True),
            clip_seconds=3601.0,
        )

        self.assertEqual(route.engine_id, "whispercpp")
        self.assertEqual(route.notice, NOTICE_CLIP_TOO_LONG)

    def test_the_trial_reaches_the_cloud_without_a_paid_entitlement(self):
        route = self.route(
            config={"cloud_mode": "murmur_cloud"},
            entitlements=_entitlements(pro=False, cloud_voice=False, trial_minutes=60),
        )

        self.assertEqual(route.engine_id, ENGINE_CLOUD)

    def test_a_spent_trial_and_no_plan_is_a_sign_in(self):
        set_current_entitlements(_entitlements(pro=False, cloud_voice=False))
        usage, _store = _usage({"cloud_trial_seconds_used": 3600.0})
        app = SimpleNamespace(
            engine_id="whispercpp",
            usage=usage,
            license_service=SimpleNamespace(current_lease_token=lambda: "token"),
            _keychain=lambda: None,
        )

        route = MurmurApp._route_for(app, _config(cloud_mode="murmur_cloud"), 10.0)

        self.assertEqual(route.engine_id, "whispercpp")
        self.assertEqual(route.notice, NOTICE_SIGN_IN)

    def test_the_soft_limit_switches_to_local_with_the_notice(self):
        # 95% of the allowance is the one fallback Murmur takes on its own.
        set_current_entitlements(_entitlements(cloud_voice=True))
        usage, _store = _usage(
            {
                "usage_allowance_minutes": 100.0,
                "usage_remote_minutes_used": 95.0,
                "usage_allowance_fetched_at": datetime.now().isoformat(),
                "usage_month": datetime.now().strftime("%Y-%m"),
            }
        )
        app = SimpleNamespace(
            engine_id="voxtral_mlx",
            usage=usage,
            license_service=SimpleNamespace(current_lease_token=lambda: "token"),
            _keychain=lambda: None,
        )

        route = MurmurApp._route_for(app, _config(cloud_mode="murmur_cloud"), 10.0)

        self.assertEqual(route.engine_id, "voxtral_mlx")
        self.assertEqual(route.notice, ALLOWANCE_MESSAGE)

    def test_just_under_the_soft_limit_still_goes_to_the_cloud(self):
        set_current_entitlements(_entitlements(cloud_voice=True))
        usage, _store = _usage(
            {
                "usage_allowance_minutes": 100.0,
                "usage_remote_minutes_used": 94.0,
                "usage_allowance_fetched_at": datetime.now().isoformat(),
                "usage_month": datetime.now().strftime("%Y-%m"),
            }
        )
        app = SimpleNamespace(
            engine_id="whispercpp",
            usage=usage,
            license_service=SimpleNamespace(current_lease_token=lambda: "token"),
            _keychain=lambda: None,
        )

        route = MurmurApp._route_for(app, _config(cloud_mode="murmur_cloud"), 10.0)

        self.assertEqual(route.engine_id, ENGINE_CLOUD)

    def test_an_unreadable_keychain_reads_as_no_key(self):
        class Broken:
            def has(self, _name):
                raise RuntimeError("locked")

        self.assertFalse(own_key_present(Broken(), _config()))
        self.assertFalse(own_key_present(None, _config()))

    def test_an_unreadable_lease_reads_as_no_lease(self):
        class Broken:
            def current_lease_token(self):
                raise RuntimeError("locked")

        self.assertFalse(lease_is_present(Broken()))
        self.assertFalse(lease_is_present(None))


class CloudFallbackTests(GateTestCase):
    """The clip the proxy refused is re-run here, once, with one notice."""

    def _app(self, remote):
        local = _FakeEngine("whispercpp", text="from this Mac")
        notices = []

        def announce(notice, *, once_key=None):
            notices.append(notice)

        app = SimpleNamespace(
            engine=local,
            engine_id="whispercpp",
            _engine_lock=threading.Lock(),
            _remote_engine_for=lambda engine_id, config: remote,
            _announce_route=announce,
        )
        app._engine_for_route = lambda engine_id, config: MurmurApp._engine_for_route(
            app, engine_id, config
        )
        app._run_engine = lambda *a, **k: MurmurApp._run_engine(app, *a, **k)
        return app, local, notices

    def _run(self, route, remote, config=None):
        app, local, notices = self._app(remote)
        transcript, engine_id = MurmurApp._transcribe_routed(
            app, route, config or _config(), "/tmp/clip.wav", language="en", hints=None
        )
        return transcript, engine_id, local, notices

    def test_an_exhausted_allowance_falls_back_and_says_so(self):
        remote = _FakeEngine("cloud", raises=CloudAllowanceExhausted("spent"))
        route = SimpleNamespace(engine_id=ENGINE_CLOUD, notice=None, reason="cloud")

        transcript, engine_id, local, notices = self._run(route, remote)

        self.assertEqual(transcript.text, "from this Mac")
        self.assertEqual(engine_id, "whispercpp")
        self.assertEqual(len(local.calls), 1)
        self.assertIn(ALLOWANCE_MESSAGE, notices)

    def test_a_rejected_lease_falls_back_and_asks_for_a_sign_in(self):
        remote = _FakeEngine("cloud", raises=CloudAuthError("no lease"))
        route = SimpleNamespace(engine_id=ENGINE_CLOUD, notice=None, reason="cloud")

        transcript, engine_id, local, notices = self._run(route, remote)

        self.assertEqual(transcript.text, "from this Mac")
        self.assertEqual(engine_id, "whispercpp")
        self.assertIn(NOTICE_SIGN_IN, notices)

    def test_a_transient_proxy_error_falls_back_without_a_notice(self):
        # The user asked for a transcript; a network blip at the proxy is not
        # theirs to act on, so it costs the round trip and nothing else.
        remote = _FakeEngine("cloud", raises=EngineError("gateway timeout"))
        route = SimpleNamespace(engine_id=ENGINE_CLOUD, notice=None, reason="cloud")

        transcript, engine_id, local, notices = self._run(route, remote)

        self.assertEqual(transcript.text, "from this Mac")
        self.assertEqual(engine_id, "whispercpp")
        self.assertEqual([n for n in notices if n], [])

    def test_a_failure_that_is_not_the_engine_still_propagates(self):
        remote = _FakeEngine("cloud", raises=RuntimeError("boom"))
        route = SimpleNamespace(engine_id=ENGINE_CLOUD, notice=None, reason="cloud")

        with self.assertRaises(RuntimeError):
            self._run(route, remote)

    def test_a_local_engine_that_fails_is_never_re_run(self):
        # Re-running the engine that just failed would only fail again, and a
        # transcript that could not be produced must not look like one that was.
        local = _FakeEngine("whispercpp", raises=EngineError("model gone"))
        app = SimpleNamespace(
            engine=local,
            engine_id="whispercpp",
            _engine_lock=threading.Lock(),
            _remote_engine_for=lambda engine_id, config: None,
            _announce_route=lambda notice, **kwargs: None,
        )
        app._engine_for_route = lambda engine_id, config: MurmurApp._engine_for_route(
            app, engine_id, config
        )
        app._run_engine = lambda *a, **k: MurmurApp._run_engine(app, *a, **k)
        route = SimpleNamespace(engine_id="whispercpp", notice=None, reason="cloud off")

        with self.assertRaises(EngineError):
            MurmurApp._transcribe_routed(
                app, route, _config(), "/tmp/clip.wav", language="en", hints=None
            )
        self.assertEqual(len(local.calls), 1)

    def test_a_successful_cloud_clip_never_touches_the_local_engine(self):
        remote = _FakeEngine("cloud")
        route = SimpleNamespace(engine_id=ENGINE_CLOUD, notice=None, reason="cloud")

        transcript, engine_id, local, _notices = self._run(route, remote)

        self.assertEqual(transcript.text, "from the cloud")
        self.assertEqual(engine_id, ENGINE_CLOUD)
        self.assertEqual(local.calls, [])

    def test_a_build_failure_that_is_not_an_engine_error_still_falls_back(self):
        # ``build_engine`` raises ValueError for a missing own-key provider and
        # the engines raise it for a non-HTTPS endpoint. That used to escape as
        # "Transcription failed" and cost the user the words they just spoke.
        app, local, notices = self._app(remote=None)
        builder = _bind_hosted_config(
            SimpleNamespace(
                cloud_base_url=DEFAULT_CLOUD_BASE_URL,
                _base_url_drift_logged=True,
                _remote_engine=None,
                _remote_engine_key=None,
                _remote_engine_lock=threading.Lock(),
                license_service=None,
                _keychain=lambda: None,
            )
        )
        app._remote_engine_for = lambda engine_id, config: MurmurApp._remote_engine_for(
            builder, engine_id, config
        )
        route = SimpleNamespace(engine_id=ENGINE_CLOUD, notice=None, reason="cloud")

        transcript, engine_id = MurmurApp._transcribe_routed(
            app, route, _config(), "/tmp/clip.wav", language="en", hints=None
        )

        # ``build_engine`` refuses "cloud" without a licence service.
        self.assertEqual(transcript.text, "from this Mac")
        self.assertEqual(engine_id, "whispercpp")


class OwnKeyFallbackTests(GateTestCase):
    """An own-key failure is the user's to fix, so it is worded as theirs.

    Routing these through ``after_cloud_failure`` matched neither proxy
    exception, so a revoked key fell back with no notice at all and a log line
    blaming Murmur Cloud — a silent, permanent downgrade.
    """

    def _run(self, error, config=None):
        remote = _FakeEngine("byok", raises=error)
        local = _FakeEngine("whispercpp", text="from this Mac")
        notices = []
        app = SimpleNamespace(
            engine=local,
            engine_id="whispercpp",
            _engine_lock=threading.Lock(),
            _remote_engine_for=lambda engine_id, cfg: remote,
            _announce_route=lambda notice, **kwargs: notices.append(notice),
        )
        app._engine_for_route = lambda engine_id, cfg: MurmurApp._engine_for_route(
            app, engine_id, cfg
        )
        app._run_engine = lambda *a, **k: MurmurApp._run_engine(app, *a, **k)
        route = SimpleNamespace(engine_id=ENGINE_BYOK, notice=None, reason="own key")
        transcript, engine_id = MurmurApp._transcribe_routed(
            app,
            route,
            config or _config(cloud_mode="own_key", byok_provider="mistral"),
            "/tmp/clip.wav",
            language="en",
            hints=None,
        )
        return transcript, engine_id, notices

    def test_a_rejected_key_says_where_to_fix_it(self):
        _t, engine_id, notices = self._run(ByokAuthError("401"))

        self.assertEqual(engine_id, "whispercpp")
        self.assertIn(NOTICE_KEY_REJECTED.format(provider="Mistral"), notices)

    def test_a_rate_limited_provider_says_so_without_blaming_the_key(self):
        _t, engine_id, notices = self._run(ByokRateLimited("429", retry_after_s=30.0))

        self.assertEqual(engine_id, "whispercpp")
        self.assertIn(NOTICE_KEY_RATE_LIMITED.format(provider="Mistral"), notices)

    def test_any_other_own_key_failure_still_tells_the_user_something(self):
        _t, engine_id, notices = self._run(EngineError("gateway timeout"))

        self.assertEqual(engine_id, "whispercpp")
        self.assertIn(NOTICE_KEY_FAILED.format(provider="Mistral"), notices)

    def test_the_notice_names_the_provider_the_user_configured(self):
        _t, _e, notices = self._run(
            ByokAuthError("401"),
            config=_config(cloud_mode="own_key", byok_provider="openai"),
        )

        self.assertIn(NOTICE_KEY_REJECTED.format(provider="OpenAI"), notices)

    def test_the_cloud_wording_never_reaches_an_own_key_failure(self):
        _t, _e, notices = self._run(ByokAuthError("401"))

        self.assertNotIn(NOTICE_SIGN_IN, notices)

    def test_the_log_names_the_engine_that_actually_failed(self):
        with self.assertLogs("murmur", level="INFO") as caught:
            self._run(ByokAuthError("401"))

        line = "\n".join(caught.output)
        self.assertIn(ENGINE_BYOK, line)
        self.assertNotIn("Murmur Cloud", line)


class OwnKeyNoticeTable(unittest.TestCase):
    """:func:`after_byok_failure` as a table, and the name it puts in it."""

    def _route(self, error, provider="Mistral"):
        return after_byok_failure(error, local_engine_id="whispercpp", provider=provider)

    def test_a_rejected_key_points_at_settings(self):
        route = self._route(ByokAuthError("401"))

        self.assertEqual(route.engine_id, "whispercpp")
        self.assertEqual(route.reason, "byok key rejected")
        self.assertIn("Settings", route.notice)

    def test_a_rate_limit_is_not_a_bad_key(self):
        route = self._route(ByokRateLimited("429"))

        self.assertEqual(route.reason, "byok rate limited")
        self.assertNotIn("rejected", route.notice)

    def test_everything_else_is_generic_but_never_silent(self):
        route = self._route(EngineError("boom"))

        self.assertEqual(route.reason, "byok failed")
        self.assertTrue(route.notice)

    def test_it_always_falls_back_to_the_engine_the_user_chose(self):
        for error in (ByokAuthError("x"), ByokRateLimited("x"), EngineError("x")):
            route = after_byok_failure(
                error, local_engine_id="voxtral_mlx", provider="OpenAI"
            )
            self.assertEqual(route.engine_id, "voxtral_mlx")

    def test_the_provider_name_comes_from_config(self):
        self.assertEqual(byok_provider_name(_config(byok_provider="mistral")), "Mistral")
        self.assertEqual(byok_provider_name(_config(byok_provider="openai")), "OpenAI")

    def test_an_unknown_or_missing_provider_still_reads_as_a_sentence(self):
        self.assertEqual(byok_provider_name(_config(byok_provider="acme")), "Acme")
        self.assertEqual(byok_provider_name({}), "Your provider")


class SessionNoticeTests(GateTestCase):
    """The own-key notices are said once a session, not once an utterance."""

    def _app(self):
        usage, _store = _usage({"usage_month": datetime.now().strftime("%Y-%m")})
        return SimpleNamespace(usage=usage, _session_notices=set())

    def test_a_kind_of_notice_is_shown_once_and_then_kept_quiet(self):
        app = self._app()
        shown = []

        with patch("murmur.rumps.notification", lambda *a: shown.append(a[-1])):
            MurmurApp._announce_route(app, "Your Mistral key was rejected", once_key="k")
            MurmurApp._announce_route(app, "Your Mistral key was rejected", once_key="k")

        self.assertEqual(shown, ["Your Mistral key was rejected"])

    def test_a_different_kind_is_still_worth_saying(self):
        app = self._app()
        shown = []

        with patch("murmur.rumps.notification", lambda *a: shown.append(a[-1])):
            MurmurApp._announce_route(app, "rejected", once_key="byok key rejected")
            MurmurApp._announce_route(app, "rate limited", once_key="byok rate limited")

        self.assertEqual(shown, ["rejected", "rate limited"])

    def test_notices_without_a_key_are_unaffected(self):
        app = self._app()
        shown = []

        with patch("murmur.rumps.notification", lambda *a: shown.append(a[-1])):
            MurmurApp._announce_route(app, NOTICE_SIGN_IN)
            MurmurApp._announce_route(app, NOTICE_SIGN_IN)

        self.assertEqual(shown, [NOTICE_SIGN_IN, NOTICE_SIGN_IN])


class RouteNoticeTests(GateTestCase):
    """The allowance notice is shown once per period; the rest whenever they apply."""

    def test_the_allowance_notice_is_shown_only_while_it_is_pending(self):
        self.assertEqual(
            notice_to_show(ALLOWANCE_MESSAGE, fallback_pending=True), ALLOWANCE_MESSAGE
        )
        self.assertIsNone(notice_to_show(ALLOWANCE_MESSAGE, fallback_pending=False))

    def test_every_other_notice_answers_a_choice_and_is_always_shown(self):
        self.assertEqual(
            notice_to_show(NOTICE_ADD_KEY, fallback_pending=False), NOTICE_ADD_KEY
        )
        self.assertEqual(
            notice_to_show(NOTICE_SIGN_IN, fallback_pending=False), NOTICE_SIGN_IN
        )

    def test_no_notice_is_no_notification(self):
        self.assertIsNone(notice_to_show(None, fallback_pending=True))
        self.assertIsNone(notice_to_show("", fallback_pending=True))

    def test_showing_the_allowance_notice_marks_it_shown_exactly_once(self):
        usage, store = _usage({"usage_month": datetime.now().strftime("%Y-%m")})
        shown = []
        app = SimpleNamespace(usage=usage)

        with patch("murmur.rumps.notification", lambda *a: shown.append(a[-1])):
            MurmurApp._announce_route(app, ALLOWANCE_MESSAGE)
            MurmurApp._announce_route(app, ALLOWANCE_MESSAGE)

        self.assertEqual(shown, [ALLOWANCE_MESSAGE])
        self.assertTrue(store.config["cloud_fallback_notice_shown"])


class UsageRecordingTests(GateTestCase):
    """What each origin adds to the meter, and what the trial costs."""

    def _app(self, usage):
        return SimpleNamespace(usage=usage)

    def test_a_cloud_clip_counts_cloud_minutes_and_words(self):
        set_current_entitlements(_entitlements(cloud_voice=True))
        usage, store = _usage()

        MurmurApp._record_usage(self._app(usage), ORIGIN_CLOUD, 30.0, 42)

        self.assertEqual(store.config["usage_cloud_seconds"], 30.0)
        self.assertEqual(store.config["usage_cloud_words"], 42)
        self.assertEqual(store.config["usage_local_seconds"], 0.0)

    def test_a_local_clip_counts_local_minutes_and_words(self):
        usage, store = _usage()

        MurmurApp._record_usage(self._app(usage), ORIGIN_LOCAL, 12.0, 7)

        self.assertEqual(store.config["usage_local_seconds"], 12.0)
        self.assertEqual(store.config["usage_cloud_seconds"], 0.0)

    def test_own_key_work_is_billed_by_the_user_and_counted_nowhere(self):
        usage, store = _usage()

        MurmurApp._record_usage(self._app(usage), ORIGIN_BYOK, 60.0, 100)

        self.assertEqual(store.config["usage_cloud_seconds"], 0.0)
        self.assertEqual(store.config["usage_local_seconds"], 0.0)
        self.assertEqual(store.config["cloud_trial_seconds_used"], 0.0)

    def test_a_trial_account_spends_the_trial_on_a_cloud_clip(self):
        set_current_entitlements(_entitlements(pro=False, cloud_voice=False, trial_minutes=60))
        usage, store = _usage()

        MurmurApp._record_usage(self._app(usage), ORIGIN_CLOUD, 45.0, 10)

        self.assertEqual(store.config["cloud_trial_seconds_used"], 45.0)

    def test_a_paying_account_never_has_its_trial_drained(self):
        set_current_entitlements(_entitlements(cloud_voice=True))
        usage, store = _usage()

        MurmurApp._record_usage(self._app(usage), ORIGIN_CLOUD, 45.0, 10)

        self.assertEqual(store.config["cloud_trial_seconds_used"], 0.0)
        self.assertFalse(should_consume_trial(get_current_entitlements()))

    def test_a_failed_write_never_costs_the_paste(self):
        class Broken:
            def record(self, *_a):
                raise OSError("disk full")

        MurmurApp._record_usage(self._app(Broken()), ORIGIN_LOCAL, 1.0, 1)  # no raise

    def test_no_usage_service_is_a_supported_state(self):
        MurmurApp._record_usage(self._app(None), ORIGIN_CLOUD, 1.0, 1)  # no raise


class AllowanceRefreshTests(GateTestCase):
    """The allowance is re-read off the dictation path, and only for the cloud."""

    def test_a_stale_reading_under_murmur_cloud_is_refreshed(self):
        usage, _store = _usage()  # nothing cached: stale by definition

        self.assertTrue(should_refresh_allowance(usage, cloud_mode="murmur_cloud"))

    def test_no_other_mode_polls_the_proxy(self):
        usage, _store = _usage()

        self.assertFalse(should_refresh_allowance(usage, cloud_mode="off"))
        self.assertFalse(should_refresh_allowance(usage, cloud_mode="own_key"))

    def test_a_fresh_reading_is_left_alone(self):
        usage, _store = _usage(
            {
                "usage_allowance_minutes": 100.0,
                "usage_allowance_fetched_at": datetime.now().isoformat(),
                "usage_month": datetime.now().strftime("%Y-%m"),
            }
        )

        self.assertFalse(should_refresh_allowance(usage, cloud_mode="murmur_cloud"))

    def test_without_a_usage_service_there_is_nothing_to_refresh(self):
        self.assertFalse(should_refresh_allowance(None, cloud_mode="murmur_cloud"))


class CloudCleanupSelectionTests(GateTestCase):
    """Which backend cleans the text, and what a refusal costs."""

    def _app(self, *, config, cloud_engine_active, client=None, local=None):
        runtime = SimpleNamespace(cleanup=local or (lambda text, prompt: "local"))
        app = SimpleNamespace(
            cleanup_runtime=runtime,
            license_service=SimpleNamespace(current_lease_token=lambda: "token"),
            engine_id="whispercpp",
            usage=None,
            cloud_base_url=DEFAULT_CLOUD_BASE_URL,
            _base_url_drift_logged=False,
            _cloud_cleanup_client=client,
            _cloud_cleanup_base_url=DEFAULT_CLOUD_BASE_URL if client else None,
            _announce_route=lambda notice, **kwargs: None,
            _record_usage=lambda *a: None,
        )
        app._hosted_config = lambda cfg: MurmurApp._hosted_config(app, cfg)
        app._cleanup_client = lambda cfg: MurmurApp._cleanup_client(app, cfg)
        app._cloud_cleanup_with_fallback = (
            lambda c: MurmurApp._cloud_cleanup_with_fallback(app, c)
        )
        chosen, is_local = MurmurApp._cleanup_callable(
            app, config, cloud_engine_active=cloud_engine_active
        )
        app.chose_local = is_local
        return app, chosen

    def test_cloud_cleanup_needs_the_switch_the_route_and_the_gate(self):
        client = SimpleNamespace(cleanup=lambda text, prompt: "cloud")
        config = _config(cleanup_cloud=True, cloud_base_url=DEFAULT_CLOUD_BASE_URL)

        _app, chosen = self._app(config=config, cloud_engine_active=True, client=client)
        self.assertIsNot(chosen, _app.cleanup_runtime.cleanup)

    def test_a_local_route_never_sends_the_text_up(self):
        # Keeping the audio here and sending the text up would break the
        # promise the Privacy tab makes.
        client = SimpleNamespace(cleanup=lambda text, prompt: "cloud")
        config = _config(cleanup_cloud=True)

        app, chosen = self._app(config=config, cloud_engine_active=False, client=client)
        self.assertIs(chosen, app.cleanup_runtime.cleanup)

    def test_a_free_install_cleans_nowhere_in_the_cloud(self):
        self.publish(pro=False)
        client = SimpleNamespace(cleanup=lambda text, prompt: "cloud")
        config = _config(cleanup_cloud=True)

        app, chosen = self._app(config=config, cloud_engine_active=True, client=client)
        self.assertIs(chosen, app.cleanup_runtime.cleanup)

    def test_the_switch_off_keeps_cleanup_on_this_mac(self):
        client = SimpleNamespace(cleanup=lambda text, prompt: "cloud")
        config = _config(cleanup_cloud=False)

        app, chosen = self._app(config=config, cloud_engine_active=True, client=client)
        self.assertIs(chosen, app.cleanup_runtime.cleanup)

    def test_a_refused_cloud_cleanup_falls_back_to_the_local_server(self):
        calls = []

        def local(text, prompt):
            calls.append(text)
            return CleanupResult(text="local cleaned", elapsed_s=0.1)

        def refusing(_text, _prompt):
            raise CloudAllowanceExhausted("spent")

        client = SimpleNamespace(cleanup=refusing)
        config = _config(cleanup_cloud=True)
        app, chosen = self._app(
            config=config, cloud_engine_active=True, client=client, local=local
        )

        result = chosen("dictated words", "be terse")

        self.assertEqual(result.text, "local cleaned")
        self.assertEqual(calls, ["dictated words"])

    def test_a_cloud_cleanup_that_ran_is_metered_in_words(self):
        metered = []
        client = SimpleNamespace(
            cleanup=lambda text, prompt: CleanupResult(text="one two three", elapsed_s=0.2)
        )
        config = _config(cleanup_cloud=True)
        app, chosen = self._app(config=config, cloud_engine_active=True, client=client)
        app._record_usage = lambda *a: metered.append(a)

        chosen("hi", "be terse")

        self.assertEqual(metered, [(ORIGIN_CLOUD, 0, 3)])

    def test_the_caller_is_told_which_backend_it_got(self):
        # It cannot work it out afterwards: ``self.cleanup_runtime.cleanup``
        # builds a fresh bound method on every access, so comparing the chosen
        # callable against it is always False and the local branch never ran.
        client = SimpleNamespace(cleanup=lambda text, prompt: "cloud")
        cloud, _ = self._app(
            config=_config(cleanup_cloud=True, cloud_base_url=DEFAULT_CLOUD_BASE_URL),
            cloud_engine_active=True,
            client=client,
        )
        local, _ = self._app(config=_config(cleanup_cloud=False), cloud_engine_active=True)

        self.assertFalse(cloud.chose_local)
        self.assertTrue(local.chose_local)

    def test_the_preparing_label_reaches_the_pill_on_a_cold_local_server(self):
        # The bound-method comparison meant this label was never shown, so the
        # pill sat on "working" through a 2 GB model load and read as a hang.
        labels = []
        runtime = SimpleNamespace(
            cleanup=lambda text, prompt: CleanupResult(text=text, elapsed_s=0.0),
            is_started=False,
        )
        app = SimpleNamespace(
            cleanup_runtime=runtime,
            _cleanup_callable=lambda cfg, cloud_engine_active: (runtime.cleanup, True),
            _set_model_menu_title=lambda *a, **k: None,
            _model_status_title="Loading model…",
            _record_usage=lambda *a: None,
            usage=None,
        )
        pill = SimpleNamespace(working=lambda label=None: labels.append(label))
        config = _config(cleanup_mode="message")

        with patch("murmur.capture_context", lambda include_selection=False: _context()):
            MurmurApp._clean_up_transcript(
                app, "hello", config, "en", vocabulary_from_config(config), pill
            )

        self.assertEqual(labels[0], CLEANUP_PREPARING_STATUS)

    def test_a_skipped_cloud_cleanup_is_not_metered(self):
        metered = []
        client = SimpleNamespace(
            cleanup=lambda text, prompt: CleanupResult(
                text="hi", elapsed_s=0.2, skipped=True, reason="rate limited"
            )
        )
        config = _config(cleanup_cloud=True)
        app, chosen = self._app(config=config, cloud_engine_active=True, client=client)
        app._record_usage = lambda *a: metered.append(a)

        chosen("hi", "be terse")

        self.assertEqual(metered, [])


class RemoteEngineCacheTests(unittest.TestCase):
    """A hosted engine is built once and follows a settings change."""

    def test_the_key_is_the_config_the_engine_was_built_from(self):
        config = _config(cloud_base_url="https://proxy.test", byok_provider="openai")

        self.assertEqual(
            remote_engine_key(ENGINE_BYOK, config),
            RemoteEngineKey(ENGINE_BYOK, "https://proxy.test", "openai", None),
        )

    def test_a_changed_provider_is_a_different_engine(self):
        first = remote_engine_key(ENGINE_BYOK, _config(byok_provider="mistral"))
        second = remote_engine_key(ENGINE_BYOK, _config(byok_provider="openai"))

        self.assertNotEqual(first, second)

    def test_a_changed_proxy_origin_is_a_different_engine(self):
        first = remote_engine_key(ENGINE_CLOUD, _config())
        second = remote_engine_key(ENGINE_CLOUD, _config(cloud_base_url="https://other.test"))

        self.assertNotEqual(first, second)

    def test_the_lease_is_not_part_of_the_key(self):
        # It is read through a callable at request time, so a sign-out and a
        # sign-in need no rebuild.
        self.assertNotIn("lease", str(remote_engine_key(ENGINE_CLOUD, _config())))

    def _app(self, base_url=DEFAULT_CLOUD_BASE_URL):
        return _bind_hosted_config(
            SimpleNamespace(
                cloud_base_url=base_url,
                _base_url_drift_logged=False,
                _remote_engine=None,
                _remote_engine_key=None,
                _remote_engine_lock=threading.Lock(),
                license_service=None,
                _keychain=lambda: None,
            )
        )

    def test_the_engine_is_built_once_and_then_reused(self):
        built = []

        def build(engine_id, **kwargs):
            built.append(engine_id)
            return _FakeEngine(engine_id)

        app = self._app()
        config = _config()
        with patch("murmur.build_engine", build):
            first = MurmurApp._remote_engine_for(app, ENGINE_CLOUD, config)
            second = MurmurApp._remote_engine_for(app, ENGINE_CLOUD, config)

        self.assertIs(first, second)
        self.assertEqual(built, [ENGINE_CLOUD])

    def test_a_changed_config_rebuilds_it(self):
        built = []

        def build(engine_id, **kwargs):
            built.append(kwargs["config"].get("byok_provider"))
            return _FakeEngine(engine_id)

        app = self._app()
        with patch("murmur.build_engine", build):
            MurmurApp._remote_engine_for(app, ENGINE_BYOK, _config(byok_provider="mistral"))
            MurmurApp._remote_engine_for(app, ENGINE_BYOK, _config(byok_provider="openai"))

        self.assertEqual(built, ["mistral", "openai"])


class PinnedProxyOriginTests(unittest.TestCase):
    """The proxy origin is read once, at launch, and every client uses that one.

    The licence service is built against the origin read at startup, but the
    engine key and the cleanup client used to re-read ``cloud_base_url`` from
    the live config on each request. Editing that key mid-session therefore
    sent the audio, the transcript and the cleanup text to another host while
    the lease still belonged to the first — and handed that host the lease.
    """

    PINNED = DEFAULT_CLOUD_BASE_URL
    ELSEWHERE = "https://elsewhere.test"

    def _app(self):
        return _bind_hosted_config(
            SimpleNamespace(
                cloud_base_url=self.PINNED,
                _base_url_drift_logged=False,
                _remote_engine=None,
                _remote_engine_key=None,
                _remote_engine_lock=threading.Lock(),
                license_service=SimpleNamespace(current_lease_token=lambda: "token"),
                _cloud_cleanup_client=None,
                _cloud_cleanup_base_url=None,
                _keychain=lambda: None,
            )
        )

    def test_an_unchanged_origin_is_the_same_dict(self):
        config = _config()

        self.assertIs(pinned_cloud_config(config, self.PINNED), config)

    def test_a_changed_origin_is_replaced_not_followed(self):
        config = _config(cloud_base_url=self.ELSEWHERE)

        pinned = pinned_cloud_config(config, self.PINNED)

        self.assertEqual(pinned[CONFIG_CLOUD_BASE_URL], self.PINNED)
        # And the caller's dict is untouched.
        self.assertEqual(config[CONFIG_CLOUD_BASE_URL], self.ELSEWHERE)

    def test_a_blank_origin_still_reads_as_the_pinned_one(self):
        pinned = pinned_cloud_config(_config(cloud_base_url="  "), self.PINNED)

        self.assertEqual(cloud_base_url(pinned), self.PINNED)

    def test_the_engine_key_never_carries_the_live_origin(self):
        app = self._app()
        seen = []

        def build(engine_id, **kwargs):
            seen.append(kwargs["config"].get(CONFIG_CLOUD_BASE_URL))
            return _FakeEngine(engine_id)

        with patch("murmur.build_engine", build):
            MurmurApp._remote_engine_for(
                app, ENGINE_CLOUD, _config(cloud_base_url=self.ELSEWHERE)
            )

        self.assertEqual(seen, [self.PINNED])
        self.assertEqual(app._remote_engine_key.base_url, self.PINNED)

    def test_the_cleanup_client_never_reaches_the_live_origin(self):
        app = self._app()
        seen = []

        class Client:
            def __init__(self, base_url, lease_provider):
                seen.append(base_url)

        with patch("murmur.CloudCleanupClient", Client):
            MurmurApp._cleanup_client(app, _config(cloud_base_url=self.ELSEWHERE))

        self.assertEqual(seen, [self.PINNED])

    def test_the_drift_is_said_once_a_session_not_once_a_dictation(self):
        app = self._app()
        config = _config(cloud_base_url=self.ELSEWHERE)

        with self.assertLogs("murmur", level="WARNING") as caught:
            MurmurApp._hosted_config(app, config)
            MurmurApp._hosted_config(app, config)

        self.assertEqual(len(caught.output), 1)
        self.assertIn("Restart", caught.output[0])

    def test_an_unchanged_origin_says_nothing_at_all(self):
        app = self._app()

        with patch("murmur.logger") as log:
            MurmurApp._hosted_config(app, _config())

        log.warning.assert_not_called()


class RemoteEngineBuildFailureTests(unittest.TestCase):
    """Building a hosted engine can fail; it must fail as an ``EngineError``.

    ``_transcribe_routed`` falls back on ``EngineError`` and nothing else, so a
    ``ValueError`` out of ``build_engine`` (a missing own-key provider) or out
    of an engine's HTTPS check took the transcript with it.
    """

    def _app(self):
        return _bind_hosted_config(
            SimpleNamespace(
                cloud_base_url=DEFAULT_CLOUD_BASE_URL,
                _base_url_drift_logged=True,
                _remote_engine=None,
                _remote_engine_key=None,
                _remote_engine_lock=threading.Lock(),
                license_service=None,
                _keychain=lambda: None,
            )
        )

    def test_a_value_error_from_the_factory_becomes_an_engine_error(self):
        def build(engine_id, **kwargs):
            raise ValueError("engine 'byok' needs config['byok_provider']")

        with patch("murmur.build_engine", build):
            with self.assertRaises(EngineError) as caught:
                MurmurApp._remote_engine_for(self._app(), ENGINE_BYOK, _config())

        self.assertIn("ValueError", str(caught.exception))

    def test_a_failure_in_load_becomes_an_engine_error_too(self):
        class Failing(_FakeEngine):
            def load(self):
                raise ValueError("base_url must be https")

        with patch("murmur.build_engine", lambda engine_id, **kw: Failing(engine_id)):
            with self.assertRaises(EngineError):
                MurmurApp._remote_engine_for(self._app(), ENGINE_CLOUD, _config())

    def test_the_wrapped_message_names_the_type_and_nothing_else(self):
        def build(engine_id, **kwargs):
            raise ValueError("cloud_base_url=https://user:sekrit@host is not https")

        with patch("murmur.build_engine", build):
            with self.assertRaises(EngineError) as caught:
                MurmurApp._remote_engine_for(self._app(), ENGINE_CLOUD, _config())

        self.assertNotIn("sekrit", str(caught.exception))

    def test_an_engine_error_keeps_its_own_subclass(self):
        # ``after_byok_failure`` branches on it: re-wrapping a rejected key
        # would turn "check your key" into "something failed".
        class Failing(_FakeEngine):
            def load(self):
                raise ByokAuthError("401")

        with patch("murmur.build_engine", lambda engine_id, **kw: Failing(engine_id)):
            with self.assertRaises(ByokAuthError):
                MurmurApp._remote_engine_for(self._app(), ENGINE_BYOK, _config())

    def test_a_failed_build_is_not_cached_as_the_engine(self):
        app = self._app()
        with patch("murmur.build_engine", lambda *a, **k: (_ for _ in ()).throw(ValueError())):
            with self.assertRaises(EngineError):
                MurmurApp._remote_engine_for(app, ENGINE_CLOUD, _config())

        self.assertIsNone(app._remote_engine)


class EntitlementRefreshTests(GateTestCase):
    """When the lease is renewed, and what the gate is told afterwards."""

    def test_the_first_pass_always_refreshes(self):
        self.assertTrue(should_refresh_entitlements(last_refresh_at=None, now=1000.0))

    def test_nothing_refreshes_again_before_the_interval(self):
        self.assertFalse(
            should_refresh_entitlements(last_refresh_at=1000.0, now=1000.0 + 60)
        )

    def test_it_refreshes_once_the_interval_has_passed(self):
        self.assertTrue(
            should_refresh_entitlements(
                last_refresh_at=1000.0, now=1000.0 + ENTITLEMENT_REFRESH_INTERVAL_S
            )
        )

    def test_a_clock_that_went_backwards_refreshes_rather_than_stalling(self):
        self.assertTrue(should_refresh_entitlements(last_refresh_at=5000.0, now=1000.0))

    def test_publishing_reaches_the_one_gate(self):
        published = _entitlements(pro=True, cloud_voice=True)
        service = SimpleNamespace(current_entitlements=lambda: published)

        result = publish_entitlements(service)

        self.assertIs(result, published)
        self.assertTrue(is_pro_feature_enabled("cleanup"))
        self.assertTrue(is_pro_feature_enabled("cloud_voice"))

    def test_no_licence_service_drops_to_the_free_tier(self):
        self.assertIsNone(publish_entitlements(None))
        self.assertFalse(is_pro_feature_enabled("cleanup"))

    def test_a_locked_keychain_keeps_whatever_the_gate_had(self):
        class Broken:
            def current_entitlements(self):
                raise RuntimeError("locked")

        self.assertIsNone(publish_entitlements(Broken()))
        self.assertTrue(is_pro_feature_enabled("cleanup"))  # the class default


class EntitlementBackoffTests(GateTestCase):
    """A renewal that failed is retried soon, not in six hours.

    The clock used to be stamped *before* the call, so a launch with no network
    counted as refreshed and the next attempt was a full interval away — an
    afternoon on the free tier for a paying Mac.
    """

    def _app(self, service):
        if service is not None:
            service.current_entitlements = lambda: _entitlements()
        return SimpleNamespace(
            license_service=service,
            _entitlements_refreshed_at=None,
            _entitlement_failures=0,
            _entitlements_retry_at=None,
            _refresh_account_menu=lambda entitlements=None: None,
        )

    def _once(self, app):
        app._entitlement_refresh_due = lambda: MurmurApp._entitlement_refresh_due(app)
        app._entitlement_refresh_succeeded = (
            lambda: MurmurApp._entitlement_refresh_succeeded(app)
        )
        app._entitlement_refresh_failed = (
            lambda error: MurmurApp._entitlement_refresh_failed(app, error)
        )
        MurmurApp._refresh_entitlements_once(app)

    def test_the_first_failure_waits_five_minutes(self):
        self.assertEqual(next_refresh_delay(1), ENTITLEMENT_RETRY_BASE_S)

    def test_the_wait_doubles_and_then_stops_at_an_hour(self):
        self.assertEqual(next_refresh_delay(2), 2 * ENTITLEMENT_RETRY_BASE_S)
        self.assertEqual(next_refresh_delay(3), 4 * ENTITLEMENT_RETRY_BASE_S)
        self.assertEqual(next_refresh_delay(9), ENTITLEMENT_RETRY_MAX_S)

    def test_the_backoff_never_exceeds_the_interval_it_replaces(self):
        self.assertLess(next_refresh_delay(99), ENTITLEMENT_REFRESH_INTERVAL_S)

    def test_a_failed_renewal_does_not_stamp_the_clock(self):
        calls = []

        def refresh():
            calls.append(1)
            raise RuntimeError("no network")

        app = self._app(SimpleNamespace(refresh_if_needed=refresh))
        with self.assertLogs("murmur", level="WARNING"):
            self._once(app)

        self.assertIsNone(app._entitlements_refreshed_at)
        self.assertEqual(app._entitlement_failures, 1)
        self.assertIsNotNone(app._entitlements_retry_at)
        self.assertEqual(len(calls), 1)

    def test_the_retry_is_not_attempted_before_its_time(self):
        app = self._app(
            SimpleNamespace(refresh_if_needed=lambda: (_ for _ in ()).throw(OSError()))
        )
        with self.assertLogs("murmur", level="WARNING"):
            self._once(app)
        self._once(app)  # a minute later, well inside the backoff

        self.assertEqual(app._entitlement_failures, 1)

    def test_the_retry_runs_once_its_time_has_come(self):
        attempts = []

        def refresh():
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("no network")

        app = self._app(SimpleNamespace(refresh_if_needed=refresh))
        with self.assertLogs("murmur", level="WARNING"):
            self._once(app)
        app._entitlements_retry_at = time.time() - 1
        self._once(app)

        self.assertEqual(len(attempts), 2)
        self.assertEqual(app._entitlement_failures, 0)
        self.assertIsNone(app._entitlements_retry_at)
        self.assertIsNotNone(app._entitlements_refreshed_at)

    def test_a_success_stamps_the_clock_and_waits_the_full_interval(self):
        app = self._app(SimpleNamespace(refresh_if_needed=lambda: None))

        self._once(app)
        stamped = app._entitlements_refreshed_at

        self.assertIsNotNone(stamped)
        self.assertFalse(app._entitlement_refresh_due())

    def test_a_build_with_no_licence_service_does_not_spin(self):
        app = self._app(None)

        self._once(app)

        self.assertIsNotNone(app._entitlements_refreshed_at)
        self.assertFalse(app._entitlement_refresh_due())

    def test_the_failure_log_names_the_type_and_not_the_message(self):
        app = self._app(
            SimpleNamespace(
                refresh_if_needed=lambda: (_ for _ in ()).throw(
                    RuntimeError("token sk-live-do-not-log-me")
                )
            )
        )

        with self.assertLogs("murmur", level="WARNING") as caught:
            self._once(app)

        self.assertIn("RuntimeError", caught.output[0])
        self.assertNotIn("sk-live-do-not-log-me", caught.output[0])


class VolatileSecretStoreTests(GateTestCase):
    """A Mac with no Keychain keeps the lease in memory, and has to say so.

    Signing in, seeing "Pro", quitting and being Free again is reported as a
    billing failure. The two places the account is shown say it up front.
    """

    def test_the_menu_line_marks_a_lease_that_will_not_survive_a_quit(self):
        self.assertEqual(
            account_menu_title(_entitlements(pro=True), store_is_volatile=True),
            ACCOUNT_STATUS_PRO + ACCOUNT_STATUS_NOT_SAVED,
        )
        self.assertEqual(
            account_menu_title(None, store_is_volatile=True),
            ACCOUNT_STATUS_FREE + ACCOUNT_STATUS_NOT_SAVED,
        )

    def test_a_working_keychain_leaves_the_line_alone(self):
        self.assertEqual(
            account_menu_title(_entitlements(pro=True), store_is_volatile=False),
            ACCOUNT_STATUS_PRO,
        )

    def test_the_flag_is_set_when_the_keychain_is_unreachable(self):
        app = SimpleNamespace(_keychain=lambda: None, secret_store_is_volatile=False)

        with self.assertLogs("murmur", level="WARNING"):
            MurmurApp._build_license_service(app, DEFAULT_CLOUD_BASE_URL)

        self.assertTrue(app.secret_store_is_volatile)

    def test_a_reachable_keychain_is_not_volatile(self):
        app = SimpleNamespace(
            _keychain=lambda: SimpleNamespace(get=lambda name: None), secret_store_is_volatile=True
        )

        MurmurApp._build_license_service(app, DEFAULT_CLOUD_BASE_URL)

        self.assertFalse(app.secret_store_is_volatile)

    def test_the_menu_redraw_carries_the_flag(self):
        titles = []
        app = SimpleNamespace(
            secret_store_is_volatile=True,
            run_on_main_thread=lambda apply: apply(),
            account_item=SimpleNamespace(title=""),
        )

        MurmurApp._refresh_account_menu(app, _entitlements(pro=True))
        titles.append(app.account_item.title)

        self.assertEqual(titles, [ACCOUNT_STATUS_PRO + ACCOUNT_STATUS_NOT_SAVED])

    def test_the_account_tab_is_told_too(self):
        app = SimpleNamespace(
            persistence=object(),
            _keychain=lambda: None,
            runtime_config=dict,
            usage=None,
            license_service=None,
            secret_store_is_volatile=True,
        )

        self.assertTrue(MurmurApp._settings_services(app)["secret_store_volatile"])


class AccountMenuTests(unittest.TestCase):
    """The one line in the menu that names the plan, and the way in."""

    def test_no_licence_reads_free(self):
        self.assertEqual(account_menu_title(None), ACCOUNT_STATUS_FREE)
        self.assertEqual(account_menu_title(Entitlements.none()), ACCOUNT_STATUS_FREE)

    def test_a_live_plan_reads_pro(self):
        self.assertEqual(account_menu_title(_entitlements(pro=True)), ACCOUNT_STATUS_PRO)

    def test_a_lapsed_plan_in_its_grace_week_says_so(self):
        self.assertEqual(
            account_menu_title(_entitlements(pro=True, in_grace=True)),
            ACCOUNT_STATUS_PRO_GRACE,
        )

    def test_the_sign_in_item_opens_settings_on_the_account_tab(self):
        opened = []
        app = SimpleNamespace(open_settings_window_safely=opened.append)

        MurmurApp.open_account_settings(app)

        self.assertEqual(opened, ["account"])
        self.assertIn("Boske", SIGN_IN_MENU_TITLE)


class LicenseProviderShapeTests(unittest.TestCase):
    """``services["license"]`` is the service; both tabs must survive that.

    The Account tab binds four of its methods, so the dict cannot hold a plain
    entitlements callable. The Engine tab only wants a status line, and used to
    call whatever it was given — which would have raised a ``TypeError`` out of
    the model's constructor and taken the whole window with it.
    """

    def _read(self, provider):
        from ui.settings.engine_tab import EngineTabModel

        return EngineTabModel._read_license(
            SimpleNamespace(_license_provider=provider)
        )

    def test_the_service_object_is_read_through_current_entitlements(self):
        published = _entitlements(pro=True, cloud_voice=True)
        service = SimpleNamespace(current_entitlements=lambda: published)

        self.assertIs(self._read(service), published)

    def test_a_plain_callable_still_works(self):
        published = _entitlements()

        self.assertIs(self._read(lambda: published), published)

    def test_no_provider_reads_as_not_signed_in(self):
        self.assertIsNone(self._read(None))

    def test_a_locked_keychain_never_takes_the_window_down(self):
        class Broken:
            def current_entitlements(self):
                raise RuntimeError("locked")

        self.assertIsNone(self._read(Broken()))


class OneGateGuardTests(unittest.TestCase):
    """"No feature check scattered in UI code" is a rule, so it is a test.

    Every gated control in ``ui/`` asks the injected ``pro_gate``. Importing the
    licence module there would be a second opinion about what "Pro" means, and
    the whole point of a single gate is that there is never a second one.
    """

    ROOT = Path(__file__).resolve().parent.parent

    def _ui_sources(self):
        for path in sorted((self.ROOT / "ui").rglob("*.py")):
            yield path, path.read_text(encoding="utf-8")

    def test_no_ui_module_imports_the_licence_service(self):
        offenders = []
        for path, source in self._ui_sources():
            for node in ast.walk(ast.parse(source, filename=str(path))):
                if isinstance(node, ast.ImportFrom) and node.module == (
                    "services.license_service"
                ):
                    offenders.append(path.name)
                elif isinstance(node, ast.Import) and any(
                    alias.name == "services.license_service" for alias in node.names
                ):
                    offenders.append(path.name)

        self.assertEqual(offenders, [], "the UI must gate through pro_gate only")

    def test_no_ui_module_reads_the_published_entitlements(self):
        for path, source in self._ui_sources():
            self.assertNotIn("get_current_entitlements", source, path.name)
            self.assertNotIn("set_current_entitlements", source, path.name)

    def test_the_app_hands_the_tabs_the_licensed_gate_itself(self):
        app = SimpleNamespace(
            persistence=object(),
            _keychain=lambda: None,
            runtime_config=dict,
            usage=None,
            license_service=None,
        )

        self.assertIs(
            MurmurApp._settings_services(app)["pro_gate"], is_pro_feature_enabled
        )


class UsageConfigStoreTests(unittest.TestCase):
    """The adapter that keeps a counter write from reverting a settings write."""

    class _Persistence:
        def __init__(self):
            self.config = dict(DEFAULT_CONFIG)
            self.updates = []

        def load_config(self, default):
            return {**default, **self.config}

        def update_config(self, changes, default=None):
            self.updates.append(dict(changes))
            self.config.update(changes)
            return dict(self.config)

    def test_it_writes_only_the_keys_the_usage_service_owns(self):
        persistence = self._Persistence()
        store = UsageConfigStore(persistence)

        config = store.load()
        config["usage_cloud_words"] = 12
        config["engine_id"] = "hijacked"
        store.save(config)

        self.assertEqual(persistence.updates, [{**USAGE_DEFAULTS, "usage_cloud_words": 12}])
        self.assertNotIn("engine_id", persistence.updates[0])

    def test_a_full_round_trip_through_the_real_service_counts(self):
        persistence = self._Persistence()
        usage = UsageService(config_store=UsageConfigStore(persistence))

        usage.record(ORIGIN_CLOUD, 60.0, 10)

        self.assertEqual(usage.summary().cloud_minutes, 1.0)
        self.assertEqual(usage.summary().cloud_words, 10)
        # And nothing else in the file moved.
        self.assertEqual(persistence.config["engine_id"], DEFAULT_CONFIG["engine_id"])


if __name__ == "__main__":
    unittest.main()
