"""``MurmurApp`` itself: what it is made of, and how it starts and stops.

AppKit at import: ``rumps``, ``objc``, ``Foundation`` and ``AppKit`` — this is
the module that subclasses ``rumps.App`` and ``NSObject``.

The class is composed rather than long. :class:`~app.menu.MenuMixin`,
:class:`~app.services.ServicesMixin` and :class:`~app.pipeline.PipelineMixin`
each carry one third of what used to be a 2,500-line class body; what is left
here is the startup order, the shortcut, the wizard, the updater and the quit.
"""

import fcntl
import itertools
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import objc
import rumps
import scipy.io.wavfile as wav
from AppKit import NSApplicationDidBecomeActiveNotification
from Cocoa import NSWorkspace
from Foundation import NSNotificationCenter, NSObject
from PyObjCTools import AppHelper

from services.audio_capture_service import AudioCaptureService
from services.hotkey_service import (
    KEY_UP_MODES,
    PressController,
    format_hotkey,
    format_hotkey_from_config,
    hotkey_diagnostics,
    hotkey_from_config,
    hotkey_mode_from_config,
    hotkey_permissions_ok,
    hotkey_registration_active,
    is_bundled_app,
    log_hotkey_diagnostics,
    open_privacy_settings,
    permission_status_message,
    register_global_hotkey,
    request_hotkey_permissions,
    reset_accessibility_permission,
    unregister_global_hotkey,
)
from services.license_service import get_current_entitlements
from services.persistence_service import DEFAULT_CONFIG
from services.text_insertion_service import TextInsertionService
from services.update_service import UpdateService, cleanup_previous_bundles, read_build_info
from ui import alerts as ui_alerts
from ui.onboarding_window import OnboardingCallbacks, should_show, show_onboarding
from ui.pill_window import PillPresenter

from app import config as app_config
from app.config import (
    APP_NAME,
    APP_VERSION,
    BUNDLE_ID,
    ICON_PATH,
    ONBOARDING_TEST_SECONDS,
    PERSISTENCE,
    SAMPLE_RATE,
    app_model_store,
    logger,
)
from app.decisions import (
    QUIT_BUDGET_S,
    QUIT_STEPS,
    SIGN_IN_MENU_TITLE,
    about_menu_title,
    account_menu_title,
    download_progress_status,
    engine_is_ready,
    history_origin_for,
    hotkey_registration_key,
    login_item_service,
    push_to_talk_degraded_message,
    quit_time_remaining,
    should_relaunch_after_install,
    should_reregister_hotkey,
    should_toggle_for_press_action,
    update_available_message,
    update_installed_message,
    update_relaunch_failed_message,
)
from app.menu import MenuMixin
from app.pipeline import CleanupRuntime, PipelineMixin
from app.services import ServicesMixin

_INSTANCE_LOCK = None



class _HotkeyActivationObserver(NSObject):
    """Re-register the shortcut when Murmur becomes active after permission grants."""

    def initWithCallback_(self, callback):
        self = objc.super(_HotkeyActivationObserver, self).init()
        if self is None:
            return None
        self._callback = callback
        NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self,
            "applicationDidBecomeActive:",
            NSApplicationDidBecomeActiveNotification,
            None,
        )
        return self

    def applicationDidBecomeActive_(self, _notification):
        self._callback()


def _activate_existing_instance():
    """Bring an already-running Murmur to the foreground."""
    try:
        activate_opts = 1 << 1  # NSApplicationActivateIgnoringOtherApps
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            bundle_id = app.bundleIdentifier() or ""
            name = app.localizedName() or ""
            if app.processIdentifier() == os.getpid():
                continue
            if bundle_id == BUNDLE_ID or name == APP_NAME:
                app.activateWithOptions_(activate_opts)
                return
    except Exception:
        pass


def ensure_single_instance():
    """Prevent duplicate menu bar icons from multiple Murmur processes."""
    global _INSTANCE_LOCK

    support_dir = os.path.expanduser("~/Library/Application Support/Murmur")
    os.makedirs(support_dir, mode=0o700, exist_ok=True)
    lock_path = os.path.join(support_dir, "murmur.lock")
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        lock_file.close()
        _activate_existing_instance()
        try:
            rumps.notification(APP_NAME, "Already running", "Murmur is already in your menu bar.")
        except Exception:
            pass
        sys.exit(0)

    lock_file.write(str(os.getpid()))
    lock_file.flush()
    _INSTANCE_LOCK = lock_file


class MurmurApp(MenuMixin, ServicesMixin, PipelineMixin, rumps.App):
    """The menu bar app.

    The mixins come first so their methods win over ``rumps.App``'s, and the
    order among them does not matter: no two of them define the same name.
    """

    def __init__(self):
        # Published on the module rather than on this one, so the window
        # modules can find the running app without importing it (see
        # ``ui.settings.window.murmur_app_instance``).
        app_config.APP_INSTANCE = self
        super(MurmurApp, self).__init__(APP_NAME, icon=ICON_PATH, quit_button=None)
        self.template = True
        self.title = None
        
        # State
        self.is_recording = False
        self.is_processing = False
        # Held for the whole of a transcription and for an engine swap, so a
        # model can never be unloaded from under a decode in flight.
        self._engine_lock = threading.Lock()
        self.persistence = PERSISTENCE
        #: Probed once, here, so Settings can offer the login-item switch only
        #: where it works. When it does not, ``set_launch_at_login`` is shadowed
        #: with ``None`` so ``ui.settings.general_tab.supports_launch_at_login``
        #: — which asks whether the app has a callable of that name — reports
        #: the truth and the tab shows "Not available in this build" instead of
        #: a switch that would raise when flipped.
        self._login_item_service = login_item_service()
        self.launch_at_login_supported = self._login_item_service is not None
        if not self.launch_at_login_supported:
            self.set_launch_at_login = None
        #: Probed once by :meth:`_keychain` and reused by everything that needs
        #: a secret store: the licence lease, the own-key API keys and the
        #: Account tab. A keychain that is unreachable is asked about once
        #: rather than on every window open.
        self._keychain_store = None
        self._keychain_probed = False
        # -- Wave 4: the licence, the counters and the hosted clients -----
        self._build_services()
        self.history = self.load_history()
        self.audio_capture = AudioCaptureService(sample_rate=SAMPLE_RATE, logger=logger)
        # Built in load_model(), where a bad config or an unresolvable engine id
        # becomes the same status/notification/alert as a failed load instead of
        # killing the app before the menu bar exists.
        self.engine = None
        self.engine_id = None
        self.model_id = None
        # Set when the block has a better explanation than "it failed to load"
        # (today: nothing is downloaded yet). Read by the two rejection paths.
        self.engine_unavailable_reason = None
        self._engine_reloading = False
        self._update_status_line = None
        #: Model ids whose checksums this process has already read. See
        #: :func:`verify_model_before_load`.
        self._verified_models = set()
        #: Last non-transient engine status line, so a temporary title (an
        #: update download) can be undone without restoring a stale one.
        self._model_status_title = "Loading model…"
        self.text_inserter = TextInsertionService(logger=logger)

        # -- Wave 2: the smart layer -------------------------------------
        # The presenter is cheap and builds no AppKit object until something is
        # actually shown, so it can exist before the run loop does.
        self.pill = PillPresenter()
        self.cleanup_runtime = CleanupRuntime(
            self._installed_cleanup_model_path,
            on_status=lambda message: self._set_model_menu_title(message, transient=True),
        )
        #: Live-stream bookkeeping. Every utterance takes a number from
        #: ``_stream_tokens``; the worker publishes ``(token, text)`` into
        #: ``_stream_result`` and only while its number is still current, so an
        #: abandoned decoder can never answer for the utterance after it.
        #: ``_stream_cancelled`` is how that abandoned worker is told to stop
        #: drawing on the pill. See :func:`stream_text_for_token`.
        self._stream_tokens = itertools.count(1)
        self._stream_lock = threading.Lock()
        #: Every worker started this session that has not finished, abandoned
        #: ones included. Read by :meth:`_stream_worker_alive`.
        self._stream_workers = []
        self._stream_thread = None
        self._stream_token = None
        self._stream_result = None
        self._stream_cancelled = None
        #: The cleanup download alert is asked once per session; a "Not now" is
        #: recoverable through the Mode submenu, so this is not a life sentence.
        self._cleanup_download_declined = False
        self._cleanup_download_controller = None
        #: Set when a cleanup pass wanted the download offer. The modal waits
        #: until the transcript has been pasted — see :func:`paste_and_settle`.
        self._cleanup_offer_pending = False

        # Menu items - SuperWhisper style
        self.start_stop_item = rumps.MenuItem("Start/Stop Recording", callback=self.toggle_recording)
        self.upload_item = rumps.MenuItem("Transcribe File", callback=self.upload_audio_file)
        self.history_item = rumps.MenuItem("History", callback=self.open_history_window)
        self.settings_item = rumps.MenuItem("Settings", callback=self.open_settings)
        
        logger.info("Menu items created with callbacks")
        if hasattr(sys, "_MEIPASS"):
            logger.info(f"Murmur running from bundle: {sys._MEIPASS}")
        else:
            logger.info(f"Murmur running from source: {os.path.dirname(os.path.abspath(__file__))}")
        
        # Microphone submenu (restore persisted device before building menu)
        self.mic_menu = rumps.MenuItem("Microphone")
        self._restore_microphone_from_config()
        self.update_microphone_menu()
        
        # Engine status line: the model in use, the swap in progress, or the
        # reason dictation is unavailable. Updated by load_model/reload_engine.
        self.model_item = rumps.MenuItem("Loading model…", callback=None)

        # Mode and tone: the two cleanup choices worth having one click away.
        # Everything else about cleanup lives in Settings.
        self.mode_menu = rumps.MenuItem("Mode")
        self.tone_menu = rumps.MenuItem("Tone")
        self._build_mode_menu()
        self._build_tone_menu()

        # Account: a status line the entitlement thread keeps current, and the
        # one click that leads to signing in. Both live next to the engine
        # status because "which plan" and "which engine" answer the same
        # question — where is this dictation going to be transcribed.
        self.account_item = rumps.MenuItem(
            account_menu_title(
                get_current_entitlements(),
                store_is_volatile=self.secret_store_is_volatile,
            ),
            callback=None,
        )
        self.sign_in_item = rumps.MenuItem(
            SIGN_IN_MENU_TITLE, callback=self.open_account_settings
        )

        self.menu = [
            self.start_stop_item,
            self.upload_item,
            self.history_item,
            self.settings_item,
            self.mic_menu,
            self.mode_menu,
            self.tone_menu,
            None,  # Separator
            self.model_item,
            self.account_item,
            self.sign_in_item,
            rumps.MenuItem(about_menu_title(APP_VERSION, read_build_info()), callback=None),
            rumps.MenuItem("Check for Updates...", callback=self.check_updates),
            rumps.MenuItem("Welcome Tour…", callback=self.open_welcome_tour),
            rumps.MenuItem("Enable Shortcut Permission...", callback=self.enable_shortcut_permission),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app, key="q"),
        ]

        # Load model in background
        self.loading = True
        threading.Thread(target=self.load_model, daemon=True).start()

        # The lease is read, renewed and published off the main thread: it
        # reaches the Keychain and, when due, the network. Started after the
        # menu exists so the first pass has an account line to write into.
        self._start_entitlement_refresh()

        # An update leaves the bundle it replaced beside the new one. This is
        # the launch that can safely delete it: nothing is loading out of it.
        threading.Thread(target=self._clean_previous_bundles, daemon=True).start()

        # First run: the wizard opens once the menu bar and run loop exist.
        self._onboarding_timer = rumps.Timer(self._maybe_show_onboarding, 0.6)
        self._onboarding_timer.start()

        # Register global shortcut after the run loop is active.
        self._hotkey_registration = None
        #: The binding, mode and key-up availability the live registration was
        #: made for, so an unchanged shortcut is not registered twice. See
        #: :func:`~app.decisions.should_reregister_hotkey`.
        self._registered_hotkey = None
        self._hotkey_retry_timer = None
        self._hotkey_permission_notified = False
        self._push_to_talk_degraded_notified = False
        # Replaced by reload_hotkey with the configured mode once the run loop starts.
        self._press_controller = PressController()
        self._hotkey_activation_observer = _HotkeyActivationObserver.alloc().initWithCallback_(
            self._on_application_active
        )
        self._hotkey_startup_timer = rumps.Timer(self._register_hotkey, 0.3)
        self._hotkey_startup_timer.start()

    def check_updates(self, _):
        """Ask the update feed off the main thread; the answer comes back as an alert."""
        threading.Thread(target=self._check_updates_worker, daemon=True).start()

    def _check_updates_worker(self):
        """Fetch release metadata only. No audio, no text, nothing uploaded.

        Reads the channel the Account tab writes to ``update_channel``.
        """
        channel = self.runtime_config().get("update_channel", "stable")
        try:
            info = UpdateService(APP_VERSION, channel=channel).check()
        except Exception as error:
            logger.error("Update check failed: %s", error)
            self._alert_on_main(
                "Could not check for updates. Check your network connection and try again."
            )
            return
        if info is None:
            self._alert_on_main(f"You're on the latest version ({APP_VERSION}).")
            return
        self.run_on_main_thread(lambda: self._offer_update(info))

    def _offer_update(self, info):
        """Ask before downloading; installing replaces the running app."""
        if not ui_alerts.show_confirm(
            APP_NAME,
            update_available_message(info.version, APP_VERSION),
            ok="Download & Install",
            cancel="Later",
        ):
            return
        threading.Thread(target=self._install_update_worker, args=(info,), daemon=True).start()

    def _install_update_worker(self, info):
        """Download, verify the signature, install. Progress goes to the menu.

        The line is only borrowed: it is restored to whatever the engine status
        is when the download ends, not to the string it held when it started. A
        download runs for minutes, and the engine can be swapped or fail in that
        time — putting the old title back would then lie about the engine.
        """

        def progress(bytes_done, bytes_total):
            line = download_progress_status(bytes_done, bytes_total)
            if line == self._update_status_line:
                return  # every chunk ticks; only redraw when the text changes
            self._update_status_line = line
            self._set_model_menu_title(line, transient=True)

        try:
            result = UpdateService(APP_VERSION).download_and_install(info, progress)
        except Exception as error:
            logger.error("Update install failed: %s", error)
            self._set_model_menu_title(self._model_status_title)
            self._alert_on_main(f"Could not install Murmur {info.version}.\n\n{error}")
            return
        finally:
            self._update_status_line = None
        self._set_model_menu_title(self._model_status_title)
        self._relaunch_into_update(info.version, result)

    def _relaunch_into_update(self, version, result):
        """Start the newly installed bundle, then quit this one.

        The installer deliberately does not launch it: only the running app can
        close its engine, its audio stream and its hotkey registration in that
        order. So the handover is here — start the new copy, say so, quit. A
        launch that fails leaves the installed app in place and tells the user
        to reopen it by hand rather than quitting into nothing.
        """
        if should_relaunch_after_install(result):
            try:
                subprocess.Popen(list(result.relaunch_cmd))
            except OSError as error:
                logger.error("Could not start the updated Murmur: %s", error)
                self._alert_on_main(update_relaunch_failed_message(version))
                return
        logger.info("Handing over to Murmur %s", version)
        self.run_on_main_thread(lambda: self._quit_into_update(version))

    def _quit_into_update(self, version):
        """Notify (never a modal, which the quit would race) and shut down."""
        rumps.notification(APP_NAME, "Update installed", update_installed_message(version))
        self.quit_app(None)

    def _clean_previous_bundles(self):
        """Remove the bundles a past update set aside. Logged, never surfaced.

        Leftover clutter is not the user's problem and must not delay or break a
        launch, so a failure here goes to the log and nowhere else.
        """
        try:
            removed = cleanup_previous_bundles()
        except Exception as error:
            logger.warning("Could not remove bundles left by a previous update: %s", error)
            return
        if removed:
            logger.info("Removed %d bundle(s) left by a previous update", len(removed))

    def _alert_on_main(self, message, *, title=APP_NAME):
        """Show an alert from any thread."""
        self.run_on_main_thread(lambda: ui_alerts.show_alert(title, message))

    def run_on_main_thread(self, func):
        """Run a function on the main thread - required for UI updates from background threads"""
        if threading.current_thread() is threading.main_thread():
            func()
        else:
            # Use performSelectorOnMainThread to safely update UI
            AppHelper.callAfter(func)
    
    def load_history(self):
        """Load transcription history from file"""
        return self.persistence.load_history()

    def runtime_config(self):
        """Load current config from disk (includes privacy retention settings)."""
        return self.persistence.load_config(dict(DEFAULT_CONFIG))
    
    def save_history(self):
        """Save transcription history to file"""
        self.persistence.save_history(self.history)
    
    def current_engine_id(self):
        """The id of the engine that produced the last transcription, or None.

        ``self.engine_id`` is what config selected; the engine actually loaded
        is the authority, because a swap writes the attribute only once the new
        engine is running. Asked here so history records what did the work.
        """
        engine = self.engine
        if engine is None:
            return self.engine_id
        try:
            return engine.info().id
        except Exception as error:  # noqa: BLE001 - history must not fail on this
            logger.warning("Could not read the engine id for history: %s", error)
            return self.engine_id

    def add_to_history(
        self,
        text,
        source_type,
        filename=None,
        audio_path=None,
        duration_s=None,
        engine_id=None,
    ):
        """Add a transcription to history.

        ``origin`` is worked out from ``engine_id`` rather than at the call
        sites: the callers know what was said, not what "cloud" means, and the
        answer must be the same for all of them.

        ``engine_id`` is passed in by the dictation paths, which know which
        engine the router actually sent the clip to — the loaded local engine is
        not the authority once a clip can go to the proxy. It defaults to the
        loaded engine for the callers that never route.
        """
        if not self.runtime_config().get("save_history", DEFAULT_CONFIG["save_history"]):
            return
        if not engine_id:
            engine_id = self.current_engine_id()
        self.history = self.persistence.add_history_entry(
            self.history,
            text=text,
            source_type=source_type,
            origin=history_origin_for(engine_id),
            engine_id=engine_id,
            duration_s=duration_s,
            filename=filename,
            audio_path=audio_path,
        )
        self.save_history()

    # -- onboarding ------------------------------------------------------

    def _maybe_show_onboarding(self, _sender=None):
        """Open the wizard on a first run, once the menu bar exists."""
        if self._onboarding_timer is not None:
            self._onboarding_timer.stop()
            self._onboarding_timer = None
        if not should_show(self.runtime_config()):
            return
        self.show_onboarding_window()

    def open_welcome_tour(self, _):
        """Menu item: reopen the wizard whenever the user wants it."""
        self.show_onboarding_window()

    def show_onboarding_window(self):
        """Show the wizard, wired to this app's recorder, engine and settings."""
        try:
            show_onboarding(
                OnboardingCallbacks(
                    download=app_model_store().download,
                    record_and_transcribe=self._record_test_sentence,
                    open_settings=lambda: self.run_on_main_thread(
                        self.open_settings_window_safely
                    ),
                    on_finished=self._onboarding_finished,
                )
            )
        except Exception:
            logger.error("Could not open the welcome tour", exc_info=True)
            ui_alerts.show_alert(APP_NAME, "Could not open the welcome tour.")

    def _onboarding_finished(self, updates):
        """Persist what the wizard decided, and pick up a model it downloaded.

        Only the wizard's own keys are written. The wizard is open for minutes
        and the app keeps writing config behind it, so saving a merged snapshot
        would revert whatever landed in between.
        """
        self.persistence.update_config(updates)
        if engine_is_ready(self.engine) or self._engine_reloading:
            return
        self.loading = True
        threading.Thread(target=self.load_model, daemon=True).start()

    def _record_test_sentence(self):
        """Record a few seconds and transcribe them for the wizard's own field.

        The result goes back to the wizard and nowhere else: it is not pasted,
        not saved to history, and never logged.

        It marks the app busy for its whole duration, exactly as a normal
        recording does. Otherwise a hotkey press during the wizard's five
        seconds would open a second input stream, and an engine switch would
        see an idle app and unload the engine mid-transcription.
        """
        if self.is_recording or self.is_processing:
            raise RuntimeError("Murmur is busy with another recording. Try again in a moment.")
        # One read under the lock: the engine checked and the engine used must
        # be the same object, or a swap in between transcribes on an unloaded one.
        with self._engine_lock:
            engine = self.engine
            if not engine_is_ready(engine):
                raise RuntimeError(
                    "The speech model is not loaded yet. Download it on the previous "
                    "step, then try this again."
                )

        self.is_processing = True
        try:
            capture = AudioCaptureService(sample_rate=SAMPLE_RATE, logger=logger)
            capture.start()
            try:
                time.sleep(ONBOARDING_TEST_SECONDS)
            finally:
                capture.stop()
            chunks = capture.chunks
            if not chunks:
                raise RuntimeError(
                    "No audio was captured. Check the microphone permission and try again."
                )

            audio = np.concatenate(chunks, axis=0).flatten()
            handle, audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(handle)
            try:
                wav.write(audio_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
                with self._engine_lock:
                    transcript = engine.transcribe(Path(audio_path), language=None)
            finally:
                try:
                    os.unlink(audio_path)
                except OSError as error:
                    logger.error("Failed to delete the wizard's temp audio: %s", error)
            return transcript.text
        finally:
            self.is_processing = False

    def enable_shortcut_permission(self, _):
        """Prompt for and explain the macOS permissions required for the shortcut."""
        diagnostics = hotkey_diagnostics()
        if is_bundled_app() and diagnostics.get("signature") == "adhoc":
            reset_accessibility_permission()
            logger.warning("Reset Accessibility TCC for ad-hoc Murmur install")
        request_hotkey_permissions(logger, prompt=True)
        open_privacy_settings()
        ui_alerts.show_alert(
            APP_NAME,
            permission_status_message(diagnostics=diagnostics),
        )
        self.reload_hotkey(prompt=False)

    def _on_application_active(self):
        """Pick up Accessibility grants as soon as the user returns to Murmur."""
        if hotkey_permissions_ok() and self._hotkey_registration is None:
            self.reload_hotkey(prompt=False)

    def _schedule_hotkey_retry(self):
        if self._hotkey_retry_timer is not None:
            return
        self._hotkey_retry_timer = rumps.Timer(self._retry_hotkey_if_needed, 1)
        self._hotkey_retry_timer.start()

    def _stop_hotkey_retry(self):
        if self._hotkey_retry_timer is None:
            return
        self._hotkey_retry_timer.stop()
        self._hotkey_retry_timer = None

    def _retry_hotkey_if_needed(self, _sender=None):
        if self._hotkey_registration is not None:
            self._stop_hotkey_retry()
            return
        self.reload_hotkey(prompt=False)

    def _register_hotkey(self, _sender=None):
        """Register the configured global shortcut once the run loop is active."""
        if getattr(self, "_hotkey_startup_timer", None) is not None:
            self._hotkey_startup_timer.stop()
            self._hotkey_startup_timer = None
        self.reload_hotkey(prompt=True)

    def reload_hotkey(self, *, prompt: bool = False):
        """Apply the shortcut from settings, replacing any previous registration.

        A shortcut that is already registered exactly as configured — and is
        actually delivering what its mode needs — is left alone. Two callers
        reach this within the first second of every launch (the 0.3 s startup
        timer and ``applicationDidBecomeActive``), and re-registering the same
        binding cost a duplicate Carbon hotkey, a duplicate warning in the log,
        and a press controller reset under a key that might be held down.
        """
        config = self.runtime_config()
        binding = hotkey_from_config(config)
        mode = hotkey_mode_from_config(config)
        desired = hotkey_registration_key(binding, mode)
        if not should_reregister_hotkey(
            self._registered_hotkey,
            desired,
            registered=hotkey_registration_active(self._hotkey_registration),
        ):
            logger.info(
                "Shortcut %s is already registered as configured; leaving it alone",
                format_hotkey(binding),
            )
            return
        unregister_global_hotkey(self._hotkey_registration)
        self._hotkey_registration = None
        self._registered_hotkey = None
        self._press_controller = PressController(mode)

        def trigger_key_down():
            self.run_on_main_thread(self._on_hotkey_key_down)

        def trigger_key_up():
            self.run_on_main_thread(self._on_hotkey_key_up)

        def handle_error(error):
            logger.error(f"Hotkey callback error: {error}")

        # 1. Try to register the hotkey (Carbon will succeed without Accessibility)
        try:
            self._hotkey_registration = register_global_hotkey(
                binding,
                on_trigger=trigger_key_down,
                on_error=handle_error,
                logger=logger,
                on_key_up=trigger_key_up,
                mode=mode,
            )
            self._apply_key_up_availability(mode)
            self._registered_hotkey = self._live_hotkey_key(binding, mode)
            self._stop_hotkey_retry()
            self._hotkey_permission_notified = False
            if is_bundled_app():
                log_hotkey_diagnostics(
                    logger,
                    event=(
                        "Shortcut active: "
                        f"{format_hotkey_from_config(self.runtime_config())}"
                    ),
                )
            else:
                logger.info(
                    "Shortcut active: %s",
                    format_hotkey_from_config(self.runtime_config()),
                )
            return
        except Exception as error:
            # If registration failed (e.g., Carbon failed and NSEvent fallback failed),
            # check Accessibility permissions
            logger.warning("Direct hotkey registration failed: %s", error)

        # 2. Handle Accessibility permissions if fallback is needed
        request_hotkey_permissions(logger, prompt=prompt)
        if not hotkey_permissions_ok():
            logger.warning(
                "Deferring hotkey registration for %s until Accessibility is granted",
                format_hotkey(binding),
            )
            self._schedule_hotkey_retry()
            if not self._hotkey_permission_notified:
                self._hotkey_permission_notified = True
                rumps.notification(
                    APP_NAME,
                    "Shortcut permission needed",
                    "Enable Accessibility for Murmur.app, then return to Murmur.",
                )
            return

        # Try registering again after prompting/checking Accessibility
        try:
            self._hotkey_registration = register_global_hotkey(
                binding,
                on_trigger=trigger_key_down,
                on_error=handle_error,
                logger=logger,
                on_key_up=trigger_key_up,
                mode=mode,
            )
            self._apply_key_up_availability(mode)
            self._registered_hotkey = self._live_hotkey_key(binding, mode)
            self._stop_hotkey_retry()
            self._hotkey_permission_notified = False
        except Exception as error:
            logger.error(str(error))
            if is_bundled_app():
                log_hotkey_diagnostics(logger, event="Shortcut registration failed")
            self._schedule_hotkey_retry()
            if not self._hotkey_permission_notified:
                self._hotkey_permission_notified = True
                message = (
                    "Enable Accessibility for the current Murmur.app, then quit and reopen."
                )
                if is_bundled_app():
                    diagnostics = hotkey_diagnostics()
                    if diagnostics.get("signature") == "adhoc":
                        message = (
                            "This DMG build needs a fresh Accessibility grant. "
                            "Use Enable Shortcut Permission…, allow access, quit, and reopen."
                        )
                rumps.notification(
                    APP_NAME,
                    "Shortcut unavailable",
                    message,
                )
    
    def _live_hotkey_key(self, binding, mode):
        """What the registration in hand is, key-up availability included.

        Compared against the configured one on the next reload: a registration
        that is missing the key-up its mode needs is worth making again once
        Accessibility is granted, and an identical working one is not.
        """
        registration = self._hotkey_registration
        return hotkey_registration_key(
            binding,
            mode,
            key_up_available=bool(
                registration is not None and registration.key_up_available
            ),
        )

    def _apply_key_up_availability(self, mode):
        """Tell the press controller whether key-up will really be delivered.

        Without a key-up source, ``hold`` and ``auto`` would start a recording
        that no later press can stop, so the controller runs them as ``toggle``.
        The degradation is logged every time and shown to the user once: it is
        the difference between the shortcut they configured and the one they
        have, and it comes back the moment Accessibility is granted.
        """
        registration = self._hotkey_registration
        available = bool(registration is not None and registration.key_up_available)
        self._press_controller.set_key_up_available(available)
        if available or mode not in KEY_UP_MODES:
            self._push_to_talk_degraded_notified = False
            return
        message = push_to_talk_degraded_message(mode)
        logger.warning("Push-to-talk degraded to toggle: %s", message)
        if self._push_to_talk_degraded_notified:
            return
        self._push_to_talk_degraded_notified = True
        rumps.notification(APP_NAME, "Shortcut runs as toggle", message)

    def _safe_toggle(self):
        """Toggle recording safely"""
        self.toggle_recording(None)

    def _on_hotkey_key_down(self, _sender=None):
        """Shortcut pressed. The mode decides what that means."""
        self._press_controller.sync(self.is_recording)
        self._apply_press_action(self._press_controller.on_key_down(time.time()))

    def _on_hotkey_key_up(self, _sender=None):
        """Shortcut released. Only hold and auto act on this."""
        self._apply_press_action(self._press_controller.on_key_up(time.time()))

    def _apply_press_action(self, action):
        if should_toggle_for_press_action(action, is_recording=self.is_recording):
            self._safe_toggle()
        # The app may have refused the toggle (loading, processing, mic error), so
        # let the controller see what actually happened.
        self._press_controller.sync(self.is_recording)


    def update_status(self, status):
        """Update status (for internal use)"""
        logger.debug(f"Status update: {status}")
        pass

    def clear_history(self, _):
        """Clear all history"""
        if ui_alerts.show_confirm(
            "Clear History",
            "Are you sure you want to clear all transcription history?",
            ok="Clear",
            cancel="Cancel",
        ):
            self.history = []
            self.save_history()
            self.update_history_menu()
            logger.info("History cleared")

    def quit_app(self, _):
        """Quit the application: free what outlives the process, then go.

        Five things outlive a quit if nobody hands them back — a live decoder
        inside ``engine.stream()``, the ``llama-server`` child holding a 2 GB
        model, the floating pill's panel, the loaded model itself, and the
        Carbon hotkey macOS keeps until the app dies. The order is
        :data:`~app.decisions.QUIT_STEPS` and it is not arbitrary: the decoders
        are told to stop before the engine is unloaded, because unloading a
        model out from under a running decode is a crash in native code rather
        than an exception.

        Nothing here may hang. Every step is best effort, and the one thing
        that waits at all — joining the live decoders — shares a single
        :data:`~app.decisions.QUIT_BUDGET_S` budget across the whole quit. Past
        it, the threads are abandoned: every one of them is a daemon, so the
        interpreter takes them with it.
        """
        started = time.monotonic()
        steps = {
            "cancel_stream_workers": lambda: self._cancel_stream_workers_for_quit(started),
            "stop_cleanup_runtime": self.cleanup_runtime.stop,
            "close_pill": self.pill.close,
            "unload_engine": self._unload_engine_for_quit,
            "unregister_hotkey": self._unregister_hotkey_for_quit,
        }
        for name in QUIT_STEPS:
            try:
                steps[name]()
            except Exception as error:  # noqa: BLE001 - a quit always finishes
                logger.warning("Quit step %s did not finish cleanly: %s", name, error)
        logger.info("Quit teardown took %.2fs", time.monotonic() - started)
        rumps.quit_application()

    def _cancel_stream_workers_for_quit(self, started):
        """Tell every live decoder to stop, then wait out what is left of the budget.

        The cancel Event is what a worker checks between partials, so setting it
        is the whole of the ask; the join only exists so the engine is not
        unloaded while one is still inside it. A worker that ignores the budget
        is abandoned rather than waited on — it is a daemon thread, and the
        model it is holding is about to go with the process anyway.
        """
        with self._stream_lock:
            cancelled, self._stream_cancelled = self._stream_cancelled, None
            workers = [worker for worker in self._stream_workers if worker.is_alive()]
            self._stream_workers = []
            self._stream_thread = None
            self._stream_token = None
        if cancelled is not None:
            cancelled.set()
        for worker in workers:
            remaining = quit_time_remaining(time.monotonic() - started)
            if remaining <= 0:
                break
            worker.join(remaining)
        still_running = [worker for worker in workers if worker.is_alive()]
        if still_running:
            logger.warning(
                "%d live decoder(s) still running after %.0fs; quitting without them",
                len(still_running),
                QUIT_BUDGET_S,
            )

    def _unload_engine_for_quit(self):
        """Give the model back, without ever waiting for the engine lock.

        The lock is held for the whole of a transcription. Blocking on it here
        would make a quit during a slow decode look like a hang, and the memory
        is freed by the exiting process either way — so a busy engine is simply
        left to it.
        """
        engine, self.engine = self.engine, None
        if engine is None:
            return
        if not self._engine_lock.acquire(blocking=False):
            logger.info("The engine is busy; leaving it to the exiting process")
            return
        try:
            engine.unload()
        finally:
            self._engine_lock.release()

    def _unregister_hotkey_for_quit(self):
        """Hand the Carbon hotkey back; macOS holds it for the life of the process."""
        registration, self._hotkey_registration = self._hotkey_registration, None
        self._registered_hotkey = None
        unregister_global_hotkey(registration)
