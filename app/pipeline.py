"""What happens between the shortcut and the paste.

AppKit at import: ``rumps`` for the notifications and ``NSOpenPanel``/
``NSWorkspace`` for the file picker and the front-app probe.

The order this module runs in, once per utterance: record → (live decode) →
route → transcribe → snippets → cleanup → replacements → paste → history →
meter. :class:`CleanupRuntime` owns the ``llama-server`` child process that the
cleanup step talks to, and lives here because nothing else may start or stop it.
"""

import os
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyperclip
import rumps
import scipy.io.wavfile as wav
from AppKit import NSOpenPanel, NSWorkspace

from cleanup.cloud_cleanup import CloudCleanupClient, should_use_cloud_cleanup, words_in
from cleanup.context import AppContext, capture_context
from cleanup.llama_server import (
    CLEANUP_MODEL_ID,
    CleanupClient,
    CleanupResult,
    LlamaServer,
    LlamaServerError,
)
from cleanup.transcription_filters import should_skip_audio
from cleanup.vocabulary import (
    apply_replacements,
    hints_from_vocabulary,
    vocabulary_from_config,
)
from engines import create_engine
from engines.base import EngineError
from engines.cloud import (
    ALLOWANCE_MESSAGE,
    CloudAllowanceExhausted,
    CloudAuthError,
    fetch_usage,
    wav_duration_seconds,
)
from engines.factory import build_engine, cloud_base_url
from engines.model_store import models_for_engine
from services.engine_router import (
    ENGINE_BYOK,
    ENGINE_CLOUD,
    REMOTE_ENGINE_IDS,
    after_cloud_failure,
    route_engine,
)
from services.hotkey_service import format_hotkey_from_config
from services.language_service import resolve_language
from services.license_service import get_current_entitlements, is_pro_feature_enabled
from services.model_profile_service import (
    default_engine_for_current_machine,
    detect_ram_gb,
)
from services.persistence_service import (
    DEFAULT_CONFIG,
    ORIGIN_CLOUD,
    resolve_cleanup_enabled,
    should_log_sensitive,
)
from ui import alerts as ui_alerts
from ui.download_sheet import PHASE_DONE, PHASE_FAILED, PHASE_IDLE, download_model

from app.config import (
    APP_NAME,
    AUDIO_DIR,
    SAMPLE_RATE,
    STATE_ICON_PATHS,
    STREAM_JOIN_TIMEOUT_S,
    app_model_store,
    logger,
)
from app.decisions import (
    CLEANUP_FIRST_USE_WAIT_S,
    CLEANUP_MODEL_MISSING_REASON,
    CLEANUP_NOTICE_NOTIFY,
    CLEANUP_NOTICE_OFFER,
    CLEANUP_NOT_READY_REASON,
    CLEANUP_PREPARING_STATUS,
    CLEANUP_START_FAILED_REASON,
    CLEANUP_STOPPING_REASON,
    CLEANUP_UNSTABLE_REASON,
    CONFIG_ENGINE_ID,
    CONFIG_MODEL_ID,
    MISSING_MODEL_ONBOARDING,
    NO_MODEL_STATUS,
    RELOAD_REFUSAL_MESSAGES,
    RELOAD_START,
    RELOAD_UNCHANGED,
    SWITCHING_STATUS,
    after_byok_failure,
    byok_provider_name,
    cleanup_download_status,
    cleanup_model_missing_message,
    cleanup_notice_kind,
    cleanup_plan,
    cleanup_skipped_message,
    engine_is_ready,
    expand_gated_snippets,
    finalize_transcript,
    gated_vocabulary,
    hints_notice_changes,
    hints_notice_message,
    history_origin_for,
    language_is_auto,
    lease_is_present,
    missing_model_action,
    model_status_title,
    model_unavailable_message,
    notice_to_show,
    own_key_present,
    paste_and_settle,
    reapply_replacements,
    reload_engine_decision,
    remote_engine_key,
    resolve_engine_selection,
    run_cleanup,
    should_consume_trial,
    should_offer_cleanup_download,
    should_prewarm_cleanup,
    should_refresh_allowance,
    should_reject_toggle,
    should_reject_upload,
    should_show_hints_notice,
    skip_audio_user_message,
    stream_text_for_token,
    verify_model_before_load,
)


class CleanupRuntime:
    """Owns the ``llama-server`` child process for the life of the session.

    Started lazily on the first utterance that actually needs it — a user who
    only ever dictates verbatim never pays for a 2 GB model — and kept
    afterwards, because the cold start is measured in seconds and paying it per
    utterance would be worse than not cleaning at all.

    Three failure shapes, three answers:

    * the GGUF is not installed → :data:`CLEANUP_MODEL_MISSING_REASON`, and the
      caller offers the download. Never a silent skip.
    * the child died since the last call → start a fresh one and retry once.
      :class:`~cleanup.llama_server.LlamaServerError` is exactly that signal.
    * the request itself failed (timeout, HTTP error, empty reply) → the client
      already degrades that to a ``skipped`` result carrying the original text.

    Every public method is safe to call from the transcription thread. Two locks,
    and the split between them is the whole design:

    * ``_lock`` serialises *requests*, so two recordings cannot talk to the
      server at once. It is held across the HTTP call, which can take seconds.
    * ``_state_lock`` guards the server and client references and is only ever
      held for a pointer swap — never across a start, a request or a stop.

    :meth:`stop` therefore never queues behind anything. It used to take the one
    lock that :meth:`cleanup` held across a 60 s ``server.start()``, so quitting
    during the first cleanup froze the main thread for up to two minutes.

    The start itself runs on its own thread for the same reason: an utterance
    waits :data:`CLEANUP_FIRST_USE_WAIT_S` for it and then gives up and pastes
    the raw text, while the load carries on and the next utterance is cleaned.
    """

    def __init__(
        self,
        model_path_provider,
        *,
        server_factory=LlamaServer,
        client_factory=CleanupClient,
        on_status=None,
        first_use_wait_s: float = CLEANUP_FIRST_USE_WAIT_S,
    ) -> None:
        assert callable(model_path_provider), "model_path_provider must be callable"
        assert first_use_wait_s > 0, "first_use_wait_s must be positive"
        self._model_path_provider = model_path_provider
        self._server_factory = server_factory
        self._client_factory = client_factory
        self._on_status = on_status
        self._first_use_wait_s = float(first_use_wait_s)
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._server = None
        self._client = None
        #: The server object a start thread is holding but has not published, so
        #: :meth:`stop` can reach a child that is still loading its GGUF.
        self._starting_server = None
        self._starter = None
        self._start_error = None
        #: Set when a start attempt has finished, either way. Waited on by
        #: :meth:`_wait_for_client`, and by :meth:`stop` to release the waiters.
        self._settled = threading.Event()
        self._stopping = threading.Event()

    @property
    def is_started(self) -> bool:
        """True while a server object is held, whether or not its child is alive."""
        return self._server is not None

    def prewarm(self) -> bool:
        """Start the server in the background, before any utterance needs it.

        Returns True when a start was kicked off. False means there was nothing
        to start: no model on disk, one already running, or the app is quitting.
        Never raises and never waits — a failure here simply leaves the first
        utterance to pay what it would have paid anyway.
        """
        model_path = self._model_path_provider()
        if model_path is None:
            return False
        return self._begin_start(model_path)

    def wait_until_ready(self, timeout: float) -> bool:
        """Block until a start attempt has finished. True when one is serving."""
        self._settled.wait(timeout)
        with self._state_lock:
            return self._client is not None

    def cleanup(self, text: str, system_prompt: str) -> CleanupResult:
        """Clean ``text``, starting or restarting the server if it has to.

        Nothing here escapes as an exception. The caller is holding the user's
        only copy of what they just said, so every failure comes back as a
        ``skipped`` result carrying that text unchanged and a reason the UI can
        show.
        """
        assert text is not None, "text is required"
        assert system_prompt, "system_prompt is required"
        for attempt in (1, 2):
            model_path = self._model_path_provider()
            if model_path is None:
                return CleanupResult(
                    text=text, skipped=True, reason=CLEANUP_MODEL_MISSING_REASON
                )
            if self._stopping.is_set():
                return CleanupResult(text=text, skipped=True, reason=CLEANUP_STOPPING_REASON)
            client, reason = self._wait_for_client(model_path)
            if client is None:
                return CleanupResult(text=text, skipped=True, reason=reason)
            try:
                with self._lock:
                    return client.cleanup(text, system_prompt)
            except LlamaServerError as error:
                # The child exited under us (an OOM kill is the realistic case).
                # One restart, one retry: a second death is a real problem and
                # must not loop while the user waits to paste.
                logger.warning("Cleanup server stopped (attempt %d): %s", attempt, error)
                self._discard()
        return CleanupResult(text=text, skipped=True, reason=CLEANUP_UNSTABLE_REASON)

    def stop(self) -> None:
        """Stop the child and forget it. Idempotent; called on quit.

        Takes no lock a request or a start could be holding, so it returns in
        the time one ``terminate``-then-``kill`` takes, whatever else is in
        flight. A start still running finds ``_stopping`` set and shuts down the
        child it was waiting on rather than publishing it.
        """
        self._stopping.set()
        # Anyone waiting on a start that will now never publish is released.
        self._settled.set()
        self._discard()

    # -- internals ---------------------------------------------------------

    def _wait_for_client(self, model_path):
        """``(client, reason)``: a live client, or why there is not one yet."""
        with self._state_lock:
            client = self._client
            server = self._server
        if server is not None and not server.is_running:
            # ``is_running`` reaps a dead child; the object cannot be reused.
            self._discard()
            client = None
        if client is not None:
            return client, None

        self._begin_start(model_path)
        settled = self._settled.wait(self._first_use_wait_s)
        with self._state_lock:
            client = self._client
            error = self._start_error
        if client is not None:
            return client, None
        if self._stopping.is_set():
            return None, CLEANUP_STOPPING_REASON
        if not settled:
            # Still loading. Not a failure: the start carries on behind us and
            # the next utterance finds it ready.
            logger.info(
                "Cleanup server not ready within %.0fs; pasting the raw text",
                self._first_use_wait_s,
            )
            return None, CLEANUP_NOT_READY_REASON
        logger.warning("Cleanup server could not start: %s", error)
        return None, CLEANUP_START_FAILED_REASON

    def _begin_start(self, model_path) -> bool:
        """Kick off a background start unless one is running or pointless."""
        with self._state_lock:
            if self._stopping.is_set() or self._client is not None:
                return False
            if self._starter is not None and self._starter.is_alive():
                return False
            self._start_error = None
            self._settled.clear()
            thread = threading.Thread(
                target=self._start_server,
                args=(model_path,),
                daemon=True,
                name="murmur-cleanup-start",
            )
            self._starter = thread
        self._status(CLEANUP_PREPARING_STATUS)
        thread.start()
        return True

    def _start_server(self, model_path) -> None:
        """Body of the start thread. Publishes a client, or the reason there is none."""
        started_at = time.monotonic()
        server = None
        try:
            server = self._server_factory(model_path)
            with self._state_lock:
                self._starting_server = server
            server.start()
        except Exception as error:
            with self._state_lock:
                self._starting_server = None
                self._start_error = error
            self._stop_server(server)
            self._settled.set()
            return

        with self._state_lock:
            self._starting_server = None
            stopping = self._stopping.is_set()
            if not stopping:
                self._server = server
                self._client = self._client_factory(server)
        if stopping:
            # The app asked to quit while the model was loading. Nothing will
            # ever use this child, so it goes now rather than outliving Murmur.
            self._stop_server(server)
            self._settled.set()
            return
        logger.info("Cleanup server ready in %.1fs", time.monotonic() - started_at)
        self._settled.set()

    def _discard(self) -> None:
        """Drop the server objects, stopping their children first. Never raises."""
        with self._state_lock:
            server, self._server = self._server, None
            starting, self._starting_server = self._starting_server, None
            self._client = None
        self._stop_server(server)
        if starting is not None and starting is not server:
            self._stop_server(starting)

    @staticmethod
    def _stop_server(server) -> None:
        """Stop one server object outside every lock. Never raises."""
        if server is None:
            return
        try:
            server.stop()
        except Exception as error:
            logger.warning("Could not stop the cleanup server: %s", error)

    def _status(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)


def front_app_bundle_id() -> str | None:
    """Bundle id of the frontmost app, or None when macOS will not say.

    Used to pick a per-app language. Wave 2 moves this into ``cleanup/context.py``
    alongside the window title and selection probes.
    """
    try:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
    except Exception as error:
        logger.debug("Could not read the frontmost application: %s", error)
        return None
    if app is None:
        return None
    bundle_id = app.bundleIdentifier()
    return str(bundle_id) if bundle_id else None


class PipelineMixin:
    """Recording, transcription, routing, cleanup and paste, mixed into the app.

    Split out of ``MurmurApp`` in Wave 5, unchanged. The engine lifecycle lives
    here too: loading a model, swapping it and reporting when there is none are
    the same concern as using it.
    """

    # -- speech engine ---------------------------------------------------

    def _model_ids_for_engine(self, engine_id):
        """Catalog model ids belonging to ``engine_id``, in catalog order."""
        return tuple(spec.id for spec in models_for_engine(engine_id))

    def _resolve_selection(self, config):
        """Resolve engine and model from config, writing back defaults once."""
        selection = resolve_engine_selection(
            config,
            default_engine_id=default_engine_for_current_machine(),
            model_ids_for_engine=self._model_ids_for_engine,
        )
        if selection.needs_persist:
            self.persistence.update_config(
                {
                    CONFIG_ENGINE_ID: selection.engine_id,
                    CONFIG_MODEL_ID: selection.model_id,
                }
            )
            if selection.from_legacy_model_key:
                logger.info(
                    "Migrated legacy model setting to engine %s with model %s",
                    selection.engine_id,
                    selection.model_id,
                )
        return selection

    def _notify_settings_engine_reloaded(self):
        """Let an open Settings window re-offer the new engine's languages.

        Its Language popup is built when the window opens; after a live swap it
        would otherwise keep listing the languages of an engine that is no
        longer running. Best effort: no Settings window is the normal case.
        """
        engine = self.engine
        if engine is None:
            return
        try:
            info = engine.info()
        except Exception as error:
            logger.warning("Could not read the new engine's info: %s", error)
            return

        def notify():
            from ui.settings.window import current_controller

            controller = current_controller()
            if controller is None:
                return
            try:
                controller.engine_reloaded(info)
            except Exception as error:
                logger.warning("Settings could not follow the engine swap: %s", error)

        self.run_on_main_thread(notify)

    def _note_hints_support(self, config, transcript, vocabulary):
        """Say once, per engine, when the user's terms could not bias the decode.

        Voxtral Realtime takes no biasing argument at all, so a user who typed
        a vocabulary would otherwise wonder why it changed nothing. The notice
        names the engine and never the transcript.
        """
        engine_id = self.engine_id or transcript.engine_id
        if not should_show_hints_notice(
            config,
            engine_id,
            hints_applied=transcript.hints_applied,
            has_terms=bool(vocabulary.terms),
        ):
            return
        self.persistence.update_config(
            hints_notice_changes(self.runtime_config(), engine_id)
        )
        rumps.notification(
            APP_NAME, "Vocabulary", hints_notice_message(self._engine_display_name())
        )

    # -- cleanup pipeline ------------------------------------------------

    def _settle_cleanup_default(self, config):
        """Decide ``cleanup_enabled`` from this machine once, then store it.

        Runs on the model-loading thread because the probe shells out to
        ``sysctl``. Afterwards the file states the real setting, so Settings can
        show a switch rather than a machine-dependent "it depends".
        """
        if isinstance(config.get("cleanup_enabled"), bool):
            return config
        enabled = resolve_cleanup_enabled(config)
        logger.info("Cleanup default for this Mac: %s", "on" if enabled else "off")
        return self.persistence.update_config({"cleanup_enabled": enabled})

    def _cleanup_model_id(self, config=None):
        """Catalog id of the GGUF the cleanup server loads."""
        if config is None:
            config = self.runtime_config()
        return config.get("cleanup_model_id") or CLEANUP_MODEL_ID

    def _installed_cleanup_model_path(self):
        """Path of the cleanup GGUF, or None when it is not downloaded.

        None is the runtime's signal to skip visibly and let the app offer the
        download; it is never a reason to guess at another file.
        """
        model_id = self._cleanup_model_id()
        try:
            store = app_model_store()
            if not store.is_installed(model_id):
                return None
            return store.engine_model_path(model_id)
        except Exception as error:
            logger.warning("Could not locate the cleanup model: %s", error)
            return None

    def _cleanup_client(self, config):
        """The cloud cleanup client for this proxy origin, built once and kept.

        The origin is the one pinned at launch, never the live config's: the
        text goes to the host the lease belongs to. See
        :func:`pinned_cloud_config`.
        """
        base_url = cloud_base_url(self._hosted_config(config))
        if self._cloud_cleanup_client is not None and self._cloud_cleanup_base_url == base_url:
            return self._cloud_cleanup_client
        if self.license_service is None:
            return None
        try:
            client = CloudCleanupClient(
                base_url, lambda: self.license_service.current_lease_token()
            )
        except Exception as error:  # noqa: BLE001 - a bad base URL raises ValueError
            logger.warning(
                "Cloud cleanup is unavailable (%s); cleaning on this Mac instead: %s",
                type(error).__name__,
                error,
            )
            return None
        self._cloud_cleanup_client = client
        self._cloud_cleanup_base_url = base_url
        return client

    def _cleanup_callable(self, config, *, cloud_engine_active):
        """Which backend cleans this utterance: the proxy, or the local server.

        Returns ``(cleanup, is_local)``. The flag is not a convenience: the
        caller used to recover it with ``cleanup is self.cleanup_runtime.cleanup``,
        and comparing two freshly bound methods of the same object is always
        False in Python, so the local branch it guards — the "Preparing
        cleanup…" pill through a 2 GB model load — never once ran.

        Cloud cleanup travels with cloud transcription and never on its own —
        :func:`~cleanup.cloud_cleanup.should_use_cloud_cleanup` refuses it when
        the dictation stayed on this Mac, because sending the text up after
        keeping the audio down would break the promise the Privacy tab makes.
        """
        if not should_use_cloud_cleanup(
            config,
            pro_gate=is_pro_feature_enabled,
            cloud_engine_active=cloud_engine_active,
        ):
            return self.cleanup_runtime.cleanup, True
        client = self._cleanup_client(config)
        if client is None:
            return self.cleanup_runtime.cleanup, True
        return self._cloud_cleanup_with_fallback(client), False

    def _cloud_cleanup_with_fallback(self, client):
        """Wrap the proxy's cleanup so a refusal lands on the local server.

        The same two failures the transcription path falls back on, with the
        same notice: a spent allowance or a rejected lease costs the round trip,
        not the cleanup. Metering happens here too — the proxy charges words for
        chat, and a skip is not charged, so a skip is not counted.
        """

        def cleanup(text, system_prompt):
            try:
                result = client.cleanup(text, system_prompt)
            except (CloudAllowanceExhausted, CloudAuthError) as error:
                fallback = after_cloud_failure(
                    error,
                    local_engine_id=self.engine_id or default_engine_for_current_machine(),
                )
                logger.info(
                    "Murmur Cloud did not clean the text (%s); cleaning on this Mac",
                    fallback.reason,
                )
                self._announce_route(fallback.notice)
                return self.cleanup_runtime.cleanup(text, system_prompt)
            if not result.skipped:
                self._record_usage(ORIGIN_CLOUD, 0, words_in(result.text))
            return result

        return cleanup

    def _clean_up_transcript(
        self, text, config, language, vocabulary, pill=None, *, cloud_engine_active=False
    ):
        """Run the cleanup pass for one utterance and report what happened.

        Returns the text to paste. Cleanup is an improvement on top of a
        transcript that is already correct, so every failure here ends with the
        original text and a notice — never with nothing.

        ``cloud_engine_active`` is whether *this* dictation was transcribed by
        Murmur Cloud, which is one of the three conditions cloud cleanup needs.
        """
        try:
            context = capture_context(
                include_selection=bool(
                    config.get("include_selection", DEFAULT_CONFIG["include_selection"])
                )
            )
        except Exception as error:
            # No front app, no Accessibility, no PyObjC: an empty context makes
            # resolve_mode fall back to the configured mode, which is the right
            # answer. Losing the cleanup pass over it would not be.
            logger.warning("Could not read the front app context: %s", error)
            context = AppContext(
                bundle_id=None, app_name=None, window_title=None, selected_text=None
            )

        try:
            plan = cleanup_plan(config, context)
        except Exception as error:
            # A config naming a mode this build does not have, most likely.
            logger.warning("Could not resolve the cleanup mode: %s", error)
            return text
        if not plan.enabled:
            logger.debug("Cleanup not run: %s", plan.reason)
            return text

        cleanup, local_cleanup = self._cleanup_callable(
            config, cloud_engine_active=cloud_engine_active
        )
        if pill is not None and local_cleanup and not self.cleanup_runtime.is_started:
            # The first cleaned utterance waits on a 2 GB model load. The menu
            # bar says so, but the user is looking at the pill, and a pill that
            # sits on the plain "working" state through several seconds reads as
            # a hang. Naming the wait is the difference between "it's broken"
            # and "it's coming". The proxy has no such wait, so it gets no label.
            pill.working(label=CLEANUP_PREPARING_STATUS)
        try:
            outcome = run_cleanup(
                text,
                plan,
                cleanup=cleanup,
                language=language,
                vocabulary_terms=tuple(vocabulary.terms),
            )
        finally:
            # The runtime borrows the engine status line for "Preparing
            # cleanup…"; give it back whether the pass worked or not.
            self._set_model_menu_title(self._model_status_title)
            if pill is not None:
                pill.working()  # back to the transcript for the paste
        # Lengths and timings only: transcript text never reaches a log.
        logger.info(
            "Cleanup %s in mode %s: %d chars in, %d chars out, %.2fs",
            "ran" if outcome.ran else "skipped",
            plan.mode_id,
            len(text),
            len(outcome.text),
            outcome.elapsed_s,
        )
        if outcome.skipped_reason:
            self._report_cleanup_skipped(outcome.skipped_reason)
        return outcome.text

    def _report_cleanup_skipped(self, reason):
        """Say out loud that cleanup did not run, and queue the fix when there is one."""
        kind = cleanup_notice_kind(reason)
        if kind == CLEANUP_NOTICE_OFFER:
            # Queued, not shown. The transcript has not been pasted yet, and a
            # modal alert here takes key focus — so the ⌘V lands in the alert
            # and the user loses what they just said. Released by
            # :func:`paste_and_settle` once the text is in.
            self._cleanup_offer_pending = True
            return
        if kind == CLEANUP_NOTICE_NOTIFY:
            rumps.notification(APP_NAME, "Cleanup skipped", cleanup_skipped_message(reason))

    def _flush_cleanup_offer(self):
        """Show any queued cleanup-download offer, now that the paste has landed."""
        if not self._cleanup_offer_pending:
            return
        self._cleanup_offer_pending = False
        self.run_on_main_thread(self._offer_cleanup_download)

    def _cleanup_download_running(self) -> bool:
        controller = self._cleanup_download_controller
        return controller is not None and controller.is_running

    def _offer_cleanup_download(self):
        """Offer the cleanup model once per session; download it on a yes."""
        model_id = self._cleanup_model_id()
        if not should_offer_cleanup_download(
            declined=self._cleanup_download_declined,
            downloading=self._cleanup_download_running(),
            installed=self._installed_cleanup_model_path() is not None,
        ):
            return
        if not ui_alerts.show_confirm(
            APP_NAME,
            cleanup_model_missing_message(self._model_display_name(model_id)),
            ok="Download",
            cancel="Not now",
        ):
            # Remembered so the alert does not return every utterance, but the
            # Mode submenu keeps the download one click away.
            self._cleanup_download_declined = True
            self._refresh_cleanup_download_item()
            return
        self._start_cleanup_download(model_id)

    def download_cleanup_model(self, _=None):
        """Menu item: fetch the cleanup model, whatever was said to the alert.

        The Settings "Speech engine" popup lists speech models only, so without
        this entry a single "Not now" left the cleanup model with no route to
        the disk for the rest of the session.
        """
        self._start_cleanup_download(self._cleanup_model_id())

    def _start_cleanup_download(self, model_id):
        """Fetch the cleanup model with the same controller the sheet uses.

        The Settings window owns the modal sheet; this path has no window to
        attach one to, so it drives the same :class:`DownloadController` and
        shows its status line in the menu, exactly as an app update does.
        """
        if self._cleanup_download_running():
            return
        try:
            total = app_model_store().spec(model_id).size_bytes
        except Exception:
            total = 0
        self._cleanup_download_controller = download_model(
            app_model_store(),
            model_id,
            total_bytes=total,
            on_change=self._cleanup_download_changed,
        )
        self._refresh_cleanup_download_item()

    def _cleanup_download_changed(self, state):
        """Progress ticks from the cleanup model download, in the menu line."""
        if state.is_active or state.phase == PHASE_IDLE:
            self._set_model_menu_title(cleanup_download_status(state), transient=True)
            return
        self._set_model_menu_title(self._model_status_title)
        self._refresh_cleanup_download_item()
        if state.phase == PHASE_DONE:
            rumps.notification(
                APP_NAME, "Cleanup ready", "The cleanup model is installed."
            )
        elif state.phase == PHASE_FAILED:
            rumps.notification(
                APP_NAME, "Cleanup model", f"Download failed: {state.error}"
            )

    def load_model(self):
        """Load the configured speech engine. Runs on a background thread."""
        self.update_status("Loading model...")
        self._set_menu_bar_state("processing")
        try:
            config = self.runtime_config()
            config = self._settle_cleanup_default(config)
            selection = self._resolve_selection(config)
            store = app_model_store()
            self._maybe_prewarm_cleanup(config)
            if not store.is_installed(selection.model_id):
                self._report_missing_model(config, selection)
                return
            self._activate_engine(selection.engine_id, selection.model_id, store)
        except Exception as error:
            self._report_engine_failure(error)

    def _maybe_prewarm_cleanup(self, config):
        """Start the cleanup server at launch on a Mac that can spare the memory.

        Runs from the model-loading thread, which is already off the main one,
        and the start itself is another background thread — so nothing here can
        delay the menu bar. A failure is not reported: the first utterance would
        have hit the same failure and has the notice for it.
        """
        try:
            if not should_prewarm_cleanup(
                config,
                pro=is_pro_feature_enabled("cleanup"),
                cleanup_enabled=resolve_cleanup_enabled(config),
                installed=self._installed_cleanup_model_path() is not None,
                ram_gb=detect_ram_gb(),
            ):
                return
            if self.cleanup_runtime.prewarm():
                logger.info("Pre-warming the cleanup server at launch")
        except Exception as error:
            logger.warning("Could not pre-warm the cleanup server: %s", error)

    def _activate_engine(self, engine_id, model_id, store):
        """Unload whatever is loaded, then build and publish the new engine.

        The old engine goes first: two speech models resident at once run to
        several gigabytes, and the Macs most likely to switch models are the
        ones that can least afford holding both. The caller reports failures.

        The files are checksummed before any of that: ``is_installed`` only
        compares sizes, so it cannot tell a good model from a swapped one.
        """
        logger.info("Loading engine %s with model %s", engine_id, model_id)
        verify_model_before_load(store, model_id, self._verified_models)
        with self._engine_lock:
            previous = self.engine
            self.engine = None
            self.engine_id = None
            self.model_id = None
            if previous is not None:
                previous.unload()
            engine = create_engine(engine_id, model_path=store.engine_model_path(model_id))
            engine.load()
            self.engine = engine
            self.engine_id = engine_id
            self.model_id = model_id
        logger.info("Model loaded successfully %s", engine.runtime_summary())
        self.loading = False
        self.engine_unavailable_reason = None
        self._set_model_menu_title(model_status_title(self._model_display_name(model_id)))
        self.update_status(
            f"Ready ({format_hotkey_from_config(self.runtime_config())} to record)"
        )
        self._set_menu_bar_state("ready")

    def _report_missing_model(self, config, selection):
        """No model on disk: say so plainly, then offer the way to get one.

        Deliberately not a fallback to some other engine — the user chose this
        one, and a silent substitution would misreport what is transcribing.
        """
        logger.info(
            "Speech model %s for engine %s is not installed",
            selection.model_id,
            selection.engine_id,
        )
        self.loading = False
        self.engine = None
        self.engine_id = None
        self.model_id = None
        self.engine_unavailable_reason = NO_MODEL_STATUS
        self.update_status(NO_MODEL_STATUS)
        self._set_model_menu_title(NO_MODEL_STATUS)
        error_state = "error" if "error" in STATE_ICON_PATHS else "ready"
        self._set_menu_bar_state(error_state)
        if missing_model_action(config) == MISSING_MODEL_ONBOARDING:
            self.run_on_main_thread(self.show_onboarding_window)
        else:
            self.run_on_main_thread(self.open_settings_window_safely)

    def _report_engine_failure(self, error):
        """One status/notification/alert path for a failed load or a failed swap."""
        logger.error("Failed to load the speech engine: %s", error, exc_info=True)
        self.loading = False
        self.engine_unavailable_reason = None
        self.update_status(f"Error: {str(error)[:30]}")
        self._set_model_menu_title("Speech engine unavailable")
        error_state = "error" if "error" in STATE_ICON_PATHS else "ready"
        self._set_menu_bar_state(error_state)
        rumps.notification(
            APP_NAME,
            "Model failed to load",
            "Recording is unavailable until the model loads successfully.",
        )
        self._alert_on_main(f"Could not load the speech model.\n\n{error}")

    def reload_engine(self, engine_id: str, model_id: str) -> str | None:
        """Swap the speech engine without a restart. Called by Settings.

        Settings calls this on the main thread; the work itself runs on a
        background thread so the popup does not freeze behind a model load.

        Returns None when the swap started (or the pair is already loaded), and
        the refusal message when it cannot happen now. The caller needs that
        answer before it writes anything: config must never name an engine the
        app declined to load.
        """
        assert engine_id, "engine_id is required"
        assert model_id, "model_id is required"
        decision = reload_engine_decision(
            requested=(engine_id, model_id),
            active=(self.engine_id, self.model_id),
            is_reloading=self._engine_reloading,
            is_recording=self.is_recording,
            is_processing=self.is_processing,
            engine_ready=engine_is_ready(self.engine),
            stream_active=self._stream_worker_alive(),
        )
        if decision == RELOAD_UNCHANGED:
            return None
        if decision != RELOAD_START:
            message = RELOAD_REFUSAL_MESSAGES[decision]
            logger.info("Engine switch refused (%s)", decision)
            self.update_status(message)
            rumps.notification(APP_NAME, "Engine unchanged", message)
            return message
        # A switch is also how a freshly downloaded model arrives, so its
        # checksums are read again rather than trusted from an earlier run.
        self._verified_models.discard(model_id)
        self._engine_reloading = True
        threading.Thread(
            target=self._reload_engine_worker,
            args=(engine_id, model_id),
            daemon=True,
        ).start()
        return None

    def _reload_engine_worker(self, engine_id, model_id):
        """Unload the old engine, load the new one, then persist the choice."""
        self.update_status(SWITCHING_STATUS)
        self._set_model_menu_title(SWITCHING_STATUS)
        self._set_menu_bar_state("processing")
        try:
            store = app_model_store()
            if not store.is_installed(model_id):
                raise RuntimeError(
                    f"{self._model_display_name(model_id)} is not downloaded yet."
                )
            self._activate_engine(engine_id, model_id, store)
            # Written only once the engine really came up, and only these two
            # keys: config must never claim an engine the app is not running.
            self.persistence.update_config(
                {CONFIG_ENGINE_ID: engine_id, CONFIG_MODEL_ID: model_id}
            )
            self._notify_settings_engine_reloaded()
            logger.info("Switched to engine %s with model %s", engine_id, model_id)
        except Exception as error:
            self._report_engine_failure(error)
        finally:
            self._engine_reloading = False

    def _remote_engine_for(self, engine_id, config):
        """The hosted engine for this dictation, built once and kept.

        Rebuilt only when the config it was built from changed — the own-key
        provider and model. The lease is not part of that key: it is read
        through a callable at request time, so signing out and back in needs no
        rebuild. Neither is the live proxy origin, which :meth:`_hosted_config`
        has already replaced with the pinned one.

        Everything that can fail here comes back as an
        :class:`~engines.base.EngineError`, because that is the one exception
        the caller falls back on. ``build_engine`` raises ``ValueError`` for a
        missing own-key provider and the engines raise it for a non-HTTPS
        endpoint; those used to escape :meth:`_transcribe_routed` untouched and
        cost the user the transcript they had just dictated.
        """
        config = self._hosted_config(config)
        key = remote_engine_key(engine_id, config)
        with self._remote_engine_lock:
            if self._remote_engine is not None and self._remote_engine_key == key:
                return self._remote_engine
            try:
                engine = build_engine(
                    engine_id,
                    config=config,
                    model_store=app_model_store(),
                    license_service=self.license_service,
                    keychain=self._keychain(),
                )
                engine.load()
            except EngineError:
                # Already the right kind, and the subclass carries the wording:
                # re-wrapping would turn a rejected key into "something failed".
                raise
            except Exception as exc:  # noqa: BLE001 - config and the factory raise widely
                # The type only: the message can quote the config it choked on.
                raise EngineError(
                    f"the {engine_id} engine could not be built ({type(exc).__name__})"
                ) from exc
            self._remote_engine = engine
            self._remote_engine_key = key
            logger.info("Built the %s engine (%s)", engine_id, engine.runtime_summary())
            return engine

    def _route_for(self, config, clip_seconds):
        """Where this dictation goes, from the app's state and the router's table."""
        return route_engine(
            cloud_mode=config.get("cloud_mode", DEFAULT_CONFIG["cloud_mode"]),
            local_engine_id=self.engine_id or default_engine_for_current_machine(),
            entitlements=get_current_entitlements(),
            has_lease=lease_is_present(self.license_service),
            usage=self.usage,
            key_present=own_key_present(self._keychain(), config),
            clip_seconds=clip_seconds,
        )

    def _engine_for_route(self, engine_id, config):
        """The engine object a routed engine id names.

        Anything the router did not send to a hosted engine is *the* local
        engine — the one the user chose and this app already loaded — so the
        fallback never quietly starts a second speech model.
        """
        if engine_id in REMOTE_ENGINE_IDS:
            return self._remote_engine_for(engine_id, config)
        return self.engine

    def _run_engine(self, engine_id, config, wav_path, *, language, hints, long_form):
        """One decode on the engine ``engine_id`` names.

        The engine lock is held for the local engine only: it exists so a model
        cannot be unloaded from under a decode, and a hosted engine has no model
        to unload. Holding it across an upload would block every engine swap for
        the length of a network round trip.
        """
        engine = self._engine_for_route(engine_id, config)
        if engine is None:
            raise RuntimeError("no speech engine is loaded")
        if engine is self.engine:
            with self._engine_lock:
                return engine.transcribe(
                    Path(wav_path), language=language, hints=hints, long_form=long_form
                )
        return engine.transcribe(
            Path(wav_path), language=language, hints=hints, long_form=long_form
        )

    def _transcribe_routed(
        self, route, config, wav_path, *, language, hints, long_form=False
    ):
        """Run one clip through the routed engine; returns ``(transcript, engine_id)``.

        The one deliberate fallback in the app lives here, and only for a clip
        that went to a hosted engine: a spent allowance says so, a rejected
        lease asks for a sign-in, and any other proxy failure falls back
        quietly, because the user asked for a transcript and a transient
        network error is not theirs to act on. A **local** engine that fails
        propagates — a transcript that could not be produced must not look like
        one that was, and re-running the engine that just failed would only
        fail again.

        Which hosted engine failed decides the wording. The two proxy
        exceptions mean nothing to a user's own provider, so an own-key failure
        is read by :func:`after_byok_failure` instead — a rejected key used to
        match neither and fall back with no notice at all, which downgraded
        every dictation silently for as long as the key stayed revoked. Those
        notices are shown once a session: the answer is in Settings, and
        repeating it on every utterance would be nagging.
        """
        self._announce_route(route.notice)
        hosted = route.engine_id in REMOTE_ENGINE_IDS
        try:
            transcript = self._run_engine(
                route.engine_id,
                config,
                wav_path,
                language=language,
                hints=hints,
                long_form=long_form,
            )
        except EngineError as error:
            if not hosted:
                raise
            local_engine_id = self.engine_id or default_engine_for_current_machine()
            own_key = route.engine_id == ENGINE_BYOK
            if own_key:
                fallback = after_byok_failure(
                    error,
                    local_engine_id=local_engine_id,
                    provider=byok_provider_name(config),
                )
            else:
                fallback = after_cloud_failure(error, local_engine_id=local_engine_id)
            logger.info(
                "The %s engine did not take the clip (%s); transcribing on this Mac",
                route.engine_id,
                fallback.reason,
            )
            self._announce_route(
                fallback.notice, once_key=fallback.reason if own_key else None
            )
            transcript = self._run_engine(
                fallback.engine_id,
                config,
                wav_path,
                language=language,
                hints=hints,
                long_form=long_form,
            )
            return transcript, fallback.engine_id
        return transcript, route.engine_id

    def _announce_route(self, notice, *, once_key=None):
        """Say a routing notice out loud, at most as often as it may be said.

        A notification, never an alert: the user is mid-dictation and a modal
        would take the key focus the paste needs. The allowance notice is marked
        shown only once it has actually been shown, so a route computed and
        discarded never burns the one notice the user gets per period.

        ``once_key`` names a class of notice worth saying once a session — the
        own-key failures, where the fix is a trip to Settings and every further
        utterance would only repeat it. Like the allowance notice, the key is
        recorded only after the notification actually went out.
        """
        pending = True
        if self.usage is not None:
            try:
                pending = self.usage.fallback_notice_pending
            except Exception as error:  # noqa: BLE001 - a corrupt config
                logger.warning(
                    "Could not read the fallback notice flag: %s", type(error).__name__
                )
        message = notice_to_show(notice, fallback_pending=pending)
        if message is None:
            return
        if once_key is not None:
            shown = getattr(self, "_session_notices", None)
            if shown is None:
                shown = self._session_notices = set()
            if once_key in shown:
                return
            shown.add(once_key)
        rumps.notification(APP_NAME, "Murmur Cloud", message)
        if message == ALLOWANCE_MESSAGE and self.usage is not None:
            try:
                self.usage.mark_fallback_notice_shown()
            except Exception as error:  # noqa: BLE001 - a failed write is not fatal
                logger.warning(
                    "Could not remember the fallback notice: %s", type(error).__name__
                )

    def _record_usage(self, origin, seconds, words):
        """Meter one finished transcription, and spend the trial when it applies.

        Own-key work reaches :meth:`UsageService.record` and is counted nowhere —
        it is billed by the user's own provider. Cloud work also spends the free
        trial, but only on an account without ``cloud_voice``: a paying account
        must not have its one-time hour drained by minutes it already paid for.
        """
        if self.usage is None:
            return
        try:
            self.usage.record(origin, max(0.0, float(seconds or 0.0)), int(words or 0))
            if origin == ORIGIN_CLOUD and should_consume_trial(get_current_entitlements()):
                self.usage.consume_trial(max(0.0, float(seconds or 0.0)))
        except Exception as error:  # noqa: BLE001 - counters must not lose a paste
            logger.warning("Could not record usage: %s", type(error).__name__)

    def _maybe_refresh_allowance(self, config):
        """Re-read the proxy's allowance off the dictation path, when it is stale."""
        if not should_refresh_allowance(
            self.usage, cloud_mode=config.get("cloud_mode", DEFAULT_CONFIG["cloud_mode"])
        ):
            return
        threading.Thread(target=self._refresh_allowance_worker, daemon=True).start()

    def _refresh_allowance_worker(self):
        """One ``GET /v1/voice/usage``, on its own thread. Never raises."""
        if self.license_service is None or self.usage is None:
            return
        try:
            remote = fetch_usage(
                self.cloud_base_url, self.license_service.current_lease_token()
            )
        except Exception as error:  # noqa: BLE001 - a failed read keeps the cache
            logger.info(
                "Could not refresh the cloud allowance: %s", type(error).__name__
            )
            return
        self.usage.refresh_allowance(remote)

    def toggle_recording(self, _):
        """Start or stop recording"""
        logger.info(
            "toggle_recording called. loading=%s, is_recording=%s, is_processing=%s",
            self.loading,
            self.is_recording,
            self.is_processing,
        )
        if should_reject_toggle(
            loading=self.loading,
            is_processing=self.is_processing,
            model_ready=engine_is_ready(self.engine),
        ):
            if self.loading:
                logger.warning("Model still loading, cannot record")
                rumps.notification(APP_NAME, "Please wait", "Model is still loading...")
            elif not engine_is_ready(self.engine):
                logger.warning("Model unavailable, cannot record")
                rumps.notification(
                    APP_NAME,
                    "Model unavailable",
                    model_unavailable_message(self.engine_unavailable_reason),
                )
            else:
                logger.warning("Transcription in progress, cannot toggle recording")
                rumps.notification(APP_NAME, "Please wait", "Transcription in progress...")
            return
        
        if self.is_recording:
            logger.info("Stopping recording")
            self.stop_recording()
        else:
            logger.info("Starting recording")
            self.start_recording()

    def _active_pill(self, config=None):
        """The pill presenter, or None when the user switched the pill off."""
        if config is None:
            config = self.runtime_config()
        if not config.get("pill_enabled", DEFAULT_CONFIG["pill_enabled"]):
            return None
        return self.pill

    def _start_stream_worker(self, pill, language=None, hints=None):
        """Decode this utterance live, pushing partials into the pill.

        No engine lock: a recording already blocks every engine swap
        (:func:`reload_engine_decision` refuses while ``is_recording``, and
        while this worker is alive), and taking the lock here for the whole
        recording would deadlock the wizard's own capture path against it.

        The utterance's token is the safety rail. The worker publishes its text
        only while that token is still the current one, so a worker abandoned at
        the join timeout cannot hand its sentence to the utterance after it.
        """
        engine = self.engine
        chunks = self.audio_capture.pcm_chunks()
        token = next(self._stream_tokens)
        cancelled = threading.Event()
        with self._stream_lock:
            self._stream_token = token
            self._stream_result = None
            self._stream_cancelled = cancelled

        def run():
            try:
                text = pill.feed_stream(
                    engine.stream(chunks, language=language, hints=hints),
                    cancelled=cancelled,
                )
            except Exception as error:
                # The WAV is written either way, so a broken stream costs the
                # live text and nothing else: transcribe() falls back to it.
                text = None
                logger.warning(
                    "Live transcription failed; using the recorded file instead: %s", error
                )
            with self._stream_lock:
                if self._stream_token == token:
                    self._stream_result = (token, text)

        thread = threading.Thread(target=run, daemon=True, name="murmur-live-stream")
        self._stream_thread = thread
        with self._stream_lock:
            # Pruned here as well as on read, so a long session does not
            # accumulate one dead Thread object per utterance.
            self._stream_workers = [
                worker for worker in self._stream_workers if worker.is_alive()
            ]
            self._stream_workers.append(thread)
        thread.start()

    def _stream_worker_alive(self) -> bool:
        """True while any live decoder — abandoned ones included — holds the engine.

        An abandoned worker is dropped from ``_stream_thread`` but is still
        inside ``engine.stream()``, so it is tracked separately: this is what
        stops an engine swap unloading the model out from under it.
        """
        with self._stream_lock:
            self._stream_workers = [
                worker for worker in self._stream_workers if worker.is_alive()
            ]
            return bool(self._stream_workers)

    def _collect_stream_text(self):
        """Take the live decoder's final text, or None when there is none.

        Called once per utterance, before the batch path runs. A stream that is
        still going after :data:`STREAM_JOIN_TIMEOUT_S` is abandoned rather than
        waited on: the recorded file is right there and always answers. Being
        abandoned means two things at once — its text is refused from here on
        (the token stops matching) and it is told to stop drawing on the pill,
        which by then belongs to whatever the user does next.
        """
        thread, self._stream_thread = self._stream_thread, None
        with self._stream_lock:
            token, self._stream_token = self._stream_token, None
            cancelled, self._stream_cancelled = self._stream_cancelled, None
        if thread is not None:
            thread.join(STREAM_JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.warning(
                    "Live transcription did not finish within %.0fs; using the recorded file",
                    STREAM_JOIN_TIMEOUT_S,
                )
                if cancelled is not None:
                    cancelled.set()
        with self._stream_lock:
            result, self._stream_result = self._stream_result, None
        return stream_text_for_token(result, token)

    def start_recording(self):
        """Start audio recording"""
        logger.info("start_recording called")
        config = self.runtime_config()
        self.is_recording = True
        self.recording_start_time = time.time()  # Track when recording started
        self._set_menu_bar_state("recording")
        self.start_stop_item.title = "Stop Recording"
        self.upload_item.set_callback(None)  # Disable transcribe file

        self._stream_thread = None
        pill = self._active_pill(config)
        # Only an engine that can stream gets the live feed; whisper.cpp shows
        # state only, which is what the pill's phases are for. Asked of the
        # engine's own declared capability (``Engine.supports_streaming``),
        # never of its id: a new streaming engine must not need a change here.
        engine = self.engine
        streaming = pill is not None and engine is not None and engine.supports_streaming

        try:
            logger.info(f"Starting audio capture with sample rate {SAMPLE_RATE}")
            self.audio_capture.enable_streaming(streaming)
            self.audio_capture.start()
            logger.info("Audio stream started successfully")
        except Exception as e:
            logger.error(f"Mic error: {e}")
            self.is_recording = False
            self.update_status(f"Mic error: {str(e)[:20]}")
            self._reset_menu_state()
            if pill is not None:
                pill.error("Microphone unavailable")
            rumps.notification(
                APP_NAME,
                "Microphone error",
                "Could not start recording. Check microphone permissions and device.",
            )
            return

        if pill is not None:
            pill.listening()
        if streaming:
            # The live decode is described exactly as the batch one will be.
            # Voxtral honours neither, but the settings must reach the engine
            # for the day one does, and transcribe() decides what to trust.
            vocabulary = vocabulary_from_config(config)
            self._start_stream_worker(
                pill,
                language=resolve_language(config, front_app_bundle_id()),
                hints=hints_from_vocabulary(vocabulary),
            )

    def stop_recording(self):
        """Stop recording and transcribe"""
        logger.info(f"stop_recording called, is_recording={self.is_recording}")
        if not self.is_recording:
            return
        
        # Calculate recording duration
        recording_duration = time.time() - getattr(self, 'recording_start_time', time.time())
        logger.info(f"Recording duration: {recording_duration:.2f} seconds")
        
        self.is_recording = False
        self.is_processing = True
        self._set_menu_bar_state("processing")
        self.start_stop_item.title = "Processing..."
        self.start_stop_item.set_callback(None)  # Disable while processing

        self.audio_capture.stop()
        logger.info("Audio stream stopped")
        audio_chunks = self.audio_capture.chunks
        logger.info(f"Audio data chunks: {len(audio_chunks)}")
        # Process in background
        threading.Thread(target=self.transcribe, args=(audio_chunks,), daemon=True).start()
    
    def transcribe(self, audio_chunks):
        """Transcribe recorded audio"""
        logger.info(f"transcribe called with {len(audio_chunks)} audio chunks")
        # First, always: the live decoder holds a generator over the capture
        # queue and has to be reaped whichever way this call ends.
        stream_text = self._collect_stream_text()
        pill = self._active_pill()
        if pill is not None:
            # After the stream, never before it: a final partial moves the pill
            # to *done* with a 1.2 s fade, and the transcript is not pasted yet.
            pill.working()
        if not audio_chunks:
            logger.warning("No audio data to transcribe")
            self.update_status("No audio recorded")
            if pill is not None:
                pill.error("No audio recorded")
            self._reset_menu_state()
            return

        audio_path = None
        cleanup_audio = False
        try:
            # Combine audio chunks
            audio = np.concatenate(audio_chunks, axis=0).flatten()
            duration_seconds = len(audio) / SAMPLE_RATE
            logger.info(f"Audio length: {len(audio)} samples ({duration_seconds:.2f} seconds)")
            
            # Check audio level to see if there's actual content
            audio_level = np.abs(audio).mean()
            max_level = np.abs(audio).max()
            logger.info(f"Audio level - mean: {audio_level:.6f}, max: {max_level:.6f}")
            
            # Skip if too short or too quiet for reliable transcription.
            if should_skip_audio(duration_seconds, max_level):
                if duration_seconds < 1.0:
                    logger.warning(f"Audio too short ({duration_seconds:.2f}s), skipping transcription")
                else:
                    logger.warning(f"Audio too quiet (max level: {max_level:.6f}), skipping transcription")
                message = skip_audio_user_message(duration_seconds, max_level)
                if pill is not None:
                    pill.error(message)
                rumps.notification(APP_NAME, "Recording skipped", message)
                self._reset_menu_state()
                return

            config = self.runtime_config()
            save_audio = config.get("save_audio", DEFAULT_CONFIG["save_audio"])
            cleanup_audio = not save_audio
            if save_audio:
                self.persistence.ensure_audio_dir(AUDIO_DIR)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_filename = f"recording_{timestamp}.wav"
                audio_path = os.path.join(AUDIO_DIR, audio_filename)
            else:
                fd, audio_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)

            wav.write(audio_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
            if should_log_sensitive(config):
                logger.info(f"Audio saved to {audio_path}")
            
            # Language follows the front app when it has an override; the
            # vocabulary biases the decode and then fixes what it got wrong,
            # with the terms past the free limit dropped for this pass.
            language = resolve_language(config, front_app_bundle_id())
            vocabulary = gated_vocabulary(vocabulary_from_config(config))
            hints = hints_from_vocabulary(vocabulary)

            # Where this dictation goes, decided before a byte is uploaded.
            route = self._route_for(config, clip_seconds=duration_seconds)
            logger.info(
                "Routing this dictation to %s (%s)", route.engine_id, route.reason
            )
            cloud_engine_active = route.engine_id == ENGINE_CLOUD

            if (
                stream_text is not None
                and language_is_auto(language)
                and route.engine_id not in REMOTE_ENGINE_IDS
            ):
                # The live decoder already produced the whole utterance while
                # the user was speaking; running the file through the same
                # engine again would only cost a second and say the same thing.
                #
                # Only when the language is auto, though. The streaming engine
                # cannot honour a pinned language (see Engine.stream), so a user
                # who chose French would otherwise get whatever the decoder
                # guessed. The pill still showed them the live words; the batch
                # pass below is what actually gets pasted.
                #
                # And only on a local route: the live decoder is the local
                # engine, so reusing its words while the user chose the cloud
                # would silently transcribe somewhere other than the menu, the
                # history and the meter all say it did.
                logger.info("Using the live stream result (%d chars)", len(stream_text))
                self._announce_route(route.notice)
                raw_text = stream_text
                engine_id = route.engine_id
            else:
                if stream_text is not None:
                    logger.info("Ignoring the live stream result: it is not this route")
                logger.info("Starting transcription in language %s", language)
                transcript, engine_id = self._transcribe_routed(
                    route, config, audio_path, language=language, hints=hints
                )
                cloud_engine_active = engine_id == ENGINE_CLOUD
                self._note_hints_support(config, transcript, vocabulary)
                raw_text = transcript.text
            origin = history_origin_for(engine_id)
            # Metered on what the engine returned, before the filters: the
            # minutes were spent whatever the hallucination filter then says.
            self._record_usage(origin, duration_seconds, words_in(raw_text))
            # The hallucination filter reads the engine's raw words; the user's
            # replacements are applied to what survives.
            text, is_hallucination = finalize_transcript(raw_text, vocabulary)
            if should_log_sensitive(config):
                logger.info("Transcription completed")

            if text and not is_hallucination:
                # Snippets first: the model should punctuate around the expanded
                # text, not around the trigger phrase that produced it.
                text = expand_gated_snippets(text, config)
                # Cleanup sits between the replacements and the paste: it reads
                # what the user actually meant to write, terms included.
                text = self._clean_up_transcript(
                    text,
                    config,
                    language,
                    vocabulary,
                    pill,
                    cloud_engine_active=cloud_engine_active,
                )
                # …and the replacements run once more over what came back: the
                # model rewrites sentences, and a rewrite re-cases terms.
                text = reapply_replacements(text, vocabulary)
                # Small delay then paste (paste_text copies, pastes, then restores clipboard)
                time.sleep(0.15)
                # The paste comes first and the queued download offer second:
                # a modal raised before it would swallow the keystrokes.
                paste_and_settle(
                    text,
                    type_text=self.type_text,
                    pill=pill,
                    offer=self._flush_cleanup_offer,
                )

                # Transcription complete - text is pasted, no notification needed
                logger.info("Transcribed and pasted")

                # Save to history with audio path when retention is enabled
                history_audio_path = audio_path if save_audio else None
                self.add_to_history(
                    text,
                    "live",
                    audio_path=history_audio_path,
                    duration_s=duration_seconds,
                    engine_id=engine_id,
                )
            else:
                if is_hallucination:
                    logger.info("Filtered hallucination")
                    history_text = f"(Filtered) {text[:120]}"
                else:
                    logger.info("No speech detected")
                    history_text = "(No speech detected)"
                history_audio_path = audio_path if save_audio else None
                self.add_to_history(
                    history_text,
                    "live",
                    audio_path=history_audio_path,
                    duration_s=duration_seconds,
                    engine_id=engine_id,
                )
                if pill is not None:
                    pill.error("No speech detected")
                rumps.notification(
                    APP_NAME,
                    "No speech detected",
                    "Nothing clear enough to paste. Try again closer to the mic.",
                )

            # The allowance reading ages out; re-read it here, with the
            # transcript already pasted, so no recording ever waits on it.
            self._maybe_refresh_allowance(config)

            # Re-enable menu items
            self._reset_menu_state()

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            if pill is not None:
                pill.error("Transcription failed")
            self._reset_menu_state()
            self.update_status(f"Error: {str(e)[:30]}")
        finally:
            if cleanup_audio and audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except OSError as cleanup_error:
                    logger.error(f"Failed to delete temp audio {audio_path}: {cleanup_error}")

    def type_text(self, text):
        """Type text at the cursor with native macOS events. True when it landed."""
        try:
            self.text_inserter.paste_text(text)
        except Exception as e:
            logger.error(f"Paste failed: {e}")
            rumps.notification(
                APP_NAME,
                "Could not paste",
                "Enable Accessibility for Murmur, then try again.",
            )
            return False
        return True

    def upload_audio_file(self, _):
        """Open file dialog to select audio file for transcription"""
        if should_reject_upload(
            loading=self.loading,
            is_recording=self.is_recording,
            is_processing=self.is_processing,
            model_ready=engine_is_ready(self.engine),
        ):
            if self.loading:
                rumps.notification(APP_NAME, "Please wait", "Model is still loading...")
            elif not engine_is_ready(self.engine):
                rumps.notification(
                    APP_NAME,
                    "Model unavailable",
                    model_unavailable_message(self.engine_unavailable_reason),
                )
            return
        
        try:
            file_path = self._pick_audio_file_path()
            if not file_path:
                return

            self.is_processing = True
            self._set_menu_bar_state("processing")
            self.start_stop_item.title = "Processing..."
            self.start_stop_item.set_callback(None)
            self.upload_item.set_callback(None)
            self.update_status("Transcribing file...")
            logger.info(f"Processing file: {file_path}")
            threading.Thread(target=self.transcribe_file, args=(file_path,), daemon=True).start()
        except Exception:
            logger.error("File picker failed", exc_info=True)
            self.update_status("File error")
            self._reset_menu_state()

    def _pick_audio_file_path(self):
        """Show native NSOpenPanel file picker; return POSIX path or None if cancelled."""
        panel = NSOpenPanel.openPanel()
        panel.setTitle_("Select an audio file to transcribe")
        panel.setPrompt_("Open")
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(False)
        panel.setAllowedFileTypes_([
            "mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac",
            "aac", "aiff", "caf", "m4b",
        ])
        if panel.runModal() != 1:
            return None
        urls = panel.URLs()
        if not urls:
            return None
        return urls[0].path()
    
    def transcribe_file(self, file_path):
        """Transcribe an audio file - runs in background thread"""
        try:
            config = self.runtime_config()
            save_audio = config.get("save_audio", DEFAULT_CONFIG["save_audio"])
            audio_path = None
            if save_audio:
                self.persistence.ensure_audio_dir(AUDIO_DIR)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                original_ext = os.path.splitext(file_path)[1] or '.wav'
                audio_filename = f"file_{timestamp}{original_ext}"
                audio_path = os.path.join(AUDIO_DIR, audio_filename)
                shutil.copy2(file_path, audio_path)
            
            language = resolve_language(config, front_app_bundle_id())
            vocabulary = gated_vocabulary(vocabulary_from_config(config))
            hints = hints_from_vocabulary(vocabulary)
            # An imported file routes like a dictation, and its length decides
            # whether the proxy will take it at all. An unreadable header is not
            # a reason to refuse the import: an unknown length never blocks the
            # cloud, because the engine enforces its own cap anyway.
            try:
                clip_seconds = wav_duration_seconds(Path(file_path))
            except Exception as error:  # noqa: BLE001 - not every import is a WAV
                logger.info(
                    "Could not read the clip length (%s); routing without it",
                    type(error).__name__,
                )
                clip_seconds = None
            route = self._route_for(config, clip_seconds=clip_seconds)
            # A whole-file import, not dictation: the decoder may condition on
            # the text it already produced for earlier windows.
            transcript, engine_id = self._transcribe_routed(
                route, config, file_path, language=language, hints=hints, long_form=True
            )
            text = apply_replacements(transcript.text, vocabulary)
            self._note_hints_support(config, transcript, vocabulary)
            self._record_usage(
                history_origin_for(engine_id),
                transcript.duration_s or clip_seconds or 0.0,
                words_in(transcript.text),
            )

            if text:
                # Copy to clipboard
                pyperclip.copy(text)

                # Show result length info
                word_count = len(text.split())

                # Save to history with audio path when retention is enabled
                self.add_to_history(
                    text,
                    "file",
                    os.path.basename(file_path),
                    audio_path=audio_path if save_audio else None,
                    duration_s=transcript.duration_s,
                    engine_id=engine_id,
                )
                
                # Update UI on main thread
                self._reset_menu_state()
                self.update_status(f"✓ {word_count} words transcribed")
                logger.info(f"File transcribed: {word_count} words")
            else:
                # Delete copied audio if no speech detected
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                self._reset_menu_state()
                logger.info("No speech detected in file")
                rumps.notification(
                    APP_NAME,
                    "No speech detected",
                    "Nothing clear enough to paste from that file.",
                )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"File transcription error: {error_msg}")
            self._reset_menu_state()
