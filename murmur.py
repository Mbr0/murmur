#!/usr/bin/env python3
"""
Murmur - A simple local speech-to-text menu bar app
Shortcut: Option+Space to start/stop recording
"""

import sys
import os

# Put the bundled resources directory on PATH before anything that shells out.
# The bundled `whisper-server` lives there, so a plain command name resolves to
# the bundled copy without any monkey-patching of subprocess.
if hasattr(sys, '_MEIPASS'):
    os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ.get('PATH', '')

import fcntl
import json
import rumps
import sounddevice as sd
import numpy as np
import logging


def _configure_logging() -> logging.Logger:
    """Configure production-safe logging (no transcription text in logs)."""
    is_bundled = hasattr(sys, "_MEIPASS")
    debug_flag_file = os.path.expanduser("~/.murmur_debug")
    debug_enabled = (
        os.environ.get("MURMUR_DEBUG", "").lower() in ("1", "true", "yes")
        or os.path.isfile(debug_flag_file)
    )
    level = logging.DEBUG if debug_enabled else logging.WARNING

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if debug_enabled and not is_bundled:
        handlers.append(logging.FileHandler("/tmp/murmur_debug.log"))
    else:
        log_dir = os.path.expanduser("~/Library/Logs/Murmur")
        try:
            os.makedirs(log_dir, mode=0o700, exist_ok=True)
            log_path = os.path.join(log_dir, "murmur.log")
            handlers.append(logging.FileHandler(log_path))
        except OSError:
            pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    app_logger = logging.getLogger(__name__)

    for handler in handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        log_path = handler.baseFilename
        try:
            os.chmod(log_path, 0o600)
        except OSError:
            pass
        log_dir = os.path.dirname(log_path)
        if os.path.isdir(log_dir):
            try:
                os.chmod(log_dir, 0o700)
            except OSError:
                pass

    return app_logger


logger = _configure_logging()

if hasattr(sys, '_MEIPASS'):
    logger.info(f"Added bundled resources to PATH: {sys._MEIPASS}")

import itertools
import pyperclip
import threading
import subprocess
import scipy.io.wavfile as wav
import time
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from cleanup.coding_mode import transform_spoken_code
from cleanup.context import AppContext, capture_context, resolve_mode
from cleanup.llama_server import (
    CLEANUP_MODEL_ID,
    CLEANUP_MODEL_SPEC,
    CleanupClient,
    CleanupResult,
    LlamaServer,
    LlamaServerError,
)
from cleanup.modes import (
    MODE_IDS,
    MODES,
    TONE_IDS,
    TONES,
    render_system_prompt,
    tone_from_config,
)
from cleanup.vocabulary import (
    apply_replacements,
    hints_from_vocabulary,
    vocabulary_from_config,
)
from engines import create_engine
from engines.model_store import CATALOG, ModelIntegrityError, ModelStore, models_for_engine
from transcription_filters import is_likely_hallucination, should_skip_audio
from ui.download_sheet import (
    PHASE_DONE,
    PHASE_FAILED,
    PHASE_IDLE,
    download_model,
)
from ui.pill_window import PillPresenter
from ui.onboarding_window import OnboardingCallbacks, should_show, show_onboarding
from services.audio_capture_service import AudioCaptureService
from services.language_service import resolve_language
from services.hotkey_service import (
    ACTION_START,
    ACTION_STOP,
    KEY_UP_MODES,
    PressController,
    format_hotkey,
    format_hotkey_from_config,
    hotkey_diagnostics,
    hotkey_from_config,
    hotkey_mode_from_config,
    hotkey_permissions_ok,
    is_bundled_app,
    log_hotkey_diagnostics,
    open_privacy_settings,
    permission_status_message,
    register_global_hotkey,
    request_hotkey_permissions,
    reset_accessibility_permission,
    unregister_global_hotkey,
)
from services.model_profile_service import (
    default_engine_for_current_machine,
    detect_ram_gb,
)
from services.persistence_service import (
    DEFAULT_CONFIG,
    PersistencePaths,
    PersistenceService,
    resolve_cleanup_enabled,
    should_log_sensitive,
)
from services.text_insertion_service import TextInsertionService
from services.update_service import UpdateService, cleanup_previous_bundles, read_build_info
import ui_alerts

import objc
import Cocoa
from Cocoa import (
    NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSBackingStoreBuffered, NSMakeRect, NSApp, NSScrollView,
    NSTextView, NSFont, NSColor, NSBezelBorder, NSViewWidthSizable,
    NSViewHeightSizable, NSFloatingWindowLevel, NSButton,
    NSRoundedBezelStyle, NSMomentaryLightButton, NSCenterTextAlignment,
    NSApplication, NSApplicationActivationPolicyAccessory, NSImage,
)
from objc import python_method
from datetime import datetime
from Foundation import NSNotificationCenter, NSObject
from AppKit import NSApplicationDidBecomeActiveNotification, NSOpenPanel, NSWorkspace
from PyObjCTools import AppHelper
# Settings
SAMPLE_RATE = 16000
APP_NAME = "Murmur"
APP_VERSION = "1.0.0"

#: Length of the wizard's "Try it" recording, in seconds.
ONBOARDING_TEST_SECONDS = 4.0

#: How long the batch path waits for the live decoder to finish its last
#: partial before giving up on it and transcribing the recorded file instead.
STREAM_JOIN_TIMEOUT_S = 10.0

# Config file for settings
CONFIG_FILE = os.path.expanduser("~/.murmur_config.json")

# History and audio storage
HISTORY_FILE = os.path.expanduser("~/.murmur_history.json")
AUDIO_DIR = os.path.expanduser("~/.murmur_audio")
LEGACY_CONFIG_FILE = os.path.expanduser("~/.mywhisper_config.json")
LEGACY_HISTORY_FILE = os.path.expanduser("~/.mywhisper_history.json")
LEGACY_AUDIO_DIR = os.path.expanduser("~/.mywhisper_audio")
PERSISTENCE_PATHS = PersistencePaths(config_file=CONFIG_FILE, history_file=HISTORY_FILE)
PERSISTENCE = PersistenceService(paths=PERSISTENCE_PATHS, logger=logger)
APP_INSTANCE = None
# `python murmur.py` runs as __main__; window modules look up APP_INSTANCE via "murmur".
sys.modules.setdefault("murmur", sys.modules[__name__])


def migrate_legacy_data():
    """Migrate legacy MyWhisper local data to Murmur paths once."""
    migrations = [
        (LEGACY_CONFIG_FILE, CONFIG_FILE, False),
        (LEGACY_HISTORY_FILE, HISTORY_FILE, False),
        (LEGACY_AUDIO_DIR, AUDIO_DIR, True),
    ]
    for legacy_path, new_path, is_directory in migrations:
        if os.path.exists(new_path) or not os.path.exists(legacy_path):
            continue
        try:
            if is_directory:
                shutil.move(legacy_path, new_path)
            else:
                shutil.copy2(legacy_path, new_path)
            logger.info(f"Migrated local data from {legacy_path} to {new_path}")
        except OSError as error:
            logger.error(f"Failed to migrate local data from {legacy_path} to {new_path}: {error}")


migrate_legacy_data()

# The engine and model come from config at load time (see MurmurApp.load_model);
# nothing about the speech engine is decided at import.
PERSISTENCE.ensure_audio_dir(AUDIO_DIR)

# Get resource path (works for both dev and PyInstaller bundle)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        # Running in PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    # Running in normal Python environment
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

# Store window controller references to prevent garbage collection
_window_controllers = []
_history_module = None
_settings_module = None

def _load_window_module(module_name, script_path):
    """Load a window module once; PyObjC classes cannot be safely reloaded."""
    import importlib.util

    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    logger.info(f"{module_name} module loaded")
    return module


def _get_history_module():
    """Load the history window module."""
    global _history_module
    script_path = resource_path("history_window.py")
    _history_module = _load_window_module("history_window", script_path)
    return _history_module


def _get_settings_module():
    """Load the settings window module."""
    global _settings_module
    script_path = resource_path("settings_window.py")
    _settings_module = _load_window_module("settings_window", script_path)
    return _settings_module


def _cleanup_window_controllers():
    """Remove closed windows from the list"""
    global _window_controllers
    valid_controllers = []
    for c in _window_controllers:
        try:
            if hasattr(c, 'window') and c.window is not None and c.window.isVisible():
                valid_controllers.append(c)
        except Exception as error:
            logger.warning(f"Failed to validate window controller: {error}")
    _window_controllers = valid_controllers

def _reload_ui_theme():
    """Pick up theme changes without restarting the whole app."""
    import importlib

    if "ui_theme" in sys.modules:
        importlib.reload(sys.modules["ui_theme"])
    import ui_theme
    config = PERSISTENCE.load_config(default=dict(DEFAULT_CONFIG))
    ui_theme.set_appearance_mode(config.get("appearance_mode", "system"))
    logger.info(
        f"UI theme {ui_theme.THEME_VERSION} ({ui_theme.appearance_mode()}) "
        f"from {getattr(ui_theme, '__file__', 'unknown')}"
    )


def _close_window_controllers(class_name):
    """Close and drop cached controllers so windows are rebuilt with current theme."""
    global _window_controllers
    remaining = []
    for controller in _window_controllers:
        if controller.__class__.__name__ == class_name:
            try:
                if hasattr(controller, "window") and controller.window is not None:
                    controller.window.close()
            except Exception as error:
                logger.warning(f"Failed to close {class_name}: {error}")
            continue
        remaining.append(controller)
    _window_controllers = remaining


def show_history_window_direct():
    """Show history window directly in the same process"""
    global _window_controllers
    logger.info("show_history_window_direct called")
    _reload_ui_theme()
    
    _cleanup_window_controllers()
    _close_window_controllers("HistoryWindowController")
    
    # Create new window
    try:
        history_module = _get_history_module()
        controller = history_module.HistoryWindowController.alloc().init()
        controller.createWindow()
        _window_controllers.append(controller)
        logger.info("Created new history window")
    except Exception as e:
        logger.error(f"Error creating history window: {e}")
        raise

def show_settings_window_direct():
    """Show settings window directly in the same process"""
    global _window_controllers
    logger.info("show_settings_window_direct called")
    _reload_ui_theme()
    
    _cleanup_window_controllers()
    _close_window_controllers("SettingsWindowController")
    
    # Create new window
    try:
        settings_module = _get_settings_module()
        controller = settings_module.SettingsWindowController.alloc().init()
        controller.createWindow()
        _window_controllers.append(controller)
        logger.info("Created new settings window")
    except Exception as e:
        logger.error(f"Error creating settings window: {e}")
        raise

ICON_PATH = resource_path("assets/icons/logo_menu_template.png")
ICON_RECORDING = resource_path("assets/icons/icon_recording.png")
ICON_PROCESSING = resource_path("assets/icons/icon_processing.png")
ICON_ERROR = resource_path("assets/icons/icon_error.png")
STATE_ICON_PATHS = {
    "ready": ICON_PATH,
    "recording": ICON_RECORDING,
    "processing": ICON_PROCESSING,
}
if os.path.exists(ICON_ERROR):
    STATE_ICON_PATHS["error"] = ICON_ERROR
_MENU_BAR_IMAGES = {}
_INSTANCE_LOCK = None
BUNDLE_ID = "com.canopystudio.murmur"


def engine_is_ready(engine) -> bool:
    """Whether the engine exists and has finished loading.

    It is built inside :meth:`MurmurApp.load_model`, so it is None until that runs
    and a construction failure is reported through the same UI as a load failure.
    """
    return engine is not None and engine.is_loaded


def should_reject_toggle(*, loading: bool, is_processing: bool, model_ready: bool) -> bool:
    """Whether hotkey/menu toggle must be ignored."""
    return loading or is_processing or not model_ready


def should_toggle_for_press_action(action: str | None, *, is_recording: bool) -> bool:
    """Whether a PressController action needs the recorder toggled.

    The controller decides what the press means; this decides whether the app is
    already in that state. Unknown actions raise instead of being ignored.
    """
    if action is None:
        return False
    if action == ACTION_START:
        return not is_recording
    if action == ACTION_STOP:
        return is_recording
    raise ValueError(f"Unknown press action: {action!r}")


def should_reject_upload(
    *, loading: bool, is_recording: bool, is_processing: bool, model_ready: bool
) -> bool:
    """Whether file upload/transcribe must be ignored."""
    return loading or is_recording or is_processing or not model_ready


def should_apply_ready_on_reset(*, is_recording: bool) -> bool:
    """Whether menu reset may force the ready/idle UI state."""
    return not is_recording


def resolve_mic_device_index(
    saved_index: object, input_device_indices: set[int] | frozenset[int]
) -> int | None:
    """Return persisted mic index, or None for system default. Fail fast if invalid."""
    if saved_index is None:
        return None
    if isinstance(saved_index, bool) or not isinstance(saved_index, int):
        raise ValueError(f"Invalid mic_device_index type: {type(saved_index).__name__}")
    if saved_index not in input_device_indices:
        raise ValueError(f"Microphone device index {saved_index} is not available")
    return saved_index


def resolve_mic_device(
    saved_index: object,
    saved_name: object,
    input_devices: dict[int, str],
) -> int | None:
    """Resolve persisted mic by index+name. Prefer name when index drifted; never accept a mismatched device at the saved index."""
    if saved_name is not None and not isinstance(saved_name, str):
        raise ValueError(f"Invalid mic_device_name type: {type(saved_name).__name__}")
    name = saved_name.strip() if isinstance(saved_name, str) and saved_name.strip() else None

    if saved_index is None and name is None:
        return None
    if saved_index is not None and (
        isinstance(saved_index, bool) or not isinstance(saved_index, int)
    ):
        raise ValueError(f"Invalid mic_device_index type: {type(saved_index).__name__}")

    if saved_index is not None and saved_index in input_devices:
        device_name = input_devices[saved_index]
        if name is None or device_name == name:
            return saved_index
        # Name mismatch at this index — do not accept; try resolve by name below.

    if name is not None:
        matches = [idx for idx, device_name in input_devices.items() if device_name == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple microphones named {name!r}")
        raise ValueError(f"Microphone {name!r} is not available")

    raise ValueError(f"Microphone device index {saved_index} is not available")


def clear_mic_device_selection(config: dict) -> dict:
    """Return config with mic selection cleared (fail-fast; no stale device)."""
    updated = dict(config)
    updated["mic_device_index"] = None
    updated["mic_device_name"] = None
    return updated


def skip_audio_user_message(duration_seconds: float, max_level: float) -> str:
    """Calm user-facing reason when short/quiet audio is skipped (no transcription text)."""
    if duration_seconds < 1.0:
        return "Recording was too short to transcribe."
    return "Recording was too quiet to transcribe."


#: Everything Murmur can download: the speech models plus the cleanup GGUF.
#:
#: The cleanup model is not a speech engine, so ``engines.model_store.CATALOG``
#: deliberately does not carry it and the app composes the two here instead.
#: ``ui.download_sheet.EngineSectionModel`` filters back down to
#: ``engines.ENGINE_IDS``, so the Settings popup never offers a chat model as a
#: transcriber — but the same store, downloader and integrity checks serve both.
APP_CATALOG = CATALOG + (CLEANUP_MODEL_SPEC,)


def app_model_store() -> ModelStore:
    """The store the app uses everywhere: speech models and the cleanup model."""
    return ModelStore(catalog=APP_CATALOG)


#: Config key naming the engine's model; ``None`` until the user or the
#: defaults below fill it in.
CONFIG_ENGINE_ID = "engine_id"
CONFIG_MODEL_ID = "model_id"

#: The pre-Wave-1 config key, a bare openai-whisper size such as ``"medium"``.
#: Its presence is what marks a config as needing the one-off migration.
LEGACY_MODEL_KEY = "model"

#: Menu status while no speech model is on disk.
NO_MODEL_STATUS = "No speech model installed"

#: Menu status while the engine is being swapped.
SWITCHING_STATUS = "Switching engine…"

#: Where :func:`missing_model_action` sends a user with no model.
MISSING_MODEL_ONBOARDING = "onboarding"
MISSING_MODEL_SETTINGS = "settings"

#: Outcomes of :func:`reload_engine_decision`.
RELOAD_START = "start"
RELOAD_UNCHANGED = "unchanged"
RELOAD_BUSY = "busy"
RELOAD_RECORDING = "recording"

#: Why a reload was refused, in the user's words. A refusal is never silent.
RELOAD_REFUSAL_MESSAGES = {
    RELOAD_BUSY: "Murmur is still busy. Choose the model again in a moment.",
    RELOAD_RECORDING: "Stop recording before switching the speech engine.",
}

#: Config key remembering which engines already showed the "hints ignored"
#: notice: ``{engine_id: True}``. Shown once per engine, never per recording.
HINTS_NOTICE_KEY = "hints_notice_shown"


@dataclass(frozen=True)
class EngineSelection:
    """Which engine and model to load, and whether config has to catch up."""

    engine_id: str
    model_id: str
    #: True when config did not name both keys and must be written back.
    needs_persist: bool
    #: True when that write is the one-off migration off ``LEGACY_MODEL_KEY``.
    from_legacy_model_key: bool


def resolve_engine_selection(
    config: dict,
    *,
    default_engine_id: str,
    model_ids_for_engine,
) -> EngineSelection:
    """Resolve the engine and model to load from config, filling in defaults.

    A config that names both keys is honoured as-is. A missing engine falls
    back to this machine's default (chip and RAM, decision D1); a missing model
    falls back to the first catalog model of whichever engine won. Either gap
    means the resolved pair is written back, so the choice is made exactly once
    — including for a legacy config that only carried ``model``.

    An engine with no catalog model at all is a packaging error, not a user
    state, so it raises rather than silently picking another engine.
    """
    assert config is not None, "config is required"
    assert default_engine_id, "default_engine_id is required"

    engine_id = config.get(CONFIG_ENGINE_ID)
    if not isinstance(engine_id, str) or not engine_id:
        engine_id = default_engine_id

    model_id = config.get(CONFIG_MODEL_ID)
    if not isinstance(model_id, str) or not model_id:
        candidates = tuple(model_ids_for_engine(engine_id))
        if not candidates:
            raise ValueError(f"No catalog model for engine {engine_id!r}")
        model_id = candidates[0]

    needs_persist = (
        config.get(CONFIG_ENGINE_ID) != engine_id or config.get(CONFIG_MODEL_ID) != model_id
    )
    return EngineSelection(
        engine_id=engine_id,
        model_id=model_id,
        needs_persist=needs_persist,
        from_legacy_model_key=needs_persist and LEGACY_MODEL_KEY in config,
    )


def missing_model_action(config: dict) -> str:
    """Where to send a user whose chosen model is not downloaded.

    A Mac that never finished the wizard gets the wizard, which can download
    the model in place; anyone else gets Settings, where the same download
    lives. Neither path falls back to another engine behind the user's back.
    """
    assert config is not None, "config is required"
    return MISSING_MODEL_ONBOARDING if should_show(config) else MISSING_MODEL_SETTINGS


def model_unavailable_message(reason: str | None) -> str:
    """Body of the "cannot record/transcribe" notification.

    ``reason`` is the menu status when one explains the block (no model
    installed), and None when the engine simply failed to load.
    """
    if not reason:
        return "Recording is unavailable until the model loads successfully."
    return f"{reason}. Download one from Settings → Speech engine."


def model_status_title(display_name: str | None) -> str:
    """Title of the menu's engine status line."""
    return f"Model: {display_name}" if display_name else NO_MODEL_STATUS


def reload_engine_decision(
    *,
    requested: tuple[str, str],
    active: tuple[str | None, str | None],
    is_reloading: bool,
    is_recording: bool,
    is_processing: bool,
    engine_ready: bool,
    stream_active: bool = False,
) -> str:
    """Whether a requested engine swap may start now.

    Policy: refuse rather than queue. A queued swap would fire minutes later,
    long after the user stopped thinking about it, and a refusal that says so
    is easier to act on than a delayed surprise. Recording and transcription
    both hold the engine, so both block; a second request while one is already
    in flight is refused too.

    ``stream_active`` is the fourth holder and the least obvious one: when the
    batch path gives up waiting for the live decoder it clears both flags and
    finishes the utterance, while the abandoned worker is still inside
    ``engine.stream()``. The app looks idle and is not, so a swap there would
    call ``unload()`` on a model being read.
    """
    assert requested and len(requested) == 2, "requested is (engine_id, model_id)"
    if is_reloading:
        return RELOAD_BUSY
    if is_recording:
        return RELOAD_RECORDING
    if is_processing or stream_active:
        return RELOAD_BUSY
    if engine_ready and tuple(active) == tuple(requested):
        return RELOAD_UNCHANGED
    return RELOAD_START


def should_show_hints_notice(
    config: dict, engine_id: str, *, hints_applied: bool | None, has_terms: bool
) -> bool:
    """Whether to tell the user this engine ignored their vocabulary terms.

    Only when there were terms to ignore, only when the engine said outright
    that it did not use them (``False``, not the ``None`` that means "nothing
    to apply"), and only the first time for that engine.
    """
    assert config is not None, "config is required"
    assert engine_id, "engine_id is required"
    if not has_terms or hints_applied is not False:
        return False
    shown = config.get(HINTS_NOTICE_KEY) or {}
    return not bool(shown.get(engine_id))


def remember_hints_notice(config: dict, engine_id: str) -> dict:
    """Return a copy of ``config`` marking the notice as shown for ``engine_id``."""
    assert config is not None, "config is required"
    assert engine_id, "engine_id is required"
    shown = dict(config.get(HINTS_NOTICE_KEY) or {})
    shown[engine_id] = True
    return {**config, HINTS_NOTICE_KEY: shown}


def hints_notice_message(engine_name: str) -> str:
    """The one-time notice itself. Names the engine, never the transcript."""
    assert engine_name, "engine_name is required"
    return f"Vocabulary hints are not supported by {engine_name}"


def push_to_talk_degraded_message(mode: str) -> str | None:
    """What to tell the user when the chosen press mode cannot run as chosen.

    ``hold`` and ``auto`` both need to see the key release, which macOS only
    delivers through an NSEvent monitor, which needs Accessibility. Without it
    Murmur runs the shortcut as ``toggle``. Saying so beats leaving the user
    holding a key that will never stop the recording. None when there is
    nothing to explain.
    """
    if mode not in KEY_UP_MODES:
        return None
    return (
        f"Push-to-talk “{mode}” needs Accessibility to see the key release. "
        "Until it is granted, the shortcut toggles recording on and off instead."
    )


def finalize_transcript(
    raw_text: str,
    vocabulary,
    *,
    detect_hallucination=is_likely_hallucination,
    replace=apply_replacements,
) -> tuple[str, bool]:
    """Return ``(text to paste, was a hallucination)`` for one raw transcript.

    The filter reads the engine's own words, before the user's replacements
    rewrite them. Running it afterwards let a replacement hide a classic
    silence hallucination from the filter — and let one whose output happened
    to look like a hallucination suppress a real transcript.
    """
    assert raw_text is not None, "raw_text is required"
    hallucination = bool(detect_hallucination(raw_text))
    return replace(raw_text, vocabulary), hallucination


def reapply_replacements(text: str, vocabulary, *, replace=apply_replacements) -> str:
    """Run the user's replacements again over cleaned-up text.

    The cleanup pass rewrites sentences, and a rewrite re-cases words: a term
    the user spelled "Murmur" comes back "murmur" the moment the model starts a
    clause with it. The replacements are cheap and idempotent, so the cheapest
    honest fix is to apply them once more on the way to the clipboard.

    Deliberately *not* the hallucination filter. That reads the engine's own
    words (see :func:`finalize_transcript`); re-judging a sentence the model
    wrote would let its phrasing suppress a real transcript.
    """
    assert text is not None, "text is required"
    return replace(text, vocabulary)


def stream_text_for_token(result, token) -> str | None:
    """The live decoder's text, but only when it belongs to ``token``.

    Every utterance takes a number. The worker publishes ``(token, text)`` and
    the collector accepts it only while that number is still the current one.

    Without it: a stream that overran its join timeout was abandoned, kept
    running, and eventually wrote its text into the same slot — which the *next*
    utterance then read and pasted. The user said one thing and got the previous
    sentence. Returns None for a stale token, a stream that failed, or one that
    produced nothing but whitespace.
    """
    if result is None or token is None:
        return None
    result_token, text = result
    if result_token != token:
        return None
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


# ---------------------------------------------------------------------------
# Cleanup pipeline (Wave 2)
# ---------------------------------------------------------------------------

#: TEMPORARY, until Wave 4 lands ``services/license_service.py``. That wave
#: replaces every call below with the one real gate,
#: ``is_pro_feature_enabled(feature)``; this stand-in exists so the Wave 2
#: wiring can be written and tested against the shape of that gate rather than
#: against nothing. The hidden ``pro_override_for_dev`` key is a developer
#: switch, deliberately absent from ``DEFAULT_CONFIG`` so it never appears in a
#: user's file, and it goes away with this function.
PRO_OVERRIDE_KEY = "pro_override_for_dev"

#: Menu status while the cleanup server is coming up for the first time.
CLEANUP_PREPARING_STATUS = "Preparing cleanup…"

#: Reason given when the GGUF the cleanup server needs is not on disk.
CLEANUP_MODEL_MISSING_REASON = "the cleanup model is not downloaded"

#: Reason given when the cleanup server would not come up at all.
CLEANUP_START_FAILED_REASON = "the cleanup server could not start"

#: Reason given when it comes up and dies again on the very next request.
CLEANUP_UNSTABLE_REASON = "the cleanup server keeps stopping"

#: Reason given when the server is still loading the model. Not a failure: the
#: start carries on in the background, so the next utterance is cleaned.
CLEANUP_NOT_READY_REASON = "the cleanup model is still loading"

#: Reason given for a request that arrives after the app has begun quitting.
CLEANUP_STOPPING_REASON = "Murmur is shutting the cleanup server down"

#: How long one utterance may wait for the cleanup server's *first* start.
#: The load itself is allowed up to ``LlamaServer.startup_timeout_s`` (60 s, and
#: the client retries once on a dead child, so ~120 s in the worst case) and it
#: keeps running in the background past this — but the user is standing there
#: holding a finished sentence, and eight seconds is already a long time to
#: watch a pill say nothing. Past it the raw text is pasted with a visible
#: notice and the *next* utterance gets the cleaned version.
CLEANUP_FIRST_USE_WAIT_S = 8.0

#: What to do about a cleanup that did not run. See :func:`cleanup_notice_kind`.
CLEANUP_NOTICE_NOTIFY = "notify"
CLEANUP_NOTICE_OFFER = "offer"

#: Menu entry that fetches the cleanup GGUF. It is not a speech engine, so the
#: Settings popup filters it out and this is its only permanent home.
CLEANUP_DOWNLOAD_MENU_TITLE = "Download cleanup model…"

#: Hidden config key: start the cleanup server at launch rather than on the
#: first utterance. Absent from ``DEFAULT_CONFIG`` on purpose — it costs 2 GB of
#: resident memory for a feature the user may not touch this session, so it is
#: opt-out for machines that can clearly afford it and invisible elsewhere.
CLEANUP_PREWARM_KEY = "cleanup_prewarm"
CLEANUP_PREWARM_DEFAULT = True

#: Below this, pre-warming competes with the speech model for RAM and the Mac
#: starts swapping mid-dictation. Matches the cleanup feature's own floor.
CLEANUP_PREWARM_MIN_RAM_GB = 16

#: Why cleanup did not run, when the answer is configuration rather than a
#: failure. These are the user's own settings, so they are logged, never shown.
CLEANUP_OFF_PRO = "Pro is not active"
CLEANUP_OFF_DISABLED = "cleanup is switched off"
CLEANUP_OFF_PASSTHROUGH = "the mode is verbatim dictation"

#: Menu item that toggles ``context_awareness`` rather than naming a mode.
MODE_MENU_AUTOMATIC = "Automatic (by app)"

#: The two language codes ``transform_spoken_code`` has trigger words for.
CODE_TRANSFORM_LANGUAGES = ("en", "fr")


def pro_enabled(feature: str, config: dict | None = None) -> bool:
    """Whether a Pro feature is unlocked. Placeholder — see :data:`PRO_OVERRIDE_KEY`.

    Wave 4 replaces the body with the licensed gate. Everything that gates on
    Pro calls this one function today so that replacement is a single edit, and
    so no feature check ever grows its own opinion of what "Pro" means.
    """
    assert feature, "feature is required"
    if config is None:
        config = PERSISTENCE.load_config(dict(DEFAULT_CONFIG))
    return bool(config.get(PRO_OVERRIDE_KEY, False))


def language_is_auto(language: str | None) -> bool:
    """Whether the configured language leaves detection to the engine.

    The one place that decides what "auto" means, because two very different
    things depend on the answer and they must never disagree:

    * the cleanup prompt (:func:`prompt_language`) says "the same language as
      the dictation" instead of naming one;
    * the live decode may stand in for the batch pass. Voxtral's ``stream()``
      accepts a language and cannot honour it, so a user who pinned French must
      get the batch result — the pill still showed them the live words while
      they spoke, which is what the pill is for.
    """
    if not language:
        return True
    return str(language).strip().lower() in ("", "auto")


def prompt_language(language: str | None) -> str | None:
    """Language for the cleanup prompt: ``"auto"`` and ``""`` both mean None.

    ``render_system_prompt`` turns None into "the same language as the
    dictation", which is exactly what auto-detect means. Passing the literal
    string "auto" would ask the model to write in a language called "auto".
    """
    if language_is_auto(language):
        return None
    return str(language).strip()


def code_transform_language(language: str | None) -> str:
    """Trigger vocabulary for :func:`transform_spoken_code`; anything unknown is English.

    The transform only ships English and French trigger words and raises on any
    other code. A user dictating code in German must not lose their transcript
    to that, so an unsupported language falls back to the English table, which
    simply matches nothing in their speech.
    """
    stripped = (language or "").strip().lower()
    base = stripped.split("-")[0]
    return base if base in CODE_TRANSFORM_LANGUAGES else "en"


@dataclass(frozen=True)
class CleanupPlan:
    """Whether cleanup runs for this utterance, and under which mode and tone."""

    mode_id: str
    tone_id: str
    enabled: bool
    #: Why it is not running, for the log. Configuration, never an error, so it
    #: is deliberately not shown to the user — only a *failed* pass is.
    reason: str | None = None


def cleanup_plan(config: dict, context, *, pro=pro_enabled) -> CleanupPlan:
    """Resolve mode and tone for this utterance and decide whether to clean it.

    Three gates, in the order the plan names them: the Pro entitlement, the
    user's on/off switch, and the mode itself — Dictation is verbatim by
    definition, so it is a skip of the LLM, not a skip of cleanup.
    """
    assert config is not None, "config is required"
    mode_id = resolve_mode(context, config)
    tone_id = tone_from_config(config).id
    if not pro("cleanup", config):
        return CleanupPlan(mode_id, tone_id, False, CLEANUP_OFF_PRO)
    if not resolve_cleanup_enabled(config):
        return CleanupPlan(mode_id, tone_id, False, CLEANUP_OFF_DISABLED)
    if MODES[mode_id].is_passthrough:
        return CleanupPlan(mode_id, tone_id, False, CLEANUP_OFF_PASSTHROUGH)
    return CleanupPlan(mode_id, tone_id, True)


@dataclass(frozen=True)
class CleanupOutcome:
    """What the cleanup pass produced. ``text`` is always safe to paste."""

    text: str
    ran: bool
    #: Set only when cleanup was attempted and did not deliver. This is what
    #: becomes the visible "cleanup skipped" notice; None means nothing to say.
    skipped_reason: str | None = None
    elapsed_s: float = 0.0


def cleanup_skipped_message(reason: str) -> str:
    """The visible notice for a cleanup that was attempted and did not deliver."""
    assert reason, "reason is required"
    return f"Cleanup skipped — {reason}. Your text was pasted unchanged."


def cleanup_notice_kind(reason: str | None) -> str | None:
    """What to do about a cleanup that did not run: nothing, a notice, or an offer.

    A missing model is the one failure with a fix the user can act on, so it
    earns the modal; everything else is a notification that says what happened
    and gets out of the way. None means the pass delivered and there is nothing
    to say.
    """
    if not reason:
        return None
    if reason == CLEANUP_MODEL_MISSING_REASON:
        return CLEANUP_NOTICE_OFFER
    return CLEANUP_NOTICE_NOTIFY


def paste_and_settle(text: str, *, type_text, pill=None, offer=None) -> bool:
    """Paste ``text``, tell the pill how it went, and only then run ``offer``.

    The order is the whole point. ``offer`` raises a modal alert, and a modal
    raised *before* the paste takes key focus — so the synthesised ⌘V lands in
    the alert instead of the user's document and the transcript is gone. The
    offer is therefore queued during the cleanup pass and released here, once
    :func:`type_text` has returned.

    Returns True when the text landed.
    """
    assert text is not None, "text is required"
    assert callable(type_text), "type_text must be callable"
    pasted = bool(type_text(text))
    if pill is not None:
        if pasted:
            pill.done(len(text))
        else:
            pill.error("Could not paste")
    if offer is not None:
        offer()
    return pasted


def should_offer_cleanup_download(
    *, declined: bool, downloading: bool, installed: bool
) -> bool:
    """Whether the automatic "download the cleanup model?" alert may be shown.

    Asked at most once a session: a modal on every utterance would be worse than
    no cleanup at all. But a decline is not permanent — that is what
    :data:`CLEANUP_DOWNLOAD_MENU_TITLE` is for. Setting the flag *before* the
    alert (so a "Not now" burned the one chance) is exactly what left the model
    unreachable for the rest of the session.
    """
    return not (declined or downloading or installed)


def cleanup_download_menu_enabled(*, installed: bool, downloading: bool) -> bool:
    """Whether the "Download cleanup model…" entry is clickable.

    Always present, because a user who said "Not now" needs a way back and the
    Settings popup deliberately does not list this model. Dead when there is
    nothing to do: already here, or already coming down.
    """
    return not (installed or downloading)


def should_prewarm_cleanup(
    config: dict, *, pro: bool, cleanup_enabled: bool, installed: bool, ram_gb: int | None
) -> bool:
    """Whether to start the cleanup server at launch instead of on first use.

    Pre-warming trades 2 GB of resident memory, from launch, for the multi-second
    wait the first cleaned utterance would otherwise pay. Worth it only when
    every one of these holds: the feature is licensed and switched on, the model
    is actually on disk, and the Mac has memory to spare. Anything else waits for
    the first utterance, where :data:`CLEANUP_FIRST_USE_WAIT_S` bounds the cost.
    """
    assert config is not None, "config is required"
    if not (pro and cleanup_enabled and installed):
        return False
    if not config.get(CLEANUP_PREWARM_KEY, CLEANUP_PREWARM_DEFAULT):
        return False
    return ram_gb is not None and ram_gb >= CLEANUP_PREWARM_MIN_RAM_GB


def run_cleanup(
    text: str,
    plan: CleanupPlan,
    *,
    cleanup,
    language: str | None = None,
    vocabulary_terms: tuple[str, ...] = (),
    transform_code=transform_spoken_code,
    render=render_system_prompt,
) -> CleanupOutcome:
    """Run the LLM pass for one utterance. Never raises on a bad reply.

    Code mode runs the deterministic spoken-punctuation transform *first*: it is
    idempotent and rule-based, so doing it before the model means the model sees
    real code tokens (``--force``) rather than the words for them, and cannot
    "correct" them back into prose.

    ``cleanup`` is the callable that actually talks to the server —
    ``CleanupRuntime.cleanup`` in the app, a fake in the tests. It returns a
    :class:`~cleanup.llama_server.CleanupResult`, whose ``skipped`` flag carries
    the original text: a skip costs the improvement, never the transcript.
    """
    assert text is not None, "text is required"
    assert plan is not None, "plan is required"
    assert callable(cleanup), "cleanup must be callable"
    if not plan.enabled:
        return CleanupOutcome(text=text, ran=False)

    if plan.mode_id == "code":
        text = transform_code(text, language=code_transform_language(language))

    system_prompt = render(
        plan.mode_id, plan.tone_id, prompt_language(language), tuple(vocabulary_terms)
    )
    result = cleanup(text, system_prompt)
    if result.skipped:
        return CleanupOutcome(
            text=text,
            ran=False,
            skipped_reason=result.reason or "the cleanup pass did not answer",
            elapsed_s=result.elapsed_s,
        )
    return CleanupOutcome(text=result.text, ran=True, elapsed_s=result.elapsed_s)


def mode_menu_state(config: dict) -> dict[str, bool]:
    """Which "Mode" submenu entries carry a checkmark.

    Every mode id maps to whether it is the configured fallback, and
    :data:`MODE_MENU_AUTOMATIC` to whether the bundle-id table applies. Both can
    be ticked at once, and that is the truth: the table decides per app and the
    ticked mode is what applies everywhere the table says nothing.
    """
    assert config is not None, "config is required"
    active = config.get("cleanup_mode", DEFAULT_CONFIG["cleanup_mode"])
    state = {mode_id: mode_id == active for mode_id in MODE_IDS}
    state[MODE_MENU_AUTOMATIC] = bool(
        config.get("context_awareness", DEFAULT_CONFIG["context_awareness"])
    )
    return state


def tone_menu_state(config: dict) -> dict[str, bool]:
    """Which "Tone" submenu entry carries a checkmark. Exactly one, always."""
    assert config is not None, "config is required"
    active = config.get("cleanup_tone", DEFAULT_CONFIG["cleanup_tone"])
    if active not in TONE_IDS:
        active = DEFAULT_CONFIG["cleanup_tone"]
    return {tone_id: tone_id == active for tone_id in TONE_IDS}


def cleanup_model_missing_message(display_name: str) -> str:
    """Offered once per session when a mode needs the GGUF and it is not here."""
    assert display_name, "display_name is required"
    return (
        f"Cleanup needs {display_name}, which is not downloaded yet.\n\n"
        "Your text was pasted exactly as dictated. Download it now to let "
        "Murmur clean up what you say."
    )


def cleanup_download_status(state) -> str:
    """Menu status while the cleanup model downloads, from the sheet's own state."""
    assert state is not None, "state is required"
    return f"Cleanup model: {state.status_line()}"


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


def model_integrity_message(display_name: str) -> str:
    """What to say when a model's files no longer match their checksums."""
    assert display_name, "display_name is required"
    return (
        f"{display_name} failed verification: its files do not match the "
        "checksums on record. Delete and re-download the model from "
        "Settings → Speech engine."
    )


def verify_model_before_load(store, model_id: str, verified: set) -> None:
    """Re-hash ``model_id`` before an engine is pointed at it, once per process.

    ``ModelStore.is_installed`` compares file sizes only, so a truncated,
    swapped or tampered model passes it and the engine happily loads whatever
    is on disk. Verification reads every byte, which costs a few seconds per
    gigabyte, so ``verified`` remembers the ids already checked in this process;
    the caller drops an id again after a download or an engine switch.

    A mismatch is re-raised as a plain :class:`RuntimeError` so it reaches the
    user through the same alert as any other failed load.
    """
    assert store is not None, "store is required"
    assert model_id, "model_id is required"
    assert verified is not None, "verified is required"
    if model_id in verified:
        return
    try:
        store.verify(model_id)
    except ModelIntegrityError as error:
        raise RuntimeError(model_integrity_message(model_id)) from error
    verified.add(model_id)


def about_menu_title(version: str, build_info: dict) -> str:
    """The About line: version, plus a warning when the build is not signed.

    ``build_info`` is ``{}`` outside a bundle, so a source run says nothing
    about signing; only a real bundle that reports ``signed: false`` is
    labelled, matching what CI writes into ``build_info.json``.
    """
    assert version, "version is required"
    title = f"{APP_NAME} {version}"
    if build_info.get("signed") is False:
        return f"{title} · internal build"
    return title


def update_available_message(latest_version: str, current_version: str) -> str:
    """Alert body offering an update. Version metadata only."""
    assert latest_version, "latest_version is required"
    assert current_version, "current_version is required"
    return (
        f"Murmur {latest_version} is available (you have {current_version}).\n\n"
        "Murmur will download it, check its signature, replace this copy, and "
        "restart itself."
    )


def should_relaunch_after_install(result) -> bool:
    """Whether this process must start the new bundle before it quits.

    ``install_update`` puts the new app in place but does not run it, because
    only the running app knows how to shut itself down cleanly. So Murmur
    launches the new bundle itself and then quits, and the user never has to
    reopen anything. Skipped only when the installer already relaunched.
    """
    assert result is not None, "result is required"
    return not bool(getattr(result, "relaunched", False))


def update_installed_message(version: str) -> str:
    """Said as the app hands over to the version it just installed."""
    assert version, "version is required"
    return f"Murmur {version} is installed. Restarting now."


def update_relaunch_failed_message(version: str) -> str:
    """Said when the new bundle is in place but would not start."""
    assert version, "version is required"
    return (
        f"Murmur {version} is installed, but it could not be started. "
        "Quit Murmur and open it again."
    )


def download_progress_status(bytes_done: int, bytes_total: int | None) -> str:
    """Menu status while an update downloads. Percent when the size is known."""
    assert bytes_done >= 0, f"bytes_done cannot be negative: {bytes_done}"
    if not bytes_total or bytes_total <= 0:
        return f"Downloading update… {bytes_done // 1_000_000} MB"
    percent = min(100, int(bytes_done * 100 / bytes_total))
    return f"Downloading update… {percent}%"


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


def _menu_bar_image(path):
    """Load and cache a menu bar NSImage."""
    cached = _MENU_BAR_IMAGES.get(path)
    if cached is not None:
        return cached

    image = NSImage.alloc().initByReferencingFile_(os.path.abspath(path))
    image.setScalesWhenResized_(True)
    image.setSize_((20, 20))
    if path == ICON_PATH:
        image.setTemplate_(True)
    _MENU_BAR_IMAGES[path] = image
    return image


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


class MurmurApp(rumps.App):
    def __init__(self):
        global APP_INSTANCE
        APP_INSTANCE = self
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

        # An update leaves the bundle it replaced beside the new one. This is
        # the launch that can safely delete it: nothing is loading out of it.
        threading.Thread(target=self._clean_previous_bundles, daemon=True).start()

        # First run: the wizard opens once the menu bar and run loop exist.
        self._onboarding_timer = rumps.Timer(self._maybe_show_onboarding, 0.6)
        self._onboarding_timer.start()

        # Register global shortcut after the run loop is active.
        self._hotkey_registration = None
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

    def _apply_menu_bar_state(self, state):
        """Update the existing NSStatusItem image directly (avoids rumps icon swap bugs)."""
        path = STATE_ICON_PATHS.get(state, ICON_PATH)
        image = _menu_bar_image(path)
        self.title = None
        self._icon = path
        self._icon_nsimage = image
        nsapp = getattr(self, "_nsapp", None)
        if nsapp is not None:
            nsapp.nsstatusitem.setImage_(image)
            nsapp.fallbackOnName()

    def _set_menu_bar_state(self, state):
        """Update the lone menu bar status icon (ready/recording/processing)."""
        self.run_on_main_thread(lambda: self._apply_menu_bar_state(state))

    def _input_device_indices(self) -> set[int]:
        return set(self._input_devices_by_index())

    def _input_devices_by_index(self) -> dict[int, str]:
        devices = sd.query_devices()
        return {
            i: device["name"]
            for i, device in enumerate(devices)
            if device.get("max_input_channels", 0) > 0
        }

    def _restore_microphone_from_config(self) -> None:
        """Apply persisted mic on startup; match name when present; clear stale config on failure."""
        config = self.runtime_config()
        saved_index = config.get("mic_device_index", DEFAULT_CONFIG["mic_device_index"])
        saved_name = config.get("mic_device_name", DEFAULT_CONFIG["mic_device_name"])
        try:
            devices = self._input_devices_by_index()
            device_idx = resolve_mic_device(saved_index, saved_name, devices)
            if device_idx is None:
                return
            sd.default.device = (device_idx, sd.default.device[1])
            device_name = devices[device_idx]
            if device_idx != saved_index or device_name != saved_name:
                self._persist_microphone_selection(device_idx, device_name)
            logger.info("Restored microphone device index %s (%s)", device_idx, device_name)
        except Exception as error:
            logger.error("Could not restore microphone device: %s", error)
            self.persistence.save_config(clear_mic_device_selection(config))
            rumps.notification(
                APP_NAME,
                "Microphone unavailable",
                "Saved microphone is no longer available. Choose one from the menu.",
            )

    def _persist_microphone_selection(self, device_idx: int, device_name: str) -> None:
        config = self.runtime_config()
        config["mic_device_index"] = device_idx
        config["mic_device_name"] = device_name
        self.persistence.save_config(config)

    def update_microphone_menu(self):
        """Update the microphone selection submenu"""
        # Clear existing items
        keys_to_remove = list(self.mic_menu.keys())
        for key in keys_to_remove:
            del self.mic_menu[key]
        
        try:
            devices = sd.query_devices()
            input_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            default_device = sd.default.device[0]
            
            for idx, name in input_devices[:10]:  # Limit to 10 devices
                # Truncate long names
                display_name = name[:40] + "..." if len(name) > 40 else name
                item = rumps.MenuItem(display_name, callback=lambda _, i=idx: self.select_microphone(i))
                if idx == default_device:
                    item.state = 1  # Checkmark
                self.mic_menu.add(item)
        except Exception as e:
            self.mic_menu.add(rumps.MenuItem("Default Microphone", callback=None))
    
    def select_microphone(self, device_idx):
        """Select a microphone device and persist the choice."""
        try:
            device_idx = resolve_mic_device_index(device_idx, self._input_device_indices())
            if device_idx is None:
                raise ValueError("Microphone device index is required")
            sd.default.device = (device_idx, sd.default.device[1])
            device_name = sd.query_devices(device_idx)["name"]
            self._persist_microphone_selection(device_idx, device_name)
            self.update_microphone_menu()
            logger.info(f"Microphone changed to: {device_name}")
        except Exception as e:
            logger.error(f"Error changing microphone: {e}")
            rumps.notification(
                APP_NAME,
                "Microphone unavailable",
                "Could not switch to that microphone. Choose another device.",
            )
    
    # -- cleanup mode and tone menus -------------------------------------

    def _build_mode_menu(self):
        """Five modes, then the switch that lets the front app choose for you."""
        for mode_id in MODE_IDS:
            self.mode_menu.add(
                rumps.MenuItem(
                    MODES[mode_id].display_name,
                    callback=lambda _, chosen=mode_id: self.select_mode(chosen),
                )
            )
        self.mode_menu.add(None)  # separator
        self.mode_menu.add(
            rumps.MenuItem(MODE_MENU_AUTOMATIC, callback=self.toggle_context_awareness)
        )
        # Every mode above except Dictation needs the cleanup model, and the
        # Settings popup will not offer it (it lists speech engines only), so
        # this is where it can always be fetched from.
        self.mode_menu.add(
            rumps.MenuItem(CLEANUP_DOWNLOAD_MENU_TITLE, callback=self.download_cleanup_model)
        )
        self._refresh_cleanup_download_item()
        self._refresh_mode_menu()

    def _build_tone_menu(self):
        for tone_id in TONE_IDS:
            self.tone_menu.add(
                rumps.MenuItem(
                    TONES[tone_id].display_name,
                    callback=lambda _, chosen=tone_id: self.select_tone(chosen),
                )
            )
        self._refresh_tone_menu()

    def _refresh_mode_menu(self, config=None):
        """Put the checkmarks where :func:`mode_menu_state` says they belong."""
        state = mode_menu_state(self.runtime_config() if config is None else config)
        for mode_id in MODE_IDS:
            self.mode_menu[MODES[mode_id].display_name].state = int(state[mode_id])
        self.mode_menu[MODE_MENU_AUTOMATIC].state = int(state[MODE_MENU_AUTOMATIC])

    def _refresh_tone_menu(self, config=None):
        state = tone_menu_state(self.runtime_config() if config is None else config)
        for tone_id in TONE_IDS:
            self.tone_menu[TONES[tone_id].display_name].state = int(state[tone_id])

    def select_mode(self, mode_id):
        """Menu item: set the fallback cleanup mode. Writes one key."""
        config = self.persistence.update_config({"cleanup_mode": mode_id})
        logger.info("Cleanup mode set to %s", mode_id)
        self._refresh_mode_menu(config)

    def select_tone(self, tone_id):
        """Menu item: set the cleanup tone. Writes one key."""
        config = self.persistence.update_config({"cleanup_tone": tone_id})
        logger.info("Cleanup tone set to %s", tone_id)
        self._refresh_tone_menu(config)

    def toggle_context_awareness(self, _):
        """Menu item: let the front app pick the mode, or stop it doing so."""
        current = self.runtime_config().get(
            "context_awareness", DEFAULT_CONFIG["context_awareness"]
        )
        config = self.persistence.update_config({"context_awareness": not current})
        logger.info("Context awareness %s", "on" if not current else "off")
        self._refresh_mode_menu(config)

    def open_settings(self, _):
        """Menu item: open the settings window."""
        self.open_settings_window_safely()


    def open_history_window(self, _, selected_index=0):
        """Open the SuperWhisper-style history window"""
        try:
            show_history_window_direct()
        except Exception:
            logger.error("Could not open history", exc_info=True)
            ui_alerts.show_alert(APP_NAME, "Could not open History.")

    def check_updates(self, _):
        """Ask the update feed off the main thread; the answer comes back as an alert."""
        threading.Thread(target=self._check_updates_worker, daemon=True).start()

    def _check_updates_worker(self):
        """Fetch release metadata only. No audio, no text, nothing uploaded."""
        try:
            info = UpdateService(APP_VERSION).check()
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
    
    def add_to_history(self, text, source_type, filename=None, audio_path=None):
        """Add a transcription to history"""
        if not self.runtime_config().get("save_history", DEFAULT_CONFIG["save_history"]):
            return
        self.history = self.persistence.add_history_entry(
            self.history,
            text=text,
            source_type=source_type,
            filename=filename,
            audio_path=audio_path,
        )
        self.save_history()
    
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
            updated = {
                **config,
                CONFIG_ENGINE_ID: selection.engine_id,
                CONFIG_MODEL_ID: selection.model_id,
            }
            self.persistence.save_config(updated)
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
            for controller in list(_window_controllers):
                hook = getattr(controller, "engine_reloaded", None)
                if hook is None:
                    continue
                try:
                    hook(info)
                except Exception as error:
                    logger.warning("Settings could not follow the engine swap: %s", error)

        self.run_on_main_thread(notify)

    def _model_display_name(self, model_id):
        """Catalog display name for a model id, falling back to the id itself."""
        try:
            return app_model_store().spec(model_id).display_name
        except Exception:
            return model_id

    def _set_model_menu_title(self, title, *, transient=False):
        """Write the engine status line in the menu, from any thread.

        ``transient=True`` marks a title that borrows the line for something
        else (an update download), so it is not remembered as the engine's
        status and cannot be restored later.
        """
        if not transient:
            self._model_status_title = title
        self.run_on_main_thread(lambda: setattr(self.model_item, "title", title))

    def _engine_display_name(self):
        """What to call the running engine in user-facing copy."""
        engine = self.engine
        if engine is not None:
            try:
                return engine.info().name
            except Exception:
                pass
        return self.engine_id or "this engine"

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
        self.persistence.save_config(remember_hints_notice(self.runtime_config(), engine_id))
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

    def _clean_up_transcript(self, text, config, language, vocabulary, pill=None):
        """Run the cleanup pass for one utterance and report what happened.

        Returns the text to paste. Cleanup is an improvement on top of a
        transcript that is already correct, so every failure here ends with the
        original text and a notice — never with nothing.
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

        if pill is not None and not self.cleanup_runtime.is_started:
            # The first cleaned utterance waits on a 2 GB model load. The menu
            # bar says so, but the user is looking at the pill, and a pill that
            # sits on the plain "working" state through several seconds reads as
            # a hang. Naming the wait is the difference between "it's broken"
            # and "it's coming".
            pill.working(label=CLEANUP_PREPARING_STATUS)
        try:
            outcome = run_cleanup(
                text,
                plan,
                cleanup=self.cleanup_runtime.cleanup,
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

    def _refresh_cleanup_download_item(self):
        """Grey the download entry out when there is nothing for it to do."""
        item = self.mode_menu.get(CLEANUP_DOWNLOAD_MENU_TITLE)
        if item is None:
            return
        enabled = cleanup_download_menu_enabled(
            installed=self._installed_cleanup_model_path() is not None,
            downloading=self._cleanup_download_running(),
        )
        item.set_callback(self.download_cleanup_model if enabled else None)

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
                pro=pro_enabled("cleanup", config),
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

    # -- onboarding ------------------------------------------------------

    def open_settings_window_safely(self):
        """Open Settings from any code path without letting AppKit errors escape."""
        try:
            show_settings_window_direct()
        except Exception:
            logger.error("Could not open settings", exc_info=True)
            ui_alerts.show_alert(APP_NAME, "Could not open Settings.")

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
        """Apply the shortcut from settings, replacing any previous registration."""
        config = self.runtime_config()
        binding = hotkey_from_config(config)
        mode = hotkey_mode_from_config(config)
        unregister_global_hotkey(self._hotkey_registration)
        self._hotkey_registration = None
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
        # state only, which is what the pill's phases are for.
        streaming = pill is not None and bool(
            getattr(self.engine, "supports_streaming", False)
        )

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
            # vocabulary biases the decode and then fixes what it got wrong.
            language = resolve_language(config, front_app_bundle_id())
            vocabulary = vocabulary_from_config(config)
            hints = hints_from_vocabulary(vocabulary)

            if stream_text is not None and language_is_auto(language):
                # The live decoder already produced the whole utterance while
                # the user was speaking; running the file through the same
                # engine again would only cost a second and say the same thing.
                #
                # Only when the language is auto, though. The streaming engine
                # cannot honour a pinned language (see Engine.stream), so a user
                # who chose French would otherwise get whatever the decoder
                # guessed. The pill still showed them the live words; the batch
                # pass below is what actually gets pasted.
                logger.info("Using the live stream result (%d chars)", len(stream_text))
                raw_text = stream_text
            else:
                if stream_text is not None:
                    logger.info(
                        "Ignoring the live stream result: the language is pinned to %s",
                        language,
                    )
                logger.info("Starting transcription in language %s", language)
                with self._engine_lock:
                    transcript = self.engine.transcribe(
                        Path(audio_path), language=language, hints=hints
                    )
                self._note_hints_support(config, transcript, vocabulary)
                raw_text = transcript.text
            # The hallucination filter reads the engine's raw words; the user's
            # replacements are applied to what survives.
            text, is_hallucination = finalize_transcript(raw_text, vocabulary)
            if should_log_sensitive(config):
                logger.info("Transcription completed")

            if text and not is_hallucination:
                # Cleanup sits between the replacements and the paste: it reads
                # what the user actually meant to write, terms included.
                text = self._clean_up_transcript(text, config, language, vocabulary, pill)
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
                self.add_to_history(text, "live", audio_path=history_audio_path)
            else:
                if is_hallucination:
                    logger.info("Filtered hallucination")
                    history_text = f"(Filtered) {text[:120]}"
                else:
                    logger.info("No speech detected")
                    history_text = "(No speech detected)"
                history_audio_path = audio_path if save_audio else None
                self.add_to_history(history_text, "live", audio_path=history_audio_path)
                if pill is not None:
                    pill.error("No speech detected")
                rumps.notification(
                    APP_NAME,
                    "No speech detected",
                    "Nothing clear enough to paste. Try again closer to the mic.",
                )

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
    
    def _reset_menu_state(self):
        """Reset menu items to normal state - must be called on main thread"""
        def do_reset():
            self.is_processing = False
            if should_apply_ready_on_reset(is_recording=self.is_recording):
                self._apply_menu_bar_state("ready")
                self.start_stop_item.title = "Start/Stop Recording"
            self.start_stop_item.set_callback(self.toggle_recording)
            self.upload_item.set_callback(self.upload_audio_file)
        self.run_on_main_thread(do_reset)
    
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
            vocabulary = vocabulary_from_config(config)
            hints = hints_from_vocabulary(vocabulary)
            with self._engine_lock:
                # A whole-file import, not dictation: the decoder may condition on
                # the text it already produced for earlier windows.
                transcript = self.engine.transcribe(
                    Path(file_path), language=language, hints=hints, long_form=True
                )
            text = apply_replacements(transcript.text, vocabulary)
            self._note_hints_support(config, transcript, vocabulary)

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

    def update_history_menu(self):
        """Keep menu state in sync after history mutations."""
        # History currently lives in the dedicated history window.
        # We still expose this method because clear_history() invokes it.
        self.history_item.title = "History"
    
    def quit_app(self, _):
        """Quit the application.

        The cleanup server is a child process holding a 2 GB model; leaving it
        behind would keep that resident after Murmur is gone. Neither shutdown
        may block the quit, so both are best effort.
        """
        try:
            self.cleanup_runtime.stop()
        except Exception as error:
            logger.warning("Could not stop the cleanup server: %s", error)
        try:
            self.pill.close()
        except Exception as error:
            logger.warning("Could not close the pill: %s", error)
        rumps.quit_application()


if __name__ == "__main__":
    ensure_single_instance()
    ns_app = NSApplication.sharedApplication()
    ns_app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    MurmurApp().run()
