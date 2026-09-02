#!/usr/bin/env python3
"""
Murmur - A simple local speech-to-text menu bar app
Shortcut: Option+Space to start/stop recording
"""

import sys
import os

# Put the bundled resources directory on PATH before anything that shells out.
# The bundled `whisper-server` and `ffmpeg` both live there, so a plain command
# name resolves to the bundled copy without any monkey-patching of subprocess.
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

import pyperclip
import threading
import subprocess
import scipy.io.wavfile as wav
import time
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from cleanup.vocabulary import (
    apply_replacements,
    hints_from_vocabulary,
    vocabulary_from_config,
)
from engines import create_engine
from engines.model_store import ModelStore, models_for_engine
from transcription_filters import is_likely_hallucination, should_skip_audio
from ui.onboarding_window import OnboardingCallbacks, should_show, show_onboarding
from services.audio_capture_service import AudioCaptureService
from services.language_service import resolve_language
from services.hotkey_service import (
    ACTION_START,
    ACTION_STOP,
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
from services.model_profile_service import default_engine_for_current_machine
from services.persistence_service import (
    DEFAULT_CONFIG,
    PersistencePaths,
    PersistenceService,
    should_log_sensitive,
)
from services.text_insertion_service import TextInsertionService
from services.update_service import UpdateService, read_build_info
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
) -> str:
    """Whether a requested engine swap may start now.

    Policy: refuse rather than queue. A queued swap would fire minutes later,
    long after the user stopped thinking about it, and a refusal that says so
    is easier to act on than a delayed surprise. Recording and transcription
    both hold the engine, so both block; a second request while one is already
    in flight is refused too.
    """
    assert requested and len(requested) == 2, "requested is (engine_id, model_id)"
    if is_reloading:
        return RELOAD_BUSY
    if is_recording:
        return RELOAD_RECORDING
    if is_processing:
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
        "Murmur will download it, check its signature, and replace this copy."
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
        self.text_inserter = TextInsertionService(logger=logger)
        
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
        self.menu = [
            self.start_stop_item,
            self.upload_item,
            self.history_item,
            self.settings_item,
            self.mic_menu,
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

        # First run: the wizard opens once the menu bar and run loop exist.
        self._onboarding_timer = rumps.Timer(self._maybe_show_onboarding, 0.6)
        self._onboarding_timer.start()

        # Register global shortcut after the run loop is active.
        self._hotkey_registration = None
        self._hotkey_retry_timer = None
        self._hotkey_permission_notified = False
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
        """Download, verify the signature, install. Progress goes to the menu."""
        previous_title = self.model_item.title

        def progress(bytes_done, bytes_total):
            line = download_progress_status(bytes_done, bytes_total)
            if line == self._update_status_line:
                return  # every chunk ticks; only redraw when the text changes
            self._update_status_line = line
            self._set_model_menu_title(line)

        try:
            UpdateService(APP_VERSION).download_and_install(info, progress)
        except Exception as error:
            logger.error("Update install failed: %s", error)
            self._set_model_menu_title(previous_title)
            self._alert_on_main(f"Could not install Murmur {info.version}.\n\n{error}")
            return
        finally:
            self._update_status_line = None
        self._set_model_menu_title(previous_title)
        self._alert_on_main(
            f"Murmur {info.version} is installed. Quit and reopen Murmur to use it."
        )

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

    def _model_display_name(self, model_id):
        """Catalog display name for a model id, falling back to the id itself."""
        try:
            return ModelStore().spec(model_id).display_name
        except Exception:
            return model_id

    def _set_model_menu_title(self, title):
        """Write the engine status line in the menu, from any thread."""
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

    def load_model(self):
        """Load the configured speech engine. Runs on a background thread."""
        self.update_status("Loading model...")
        self._set_menu_bar_state("processing")
        try:
            config = self.runtime_config()
            selection = self._resolve_selection(config)
            store = ModelStore()
            if not store.is_installed(selection.model_id):
                self._report_missing_model(config, selection)
                return
            self._activate_engine(selection.engine_id, selection.model_id, store)
        except Exception as error:
            self._report_engine_failure(error)

    def _activate_engine(self, engine_id, model_id, store):
        """Unload whatever is loaded, then build and publish the new engine.

        The old engine goes first: two speech models resident at once run to
        several gigabytes, and the Macs most likely to switch models are the
        ones that can least afford holding both. The caller reports failures.
        """
        logger.info("Loading engine %s with model %s", engine_id, model_id)
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

    def reload_engine(self, engine_id: str, model_id: str) -> None:
        """Swap the speech engine without a restart. Called by Settings.

        Settings calls this on the main thread; the work itself runs on a
        background thread so the popup does not freeze behind a model load.
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
        )
        if decision == RELOAD_UNCHANGED:
            return
        if decision != RELOAD_START:
            message = RELOAD_REFUSAL_MESSAGES[decision]
            logger.info("Engine switch refused (%s)", decision)
            self.update_status(message)
            rumps.notification(APP_NAME, "Engine unchanged", message)
            return
        self._engine_reloading = True
        threading.Thread(
            target=self._reload_engine_worker,
            args=(engine_id, model_id),
            daemon=True,
        ).start()

    def _reload_engine_worker(self, engine_id, model_id):
        """Unload the old engine, load the new one, then persist the choice."""
        self.update_status(SWITCHING_STATUS)
        self._set_model_menu_title(SWITCHING_STATUS)
        self._set_menu_bar_state("processing")
        try:
            store = ModelStore()
            if not store.is_installed(model_id):
                raise RuntimeError(
                    f"{self._model_display_name(model_id)} is not downloaded yet."
                )
            self._activate_engine(engine_id, model_id, store)
            config = self.runtime_config()
            config[CONFIG_ENGINE_ID] = engine_id
            config[CONFIG_MODEL_ID] = model_id
            self.persistence.save_config(config)
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
                    download=ModelStore().download,
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
        """Persist what the wizard decided, and pick up a model it downloaded."""
        config = self.runtime_config()
        config.update(updates)
        self.persistence.save_config(config)
        if engine_is_ready(self.engine) or self._engine_reloading:
            return
        self.loading = True
        threading.Thread(target=self.load_model, daemon=True).start()

    def _record_test_sentence(self):
        """Record a few seconds and transcribe them for the wizard's own field.

        The result goes back to the wizard and nowhere else: it is not pasted,
        not saved to history, and never logged.
        """
        if not engine_is_ready(self.engine):
            raise RuntimeError(
                "The speech model is not loaded yet. Download it on the previous "
                "step, then try this again."
            )
        if self.is_recording or self.is_processing:
            raise RuntimeError("Murmur is busy with another recording. Try again in a moment.")

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
                transcript = self.engine.transcribe(Path(audio_path), language=None)
        finally:
            try:
                os.unlink(audio_path)
            except OSError as error:
                logger.error("Failed to delete the wizard's temp audio: %s", error)
        return transcript.text

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
    
    def start_recording(self):
        """Start audio recording"""
        logger.info("start_recording called")
        self.is_recording = True
        self.recording_start_time = time.time()  # Track when recording started
        self._set_menu_bar_state("recording")
        self.start_stop_item.title = "Stop Recording"
        self.upload_item.set_callback(None)  # Disable transcribe file

        try:
            logger.info(f"Starting audio capture with sample rate {SAMPLE_RATE}")
            self.audio_capture.start()
            logger.info("Audio stream started successfully")
        except Exception as e:
            logger.error(f"Mic error: {e}")
            self.is_recording = False
            self.update_status(f"Mic error: {str(e)[:20]}")
            self._reset_menu_state()
            rumps.notification(
                APP_NAME,
                "Microphone error",
                "Could not start recording. Check microphone permissions and device.",
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
        if not audio_chunks:
            logger.warning("No audio data to transcribe")
            self.update_status("No audio recorded")
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
                rumps.notification(
                    APP_NAME,
                    "Recording skipped",
                    skip_audio_user_message(duration_seconds, max_level),
                )
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

            logger.info("Starting transcription in language %s", language)
            with self._engine_lock:
                transcript = self.engine.transcribe(
                    Path(audio_path), language=language, hints=hints
                )
            text = apply_replacements(transcript.text, vocabulary)
            self._note_hints_support(config, transcript, vocabulary)
            if should_log_sensitive(config):
                logger.info("Transcription completed")
            
            # Filter out common hallucinations that occur with silence/noise
            is_hallucination = is_likely_hallucination(text)
            
            if text and not is_hallucination:
                # Small delay then paste (paste_text copies, pastes, then restores clipboard)
                time.sleep(0.15)
                self.type_text(text)
                
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
                rumps.notification(
                    APP_NAME,
                    "No speech detected",
                    "Nothing clear enough to paste. Try again closer to the mic.",
                )
            
            # Re-enable menu items
            self._reset_menu_state()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
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
        """Type text at current cursor position using native macOS events."""
        try:
            self.text_inserter.paste_text(text)
        except Exception as e:
            logger.error(f"Paste failed: {e}")
            rumps.notification(
                APP_NAME,
                "Could not paste",
                "Enable Accessibility for Murmur, then try again.",
            )
    
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
        """Quit the application"""
        rumps.quit_application()


if __name__ == "__main__":
    ensure_single_instance()
    ns_app = NSApplication.sharedApplication()
    ns_app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    MurmurApp().run()
