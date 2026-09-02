#!/usr/bin/env python3
"""
Murmur - A simple local speech-to-text menu bar app
Shortcut: Option+Space to start/stop recording
"""

import sys
import os

# Add bundled ffmpeg to PATH FIRST if running in PyInstaller bundle
# This must happen before any imports that might use ffmpeg
if hasattr(sys, '_MEIPASS'):
    # Add the bundled directory to PATH
    os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ.get('PATH', '')
    
    # Also patch whisper's audio module to use the bundled ffmpeg directly
    import subprocess
    _original_run = subprocess.run
    _ffmpeg_path = os.path.join(sys._MEIPASS, 'ffmpeg')
    
    def _patched_run(cmd, *args, **kwargs):
        # If the command starts with 'ffmpeg', replace it with the full path
        if cmd and isinstance(cmd, list) and cmd[0] == 'ffmpeg':
            cmd = [_ffmpeg_path] + cmd[1:]
        elif cmd and isinstance(cmd, str) and cmd.startswith('ffmpeg'):
            cmd = cmd.replace('ffmpeg', _ffmpeg_path, 1)
        return _original_run(cmd, *args, **kwargs)
    
    subprocess.run = _patched_run

import fcntl
import json
import rumps
import sounddevice as sd
import numpy as np
import urllib.error
import urllib.request
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
    logger.info(f"Patched subprocess.run to use bundled ffmpeg")

import pyperclip
import threading
import subprocess
import scipy.io.wavfile as wav
import time
import shutil
import tempfile
from pathlib import Path
from engines import create_engine
from transcription_filters import is_likely_hallucination, should_skip_audio
from services.audio_capture_service import AudioCaptureService
from services.hotkey_service import (
    format_hotkey,
    format_hotkey_from_config,
    hotkey_diagnostics,
    hotkey_from_config,
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
from services.model_profile_service import default_model_for_current_machine
from services.persistence_service import (
    DEFAULT_CONFIG,
    PersistencePaths,
    PersistenceService,
    should_log_sensitive,
)
from services.text_insertion_service import TextInsertionService
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
GITHUB_RELEASES_LATEST_URL = "https://api.github.com/repos/Mbr0/murmur/releases/latest"
UPDATE_CHECK_TIMEOUT_SECONDS = 5.0

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

# Load model from config
default_model = default_model_for_current_machine()
_config = PERSISTENCE.load_config(default={"model": default_model, **DEFAULT_CONFIG})
MODEL_SIZE = _config.get("model", default_model)
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
    config = PERSISTENCE.load_config(default={"model": default_model, **DEFAULT_CONFIG})
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


def normalize_release_tag(tag: str) -> str:
    cleaned = tag.strip()
    if len(cleaned) >= 2 and cleaned[0] in ("v", "V") and cleaned[1].isdigit():
        return cleaned[1:]
    return cleaned


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for segment in normalize_release_tag(version).split("."):
        digits = []
        for char in segment:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if not digits:
            raise ValueError(f"Invalid version: {version}")
        parts.append(int("".join(digits)))
    if not parts:
        raise ValueError(f"Invalid version: {version}")
    return tuple(parts)


def is_newer_version(latest: str, current: str) -> bool:
    """True when latest release tag is strictly newer than the installed version."""
    return _version_tuple(latest) > _version_tuple(current)


def parse_latest_release_tag(payload: dict) -> str:
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("Missing tag_name in release payload")
    return tag.strip()


def check_for_update_message(
    *, current_version: str, latest_tag: str | None, error: str | None
) -> str:
    """User-facing update status. Never includes audio or transcription content."""
    offline = (
        "Could not check for updates. Check your network connection and try again."
    )
    if error is not None or latest_tag is None:
        return offline
    current = normalize_release_tag(current_version)
    latest = normalize_release_tag(latest_tag)
    try:
        if is_newer_version(latest_tag, current_version):
            return (
                f"Update available: {latest} (you have {current}).\n"
                "Download from GitHub Releases: github.com/Mbr0/murmur/releases"
            )
        if is_newer_version(current_version, latest_tag):
            return (
                f"Your version ({current}) is ahead of the latest release ({latest})."
            )
    except ValueError:
        return offline
    return f"You're on the latest version ({current})."


def fetch_latest_release_tag(
    url: str = GITHUB_RELEASES_LATEST_URL,
    *,
    timeout: float = UPDATE_CHECK_TIMEOUT_SECONDS,
) -> str:
    """Fetch latest GitHub release tag only (version metadata; no audio/text upload)."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"Murmur/{APP_VERSION}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        raise RuntimeError("offline") from error
    if not isinstance(payload, dict):
        raise ValueError("Unexpected release payload")
    return parse_latest_release_tag(payload)


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
        self._whisper_lock = threading.Lock()
        self.persistence = PERSISTENCE
        self.history = self.load_history()
        self.audio_capture = AudioCaptureService(sample_rate=SAMPLE_RATE, logger=logger)
        # Built in load_model(), where a bad config or an unresolvable engine id
        # becomes the same status/notification/alert as a failed load instead of
        # killing the app before the menu bar exists.
        self.engine = None
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
        
        self.menu = [
            self.start_stop_item,
            self.upload_item,
            self.history_item,
            self.settings_item,
            self.mic_menu,
            None,  # Separator
            rumps.MenuItem(f"Model: {MODEL_SIZE.capitalize()}", callback=None),
            rumps.MenuItem(f"Murmur {APP_VERSION}", callback=None),
            rumps.MenuItem("Check for Updates...", callback=self.check_updates),
            rumps.MenuItem("Enable Shortcut Permission...", callback=self.enable_shortcut_permission),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app, key="q"),
        ]
        
        # Load model in background
        self.loading = True
        threading.Thread(target=self.load_model, daemon=True).start()
        
        # Register global shortcut after the run loop is active.
        self._hotkey_registration = None
        self._hotkey_retry_timer = None
        self._hotkey_permission_notified = False
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
        """Open settings window"""
        try:
            show_settings_window_direct()
        except Exception:
            logger.error("Could not open settings", exc_info=True)
            ui_alerts.show_alert(APP_NAME, "Could not open Settings.")
    
    def open_history_window(self, _, selected_index=0):
        """Open the SuperWhisper-style history window"""
        try:
            show_history_window_direct()
        except Exception:
            logger.error("Could not open history", exc_info=True)
            ui_alerts.show_alert(APP_NAME, "Could not open History.")

    def check_updates(self, _):
        """Compare installed version to latest GitHub release tag (metadata only)."""
        latest_tag = None
        error = None
        try:
            latest_tag = fetch_latest_release_tag()
        except Exception:
            error = "offline"
        ui_alerts.show_alert(
            APP_NAME,
            check_for_update_message(
                current_version=APP_VERSION,
                latest_tag=latest_tag,
                error=error,
            ),
        )
    
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
        default = {"model": default_model_for_current_machine(), **DEFAULT_CONFIG}
        return self.persistence.load_config(default)
    
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
    
    def load_model(self):
        """Load the transcription engine"""
        logger.info(f"Loading model: {MODEL_SIZE}")
        self.update_status("Loading model...")
        self._set_menu_bar_state("processing")
        try:
            self.engine = create_engine("whisper_openai", model_name=MODEL_SIZE)
            self.engine.load()
            logger.info("Model loaded successfully %s", self.engine.runtime_summary())
            self.loading = False
            self.update_status(f"Ready ({format_hotkey_from_config(self.runtime_config())} to record)")
            self._set_menu_bar_state("ready")
            # Model loaded silently - no notification needed
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.loading = False
            self.update_status(f"Error: {str(e)[:30]}")
            error_state = "error" if "error" in STATE_ICON_PATHS else "ready"
            self._set_menu_bar_state(error_state)
            rumps.notification(
                APP_NAME,
                "Model failed to load",
                "Recording is unavailable until the model loads successfully.",
            )
            self.run_on_main_thread(
                lambda: ui_alerts.show_alert(
                    APP_NAME,
                    f"Could not load the speech model.\n\n{e}",
                )
            )
    

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
        binding = hotkey_from_config(self.runtime_config())
        unregister_global_hotkey(self._hotkey_registration)
        self._hotkey_registration = None

        def trigger_toggle():
            self.run_on_main_thread(self._safe_toggle)

        def handle_error(error):
            logger.error(f"Hotkey callback error: {error}")

        # 1. Try to register the hotkey (Carbon will succeed without Accessibility)
        try:
            self._hotkey_registration = register_global_hotkey(
                binding,
                on_trigger=trigger_toggle,
                on_error=handle_error,
                logger=logger,
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
                on_trigger=trigger_toggle,
                on_error=handle_error,
                logger=logger,
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
                    "Recording is unavailable until the model loads successfully.",
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
            
            # Transcribe with better parameters
            logger.info("Starting transcription...")
            with self._whisper_lock:
                transcript = self.engine.transcribe(Path(audio_path), language=None)
            text = transcript.text
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
                    "Transcription is unavailable until the model loads successfully.",
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
            
            with self._whisper_lock:
                # A whole-file import, not dictation: the decoder may condition on
                # the text it already produced for earlier windows.
                transcript = self.engine.transcribe(
                    Path(file_path), language=None, long_form=True
                )
            text = transcript.text

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
