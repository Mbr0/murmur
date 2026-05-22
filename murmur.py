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

import rumps
import sounddevice as sd
import numpy as np
import whisper
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/murmur_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if hasattr(sys, '_MEIPASS'):
    logger.info(f"Added bundled resources to PATH: {sys._MEIPASS}")
    logger.info(f"Patched subprocess.run to use bundled ffmpeg")

import pyperclip
import threading
import subprocess
import scipy.io.wavfile as wav
import time
import shutil
from transcription_filters import is_likely_hallucination, should_skip_audio
from services.audio_capture_service import AudioCaptureService
from services.hotkey_service import register_option_space_hotkey
from services.model_profile_service import default_model_for_current_machine
from services.persistence_service import PersistencePaths, PersistenceService
from services.text_insertion_service import TextInsertionService

import objc
import Cocoa
from Cocoa import (
    NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSBackingStoreBuffered, NSMakeRect, NSApp, NSScrollView,
    NSTextView, NSFont, NSColor, NSBezelBorder, NSViewWidthSizable,
    NSViewHeightSizable, NSFloatingWindowLevel, NSButton,
    NSRoundedBezelStyle, NSMomentaryLightButton, NSCenterTextAlignment,
    NSApplication
)
from objc import python_method
from datetime import datetime
from Foundation import NSObject
from PyObjCTools import AppHelper
# Settings
SAMPLE_RATE = 16000
APP_NAME = "Murmur"

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

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load model from config
default_model = default_model_for_current_machine()
_config = PERSISTENCE.load_config(default={"model": default_model})
MODEL_SIZE = _config.get("model", default_model)

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

def _get_history_module():
    """Get the history module, loading it once"""
    global _history_module
    if _history_module is None:
        script_path = resource_path("history_window.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("history_window", script_path)
        _history_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_history_module)
        logger.info("History module loaded")
    return _history_module

def _get_settings_module():
    """Get the settings module, loading it once"""
    global _settings_module
    if _settings_module is None:
        script_path = resource_path("settings_window.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("settings_window", script_path)
        _settings_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_settings_module)
        logger.info("Settings module loaded")
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

def show_history_window_direct():
    """Show history window directly in the same process"""
    global _window_controllers
    logger.info("show_history_window_direct called")
    
    _cleanup_window_controllers()
    
    # Check if we already have an open history window and bring it to front
    for controller in _window_controllers:
        try:
            if controller.__class__.__name__ == 'HistoryWindowController':
                logger.info("Found existing history window, bringing to front")
                controller.window.makeKeyAndOrderFront_(None)
                NSApp.activateIgnoringOtherApps_(True)
                return
        except Exception as e:
            logger.error(f"Error checking existing window: {e}")
    
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
    
    _cleanup_window_controllers()
    
    # Check if we already have an open settings window and bring it to front
    for controller in _window_controllers:
        try:
            if controller.__class__.__name__ == 'SettingsWindowController':
                logger.info("Found existing settings window, bringing to front")
                controller.window.makeKeyAndOrderFront_(None)
                NSApp.activateIgnoringOtherApps_(True)
                return
        except Exception as e:
            logger.error(f"Error checking existing window: {e}")
    
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

# Custom menu bar icon
ICON_PATH = resource_path("logo_menu_white.png")
DOCK_ICON_PATH = resource_path("logo_rounded.png")  # Rounded logo for dock
ICON_RECORDING = resource_path("icon_recording.png")
ICON_PROCESSING = resource_path("icon_processing.png")
ICON_ERROR = resource_path("icon_error.png")


class MurmurApp(rumps.App):
    def __init__(self):
        super(MurmurApp, self).__init__(APP_NAME, icon=ICON_PATH, quit_button=None)
        self.title = None
        
        # State
        self.is_recording = False
        self.history = self.load_history()
        self.audio_capture = AudioCaptureService(sample_rate=SAMPLE_RATE, logger=logger)
        self.model = None
        self.persistence = PERSISTENCE
        self.text_inserter = TextInsertionService(logger=logger)
        
        # Menu items - SuperWhisper style
        self.start_stop_item = rumps.MenuItem("Start/Stop Recording", callback=self.toggle_recording)
        self.upload_item = rumps.MenuItem("Transcribe File", callback=self.upload_audio_file)
        self.history_item = rumps.MenuItem("History", callback=self.open_history_window)
        self.settings_item = rumps.MenuItem("Settings", callback=self.open_settings)
        
        logger.info("Menu items created with callbacks")
        
        # Microphone submenu
        self.mic_menu = rumps.MenuItem("Microphone")
        self.update_microphone_menu()
        
        self.menu = [
            self.start_stop_item,
            self.upload_item,
            self.history_item,
            self.settings_item,
            self.mic_menu,
            None,  # Separator
            rumps.MenuItem(f"Version {MODEL_SIZE}", callback=None),
            rumps.MenuItem("Check for Updates...", callback=self.check_updates),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app, key="q"),
        ]
        
        # Load model in background
        self.loading = True
        threading.Thread(target=self.load_model, daemon=True).start()
        
        # Setup keyboard listener
        self.setup_keyboard_listener()

        # Set Dock Icon
        self.set_dock_icon()
    
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
        """Select a microphone device"""
        try:
            sd.default.device[0] = device_idx
            self.update_microphone_menu()
            device_name = sd.query_devices(device_idx)['name']
            logger.info(f"Microphone changed to: {device_name}")
        except Exception as e:
            logger.error(f"Error changing microphone: {e}")
    
    def open_settings(self, _):
        """Open settings window"""
        try:
            show_settings_window_direct()
        except Exception as e:
            logger.error(f"Could not open settings: {e}")
    
    def check_updates(self, _):
        """Check for updates (placeholder)"""
        rumps.alert(APP_NAME, "You're running the latest version!")
    
    def set_dock_icon(self):
        """Set the application dock icon"""
        try:
            if os.path.exists(DOCK_ICON_PATH):
                image = Cocoa.NSImage.alloc().initByReferencingFile_(os.path.abspath(DOCK_ICON_PATH))
                Cocoa.NSApplication.sharedApplication().setApplicationIconImage_(image)
        except Exception as e:
            print(f"Failed to set dock icon: {e}")
    
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
    
    def save_history(self):
        """Save transcription history to file"""
        self.persistence.save_history(self.history)
    
    def add_to_history(self, text, source_type, filename=None, audio_path=None):
        """Add a transcription to history"""
        self.history = self.persistence.add_history_entry(
            self.history,
            text=text,
            source_type=source_type,
            filename=filename,
            audio_path=audio_path,
        )
        self.save_history()
    
    def load_model(self):
        """Load Whisper model"""
        logger.info(f"Loading model: {MODEL_SIZE}")
        self.update_status("Loading model...")
        self.run_on_main_thread(lambda: setattr(self, 'icon', ICON_PROCESSING))
        self.run_on_main_thread(lambda: setattr(self, 'title', None))
        try:
            self.model = whisper.load_model(MODEL_SIZE)
            self.loading = False
            logger.info("Model loaded successfully")
            self.update_status("Ready (⌥Space to record)")
            self.run_on_main_thread(lambda: setattr(self, 'icon', ICON_PATH))
            self.run_on_main_thread(lambda: setattr(self, 'title', None))
            # Model loaded silently - no notification needed
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.update_status(f"Error: {str(e)[:30]}")
            self.run_on_main_thread(lambda: setattr(self, 'icon', ICON_ERROR))
            self.run_on_main_thread(lambda: setattr(self, 'title', None))
    
    def setup_keyboard_listener(self):
        """Setup global keyboard shortcut using CGEventTap (native macOS)"""
        def trigger_toggle():
            threading.Timer(0.05, self._safe_toggle).start()

        def handle_error(error):
            logger.error(f"Hotkey callback error: {error}")

        try:
            self.event_tap = register_option_space_hotkey(
                on_trigger=trigger_toggle,
                on_error=handle_error,
                logger=logger,
            )
        except Exception as error:
            logger.error(str(error))
    
    def _safe_toggle(self):
        """Toggle recording safely"""
        self.toggle_recording(None)
    
    def load_model_menu(self, _):
        """Reload model from menu"""
        if not self.loading:
            self.loading = True
            threading.Thread(target=self.load_model, daemon=True).start()
    
    def update_status(self, status):
        """Update status (for internal use)"""
        logger.debug(f"Status update: {status}")
        pass
    
    def toggle_recording(self, _):
        """Start or stop recording"""
        logger.info(f"toggle_recording called. loading={self.loading}, is_recording={self.is_recording}")
        if self.loading:
            logger.warning("Model still loading, cannot record")
            rumps.notification(APP_NAME, "Please wait", "Model is still loading...")
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
        self.icon = ICON_RECORDING
        self.title = None
        self.start_stop_item.title = "⏹ Stop Recording"
        self.upload_item.set_callback(None)  # Disable transcribe file

        try:
            logger.info(f"Starting audio capture with sample rate {SAMPLE_RATE}")
            self.audio_capture.start()
            logger.info("Audio stream started successfully")
        except Exception as e:
            logger.error(f"Mic error: {e}")
            self.icon = ICON_ERROR
            self.title = None
            self.update_status(f"Mic error: {str(e)[:20]}")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop recording and transcribe"""
        logger.info(f"stop_recording called, is_recording={self.is_recording}")
        if not self.is_recording:
            return
        
        # Calculate recording duration
        recording_duration = time.time() - getattr(self, 'recording_start_time', time.time())
        logger.info(f"Recording duration: {recording_duration:.2f} seconds")
        
        self.is_recording = False
        self.icon = ICON_PROCESSING
        self.title = None  # Icon only during processing
        self.start_stop_item.title = "⏳ Processing..."
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
            self.run_on_main_thread(lambda: setattr(self, 'icon', ICON_PATH))
            self.run_on_main_thread(lambda: setattr(self, 'title', None))
            self.update_status("No audio recorded")
            self._reset_menu_state()
            return
        
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
                self.run_on_main_thread(lambda: setattr(self, 'icon', ICON_PATH))
                self.run_on_main_thread(lambda: setattr(self, 'title', None))
                self._reset_menu_state()
                return

            # Save audio file permanently
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"recording_{timestamp}.wav"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            wav.write(audio_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
            logger.info(f"Audio saved to {audio_path}")
            
            # Transcribe with better parameters
            logger.info("Starting transcription...")
            result = self.model.transcribe(
                audio_path,
                fp16=False,
                language=None,  # Auto-detect language
                condition_on_previous_text=False,  # Don't use previous context (prevents hallucinations)
                no_speech_threshold=0.6,  # Higher threshold to filter out no-speech segments
            )
            
            text = result["text"].strip()
            logger.info(f"Transcription result: {text[:100] if text else 'empty'}")
            
            # Filter out common hallucinations that occur with silence/noise
            is_hallucination = is_likely_hallucination(text)
            
            if text and not is_hallucination:
                # Copy to clipboard
                pyperclip.copy(text)
                
                # Small delay then type the text
                time.sleep(0.15)
                self.type_text(text)
                
                # Transcription complete - text is pasted, no notification needed
                logger.info(f"Transcribed and pasted: {text[:50]}...")
                
                # Save to history with audio path
                self.add_to_history(text, "live", audio_path=audio_path)
            else:
                if is_hallucination:
                    logger.info(f"Filtered hallucination: '{text}'")
                else:
                    logger.info("No speech detected")
                # Delete audio if no speech detected or hallucination
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            
            # Re-enable menu items
            self._reset_menu_state()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            self.run_on_main_thread(lambda: setattr(self, 'icon', ICON_ERROR))
            self.run_on_main_thread(lambda: setattr(self, 'title', None))
            self._reset_menu_state()
            self.update_status(f"Error: {str(e)[:30]}")
    
    def _reset_menu_state(self):
        """Reset menu items to normal state - must be called on main thread"""
        def do_reset():
            self.title = None
            self.icon = ICON_PATH
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
    
    def upload_audio_file(self, _):
        """Open file dialog to select audio file for transcription"""
        if self.loading:
            return
        
        if self.is_recording:
            return
        
        # Use AppleScript to open native file picker (without activate to avoid app focus issues)
        script = '''
        set theFile to choose file with prompt "Select an audio file to transcribe:" of type {"public.audio", "mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac"}
        return POSIX path of theFile
        '''
        
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0 and result.stdout.strip():
                file_path = result.stdout.strip()
                self.icon = ICON_PROCESSING
                self.title = None  # Icon only during processing
                self.start_stop_item.title = "⏳ Processing..."
                self.start_stop_item.set_callback(None)
                self.upload_item.set_callback(None)
                self.update_status(f"Transcribing file...")
                logger.info(f"Processing file: {file_path}")
                threading.Thread(target=self.transcribe_file, args=(file_path,), daemon=True).start()
        except subprocess.TimeoutExpired:
            pass  # User cancelled
        except Exception as e:
            self.update_status(f"File error: {str(e)[:20]}")
    
    def transcribe_file(self, file_path):
        """Transcribe an audio file - runs in background thread"""
        try:
            # Copy audio file to our storage directory
            import shutil
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_ext = os.path.splitext(file_path)[1] or '.wav'
            audio_filename = f"file_{timestamp}{original_ext}"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            shutil.copy2(file_path, audio_path)
            
            # Transcribe directly with Whisper
            result = self.model.transcribe(
                file_path,
                fp16=False,
                language=None  # Auto-detect language
            )
            
            text = result["text"].strip()
            
            if text:
                # Copy to clipboard
                pyperclip.copy(text)
                
                # Show result length info
                word_count = len(text.split())
                
                # Save to history with audio path
                self.add_to_history(text, "file", os.path.basename(file_path), audio_path=audio_path)
                
                # Update UI on main thread
                self._reset_menu_state()
                self.update_status(f"✓ {word_count} words transcribed")
                logger.info(f"File transcribed: {word_count} words")
            else:
                # Delete copied audio if no speech detected
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                self._reset_menu_state()
                logger.info("No speech detected in file")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"File transcription error: {error_msg}")
            self.run_on_main_thread(lambda: setattr(self, 'icon', ICON_ERROR))
            self.run_on_main_thread(lambda: setattr(self, 'title', None))
            self._reset_menu_state()
    
    def open_history_window(self, _, selected_index=0):
        """Open the SuperWhisper-style history window"""
        try:
            show_history_window_direct()
        except Exception as e:
            logger.error(f"Could not open history: {e}")
    
    def show_history_item(self, index):
        """Show a history item - opens the history window"""
        self.open_history_window(None, index)
    
    def copy_history_item(self, text):
        """Copy a history item to clipboard"""
        pyperclip.copy(text)
        logger.info("Copied to clipboard")
    
    def clear_history(self, _):
        """Clear all history"""
        response = rumps.alert(
            title="Clear History",
            message="Are you sure you want to clear all transcription history?",
            ok="Clear",
            cancel="Cancel"
        )
        if response == 1:  # OK clicked
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
    MurmurApp().run()
