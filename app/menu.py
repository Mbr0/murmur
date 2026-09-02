"""The menu bar: what it shows, what it says, and the windows it opens.

AppKit at import: ``rumps`` for the menu items and ``NSImage`` for the status
icon, which is set on the ``NSStatusItem`` directly because rumps' own icon swap
drops the template flag.
"""

import os

import rumps
import sounddevice as sd
from Cocoa import NSImage

from cleanup.modes import MODE_IDS, MODES, TONE_IDS, TONES
from services.license_service import get_current_entitlements
from services.persistence_service import DEFAULT_CONFIG
from ui import alerts as ui_alerts
from ui.settings.base import TAB_ACCOUNT

from app.config import APP_NAME, ICON_PATH, STATE_ICON_PATHS, app_model_store, logger
from app.decisions import (
    CLEANUP_DOWNLOAD_MENU_TITLE,
    MODE_MENU_AUTOMATIC,
    account_menu_title,
    cleanup_download_menu_enabled,
    clear_mic_device_selection,
    mode_menu_state,
    resolve_mic_device,
    resolve_mic_device_index,
    should_apply_ready_on_reset,
    tone_menu_state,
)
from app.windows import show_history_window_direct, show_settings_window_direct

_MENU_BAR_IMAGES = {}



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


class MenuMixin:
    """The menu bar half of the app: items, checkmarks, titles and windows.

    Split out of ``MurmurApp`` in Wave 5. Every method still writes to the
    ``rumps.MenuItem`` objects ``__init__`` built; what changed is the file.
    """

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

    def open_settings_window_safely(self, tab=None):
        """Open Settings from any code path without letting AppKit errors escape.

        ``tab`` names one of ``ui.settings.base.TAB_ORDER``; ``None`` reopens
        wherever the user left the window.
        """
        try:
            show_settings_window_direct(tab)
        except Exception:
            logger.error("Could not open settings", exc_info=True)
            ui_alerts.show_alert(APP_NAME, "Could not open Settings.")

    def _refresh_account_menu(self, entitlements=None):
        """Redraw the menu's account line. Safe to call from any thread."""
        title = account_menu_title(
            entitlements if entitlements is not None else get_current_entitlements(),
            store_is_volatile=bool(getattr(self, "secret_store_is_volatile", False)),
        )

        def apply():
            item = getattr(self, "account_item", None)
            if item is not None:
                item.title = title

        self.run_on_main_thread(apply)

    def open_account_settings(self, _=None):
        """Menu → "Sign in with Boske ID…": Settings, on the Account tab."""
        self.open_settings_window_safely(TAB_ACCOUNT)

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

    def update_history_menu(self):
        """Keep menu state in sync after history mutations."""
        # History currently lives in the dedicated history window.
        # We still expose this method because clear_history() invokes it.
        self.history_item.title = "History"
