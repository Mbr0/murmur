#!/usr/bin/env python3
"""
Murmur Settings Window - Configuration panel
"""

import sys
import os
import subprocess
from services.model_profile_service import default_model_for_current_machine
from services.hotkey_service import (
    DEFAULT_HOTKEY,
    HotkeyBinding,
    MODIFIER_KEYCODES,
    binding_from_ns_flags,
    binding_has_modifier,
    capture_label_for_binding,
    format_hotkey,
    hotkey_from_config,
    hotkey_permissions_ok,
    hotkey_to_config,
    open_privacy_settings,
    permission_status_message,
    ns_modifier_flags,
)
from services.persistence_service import DEFAULT_CONFIG, PersistencePaths, PersistenceService

# PyObjC imports
import objc
from Cocoa import (
    NSApplication, NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSBackingStoreBuffered, NSMakeRect, NSFont, NSColor,
    NSButton, NSRoundedBezelStyle, NSTextField, NSObject, NSApp, NSView,
    NSApplicationActivationPolicyAccessory, NSPopUpButton, NSBox, NSBoxSeparator,
    NSAlert, NSInformationalAlertStyle, NSWarningAlertStyle, NSOnState, NSOffState,
    NSAlertFirstButtonReturn, NSAppearance, NSScrollView,
    NSEvent, NSEventMaskKeyDown, NSEventMaskFlagsChanged, NSEventTypeFlagsChanged,
    NSEventModifierFlagCommand, NSEventModifierFlagOption,
    NSEventModifierFlagControl, NSEventModifierFlagShift, NSEventModifierFlagFunction,
)
from PyObjCTools import AppHelper
import ui_alerts
import ui_theme

APP_NAME = "Murmur"

# Config file
CONFIG_FILE = os.path.expanduser("~/.murmur_config.json")
HISTORY_FILE = os.path.expanduser("~/.murmur_history.json")
AUDIO_DIR = os.path.expanduser("~/.murmur_audio")
LEGACY_CONFIG_FILE = os.path.expanduser("~/.mywhisper_config.json")
PERSISTENCE = PersistenceService(
    PersistencePaths(config_file=CONFIG_FILE, history_file=HISTORY_FILE),
    logger=print,
)

# Model info
MODELS = {
    "tiny": {"size": "~39 MB", "vram": "~1 GB", "speed": "~32x", "quality": "Basic", "recommended_for": "Testing only"},
    "base": {"size": "~74 MB", "vram": "~1 GB", "speed": "~16x", "quality": "Good", "recommended_for": "Quick transcriptions"},
    "small": {"size": "~244 MB", "vram": "~2 GB", "speed": "~6x", "quality": "Better", "recommended_for": "Good balance"},
    "medium": {"size": "~769 MB", "vram": "~5 GB", "speed": "~2x", "quality": "Great", "recommended_for": "Most users"},
    "large": {"size": "~1550 MB", "vram": "~10 GB", "speed": "~1x", "quality": "Best", "recommended_for": "Maximum accuracy"},
}

SETTINGS_WIDTH = 520
SETTINGS_HEIGHT = 660

APPEARANCE_LABELS = {
    "system": "System",
    "dark": "Dark",
    "light": "Light",
}


def _murmur_app_instance():
    """Return the menu bar app when Settings is opened from Murmur (not standalone)."""
    for module_name in ("murmur", "__main__"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        app = getattr(module, "APP_INSTANCE", None)
        if app is not None:
            return app
    return None

def get_system_info():
    """Get system information"""
    info = {
        "chip": "Unknown",
        "ram": 0,
        "recommended_model": "medium"
    }
    
    try:
        # Get chip info
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                               capture_output=True, text=True)
        cpu = result.stdout.strip()
        
        # Check for Apple Silicon
        result2 = subprocess.run(["uname", "-m"], capture_output=True, text=True)
        arch = result2.stdout.strip()
        
        if arch == "arm64":
            # Get Apple Silicon chip name
            result3 = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True)
            model = result3.stdout.strip()
            
            # Map to chip names
            if "Mac14" in model or "Mac15" in model or "Mac16" in model:
                info["chip"] = "Apple Silicon (M2/M3/M4)"
            else:
                info["chip"] = "Apple Silicon (M1)"
        else:
            info["chip"] = cpu[:40] if cpu else "Intel"
        
        # Get RAM
        result4 = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        ram_bytes = int(result4.stdout.strip())
        info["ram"] = ram_bytes // (1024 ** 3)  # Convert to GB
        
        # Recommend model based on RAM
        if info["ram"] >= 32:
            info["recommended_model"] = "large"
        elif info["ram"] >= 16:
            info["recommended_model"] = "medium"
        elif info["ram"] >= 8:
            info["recommended_model"] = "small"
        else:
            info["recommended_model"] = "base"
            
    except Exception as e:
        print(f"Error getting system info: {e}")
    
    return info


def load_config():
    """Load configuration"""
    default = {"model": default_model_for_current_machine(), **DEFAULT_CONFIG}
    if not os.path.exists(CONFIG_FILE) and os.path.exists(LEGACY_CONFIG_FILE):
        return PERSISTENCE.load_config(default)
    return PERSISTENCE.load_config(default)


def save_config(config):
    """Save configuration"""
    PERSISTENCE.save_config(config)


class SettingsWindowController(NSObject):
    """Controller for the settings window"""
    
    def init(self):
        self = objc.super(SettingsWindowController, self).init()
        if self is None:
            return None
        
        self.config = load_config()
        self.system_info = get_system_info()
        self.hotkey_binding = hotkey_from_config(self.config)
        self.hotkey_label = self.config.get("hotkey_label")
        self._hotkey_monitor = None
        self._capture_modifiers = 0
        self.model_popup = None
        self.save_audio_switch = None
        self.save_history_switch = None
        self.appearance_popup = None
        self.needs_restart = False
        
        return self
    
    def createWindow(self):
        """Create and show the settings window"""
        window_rect = NSMakeRect(0, 0, SETTINGS_WIDTH, SETTINGS_HEIGHT)
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            window_rect, style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Murmur Settings")
        self.window.center()
        self.window.setReleasedWhenClosed_(False)
        self.window.setContentMinSize_((SETTINGS_WIDTH, SETTINGS_HEIGHT))

        ui_theme.set_appearance_mode(self.config.get("appearance_mode", "system"))
        ui_theme.apply_window_theme(self.window)

        content_view = self.window.contentView()
        width = content_view.frame().size.width
        height = content_view.frame().size.height
        scroll_height = height - ui_theme.FOOTER_HEIGHT

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, ui_theme.FOOTER_HEIGHT, width, scroll_height))
        scroll.setHasVerticalScroller_(True)
        scroll.setHasHorizontalScroller_(False)
        scroll.setBorderType_(0)
        scroll.setDrawsBackground_(False)
        scroll.setAutohidesScrollers_(False)

        document = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, scroll_height))
        document.setFlipped_(True)
        content_height = self._add_settings_sections(document, width, start_y=16)
        document.setFrame_(NSMakeRect(0, 0, width, max(content_height, scroll_height)))
        scroll.setDocumentView_(document)
        content_view.addSubview_(scroll)

        footer = ui_theme.add_footer_bar(content_view, width, ui_theme.FOOTER_HEIGHT)

        save_btn = NSButton.alloc().initWithFrame_(NSMakeRect(width - ui_theme.MARGIN - 84, 14, 84, 28))
        save_btn.setTitle_("Save")
        ui_theme.style_primary_button(save_btn)
        save_btn.setTarget_(self)
        save_btn.setAction_(objc.selector(self.saveClicked_, signature=b'v@:@'))
        footer.addSubview_(save_btn)

        cancel_btn = NSButton.alloc().initWithFrame_(NSMakeRect(width - ui_theme.MARGIN - 178, 14, 84, 28))
        cancel_btn.setTitle_("Cancel")
        ui_theme.style_dark_button(cancel_btn)
        cancel_btn.setTarget_(self)
        cancel_btn.setAction_(objc.selector(self.cancelClicked_, signature=b'v@:@'))
        footer.addSubview_(cancel_btn)

        hint = self._create_label(
            "Changes apply when you click Save.",
            x=ui_theme.MARGIN,
            y=18,
            width=width - 220,
            size=11,
            color=ui_theme.muted_text_color(),
        )
        footer.addSubview_(hint)

        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    @objc.python_method
    def _add_settings_sections(self, content_view, width, start_y=16):
        """Lay out settings sections top-to-bottom in a flipped scroll document."""
        current_model = self.config.get("model", "medium")
        rec_model = self.system_info["recommended_model"]
        content_width = width - ui_theme.MARGIN * 2
        y = start_y

        def add(label):
            nonlocal y
            content_view.addSubview_(label)
            y += label.frame().size.height + 6

        def rule():
            nonlocal y
            ui_theme.add_horizontal_rule(content_view, ui_theme.MARGIN, y, content_width)
            y += 16

        add(self._create_label(
            "System Information", x=ui_theme.MARGIN, y=y, width=content_width, bold=True, size=13
        ))
        add(self._create_label(
            f"Chip: {self.system_info['chip']}", x=ui_theme.MARGIN, y=y, width=content_width
        ))
        add(self._create_label(
            f"Memory: {self.system_info['ram']} GB RAM", x=ui_theme.MARGIN, y=y, width=content_width
        ))
        add(self._create_label(
            f"Recommended: {rec_model.capitalize()}",
            x=ui_theme.MARGIN,
            y=y,
            width=content_width,
            color=ui_theme.brand_accent_color(),
        ))
        rule()

        add(self._create_label(
            "Whisper Model", x=ui_theme.MARGIN, y=y, width=content_width, bold=True, size=13
        ))
        self.model_popup = NSPopUpButton.alloc().initWithFrame_(NSMakeRect(ui_theme.MARGIN, y, 160, 26))
        for model_name in MODELS:
            display = model_name.capitalize()
            if model_name == rec_model:
                display += " ✓"
            self.model_popup.addItemWithTitle_(display)
        for i, model_name in enumerate(MODELS.keys()):
            if model_name == current_model:
                self.model_popup.selectItemAtIndex_(i)
                break
        self.model_popup.setTarget_(self)
        self.model_popup.setAction_(objc.selector(self.modelChanged_, signature=b'v@:@'))
        ui_theme.style_popup_on_dark(self.model_popup)
        content_view.addSubview_(self.model_popup)
        y += 34

        self.model_info_label = self._create_label(
            "", x=ui_theme.MARGIN, y=y, width=content_width, size=11, color=ui_theme.muted_text_color()
        )
        add(self.model_info_label)
        self._update_model_info()
        rule()

        add(self._create_label(
            "Privacy & Local Data", x=ui_theme.MARGIN, y=y, width=content_width, bold=True, size=13
        ))
        self.save_audio_switch = self._create_switch(
            "Save audio recordings on this Mac",
            x=ui_theme.MARGIN,
            y=y,
            width=content_width,
            enabled=self.config.get("save_audio", DEFAULT_CONFIG["save_audio"]),
        )
        content_view.addSubview_(self.save_audio_switch)
        y += 30
        self.save_history_switch = self._create_switch(
            "Save transcription history on this Mac",
            x=ui_theme.MARGIN,
            y=y,
            width=content_width,
            enabled=self.config.get("save_history", DEFAULT_CONFIG["save_history"]),
        )
        content_view.addSubview_(self.save_history_switch)
        y += 34

        add(self._create_label(
            "Off by default for privacy. When enabled, data is stored locally in "
            "~/.murmur_history.json and ~/.murmur_audio/. Settings stay in ~/.murmur_config.json.",
            x=ui_theme.MARGIN,
            y=y,
            width=content_width,
            size=11,
            color=ui_theme.muted_text_color(),
            height=40,
        ))

        delete_btn = NSButton.alloc().initWithFrame_(NSMakeRect(ui_theme.MARGIN, y, 190, 28))
        delete_btn.setTitle_("Delete All Local Data")
        ui_theme.style_dark_button(delete_btn)
        delete_btn.setTarget_(self)
        delete_btn.setAction_(objc.selector(self.deleteLocalDataClicked_, signature=b'v@:@'))
        content_view.addSubview_(delete_btn)
        y += 38
        rule()

        add(self._create_label(
            "Appearance", x=ui_theme.MARGIN, y=y, width=content_width, bold=True, size=13
        ))
        self.appearance_popup = NSPopUpButton.alloc().initWithFrame_(NSMakeRect(ui_theme.MARGIN, y, 180, 26))
        for mode in ui_theme.APPEARANCE_MODES:
            self.appearance_popup.addItemWithTitle_(APPEARANCE_LABELS[mode])
        current_appearance = self.config.get("appearance_mode", "system")
        if current_appearance not in ui_theme.APPEARANCE_MODES:
            current_appearance = "system"
        self.appearance_popup.selectItemAtIndex_(ui_theme.APPEARANCE_MODES.index(current_appearance))
        ui_theme.style_popup_on_dark(self.appearance_popup)
        content_view.addSubview_(self.appearance_popup)
        y += 34

        add(self._create_label(
            "Keyboard shortcut", x=ui_theme.MARGIN, y=y, width=content_width, bold=True, size=13
        ))
        self.hotkey_button = NSButton.alloc().initWithFrame_(NSMakeRect(ui_theme.MARGIN, y, 220, 28))
        self.hotkey_button.setTitle_(format_hotkey(self.hotkey_binding, label=self.hotkey_label))
        ui_theme.style_dark_button(self.hotkey_button)
        self.hotkey_button.setTarget_(self)
        self.hotkey_button.setAction_(objc.selector(self.recordHotkeyClicked_, signature=b'v@:@'))
        content_view.addSubview_(self.hotkey_button)

        reset_hotkey_btn = NSButton.alloc().initWithFrame_(NSMakeRect(ui_theme.MARGIN + 232, y, 140, 28))
        reset_hotkey_btn.setTitle_("Default (⌥ Space)")
        ui_theme.style_dark_button(reset_hotkey_btn)
        reset_hotkey_btn.setTarget_(self)
        reset_hotkey_btn.setAction_(objc.selector(self.resetHotkeyClicked_, signature=b'v@:@'))
        content_view.addSubview_(reset_hotkey_btn)
        y += 34

        permissions_btn = NSButton.alloc().initWithFrame_(NSMakeRect(ui_theme.MARGIN, y, 190, 28))
        permissions_btn.setTitle_("Open Privacy Settings")
        ui_theme.style_dark_button(permissions_btn)
        permissions_btn.setTarget_(self)
        permissions_btn.setAction_(objc.selector(self.openShortcutPermissions_, signature=b'v@:@'))
        content_view.addSubview_(permissions_btn)
        y += 34

        add(self._create_label(
            permission_status_message(),
            x=ui_theme.MARGIN,
            y=y,
            width=content_width,
            size=11,
            color=ui_theme.muted_text_color(),
            height=48,
        ))
        return y + 16
    
    def _create_label(self, text, x, y, width, bold=False, size=12, color=None, height=20):
        """Create a text label"""
        label = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, width, height))
        label.setStringValue_(text)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        if bold:
            label.setFont_(NSFont.systemFontOfSize_weight_(size, 0.3))
        else:
            label.setFont_(NSFont.systemFontOfSize_(size))
        label.setTextColor_(color if color else ui_theme.primary_text_color())
        return label

    @objc.python_method
    def _create_switch(self, title, x, y, width, enabled):
        """Create a macOS switch control."""
        switch = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, width, 24))
        switch.setButtonType_(3)
        switch.setTitle_(title)
        switch.setState_(NSOnState if enabled else NSOffState)
        switch.setAppearance_(ui_theme.control_appearance())
        return switch
    
    @objc.python_method
    def _update_model_info(self):
        """Update model info display"""
        idx = self.model_popup.indexOfSelectedItem()
        model_name = list(MODELS.keys())[idx]
        info = MODELS[model_name]
        
        text = f"Size: {info['size']} | VRAM: {info['vram']} | {info['recommended_for']}"
        self.model_info_label.setStringValue_(text)
    
    def modelChanged_(self, sender):
        """Handle model selection change"""
        self._update_model_info()
        self.needs_restart = True

    @objc.python_method
    def _stop_hotkey_capture(self):
        if self._hotkey_monitor is not None:
            NSEvent.removeMonitor_(self._hotkey_monitor)
            self._hotkey_monitor = None
        self._capture_modifiers = 0
        self.hotkey_button.setTitle_(
            format_hotkey(self.hotkey_binding, label=self.hotkey_label)
        )

    def recordHotkeyClicked_(self, sender):
        """Capture a new global shortcut from the next key press."""
        self._stop_hotkey_capture()
        self._capture_modifiers = 0
        self.hotkey_button.setTitle_("Press shortcut…")
        mask = NSEventMaskKeyDown | NSEventMaskFlagsChanged
        self._hotkey_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            mask,
            self._captureHotkeyEvent_,
        )

    def resetHotkeyClicked_(self, sender):
        """Restore the default Option+Space shortcut."""
        self._stop_hotkey_capture()
        self.hotkey_binding = DEFAULT_HOTKEY
        self.hotkey_label = "Space"
        self.hotkey_button.setTitle_(format_hotkey(DEFAULT_HOTKEY, label="Space"))

    def openShortcutPermissions_(self, sender):
        """Open macOS privacy settings for the global shortcut."""
        open_privacy_settings()
        ui_alerts.show_alert(APP_NAME, permission_status_message())
        app = _murmur_app_instance()
        if app is not None:
            app.reload_hotkey(prompt=True)

    @objc.python_method
    def _captureHotkeyEvent_(self, event):
        if event.type() == NSEventTypeFlagsChanged:
            self._capture_modifiers = ns_modifier_flags(event.modifierFlags())
            return event

        keycode = event.keyCode()
        if keycode in MODIFIER_KEYCODES:
            return event

        combined_flags = ns_modifier_flags(event.modifierFlags() | self._capture_modifiers)
        has_modifier = bool(
            combined_flags
            & (
                NSEventModifierFlagCommand
                | NSEventModifierFlagOption
                | NSEventModifierFlagControl
                | NSEventModifierFlagShift
                | NSEventModifierFlagFunction
            )
        )
        if not has_modifier:
            return event

        binding = binding_from_ns_flags(keycode, combined_flags)
        if not binding_has_modifier(binding):
            return event

        characters = event.charactersIgnoringModifiers() or ""
        self.hotkey_binding = binding
        self.hotkey_label = capture_label_for_binding(binding, characters=characters)
        self._stop_hotkey_capture()
        return None
    
    def saveClicked_(self, sender):
        """Save settings"""
        idx = self.model_popup.indexOfSelectedItem()
        model_name = list(MODELS.keys())[idx]
        
        self.config["model"] = model_name
        save_audio = self.save_audio_switch.state() == NSOnState
        save_history = self.save_history_switch.state() == NSOnState
        self.config["save_audio"] = save_audio
        self.config["save_history"] = save_history
        self.config["privacy_mode"] = not (save_audio or save_history)
        appearance_idx = self.appearance_popup.indexOfSelectedItem()
        self.config["appearance_mode"] = ui_theme.APPEARANCE_MODES[appearance_idx]
        self.config.update(hotkey_to_config(self.hotkey_binding, label=self.hotkey_label))
        save_config(self.config)
        ui_theme.set_appearance_mode(self.config["appearance_mode"])

        app = _murmur_app_instance()
        if app is not None:
            app.reload_hotkey(prompt=False)
        
        if self.needs_restart:
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Settings Saved")
            alert.setInformativeText_(
                f"Model changed to '{model_name.capitalize()}'. Please restart Murmur for the changes to take effect."
            )
            alert.setAlertStyle_(NSInformationalAlertStyle)
            ui_alerts.configure_alert(alert)
            alert.runModal()
        
        self.window.close()
        if __name__ == "__main__":
            NSApp.terminate_(None)

    def deleteLocalDataClicked_(self, sender):
        """Delete all locally stored Murmur history and audio."""
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Delete All Local Data?")
        alert.setInformativeText_(
            "This permanently removes transcription history, saved audio, and local debug logs "
            "from this Mac. Your settings will be kept."
        )
        alert.setAlertStyle_(NSWarningAlertStyle)
        alert.addButtonWithTitle_("Delete")
        alert.addButtonWithTitle_("Cancel")
        ui_alerts.configure_alert(alert)
        if alert.runModal() != NSAlertFirstButtonReturn:
            return

        PERSISTENCE.clear_all_local_data(AUDIO_DIR)
        app = _murmur_app_instance()
        if app is not None:
            app.history = []

        done = NSAlert.alloc().init()
        done.setMessageText_("Local Data Deleted")
        done.setInformativeText_(
            "Transcription history, saved audio, and debug logs were removed from this Mac."
        )
        done.setAlertStyle_(NSInformationalAlertStyle)
        ui_alerts.configure_alert(done)
        done.runModal()
    
    def cancelClicked_(self, sender):
        """Cancel and close"""
        self._stop_hotkey_capture()
        self.window.close()
        if __name__ == "__main__":
            NSApp.terminate_(None)


# Global reference
_controller = None

def main():
    global _controller
    
    # Set up as accessory app (no dock icon)
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    
    _controller = SettingsWindowController.alloc().init()
    _controller.createWindow()
    
    # Activate to bring window to front
    app.activateIgnoringOtherApps_(True)
    
    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
