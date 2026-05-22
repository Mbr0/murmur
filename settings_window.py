#!/usr/bin/env python3
"""
Murmur Settings Window - Configuration panel
"""

import sys
import json
import os
import platform
import subprocess
from services.model_profile_service import default_model_for_current_machine

# PyObjC imports
import objc
from Cocoa import (
    NSApplication, NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSBackingStoreBuffered, NSMakeRect, NSFont, NSColor, NSFloatingWindowLevel,
    NSButton, NSRoundedBezelStyle, NSTextField, NSView, NSObject, NSApp,
    NSApplicationActivationPolicyAccessory, NSPopUpButton, NSBox, NSBoxSeparator
)
from Quartz import CGColorCreateGenericRGB
from PyObjCTools import AppHelper

# Config file
CONFIG_FILE = os.path.expanduser("~/.murmur_config.json")
LEGACY_CONFIG_FILE = os.path.expanduser("~/.mywhisper_config.json")

# Model info
MODELS = {
    "tiny": {"size": "~39 MB", "vram": "~1 GB", "speed": "~32x", "quality": "Basic", "recommended_for": "Testing only"},
    "base": {"size": "~74 MB", "vram": "~1 GB", "speed": "~16x", "quality": "Good", "recommended_for": "Quick transcriptions"},
    "small": {"size": "~244 MB", "vram": "~2 GB", "speed": "~6x", "quality": "Better", "recommended_for": "Good balance"},
    "medium": {"size": "~769 MB", "vram": "~5 GB", "speed": "~2x", "quality": "Great", "recommended_for": "Most users"},
    "large": {"size": "~1550 MB", "vram": "~10 GB", "speed": "~1x", "quality": "Best", "recommended_for": "Maximum accuracy"},
}


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
    default = {"model": default_model_for_current_machine()}
    try:
        if not os.path.exists(CONFIG_FILE) and os.path.exists(LEGACY_CONFIG_FILE):
            with open(LEGACY_CONFIG_FILE, 'r') as f:
                return {**default, **json.load(f)}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return {**default, **json.load(f)}
    except:
        pass
    return default


def save_config(config):
    """Save configuration"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")


class SettingsWindowController(NSObject):
    """Controller for the settings window"""
    
    def init(self):
        self = objc.super(SettingsWindowController, self).init()
        if self is None:
            return None
        
        self.config = load_config()
        self.system_info = get_system_info()
        self.model_popup = None
        self.needs_restart = False
        
        return self
    
    def createWindow(self):
        """Create and show the settings window"""
        # Create window
        window_rect = NSMakeRect(0, 0, 500, 480)
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            window_rect, style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Murmur Settings")
        self.window.center()
        self.window.setReleasedWhenClosed_(False)
        
        # Dark background
        dark_bg = NSColor.colorWithRed_green_blue_alpha_(0.11, 0.11, 0.12, 1.0)
        self.window.setBackgroundColor_(dark_bg)
        
        content_view = self.window.contentView()
        width = content_view.frame().size.width
        height = content_view.frame().size.height
        
        y_pos = height - 40
        
        # === SYSTEM INFO SECTION ===
        section_label = self._create_label("System Information", x=20, y=y_pos, width=200, bold=True, size=14)
        content_view.addSubview_(section_label)
        y_pos -= 30
        
        # Chip
        chip_label = self._create_label(f"Chip: {self.system_info['chip']}", x=20, y=y_pos, width=width-40)
        content_view.addSubview_(chip_label)
        y_pos -= 22
        
        # RAM
        ram_label = self._create_label(f"Memory: {self.system_info['ram']} GB RAM", x=20, y=y_pos, width=width-40)
        content_view.addSubview_(ram_label)
        y_pos -= 22
        
        # Recommended
        rec_model = self.system_info['recommended_model']
        rec_label = self._create_label(f"Recommended Model: {rec_model.capitalize()}", x=20, y=y_pos, width=width-40, color=NSColor.systemGreenColor())
        content_view.addSubview_(rec_label)
        y_pos -= 40
        
        # Separator
        sep1 = NSBox.alloc().initWithFrame_(NSMakeRect(20, y_pos, width-40, 1))
        sep1.setBoxType_(NSBoxSeparator)
        content_view.addSubview_(sep1)
        y_pos -= 30
        
        # === MODEL SELECTION ===
        model_section = self._create_label("Whisper Model", x=20, y=y_pos, width=200, bold=True, size=14)
        content_view.addSubview_(model_section)
        y_pos -= 35
        
        # Model dropdown
        model_label = self._create_label("Current Model:", x=20, y=y_pos+3, width=100)
        content_view.addSubview_(model_label)
        
        self.model_popup = NSPopUpButton.alloc().initWithFrame_(NSMakeRect(130, y_pos, 150, 26))
        for model_name in MODELS.keys():
            display = model_name.capitalize()
            if model_name == rec_model:
                display += " ✓"
            self.model_popup.addItemWithTitle_(display)
        
        # Set current selection
        current_model = self.config.get("model", "medium")
        for i, model_name in enumerate(MODELS.keys()):
            if model_name == current_model:
                self.model_popup.selectItemAtIndex_(i)
                break
        
        self.model_popup.setTarget_(self)
        self.model_popup.setAction_(objc.selector(self.modelChanged_, signature=b'v@:@'))
        content_view.addSubview_(self.model_popup)
        y_pos -= 35
        
        # Model info display
        self.model_info_label = self._create_label("", x=20, y=y_pos, width=width-40, size=11)
        content_view.addSubview_(self.model_info_label)
        self._update_model_info()
        y_pos -= 50
        
        # Separator
        sep2 = NSBox.alloc().initWithFrame_(NSMakeRect(20, y_pos, width-40, 1))
        sep2.setBoxType_(NSBoxSeparator)
        content_view.addSubview_(sep2)
        y_pos -= 30
        
        # === MODEL COMPARISON ===
        comp_section = self._create_label("Model Comparison", x=20, y=y_pos, width=200, bold=True, size=14)
        content_view.addSubview_(comp_section)
        y_pos -= 25
        
        # Header
        headers = ["Model", "Size", "Speed", "Quality"]
        header_widths = [70, 90, 70, 80]
        x_offset = 20
        for i, header in enumerate(headers):
            h_label = self._create_label(header, x=x_offset, y=y_pos, width=header_widths[i], bold=True, size=11)
            content_view.addSubview_(h_label)
            x_offset += header_widths[i]
        y_pos -= 20
        
        # Model rows
        for model_name, info in MODELS.items():
            x_offset = 20
            is_current = model_name == current_model
            is_recommended = model_name == rec_model
            
            # Highlight current/recommended
            text_color = NSColor.whiteColor()
            if is_current:
                text_color = NSColor.systemBlueColor()
            elif is_recommended:
                text_color = NSColor.systemGreenColor()
            
            name_display = model_name.capitalize()
            if is_current:
                name_display += " ●"
            elif is_recommended:
                name_display += " ✓"
            
            row_data = [name_display, info["size"], info["speed"], info["quality"]]
            for i, text in enumerate(row_data):
                cell = self._create_label(text, x=x_offset, y=y_pos, width=header_widths[i], size=11, color=text_color)
                content_view.addSubview_(cell)
                x_offset += header_widths[i]
            y_pos -= 18
        
        y_pos -= 20
        
        # Legend
        legend = self._create_label("● = Current   ✓ = Recommended for your system", x=20, y=y_pos, width=width-40, size=10, color=NSColor.secondaryLabelColor())
        content_view.addSubview_(legend)
        y_pos -= 30
        
        # === SHORTCUT INFO ===
        shortcut_label = self._create_label("Keyboard Shortcut: ⌥ Space (Option + Space)", x=20, y=y_pos, width=width-40)
        content_view.addSubview_(shortcut_label)
        y_pos -= 40
        
        # === BUTTONS ===
        # Save button
        save_btn = NSButton.alloc().initWithFrame_(NSMakeRect(width - 100, 15, 80, 32))
        save_btn.setTitle_("Save")
        save_btn.setBezelStyle_(NSRoundedBezelStyle)
        save_btn.setTarget_(self)
        save_btn.setAction_(objc.selector(self.saveClicked_, signature=b'v@:@'))
        content_view.addSubview_(save_btn)
        
        # Cancel button
        cancel_btn = NSButton.alloc().initWithFrame_(NSMakeRect(width - 190, 15, 80, 32))
        cancel_btn.setTitle_("Cancel")
        cancel_btn.setBezelStyle_(NSRoundedBezelStyle)
        cancel_btn.setTarget_(self)
        cancel_btn.setAction_(objc.selector(self.cancelClicked_, signature=b'v@:@'))
        content_view.addSubview_(cancel_btn)
        
        # Show window
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
    
    def _create_label(self, text, x, y, width, bold=False, size=12, color=None):
        """Create a text label"""
        label = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, width, 20))
        label.setStringValue_(text)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        if bold:
            label.setFont_(NSFont.boldSystemFontOfSize_(size))
        else:
            label.setFont_(NSFont.systemFontOfSize_(size))
        label.setTextColor_(color if color else NSColor.whiteColor())
        return label
    
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
    
    def saveClicked_(self, sender):
        """Save settings"""
        idx = self.model_popup.indexOfSelectedItem()
        model_name = list(MODELS.keys())[idx]
        
        self.config["model"] = model_name
        save_config(self.config)
        
        if self.needs_restart:
            # Show restart message
            from Cocoa import NSAlert, NSInformationalAlertStyle
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Settings Saved")
            alert.setInformativeText_(f"Model changed to '{model_name.capitalize()}'. Please restart Murmur for the changes to take effect.")
            alert.setAlertStyle_(NSInformationalAlertStyle)
            alert.runModal()
        
        self.window.close()
        # Only terminate if running standalone (not embedded in main app)
        if __name__ == "__main__" or (hasattr(sys, 'modules') and 'murmur' not in sys.modules):
            NSApp.terminate_(None)
    
    def cancelClicked_(self, sender):
        """Cancel and close"""
        self.window.close()
        # Only terminate if running standalone (not embedded in main app)
        if __name__ == "__main__" or (hasattr(sys, 'modules') and 'murmur' not in sys.modules):
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
