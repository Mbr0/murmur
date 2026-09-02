#!/usr/bin/env python3
"""
Murmur History Window - SuperWhisper-style history viewer
Launched as a separate process for better UI handling
"""

import sys
import json
import os
import pyperclip
import subprocess
import wave
from datetime import datetime

# PyObjC imports
import objc
from Cocoa import (
    NSApplication, NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSBackingStoreBuffered, NSMakeRect, NSMakeSize, NSMakePoint, NSScrollView, NSTextView, NSFont,
    NSColor, NSViewWidthSizable, NSViewHeightSizable, NSFloatingWindowLevel,
    NSButton, NSRoundedBezelStyle, NSTextField, NSView, NSTableView,
    NSTableColumn, NSObject, NSApp, NSApplicationActivationPolicyAccessory,
    NSTextFieldCell, NSRightTextAlignment,
    NSAlert, NSWarningAlertStyle, NSAlertFirstButtonReturn, NSFocusRingTypeNone,
    NSAppearance, NSAccessibilityButtonRole,
)
from Quartz import CGColorCreateGenericRGB
from PyObjCTools import AppHelper
from ui import alerts as ui_alerts
from ui import theme as ui_theme

# History file path
HISTORY_FILE = os.path.expanduser("~/.murmur_history.json")
LEGACY_HISTORY_FILE = os.path.expanduser("~/.mywhisper_history.json")
MURMUR_AUDIO_DIR = os.path.realpath(os.path.expanduser("~/.murmur_audio"))
LEGACY_AUDIO_DIR = os.path.realpath(os.path.expanduser("~/.mywhisper_audio"))


def _is_allowed_playback_path(audio_path):
    """Only play audio files stored under Murmur's local audio directories."""
    try:
        resolved = os.path.realpath(audio_path)
    except OSError:
        return False
    for allowed_dir in (MURMUR_AUDIO_DIR, LEGACY_AUDIO_DIR):
        if resolved == allowed_dir or resolved.startswith(allowed_dir + os.sep):
            return True
    return False

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds"""
    try:
        if audio_path and os.path.exists(audio_path):
            with wave.open(audio_path, 'r') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
    except:
        pass
    return 0

def format_duration(seconds):
    """Format seconds as m:ss or s.Xs"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

def load_history():
    """Load history from the running app or local disk."""
    module = sys.modules.get("app.config")
    app = getattr(module, "APP_INSTANCE", None) if module is not None else None
    if app is not None and getattr(app, "history", None) is not None:
        return list(app.history)

    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as file:
                payload = json.load(file)
                if isinstance(payload, list):
                    return payload
        if os.path.exists(LEGACY_HISTORY_FILE):
            with open(LEGACY_HISTORY_FILE, "r") as file:
                payload = json.load(file)
                if isinstance(payload, list):
                    return payload
    except (json.JSONDecodeError, OSError):
        return []
    return []


class HistoryWindowController(NSObject):
    """Controller for the history window"""
    
    def init(self):
        self = objc.super(HistoryWindowController, self).init()
        if self is None:
            return None
        
        self.history = load_history()
        self.selected_index = 0
        self.item_buttons = []
        self.item_accent_bars = []
        self.text_view = None
        self.date_label = None
        self.source_label = None
        self.list_container = None
        self.sidebar_scroll = None
        self.sidebar_width = 300
        self.audio_process = None  # Track audio playback
        self.is_playing = False
        
        return self
    
    def createWindow(self):
        """Create and show the main window"""
        self.history = load_history()
        # Create window
        window_rect = NSMakeRect(0, 0, 920, 560)
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            window_rect, style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Murmur - History")
        self.window.center()
        self.window.setReleasedWhenClosed_(False)
        
        # Murmur dark theme
        ui_theme.apply_window_theme(self.window)
        
        content_view = self.window.contentView()
        content_width = content_view.frame().size.width
        content_height = content_view.frame().size.height
        
        # === LEFT SIDEBAR ===
        sidebar_width = 300
        self._create_sidebar(content_view, sidebar_width, content_height)
        
        # === SEPARATOR ===
        separator = NSView.alloc().initWithFrame_(NSMakeRect(sidebar_width, 0, 1, content_height))
        separator.setWantsLayer_(True)
        separator.layer().setBackgroundColor_(ui_theme.separator_color())
        content_view.addSubview_(separator)
        
        # === RIGHT CONTENT ===
        self._create_content_area(content_view, sidebar_width, content_width, content_height)
        
        # Select first item
        if self.history:
            self._select_item(0)
        else:
            self._show_empty_state()
        
        # Show window
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
    
    def _create_sidebar(self, content_view, sidebar_width, content_height):
        """Create the left sidebar with history list"""
        self.sidebar_width = sidebar_width
        footer_height = 56
        header_height = 48
        list_top = header_height
        list_height = content_height - header_height - footer_height

        sidebar = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, sidebar_width, content_height))
        sidebar.setWantsLayer_(True)
        sidebar.layer().setBackgroundColor_(ui_theme.sidebar_background_color())

        header = NSTextField.alloc().initWithFrame_(NSMakeRect(16, content_height - 32, sidebar_width - 32, 20))
        header.setStringValue_(f"History ({len(self.history)})")
        header.setBezeled_(False)
        header.setDrawsBackground_(False)
        header.setEditable_(False)
        header.setSelectable_(False)
        header.setTextColor_(ui_theme.primary_text_color())
        header.setFont_(NSFont.systemFontOfSize_weight_(13, 0.3))
        self.history_header = header
        sidebar.addSubview_(header)
        ui_theme.add_horizontal_rule(sidebar, 0, content_height - header_height, sidebar_width)

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, footer_height, sidebar_width, list_height))
        scroll.setHasVerticalScroller_(True)
        scroll.setHasHorizontalScroller_(False)
        scroll.setBorderType_(0)
        scroll.setDrawsBackground_(False)
        scroll.setAutohidesScrollers_(True)
        self.sidebar_scroll = scroll

        self._populate_history_list(sidebar_width, list_height)
        scroll.setDocumentView_(self.list_container)
        clip = scroll.contentView()
        clip.scrollToPoint_(NSMakePoint(0, 0))
        scroll.reflectScrolledClipView_(clip)
        sidebar.addSubview_(scroll)

        footer = ui_theme.add_footer_bar(sidebar, sidebar_width, footer_height)

        copy_btn = NSButton.alloc().initWithFrame_(NSMakeRect(12, 12, 88, 28))
        copy_btn.setTitle_("Copy")
        ui_theme.style_dark_button(copy_btn)
        copy_btn.setAccessibilityLabel_("Copy transcription")
        copy_btn.setAccessibilityRole_(NSAccessibilityButtonRole)
        copy_btn.setTarget_(self)
        copy_btn.setAction_(objc.selector(self.copyClicked_, signature=b'v@:@'))
        footer.addSubview_(copy_btn)

        clear_btn = NSButton.alloc().initWithFrame_(NSMakeRect(108, 12, 130, 28))
        clear_btn.setTitle_("Clear History")
        ui_theme.style_dark_button(clear_btn)
        clear_btn.setAccessibilityLabel_("Clear History")
        clear_btn.setAccessibilityRole_(NSAccessibilityButtonRole)
        clear_btn.setTarget_(self)
        clear_btn.setAction_(objc.selector(self.clearHistoryClicked_, signature=b'v@:@'))
        footer.addSubview_(clear_btn)

        content_view.addSubview_(sidebar)

    @objc.python_method
    def _populate_history_list(self, sidebar_width, list_height):
        """Build or rebuild the scrollable history list."""
        item_height = 72
        total_height = max(len(self.history) * item_height, list_height)
        self.list_container = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, sidebar_width, total_height))
        self.list_container.setFlipped_(True)
        self.item_buttons = []
        self.item_accent_bars = []

        for i, entry in enumerate(self.history):
            item_view = self._create_item_view(i, entry, sidebar_width, item_height)
            item_view.setFrame_(NSMakeRect(0, i * item_height, sidebar_width, item_height))
            self.list_container.addSubview_(item_view)

    @objc.python_method
    def _reload_history_list(self):
        """Refresh sidebar after history changes."""
        self._stop_audio()
        self._populate_history_list(self.sidebar_width, self.sidebar_scroll.frame().size.height)
        self.sidebar_scroll.setDocumentView_(self.list_container)
        self.history_header.setStringValue_(f"History ({len(self.history)})")

        if self.history:
            self._select_item(0)
        else:
            self._show_empty_state()

    @objc.python_method
    def _show_empty_state(self):
        """Reset detail pane when there is no history."""
        self.selected_index = 0
        self.text_view.setString_("")
        self.date_label.setStringValue_("No history yet")
        self.source_label.setStringValue_("Transcriptions you save will appear here.")
        self.source_label.setTextColor_(ui_theme.muted_text_color())
        self.time_total.setStringValue_("--:--")
        self.time_current.setStringValue_("--:--")
        self.play_btn.setEnabled_(False)
        self.stop_btn.setEnabled_(False)
    
    def _create_item_view(self, index, entry, width, height):
        """Create a single history item view"""
        item = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        item.setWantsLayer_(True)
        item.layer().setBackgroundColor_(ui_theme.row_background_color())

        accent = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 3, height))
        accent.setWantsLayer_(True)
        accent.layer().setBackgroundColor_(CGColorCreateGenericRGB(0, 0, 0, 0))
        item.addSubview_(accent)
        self.item_accent_bars.append(accent)
        
        text = entry.get("text", "")
        text_preview = text[:50].replace("\n", " ") if text else "No result"
        if len(text) > 50:
            text_preview += "…"
        
        # Get audio duration
        audio_path = entry.get("audio_path", "")
        duration = get_audio_duration(audio_path)
        duration_str = format_duration(duration) if duration > 0 else ""
        
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            date_str = ts.strftime("%d %b")
            time_str = ts.strftime("%H:%M")
        except:
            date_str = "Unknown"
            time_str = ""

        meta_parts = [part for part in (date_str, time_str, duration_str) if part]
        meta_line = " · ".join(meta_parts)

        meta_label = NSTextField.alloc().initWithFrame_(NSMakeRect(14, 8, width - 28, 14))
        meta_label.setStringValue_(meta_line)
        meta_label.setBezeled_(False)
        meta_label.setDrawsBackground_(False)
        meta_label.setEditable_(False)
        meta_label.setSelectable_(False)
        meta_label.setTextColor_(ui_theme.muted_text_color())
        meta_label.setFont_(NSFont.systemFontOfSize_(11))
        item.addSubview_(meta_label)

        text_label = NSTextField.alloc().initWithFrame_(NSMakeRect(14, 26, width - 28, 36))
        text_label.setStringValue_(text_preview)
        text_label.setBezeled_(False)
        text_label.setDrawsBackground_(False)
        text_label.setEditable_(False)
        text_label.setSelectable_(False)
        text_label.setTextColor_(ui_theme.primary_text_color() if text else ui_theme.subtle_text_color())
        text_label.setFont_(NSFont.systemFontOfSize_(13))
        item.addSubview_(text_label)
        
        btn = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        btn.setBordered_(False)
        btn.setTransparent_(True)
        btn.setFocusRingType_(NSFocusRingTypeNone)
        btn.setTag_(index)
        a11y_parts = [part for part in (meta_line, text_preview) if part]
        btn.setAccessibilityLabel_("History item: " + ", ".join(a11y_parts))
        btn.setAccessibilityRole_(NSAccessibilityButtonRole)
        btn.setTarget_(self)
        btn.setAction_(objc.selector(self.itemClicked_, signature=b'v@:@'))
        item.addSubview_(btn)

        divider = NSView.alloc().initWithFrame_(NSMakeRect(12, height - 1, width - 24, 1))
        divider.setWantsLayer_(True)
        divider.layer().setBackgroundColor_(ui_theme.separator_color())
        item.addSubview_(divider)
        
        self.item_buttons.append((item, btn))
        return item
    
    def _create_content_area(self, content_view, sidebar_width, total_width, height):
        """Create the right content area with clear separation between text and player"""
        x = sidebar_width + 1
        width = total_width - sidebar_width - 1
        player_section_height = 88
        header_height = 72
        padding = ui_theme.MARGIN

        # === TOP HEADER SECTION ===
        self.date_label = NSTextField.alloc().initWithFrame_(NSMakeRect(x + padding, height - 36, width - padding * 2, 22))
        self.date_label.setStringValue_("")
        self.date_label.setBezeled_(False)
        self.date_label.setDrawsBackground_(False)
        self.date_label.setEditable_(False)
        self.date_label.setSelectable_(False)
        self.date_label.setTextColor_(ui_theme.primary_text_color())
        self.date_label.setFont_(NSFont.systemFontOfSize_weight_(15, 0.2))
        content_view.addSubview_(self.date_label)

        self.source_label = NSTextField.alloc().initWithFrame_(NSMakeRect(x + padding, height - 58, width - padding * 2, 18))
        self.source_label.setStringValue_("")
        self.source_label.setBezeled_(False)
        self.source_label.setDrawsBackground_(False)
        self.source_label.setEditable_(False)
        self.source_label.setSelectable_(False)
        self.source_label.setTextColor_(ui_theme.muted_text_color())
        self.source_label.setFont_(NSFont.systemFontOfSize_(12))
        content_view.addSubview_(self.source_label)

        ui_theme.add_horizontal_rule(content_view, x + padding, height - header_height, width - padding * 2)

        # === TEXT CONTENT SECTION ===
        text_area_bottom = player_section_height + 16
        text_height = height - header_height - 16 - text_area_bottom

        text_bg = NSView.alloc().initWithFrame_(NSMakeRect(x + padding - 4, text_area_bottom - 4, width - padding * 2 + 8, text_height + 8))
        text_bg.setWantsLayer_(True)
        text_bg.layer().setBackgroundColor_(ui_theme.card_background_color())
        text_bg.layer().setCornerRadius_(10)
        content_view.addSubview_(text_bg)

        text_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(x + padding, text_area_bottom, width - padding * 2, text_height))
        text_scroll.setHasVerticalScroller_(True)
        text_scroll.setHasHorizontalScroller_(False)
        text_scroll.setBorderType_(0)
        text_scroll.setDrawsBackground_(False)
        text_scroll.setAutohidesScrollers_(True)

        self.text_view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, width - padding * 2 - 16, max(text_height, 120)))
        self.text_view.setString_("")
        self.text_view.setEditable_(False)
        self.text_view.setSelectable_(True)
        self.text_view.setFont_(NSFont.systemFontOfSize_(15))
        self.text_view.setTextColor_(ui_theme.primary_text_color())
        self.text_view.setBackgroundColor_(NSColor.clearColor())
        self.text_view.setTextContainerInset_(NSMakeSize(14, 14))

        text_scroll.setDocumentView_(self.text_view)
        content_view.addSubview_(text_scroll)

        # === BOTTOM AUDIO PLAYER BAR ===
        player_bg = NSView.alloc().initWithFrame_(NSMakeRect(x, 0, width, player_section_height))
        player_bg.setWantsLayer_(True)
        player_bg.layer().setBackgroundColor_(ui_theme.card_background_color())
        content_view.addSubview_(player_bg)
        ui_theme.add_horizontal_rule(content_view, x, player_section_height, width)

        bar_x = x + padding
        bar_width = width - padding * 2

        self.waveform_view = NSView.alloc().initWithFrame_(NSMakeRect(bar_x, 54, bar_width, 6))
        self.waveform_view.setWantsLayer_(True)
        self.waveform_view.layer().setBackgroundColor_(CGColorCreateGenericRGB(0.28, 0.29, 0.32, 1.0))
        self.waveform_view.layer().setCornerRadius_(3)
        content_view.addSubview_(self.waveform_view)

        self.time_current = NSTextField.alloc().initWithFrame_(NSMakeRect(bar_x, 36, 45, 14))
        self.time_current.setStringValue_("0:00")
        self.time_current.setBezeled_(False)
        self.time_current.setDrawsBackground_(False)
        self.time_current.setEditable_(False)
        self.time_current.setSelectable_(False)
        self.time_current.setTextColor_(ui_theme.muted_text_color())
        self.time_current.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(11, 0.0))
        content_view.addSubview_(self.time_current)

        self.time_total = NSTextField.alloc().initWithFrame_(NSMakeRect(bar_x + bar_width - 45, 36, 45, 14))
        self.time_total.setStringValue_("0:00")
        self.time_total.setBezeled_(False)
        self.time_total.setDrawsBackground_(False)
        self.time_total.setEditable_(False)
        self.time_total.setSelectable_(False)
        self.time_total.setTextColor_(ui_theme.muted_text_color())
        self.time_total.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(11, 0.0))
        self.time_total.setAlignment_(NSRightTextAlignment)
        content_view.addSubview_(self.time_total)

        player_center_x = x + (width // 2)
        self.stop_btn = NSButton.alloc().initWithFrame_(NSMakeRect(player_center_x - 92, 10, 84, 28))
        self.stop_btn.setTitle_("Stop")
        ui_theme.style_dark_button(self.stop_btn)
        self.stop_btn.setAccessibilityLabel_("Stop playback")
        self.stop_btn.setAccessibilityRole_(NSAccessibilityButtonRole)
        self.stop_btn.setTarget_(self)
        self.stop_btn.setAction_(objc.selector(self.stopClicked_, signature=b'v@:@'))
        content_view.addSubview_(self.stop_btn)

        self.play_btn = NSButton.alloc().initWithFrame_(NSMakeRect(player_center_x + 8, 10, 84, 28))
        self.play_btn.setTitle_("Play")
        ui_theme.style_primary_button(self.play_btn)
        self.play_btn.setAccessibilityLabel_("Play audio")
        self.play_btn.setAccessibilityRole_(NSAccessibilityButtonRole)
        self.play_btn.setTarget_(self)
        self.play_btn.setAction_(objc.selector(self.playClicked_, signature=b'v@:@'))
        content_view.addSubview_(self.play_btn)

        self.duration_label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 0, 0))
        self.duration_label.setHidden_(True)
    
    @objc.python_method
    def _select_item(self, index):
        """Select and display a history item"""
        if index >= len(self.history):
            return
        
        # Stop any playing audio
        self._stop_audio()
        
        self.selected_index = index
        entry = self.history[index]
        
        # Update highlighting
        for i, (item_view, btn) in enumerate(self.item_buttons):
            accent = self.item_accent_bars[i]
            if i == index:
                item_view.layer().setBackgroundColor_(ui_theme.selected_row_background_color())
                accent.layer().setBackgroundColor_(ui_theme.accent_bar_color())
            else:
                item_view.layer().setBackgroundColor_(ui_theme.row_background_color())
                accent.layer().setBackgroundColor_(CGColorCreateGenericRGB(0, 0, 0, 0))
        
        # Update content
        text = entry.get("text", "")
        audio_path = entry.get("audio_path", "")
        
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            date_str = ts.strftime("%d %b %Y at %H:%M")
        except:
            date_str = ""
        
        source = "🎤 Live recording" if entry.get("source") == "live" else f"📁 {entry.get('filename', 'File')}"
        
        # Get audio duration
        duration = get_audio_duration(audio_path)
        has_audio = audio_path and os.path.exists(audio_path)
        
        self.text_view.setString_(text)
        self.date_label.setStringValue_(date_str)
        self.source_label.setStringValue_(source)
        self.source_label.setTextColor_(ui_theme.muted_text_color())
        
        # Update time displays
        if has_audio and duration > 0:
            total_time = self._format_time(duration)
            self.time_total.setStringValue_(total_time)
            self.time_current.setStringValue_("0:00")
        else:
            self.time_total.setStringValue_("--:--")
            self.time_current.setStringValue_("--:--")
        
        # Enable/disable buttons based on audio availability
        self.play_btn.setEnabled_(has_audio)
        self.stop_btn.setEnabled_(has_audio)
        self.play_btn.setTitle_("Play")
        self.play_btn.setAccessibilityLabel_("Play audio")
        ui_theme.style_primary_button(self.play_btn)
        self.is_playing = False
    
    @objc.python_method
    def _format_time(self, seconds):
        """Format seconds as m:ss"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    
    @objc.python_method
    def _stop_audio(self):
        """Stop audio playback"""
        if self.audio_process:
            try:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=1)
            except:
                try:
                    self.audio_process.kill()
                except:
                    pass
            self.audio_process = None
        self.is_playing = False
        if hasattr(self, 'play_btn') and self.play_btn:
            self.play_btn.setTitle_("Play")
            self.play_btn.setAccessibilityLabel_("Play audio")
            ui_theme.style_primary_button(self.play_btn)
    
    def itemClicked_(self, sender):
        """Handle item click"""
        index = sender.tag()
        self._select_item(index)
    
    def playClicked_(self, sender):
        """Play/pause the audio for the current selection"""
        if self.selected_index >= len(self.history):
            return
            
        audio_path = self.history[self.selected_index].get("audio_path", "")
        if not audio_path or not os.path.exists(audio_path):
            return
        if not _is_allowed_playback_path(audio_path):
            return
        
        if self.is_playing:
            # Pause - stop current playback
            self._stop_audio()
            self.play_btn.setTitle_("Play")
            self.play_btn.setAccessibilityLabel_("Play audio")
            ui_theme.style_primary_button(self.play_btn)
        else:
            # Play
            self._stop_audio()  # Stop any previous playback
            self.audio_process = subprocess.Popen(["afplay", audio_path])
            self.is_playing = True
            self.play_btn.setTitle_("Pause")
            self.play_btn.setAccessibilityLabel_("Pause audio")
            ui_theme.style_primary_button(self.play_btn)
            
            # Monitor playback completion in background
            import threading
            def monitor():
                if self.audio_process:
                    self.audio_process.wait()
                    self.is_playing = False
                    # Update UI on main thread
                    from PyObjCTools import AppHelper
                    def reset_play_label():
                        if self.play_btn:
                            self.play_btn.setTitle_("Play")
                            self.play_btn.setAccessibilityLabel_("Play audio")
                            ui_theme.style_primary_button(self.play_btn)
                    AppHelper.callAfter(reset_play_label)
            threading.Thread(target=monitor, daemon=True).start()
    
    def stopClicked_(self, sender):
        """Stop audio playback and reset"""
        self._stop_audio()
        self.time_current.setStringValue_("0:00")
    
    def copyContentClicked_(self, sender):
        """Copy current text to clipboard"""
        if self.selected_index < len(self.history):
            text = self.history[self.selected_index].get("text", "")
            pyperclip.copy(text)
    
    def copyClicked_(self, sender):
        """Copy current text to clipboard"""
        if self.selected_index < len(self.history):
            text = self.history[self.selected_index].get("text", "")
            pyperclip.copy(text)

    def clearHistoryClicked_(self, sender):
        """Clear transcription history from this Mac."""
        if not self.history:
            return

        alert = NSAlert.alloc().init()
        alert.setMessageText_("Clear History?")
        alert.setInformativeText_(
            "This removes all saved transcriptions from history. "
            "Saved audio files in ~/.murmur_audio/ are kept unless you delete them in Settings."
        )
        alert.setAlertStyle_(NSWarningAlertStyle)
        alert.addButtonWithTitle_("Clear")
        alert.addButtonWithTitle_("Cancel")
        ui_alerts.configure_alert(alert)
        if alert.runModal() != NSAlertFirstButtonReturn:
            return

        self.history = []
        try:
            with open(HISTORY_FILE, "w") as file:
                json.dump([], file)
        except OSError:
            return

        app_config = sys.modules.get("app.config")
        running = getattr(app_config, "APP_INSTANCE", None) if app_config is not None else None
        if running is not None:
            running.history = []
            running.save_history()

        self._reload_history_list()


# Global reference to keep controller alive
_controller = None

def main():
    global _controller
    
    # Set up as accessory app (no dock icon, window still shows)
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    
    # Create controller and window
    _controller = HistoryWindowController.alloc().init()
    _controller.createWindow()
    
    # Activate to bring window to front
    app.activateIgnoringOtherApps_(True)
    
    # Run event loop
    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()