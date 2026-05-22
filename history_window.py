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
    NSBackingStoreBuffered, NSMakeRect, NSMakeSize, NSScrollView, NSTextView, NSFont,
    NSColor, NSViewWidthSizable, NSViewHeightSizable, NSFloatingWindowLevel,
    NSButton, NSRoundedBezelStyle, NSTextField, NSView, NSTableView,
    NSTableColumn, NSObject, NSApp, NSApplicationActivationPolicyAccessory,
    NSTextFieldCell, NSRightTextAlignment
)
from Quartz import CGColorCreateGenericRGB
from PyObjCTools import AppHelper

# History file path
HISTORY_FILE = os.path.expanduser("~/.murmur_history.json")
LEGACY_HISTORY_FILE = os.path.expanduser("~/.mywhisper_history.json")

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
    """Load history from file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        if os.path.exists(LEGACY_HISTORY_FILE):
            with open(LEGACY_HISTORY_FILE, 'r') as f:
                return json.load(f)
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
        self.text_view = None
        self.date_label = None
        self.source_label = None
        self.list_container = None
        self.audio_process = None  # Track audio playback
        self.is_playing = False
        
        return self
    
    def createWindow(self):
        """Create and show the main window"""
        # Create window
        window_rect = NSMakeRect(0, 0, 900, 550)
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            window_rect, style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Murmur - History")
        self.window.center()
        self.window.setReleasedWhenClosed_(False)
        
        # Dark background
        dark_bg = NSColor.colorWithRed_green_blue_alpha_(0.11, 0.11, 0.12, 1.0)
        self.window.setBackgroundColor_(dark_bg)
        
        content_view = self.window.contentView()
        content_width = content_view.frame().size.width
        content_height = content_view.frame().size.height
        
        # === LEFT SIDEBAR ===
        sidebar_width = 280
        self._create_sidebar(content_view, sidebar_width, content_height)
        
        # === SEPARATOR ===
        separator = NSView.alloc().initWithFrame_(NSMakeRect(sidebar_width, 0, 1, content_height))
        separator.setWantsLayer_(True)
        separator.layer().setBackgroundColor_(CGColorCreateGenericRGB(0.25, 0.25, 0.25, 1.0))
        content_view.addSubview_(separator)
        
        # === RIGHT CONTENT ===
        self._create_content_area(content_view, sidebar_width, content_width, content_height)
        
        # Select first item
        if self.history:
            self._select_item(0)
        
        # Show window
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
    
    def _create_sidebar(self, content_view, sidebar_width, content_height):
        """Create the left sidebar with history list"""
        sidebar = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, sidebar_width, content_height))
        sidebar.setWantsLayer_(True)
        sidebar.layer().setBackgroundColor_(CGColorCreateGenericRGB(0.08, 0.08, 0.09, 1.0))
        
        # Scroll view for list
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 50, sidebar_width, content_height - 50))
        scroll.setHasVerticalScroller_(True)
        scroll.setHasHorizontalScroller_(False)
        scroll.setBorderType_(0)
        scroll.setDrawsBackground_(False)
        
        # Container for items
        item_height = 65
        total_height = max(len(self.history) * item_height, content_height - 50)
        self.list_container = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, sidebar_width, total_height))
        self.list_container.setFlipped_(True)
        
        # Create history items
        for i, entry in enumerate(self.history):
            item_view = self._create_item_view(i, entry, sidebar_width, item_height)
            item_view.setFrame_(NSMakeRect(0, i * item_height, sidebar_width, item_height))
            self.list_container.addSubview_(item_view)
        
        scroll.setDocumentView_(self.list_container)
        sidebar.addSubview_(scroll)
        
        # Copy button at bottom
        copy_btn = NSButton.alloc().initWithFrame_(NSMakeRect(10, 10, 100, 30))
        copy_btn.setTitle_("📋 Copy")
        copy_btn.setBezelStyle_(NSRoundedBezelStyle)
        copy_btn.setTarget_(self)
        copy_btn.setAction_(objc.selector(self.copyClicked_, signature=b'v@:@'))
        sidebar.addSubview_(copy_btn)
        
        content_view.addSubview_(sidebar)
    
    def _create_item_view(self, index, entry, width, height):
        """Create a single history item view"""
        item = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        item.setWantsLayer_(True)
        
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
            date_str = ts.strftime("%d %B")
            time_str = ts.strftime("%H:%M")
        except:
            date_str = "Unknown"
            time_str = ""
        
        # Text preview
        text_label = NSTextField.alloc().initWithFrame_(NSMakeRect(12, height - 28, width - 60, 20))
        text_label.setStringValue_(text_preview)
        text_label.setBezeled_(False)
        text_label.setDrawsBackground_(False)
        text_label.setEditable_(False)
        text_label.setSelectable_(False)
        text_label.setTextColor_(NSColor.whiteColor() if text else NSColor.grayColor())
        text_label.setFont_(NSFont.systemFontOfSize_(13))
        item.addSubview_(text_label)
        
        # Date
        date_label = NSTextField.alloc().initWithFrame_(NSMakeRect(12, 8, 100, 16))
        date_label.setStringValue_(date_str)
        date_label.setBezeled_(False)
        date_label.setDrawsBackground_(False)
        date_label.setEditable_(False)
        date_label.setSelectable_(False)
        date_label.setTextColor_(NSColor.secondaryLabelColor())
        date_label.setFont_(NSFont.systemFontOfSize_(11))
        item.addSubview_(date_label)
        
        # Time
        time_label = NSTextField.alloc().initWithFrame_(NSMakeRect(110, 8, 50, 16))
        time_label.setStringValue_(time_str)
        time_label.setBezeled_(False)
        time_label.setDrawsBackground_(False)
        time_label.setEditable_(False)
        time_label.setSelectable_(False)
        time_label.setTextColor_(NSColor.secondaryLabelColor())
        time_label.setFont_(NSFont.systemFontOfSize_(11))
        item.addSubview_(time_label)
        
        # Duration (from audio file)
        duration_label = NSTextField.alloc().initWithFrame_(NSMakeRect(width - 50, 8, 45, 16))
        duration_label.setStringValue_(duration_str)
        duration_label.setBezeled_(False)
        duration_label.setDrawsBackground_(False)
        duration_label.setEditable_(False)
        duration_label.setSelectable_(False)
        duration_label.setTextColor_(NSColor.secondaryLabelColor())
        duration_label.setFont_(NSFont.systemFontOfSize_(11))
        item.addSubview_(duration_label)
        
        # Invisible button for clicking
        btn = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        btn.setTransparent_(True)
        btn.setTag_(index)
        btn.setTarget_(self)
        btn.setAction_(objc.selector(self.itemClicked_, signature=b'v@:@'))
        item.addSubview_(btn)
        
        self.item_buttons.append((item, btn))
        return item
    
    def _create_content_area(self, content_view, sidebar_width, total_width, height):
        """Create the right content area with clear separation between text and player"""
        x = sidebar_width + 1
        width = total_width - sidebar_width - 1
        
        # === PLAYER SECTION HEIGHT ===
        player_section_height = 100  # Fixed height for player area
        separator_y = player_section_height
        text_area_bottom = separator_y + 12  # Gap above separator
        
        # === TOP HEADER SECTION ===
        # Date label
        self.date_label = NSTextField.alloc().initWithFrame_(NSMakeRect(x + 20, height - 40, width - 150, 24))
        self.date_label.setStringValue_("")
        self.date_label.setBezeled_(False)
        self.date_label.setDrawsBackground_(False)
        self.date_label.setEditable_(False)
        self.date_label.setSelectable_(False)
        self.date_label.setTextColor_(NSColor.whiteColor())
        self.date_label.setFont_(NSFont.boldSystemFontOfSize_(14))
        content_view.addSubview_(self.date_label)
        
        # Copy button (top right, near title)
        copy_btn = NSButton.alloc().initWithFrame_(NSMakeRect(x + width - 90, height - 42, 70, 28))
        copy_btn.setTitle_("📋 Copy")
        copy_btn.setBezelStyle_(NSRoundedBezelStyle)
        copy_btn.setTarget_(self)
        copy_btn.setAction_(objc.selector(self.copyContentClicked_, signature=b'v@:@'))
        content_view.addSubview_(copy_btn)
        
        # Source label (below date)
        self.source_label = NSTextField.alloc().initWithFrame_(NSMakeRect(x + 20, height - 62, width - 40, 18))
        self.source_label.setStringValue_("")
        self.source_label.setBezeled_(False)
        self.source_label.setDrawsBackground_(False)
        self.source_label.setEditable_(False)
        self.source_label.setSelectable_(False)
        self.source_label.setTextColor_(NSColor.secondaryLabelColor())
        self.source_label.setFont_(NSFont.systemFontOfSize_(11))
        content_view.addSubview_(self.source_label)
        
        # === TEXT CONTENT SECTION ===
        text_height = height - 75 - text_area_bottom  # From below header to above player
        text_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(x + 15, text_area_bottom, width - 30, text_height))
        text_scroll.setHasVerticalScroller_(True)
        text_scroll.setHasHorizontalScroller_(False)
        text_scroll.setBorderType_(0)
        text_scroll.setDrawsBackground_(False)
        
        # Rounded background for text area
        text_bg = NSView.alloc().initWithFrame_(NSMakeRect(x + 10, text_area_bottom - 5, width - 20, text_height + 10))
        text_bg.setWantsLayer_(True)
        text_bg.layer().setBackgroundColor_(CGColorCreateGenericRGB(0.08, 0.08, 0.09, 1.0))
        text_bg.layer().setCornerRadius_(8)
        content_view.addSubview_(text_bg)
        
        # Text view
        text_rect = NSMakeRect(0, 0, width - 50, text_height)
        self.text_view = NSTextView.alloc().initWithFrame_(text_rect)
        self.text_view.setString_("")
        self.text_view.setEditable_(False)
        self.text_view.setSelectable_(True)
        self.text_view.setFont_(NSFont.systemFontOfSize_(15))
        self.text_view.setTextColor_(NSColor.whiteColor())
        self.text_view.setBackgroundColor_(NSColor.clearColor())
        self.text_view.setTextContainerInset_(NSMakeSize(12, 12))
        
        text_scroll.setDocumentView_(self.text_view)
        content_view.addSubview_(text_scroll)
        
        # === SEPARATOR LINE ===
        separator = NSView.alloc().initWithFrame_(NSMakeRect(x + 20, separator_y, width - 40, 1))
        separator.setWantsLayer_(True)
        separator.layer().setBackgroundColor_(CGColorCreateGenericRGB(0.25, 0.25, 0.27, 1.0))
        content_view.addSubview_(separator)
        
        # === BOTTOM AUDIO PLAYER BAR ===
        player_center_x = x + (width // 2)
        controls_y = 35  # Vertical center of player section
        
        # Player background
        player_bg = NSView.alloc().initWithFrame_(NSMakeRect(x, 0, width, player_section_height - 5))
        player_bg.setWantsLayer_(True)
        player_bg.layer().setBackgroundColor_(CGColorCreateGenericRGB(0.09, 0.09, 0.10, 1.0))
        content_view.addSubview_(player_bg)
        
        # Waveform / progress bar (full width, subtle)
        self.waveform_view = NSView.alloc().initWithFrame_(NSMakeRect(x + 70, 70, width - 140, 8))
        self.waveform_view.setWantsLayer_(True)
        self.waveform_view.layer().setBackgroundColor_(CGColorCreateGenericRGB(0.2, 0.2, 0.22, 1.0))
        self.waveform_view.layer().setCornerRadius_(4)
        content_view.addSubview_(self.waveform_view)
        
        # Time labels (below waveform)
        self.time_current = NSTextField.alloc().initWithFrame_(NSMakeRect(x + 70, 52, 45, 16))
        self.time_current.setStringValue_("0:00")
        self.time_current.setBezeled_(False)
        self.time_current.setDrawsBackground_(False)
        self.time_current.setEditable_(False)
        self.time_current.setSelectable_(False)
        self.time_current.setTextColor_(NSColor.secondaryLabelColor())
        self.time_current.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(10, 0.0))
        content_view.addSubview_(self.time_current)
        
        self.time_total = NSTextField.alloc().initWithFrame_(NSMakeRect(x + width - 115, 52, 45, 16))
        self.time_total.setStringValue_("0:00")
        self.time_total.setBezeled_(False)
        self.time_total.setDrawsBackground_(False)
        self.time_total.setEditable_(False)
        self.time_total.setSelectable_(False)
        self.time_total.setTextColor_(NSColor.secondaryLabelColor())
        self.time_total.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(10, 0.0))
        self.time_total.setAlignment_(NSRightTextAlignment)
        content_view.addSubview_(self.time_total)
        
        # === PLAYBACK CONTROLS (centered) ===
        # Stop button (left)
        self.stop_btn = NSButton.alloc().initWithFrame_(NSMakeRect(player_center_x - 60, 12, 36, 36))
        self.stop_btn.setTitle_("⏹")
        self.stop_btn.setBezelStyle_(NSRoundedBezelStyle)
        self.stop_btn.setTarget_(self)
        self.stop_btn.setAction_(objc.selector(self.stopClicked_, signature=b'v@:@'))
        self.stop_btn.setFont_(NSFont.systemFontOfSize_(14))
        content_view.addSubview_(self.stop_btn)
        
        # Play/Pause button (center, larger)
        self.play_btn = NSButton.alloc().initWithFrame_(NSMakeRect(player_center_x - 20, 8, 44, 44))
        self.play_btn.setTitle_("▶")
        self.play_btn.setBezelStyle_(NSRoundedBezelStyle)
        self.play_btn.setTarget_(self)
        self.play_btn.setAction_(objc.selector(self.playClicked_, signature=b'v@:@'))
        self.play_btn.setFont_(NSFont.systemFontOfSize_(18))
        content_view.addSubview_(self.play_btn)
        
        # Duration label (right of controls)
        self.duration_label = NSTextField.alloc().initWithFrame_(NSMakeRect(player_center_x + 35, 20, 80, 20))
        self.duration_label.setStringValue_("")
        self.duration_label.setBezeled_(False)
        self.duration_label.setDrawsBackground_(False)
        self.duration_label.setEditable_(False)
        self.duration_label.setSelectable_(False)
        self.duration_label.setTextColor_(NSColor.secondaryLabelColor())
        self.duration_label.setFont_(NSFont.systemFontOfSize_(11))
        content_view.addSubview_(self.duration_label)
    
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
            if i == index:
                item_view.layer().setBackgroundColor_(CGColorCreateGenericRGB(0.2, 0.2, 0.25, 1.0))
            else:
                item_view.layer().setBackgroundColor_(CGColorCreateGenericRGB(0.08, 0.08, 0.09, 1.0))
        
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
        
        # Update time displays
        if has_audio and duration > 0:
            total_time = self._format_time(duration)
            self.time_total.setStringValue_(total_time)
            self.time_current.setStringValue_("0:00")
            self.duration_label.setStringValue_(format_duration(duration))
        else:
            self.time_total.setStringValue_("--:--")
            self.time_current.setStringValue_("--:--")
            self.duration_label.setStringValue_("No audio")
        
        # Enable/disable buttons based on audio availability
        self.play_btn.setEnabled_(has_audio)
        self.stop_btn.setEnabled_(has_audio)
        self.play_btn.setTitle_("▶")
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
            self.play_btn.setTitle_("▶")
    
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
        
        if self.is_playing:
            # Pause - stop current playback
            self._stop_audio()
            self.play_btn.setTitle_("▶")
        else:
            # Play
            self._stop_audio()  # Stop any previous playback
            self.audio_process = subprocess.Popen(["afplay", audio_path])
            self.is_playing = True
            self.play_btn.setTitle_("⏸")
            
            # Monitor playback completion in background
            import threading
            def monitor():
                if self.audio_process:
                    self.audio_process.wait()
                    self.is_playing = False
                    # Update UI on main thread
                    from PyObjCTools import AppHelper
                    AppHelper.callAfter(lambda: self.play_btn.setTitle_("▶") if self.play_btn else None)
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