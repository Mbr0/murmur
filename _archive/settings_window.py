#!/usr/bin/env python3
"""
Murmur Settings Window - Configuration panel
"""

import logging
import sys
import os
import subprocess
from dataclasses import replace as dataclass_replace
from cleanup.vocabulary import (
    Replacement,
    Vocabulary,
    VocabularyError,
    export_csv,
    import_csv,
    vocabulary_from_config,
    vocabulary_to_config,
)
from engines import LANGUAGE_AUTO
from services.language_service import available_languages, language_display_name
from services.hotkey_service import (
    DEFAULT_HOTKEY,
    HOTKEY_MODE_AUTO,
    HOTKEY_MODE_CONFIG_KEY,
    HOTKEY_MODE_HOLD,
    HOTKEY_MODE_TOGGLE,
    HOTKEY_MODES,
    HotkeyBinding,
    MODIFIER_KEYCODES,
    binding_from_ns_flags,
    binding_has_modifier,
    capture_label_for_binding,
    format_hotkey,
    hotkey_from_config,
    hotkey_mode_from_config,
    hotkey_permissions_ok,
    hotkey_to_config,
    open_privacy_settings,
    permission_status_message,
    ns_modifier_flags,
)
from services.persistence_service import DEFAULT_CONFIG, PersistencePaths, PersistenceService
from engines.model_store import ModelStore
from ui.download_sheet import (
    PHASE_CANCELLED,
    PHASE_DONE,
    PHASE_FAILED,
    DownloadController,
    EngineSectionModel,
)

# PyObjC imports
import objc
from Cocoa import (
    NSApplication, NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSBackingStoreBuffered, NSMakeRect, NSFont, NSColor,
    NSButton, NSRoundedBezelStyle, NSTextField, NSObject, NSApp, NSView,
    NSApplicationActivationPolicyAccessory, NSPopUpButton, NSBox, NSBoxSeparator,
    NSAlert, NSInformationalAlertStyle, NSWarningAlertStyle, NSOnState, NSOffState,
    NSAlertFirstButtonReturn, NSAppearance, NSScrollView, NSBezelBorder,
    NSPanel, NSProgressIndicator, NSTextView, NSTableView, NSTableColumn,
    NSButtonCell, NSOpenPanel, NSSavePanel,
    NSEvent, NSEventMaskKeyDown, NSEventMaskFlagsChanged, NSEventTypeFlagsChanged,
    NSEventModifierFlagCommand, NSEventModifierFlagOption,
    NSEventModifierFlagControl, NSEventModifierFlagShift, NSEventModifierFlagFunction,
)
from Foundation import NSIndexSet
from PyObjCTools import AppHelper
import ui_alerts
import ui_theme

APP_NAME = "Murmur"
logger = logging.getLogger(__name__)

# Config file
CONFIG_FILE = os.path.expanduser("~/.murmur_config.json")
HISTORY_FILE = os.path.expanduser("~/.murmur_history.json")
AUDIO_DIR = os.path.expanduser("~/.murmur_audio")
LEGACY_CONFIG_FILE = os.path.expanduser("~/.mywhisper_config.json")
PERSISTENCE = PersistenceService(
    PersistencePaths(config_file=CONFIG_FILE, history_file=HISTORY_FILE),
    logger=logger,
)

SETTINGS_WIDTH = 520
SETTINGS_HEIGHT = 660

DOWNLOAD_SHEET_WIDTH = 380
DOWNLOAD_SHEET_HEIGHT = 156

APPEARANCE_LABELS = {
    "system": "System",
    "dark": "Dark",
    "light": "Light",
}

#: What each shortcut behaviour means in the popup, in ``HOTKEY_MODES`` order.
HOTKEY_MODE_LABELS = {
    HOTKEY_MODE_TOGGLE: "Toggle",
    HOTKEY_MODE_HOLD: "Hold to talk",
    HOTKEY_MODE_AUTO: "Automatic",
}

#: Languages offered when no engine is loaded to ask. Same shape as
#: ``available_languages``: auto first, ISO codes after it, sorted.
FALLBACK_LANGUAGES = (LANGUAGE_AUTO, "de", "en", "es", "fr", "it", "nl", "pt")

#: Columns of the replacements table, in display order.
REPLACEMENT_COLUMNS = ("from", "to", "match_case")
REPLACEMENT_COLUMN_TITLES = {
    "from": "From",
    "to": "To",
    "match_case": "Match case",
}

VOCABULARY_CSV_NAME = "murmur-vocabulary.csv"

#: The one config key the Language popup owns.
CONFIG_LANGUAGE = "language"


class LanguageSectionModel:
    """Rows of the Language popup and the single key it may write.

    The rows come from the running engine, but the value already in config
    always gets a row of its own, even when the engine does not list it. Two
    bugs live in the gap: an engine that reported only ``auto`` gave the popup
    one row, so every Save wrote ``language="auto"`` over whatever the user had
    chosen; and a language chosen for a previous engine would silently vanish
    on the next Save after a live engine switch.

    Save is by code, never by row index, and only when the popup actually moved
    off the value the window opened with. An untouched popup writes nothing.
    """

    def __init__(self, config: dict, codes) -> None:
        assert config is not None, "config is required"
        codes = tuple(codes)
        assert codes, "at least one language code is required"
        current = config.get(CONFIG_LANGUAGE) or LANGUAGE_AUTO
        self.codes = codes if current in codes else (*codes, current)
        #: The value the window opened with; what "unchanged" means.
        self.initial = current

    @property
    def titles(self) -> tuple:
        """Popup titles, in row order."""
        return tuple(
            "Automatic" if code == LANGUAGE_AUTO else language_display_name(code)
            for code in self.codes
        )

    @property
    def selected_index(self) -> int:
        """Row to highlight when the popup is built."""
        return self.codes.index(self.initial)

    def code_at(self, index: int) -> str:
        assert 0 <= index < len(self.codes), f"row {index} is out of range"
        return self.codes[index]

    def changes_for_index(self, index: int) -> dict:
        """``{"language": code}`` when the popup moved, else nothing to write."""
        code = self.code_at(index)
        if code == self.initial:
            return {}
        return {CONFIG_LANGUAGE: code}

    def rebuilt(self, codes, selected_index: int) -> "LanguageSectionModel":
        """A model over a new engine's ``codes``, keeping the current choice.

        Called when the engine is swapped while Settings is open, so the popup
        stops describing the engine that is no longer running. The selected code
        carries over as the new baseline: switching engines is not the user
        changing their language.
        """
        return LanguageSectionModel({CONFIG_LANGUAGE: self.code_at(selected_index)}, codes)


class VocabularySectionModel:
    """Editing state for the Settings "Vocabulary" group.

    Plain Python, so the editing rules — how a box of text becomes a term
    list, what a fresh replacement row holds, which row is selected after a
    removal, what actually gets saved — are testable without AppKit. The
    AppKit code in :class:`SettingsWindowController` is a rendering of this.
    """

    def __init__(self, config: dict) -> None:
        assert config is not None, "config is required"
        vocabulary = vocabulary_from_config(config)
        self.terms: list[str] = list(vocabulary.terms)
        self.replacements: list[Replacement] = list(vocabulary.replacements)

    # -- terms -----------------------------------------------------------

    @property
    def terms_text(self) -> str:
        """The terms box's contents: one term per line."""
        return "\n".join(self.terms)

    def set_terms_text(self, text: str) -> None:
        """Read the terms box back. One per line; blanks and repeats dropped.

        Deduplicating here rather than at save time means the box the user
        sees next is the list Murmur actually holds.
        """
        assert text is not None, "text is required"
        seen: set[str] = set()
        terms: list[str] = []
        for line in text.splitlines():
            term = line.strip()
            if not term or term in seen:
                continue
            seen.add(term)
            terms.append(term)
        self.terms = terms

    # -- replacements ----------------------------------------------------

    @property
    def row_count(self) -> int:
        """Rows in the replacements table, blank ones included."""
        return len(self.replacements)

    def value_for(self, row: int, column: str):
        """The value one table cell displays."""
        assert 0 <= row < len(self.replacements), f"row {row} is out of range"
        assert column in REPLACEMENT_COLUMNS, f"unknown column: {column!r}"
        replacement = self.replacements[row]
        if column == "from":
            return replacement.from_text
        if column == "to":
            return replacement.to_text
        return replacement.match_case

    def set_value(self, row: int, column: str, value) -> None:
        """Write one edited table cell back into the row."""
        assert 0 <= row < len(self.replacements), f"row {row} is out of range"
        assert column in REPLACEMENT_COLUMNS, f"unknown column: {column!r}"
        replacement = self.replacements[row]
        if column == "match_case":
            assert isinstance(value, bool), f"match_case must be a bool, got {value!r}"
            self.replacements[row] = dataclass_replace(replacement, match_case=value)
            return
        assert isinstance(value, str), f"{column} must be a string, got {value!r}"
        field = "from_text" if column == "from" else "to_text"
        self.replacements[row] = dataclass_replace(replacement, **{field: value})

    def add_replacement(self) -> int:
        """Append a blank row and return its index, so the table can edit it."""
        self.replacements.append(Replacement(from_text="", to_text="", match_case=False))
        return len(self.replacements) - 1

    def remove_replacement(self, row: int) -> int:
        """Delete a row; return the row to select next, or -1 when none is left."""
        assert 0 <= row < len(self.replacements), f"row {row} is out of range"
        del self.replacements[row]
        if not self.replacements:
            return -1
        return min(row, len(self.replacements) - 1)

    # -- persistence -----------------------------------------------------

    @property
    def vocabulary(self) -> Vocabulary:
        """What gets saved. A row with no ``from`` text replaces nothing, so
        the half-typed row left behind by a stray "+" is dropped rather than
        persisted as a rule that can never match.
        """
        return Vocabulary(
            terms=tuple(self.terms),
            replacements=tuple(
                item for item in self.replacements if item.from_text.strip()
            ),
        )

    def to_config(self) -> dict:
        """The two config keys the vocabulary lives in."""
        return vocabulary_to_config(self.vocabulary)

    def export_text(self) -> str:
        """CSV for the Export button."""
        return export_csv(self.vocabulary)

    def import_text(self, text: str) -> None:
        """Replace both lists from CSV.

        Raises :class:`~cleanup.vocabulary.VocabularyError`, whose message
        names the offending line, and leaves the current lists untouched when
        it does — a bad file never half-imports.
        """
        imported = import_csv(text)
        self.terms = list(imported.terms)
        self.replacements = list(imported.replacements)


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
    return PERSISTENCE.load_config(dict(DEFAULT_CONFIG))


def update_config(changes):
    """Merge only ``changes`` into the stored config and return the result.

    Settings never writes a whole config: the window is open for as long as the
    user leaves it there, and the app keeps writing while it is — the engine a
    finished download activated, ``onboarding_completed``, the one-shot
    notices. Saving the snapshot taken at open time put all of that back.
    """
    return PERSISTENCE.update_config(changes)


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
        self.save_audio_switch = None
        self.save_history_switch = None
        self.appearance_popup = None

        self.model_store = ModelStore()
        self.engine_section = EngineSectionModel(
            self.config,
            self.model_store,
            on_engine_change=self._engine_changed,
            save_changes=update_config,
        )
        self.download_controller = DownloadController(
            self.model_store, on_change=self._download_state_changed
        )
        self.engine_popup = None
        self.engine_detail_label = None
        self.engine_download_button = None
        self.engine_delete_button = None
        self.download_sheet = None
        self.download_status_label = None
        self.download_progress_bar = None

        self.hotkey_mode = hotkey_mode_from_config(self.config)
        self.hotkey_mode_popup = None
        self.language_section = LanguageSectionModel(self.config, self._language_codes())
        self.language_popup = None
        self.vocabulary_section = VocabularySectionModel(self.config)
        self.terms_view = None
        self.replacements_table = None

        return self

    @objc.python_method
    def _language_codes(self):
        """Languages to offer: the loaded engine's own list, else the known set.

        Asking the engine keeps the picker honest about what the running model
        can actually do, and this is the one place that reads it — no engine
        branch anywhere else in the UI.
        """
        app = _murmur_app_instance()
        engine = getattr(app, "engine", None) if app is not None else None
        if engine is None:
            return FALLBACK_LANGUAGES
        try:
            return available_languages(engine.info())
        except Exception as error:
            logger.warning("Could not read languages from the engine: %s", error)
            return FALLBACK_LANGUAGES
    
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
        rule()

        add(self._create_label(
            "Speech engine", x=ui_theme.MARGIN, y=y, width=content_width, bold=True, size=13
        ))
        self.engine_popup = NSPopUpButton.alloc().initWithFrame_(
            NSMakeRect(ui_theme.MARGIN, y, content_width, 26)
        )
        self.engine_popup.setTarget_(self)
        self.engine_popup.setAction_(objc.selector(self.engineChanged_, signature=b'v@:@'))
        ui_theme.style_popup_on_dark(self.engine_popup)
        content_view.addSubview_(self.engine_popup)
        y += 34

        self.engine_detail_label = self._create_label(
            "", x=ui_theme.MARGIN, y=y, width=content_width, size=11,
            color=ui_theme.muted_text_color(),
        )
        add(self.engine_detail_label)

        self.engine_download_button = NSButton.alloc().initWithFrame_(
            NSMakeRect(ui_theme.MARGIN, y, 140, 28)
        )
        self.engine_download_button.setTitle_("Download")
        ui_theme.style_dark_button(self.engine_download_button)
        self.engine_download_button.setTarget_(self)
        self.engine_download_button.setAction_(
            objc.selector(self.downloadModelClicked_, signature=b'v@:@')
        )
        content_view.addSubview_(self.engine_download_button)

        self.engine_delete_button = NSButton.alloc().initWithFrame_(
            NSMakeRect(ui_theme.MARGIN + 152, y, 140, 28)
        )
        self.engine_delete_button.setTitle_("Delete Model")
        ui_theme.style_dark_button(self.engine_delete_button)
        self.engine_delete_button.setTarget_(self)
        self.engine_delete_button.setAction_(
            objc.selector(self.deleteModelClicked_, signature=b'v@:@')
        )
        content_view.addSubview_(self.engine_delete_button)
        y += 38

        add(self._create_label(
            "Models are downloaded once and kept on this Mac. Switching model "
            "reloads the engine in the background — no restart.",
            x=ui_theme.MARGIN,
            y=y,
            width=content_width,
            size=11,
            color=ui_theme.muted_text_color(),
            height=32,
        ))
        self._refresh_engine_controls()
        rule()

        add(self._create_label(
            "Language", x=ui_theme.MARGIN, y=y, width=content_width, bold=True, size=13
        ))
        self.language_popup = NSPopUpButton.alloc().initWithFrame_(
            NSMakeRect(ui_theme.MARGIN, y, 220, 26)
        )
        self._fill_language_popup()
        ui_theme.style_popup_on_dark(self.language_popup)
        content_view.addSubview_(self.language_popup)
        y += 34

        add(self._create_label(
            "Automatic detects the language for each recording. Picking one is "
            "faster and stops short phrases being mistaken for another language.",
            x=ui_theme.MARGIN,
            y=y,
            width=content_width,
            size=11,
            color=ui_theme.muted_text_color(),
            height=32,
        ))
        rule()

        y = self._add_vocabulary_section(content_view, y, content_width)
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

        add(self._create_label(
            "Shortcut behaviour", x=ui_theme.MARGIN, y=y, width=content_width, size=12
        ))
        self.hotkey_mode_popup = NSPopUpButton.alloc().initWithFrame_(
            NSMakeRect(ui_theme.MARGIN, y, 220, 26)
        )
        for mode in HOTKEY_MODES:
            self.hotkey_mode_popup.addItemWithTitle_(HOTKEY_MODE_LABELS[mode])
        self.hotkey_mode_popup.selectItemAtIndex_(HOTKEY_MODES.index(self.hotkey_mode))
        ui_theme.style_popup_on_dark(self.hotkey_mode_popup)
        content_view.addSubview_(self.hotkey_mode_popup)
        y += 34

        add(self._create_label(
            "Toggle starts and stops on separate presses. Hold to talk records "
            "only while the keys are down. Automatic decides from how long you "
            "hold: a tap toggles, a hold talks.",
            x=ui_theme.MARGIN,
            y=y,
            width=content_width,
            size=11,
            color=ui_theme.muted_text_color(),
            height=44,
        ))

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
    
    # -- Vocabulary section ----------------------------------------------

    @objc.python_method
    def _add_vocabulary_section(self, content_view, y, content_width):
        """Terms box, replacements table, and the CSV buttons. Returns the next y."""
        label = self._create_label(
            "Vocabulary", x=ui_theme.MARGIN, y=y, width=content_width, bold=True, size=13
        )
        content_view.addSubview_(label)
        y += label.frame().size.height + 6

        hint = self._create_label(
            "Terms bias what the engine hears — names, jargon, product names. "
            "One per line.",
            x=ui_theme.MARGIN,
            y=y,
            width=content_width,
            size=11,
            color=ui_theme.muted_text_color(),
            height=30,
        )
        content_view.addSubview_(hint)
        y += 32

        terms_scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(ui_theme.MARGIN, y, content_width, 84)
        )
        terms_scroll.setHasVerticalScroller_(True)
        terms_scroll.setHasHorizontalScroller_(False)
        terms_scroll.setBorderType_(NSBezelBorder)
        self.terms_view = NSTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, content_width, 84)
        )
        self.terms_view.setString_(self.vocabulary_section.terms_text)
        self.terms_view.setFont_(NSFont.systemFontOfSize_(12))
        self.terms_view.setRichText_(False)
        self.terms_view.setAutomaticQuoteSubstitutionEnabled_(False)
        self.terms_view.setAppearance_(ui_theme.control_appearance())
        self.terms_view.setAccessibilityLabel_("Vocabulary terms, one per line")
        terms_scroll.setDocumentView_(self.terms_view)
        content_view.addSubview_(terms_scroll)
        y += 92

        replacements_hint = self._create_label(
            "Replacements rewrite finished text, whole words only.",
            x=ui_theme.MARGIN,
            y=y,
            width=content_width,
            size=11,
            color=ui_theme.muted_text_color(),
        )
        content_view.addSubview_(replacements_hint)
        y += 24

        table_scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(ui_theme.MARGIN, y, content_width, 110)
        )
        table_scroll.setHasVerticalScroller_(True)
        table_scroll.setBorderType_(NSBezelBorder)
        self.replacements_table = NSTableView.alloc().initWithFrame_(
            NSMakeRect(0, 0, content_width, 110)
        )
        self.replacements_table.setUsesAlternatingRowBackgroundColors_(True)
        self.replacements_table.setAllowsMultipleSelection_(False)
        self.replacements_table.setAppearance_(ui_theme.control_appearance())
        self.replacements_table.setAccessibilityLabel_("Text replacements")
        for identifier, width in (("from", 170), ("to", 170), ("match_case", 90)):
            column = NSTableColumn.alloc().initWithIdentifier_(identifier)
            column.headerCell().setStringValue_(REPLACEMENT_COLUMN_TITLES[identifier])
            column.setWidth_(width)
            if identifier == "match_case":
                cell = NSButtonCell.alloc().init()
                cell.setButtonType_(3)  # NSSwitchButton
                cell.setTitle_("")
                column.setDataCell_(cell)
            else:
                column.dataCell().setEditable_(True)
            self.replacements_table.addTableColumn_(column)
        self.replacements_table.setDataSource_(self)
        table_scroll.setDocumentView_(self.replacements_table)
        content_view.addSubview_(table_scroll)
        y += 118

        for title, action, offset, width in (
            ("+", self.addReplacementClicked_, 0, 40),
            ("−", self.removeReplacementClicked_, 48, 40),
            ("Import CSV…", self.importVocabularyClicked_, 100, 120),
            ("Export CSV…", self.exportVocabularyClicked_, 232, 120),
        ):
            button = NSButton.alloc().initWithFrame_(
                NSMakeRect(ui_theme.MARGIN + offset, y, width, 28)
            )
            button.setTitle_(title)
            ui_theme.style_dark_button(button)
            button.setTarget_(self)
            button.setAction_(objc.selector(action, signature=b'v@:@'))
            content_view.addSubview_(button)
        return y + 38

    def numberOfRowsInTableView_(self, table_view):
        """Replacements table data source."""
        return self.vocabulary_section.row_count

    def tableView_objectValueForTableColumn_row_(self, table_view, column, row):
        """One cell of the replacements table."""
        return self.vocabulary_section.value_for(int(row), str(column.identifier()))

    def tableView_setObjectValue_forTableColumn_row_(self, table_view, value, column, row):
        """Write an edited cell back, converting AppKit's value to Python's."""
        identifier = str(column.identifier())
        if identifier == "match_case":
            self.vocabulary_section.set_value(int(row), identifier, bool(int(value)))
            return
        self.vocabulary_section.set_value(int(row), identifier, str(value))

    def addReplacementClicked_(self, sender):
        """Append a blank replacement and put the cursor in its From cell."""
        row = self.vocabulary_section.add_replacement()
        self.replacements_table.reloadData()
        self.replacements_table.selectRowIndexes_byExtendingSelection_(
            NSIndexSet.indexSetWithIndex_(row), False
        )
        self.replacements_table.editColumn_row_withEvent_select_(0, row, None, True)

    def removeReplacementClicked_(self, sender):
        """Delete the selected replacement, then select its neighbour."""
        row = self.replacements_table.selectedRow()
        if row < 0:
            return
        next_row = self.vocabulary_section.remove_replacement(int(row))
        self.replacements_table.reloadData()
        if next_row >= 0:
            self.replacements_table.selectRowIndexes_byExtendingSelection_(
                NSIndexSet.indexSetWithIndex_(next_row), False
            )

    def importVocabularyClicked_(self, sender):
        """Load terms and replacements from a CSV, replacing what is here."""
        panel = NSOpenPanel.openPanel()
        panel.setTitle_("Import vocabulary")
        panel.setPrompt_("Import")
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(False)
        panel.setAllowedFileTypes_(["csv"])
        if panel.runModal() != 1:
            return
        urls = panel.URLs()
        if not urls:
            return
        path = str(urls[0].path())
        try:
            with open(path, encoding="utf-8") as handle:
                self.vocabulary_section.import_text(handle.read())
        except VocabularyError as error:
            ui_alerts.show_alert(
                APP_NAME,
                f"That file could not be imported.\n\n{error}",
                style=NSWarningAlertStyle,
            )
            return
        except OSError as error:
            ui_alerts.show_alert(
                APP_NAME,
                f"That file could not be read.\n\n{error}",
                style=NSWarningAlertStyle,
            )
            return
        self.terms_view.setString_(self.vocabulary_section.terms_text)
        self.replacements_table.reloadData()

    def exportVocabularyClicked_(self, sender):
        """Write the current terms and replacements out as CSV."""
        self.vocabulary_section.set_terms_text(str(self.terms_view.string()))
        panel = NSSavePanel.savePanel()
        panel.setTitle_("Export vocabulary")
        panel.setPrompt_("Export")
        panel.setNameFieldStringValue_(VOCABULARY_CSV_NAME)
        panel.setAllowedFileTypes_(["csv"])
        if panel.runModal() != 1:
            return
        url = panel.URL()
        if url is None:
            return
        try:
            with open(str(url.path()), "w", encoding="utf-8") as handle:
                handle.write(self.vocabulary_section.export_text())
        except OSError as error:
            ui_alerts.show_alert(
                APP_NAME,
                f"That file could not be written.\n\n{error}",
                style=NSWarningAlertStyle,
            )

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
    
    # -- Speech engine section -------------------------------------------

    @objc.python_method
    def _refresh_engine_controls(self):
        """Redraw the popup, the detail line and the two buttons from the model."""
        section = self.engine_section
        section.refresh()
        if self.engine_popup is not None:
            self.engine_popup.removeAllItems()
            for choice in section.choices:
                self.engine_popup.addItemWithTitle_(choice.title)
            self.engine_popup.selectItemAtIndex_(section.selected_index)
        if self.engine_detail_label is not None:
            self.engine_detail_label.setStringValue_(section.detail_line)
        if self.engine_download_button is not None:
            self.engine_download_button.setEnabled_(
                section.can_download and not self.download_controller.is_running
            )
        if self.engine_delete_button is not None:
            self.engine_delete_button.setEnabled_(
                section.can_delete and not self.download_controller.is_running
            )

    @objc.python_method
    def _fill_language_popup(self):
        """Rebuild the popup rows from the language model and select its row."""
        if self.language_popup is None:
            return
        self.language_popup.removeAllItems()
        for title in self.language_section.titles:
            self.language_popup.addItemWithTitle_(title)
        self.language_popup.selectItemAtIndex_(self.language_section.selected_index)

    @objc.python_method
    def engine_reloaded(self, info):
        """The app finished swapping engines: re-offer that engine's languages.

        Without this the popup still lists the previous engine's languages, and
        the user picks from a menu the running model cannot honour.
        """
        try:
            codes = available_languages(info)
        except Exception as error:
            logger.warning("Could not read languages from the new engine: %s", error)
            return
        index = (
            self.language_popup.indexOfSelectedItem()
            if self.language_popup is not None
            else self.language_section.selected_index
        )
        self.language_section = self.language_section.rebuilt(codes, index)
        self._fill_language_popup()

    @objc.python_method
    def _engine_changed(self, engine_id, model_id):
        """Ask the running app to swap engines in the background. No restart.

        Returns None when the app accepted, or the refusal message when it did
        not — the section uses that to leave config alone and put the highlight
        back on the engine that is really running.
        """
        app = _murmur_app_instance()
        reload_engine = getattr(app, "reload_engine", None) if app is not None else None
        if reload_engine is None:
            logger.info(
                "Speech engine set to %s/%s; no running app to reload",
                engine_id,
                model_id,
            )
            return None
        return reload_engine(engine_id, model_id)

    def engineChanged_(self, sender):
        """Popup selection changed: activate the model when it is installed."""
        self.engine_section.select_index(self.engine_popup.indexOfSelectedItem())
        self._refresh_engine_controls()
        refusal = self.engine_section.refusal
        if refusal:
            ui_alerts.show_alert(
                "Speech engine unchanged", refusal, style=NSWarningAlertStyle
            )

    def downloadModelClicked_(self, sender):
        """Fetch the highlighted model, showing a progress sheet."""
        if self.download_controller.is_running:
            return
        choice = self.engine_section.selected_choice
        self._begin_download_sheet(choice)
        self.download_controller.start(choice.model_id, total_bytes=choice.size_bytes)
        self._refresh_engine_controls()

    def cancelDownloadClicked_(self, sender):
        """Stop the running download; the partial file is kept for a resume."""
        self.download_controller.cancel()

    def deleteModelClicked_(self, sender):
        """Remove the highlighted model's files, unless it is the one in use."""
        refusal = self.engine_section.delete()
        if refusal is not None:
            ui_alerts.show_alert(APP_NAME, refusal)
            return
        self._refresh_engine_controls()

    @objc.python_method
    def _begin_download_sheet(self, choice):
        """Show a sheet with a determinate progress bar and a Cancel button."""
        width = DOWNLOAD_SHEET_WIDTH
        height = DOWNLOAD_SHEET_HEIGHT
        sheet = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height),
            NSWindowStyleMaskTitled,
            NSBackingStoreBuffered,
            False,
        )
        sheet.setTitle_("Downloading")
        ui_theme.apply_window_theme(sheet)
        view = sheet.contentView()

        view.addSubview_(self._create_label(
            choice.display_name,
            x=ui_theme.MARGIN,
            y=height - 46,
            width=width - ui_theme.MARGIN * 2,
            bold=True,
            size=13,
        ))

        self.download_status_label = self._create_label(
            "Ready to download",
            x=ui_theme.MARGIN,
            y=height - 70,
            width=width - ui_theme.MARGIN * 2,
            size=11,
            color=ui_theme.muted_text_color(),
        )
        view.addSubview_(self.download_status_label)

        bar = NSProgressIndicator.alloc().initWithFrame_(
            NSMakeRect(ui_theme.MARGIN, height - 96, width - ui_theme.MARGIN * 2, 16)
        )
        bar.setStyle_(0)  # NSProgressIndicatorBarStyle
        bar.setIndeterminate_(False)
        bar.setMinValue_(0.0)
        bar.setMaxValue_(100.0)
        bar.setDoubleValue_(0.0)
        bar.setAppearance_(ui_theme.control_appearance())
        view.addSubview_(bar)
        self.download_progress_bar = bar

        cancel = NSButton.alloc().initWithFrame_(
            NSMakeRect(width - ui_theme.MARGIN - 96, 18, 96, 28)
        )
        cancel.setTitle_("Cancel")
        ui_theme.style_dark_button(cancel)
        cancel.setTarget_(self)
        cancel.setAction_(objc.selector(self.cancelDownloadClicked_, signature=b'v@:@'))
        view.addSubview_(cancel)

        self.download_sheet = sheet
        self.window.beginSheet_completionHandler_(sheet, None)

    @objc.python_method
    def _end_download_sheet(self):
        """Take the sheet down and forget its controls."""
        if self.download_sheet is not None:
            self.window.endSheet_(self.download_sheet)
            self.download_sheet.orderOut_(None)
        self.download_sheet = None
        self.download_status_label = None
        self.download_progress_bar = None

    @objc.python_method
    def _download_state_changed(self, state):
        """Render one download state change. Runs on the main thread."""
        if self.download_status_label is not None:
            self.download_status_label.setStringValue_(state.status_line())
        if self.download_progress_bar is not None:
            self.download_progress_bar.setDoubleValue_(state.percent)
        if state.phase not in (PHASE_DONE, PHASE_FAILED, PHASE_CANCELLED):
            return

        self._end_download_sheet()
        if state.phase == PHASE_DONE:
            self.engine_section.on_download_finished(state.model_id)
        elif state.phase == PHASE_FAILED:
            ui_alerts.show_alert(
                APP_NAME,
                f"{state.model_id} could not be downloaded.\n\n{state.error}",
                style=NSWarningAlertStyle,
            )
        self._refresh_engine_controls()
        # A model can finish downloading while the app is mid-recording, and the
        # swap is then refused. Say so rather than leaving the popup looking
        # like the new model took over.
        refusal = self.engine_section.refusal
        if refusal:
            ui_alerts.show_alert(
                "Speech engine unchanged", refusal, style=NSWarningAlertStyle
            )

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
        """Save settings.

        Only the keys this window owns are written, merged into whatever is on
        disk right now. Writing back the whole config the window loaded when it
        opened reverted everything the app had written in the meantime — the
        engine a finished download activated, ``onboarding_completed`` (so the
        wizard reopened forever), the one-shot notices.

        The speech engine is not saved here either: choosing a model applies at
        once and reloads the engine in the background, so there is nothing to
        defer and no restart to ask for.
        """
        save_audio = self.save_audio_switch.state() == NSOnState
        save_history = self.save_history_switch.state() == NSOnState
        appearance_idx = self.appearance_popup.indexOfSelectedItem()
        self.vocabulary_section.set_terms_text(str(self.terms_view.string()))

        changes = {
            "save_audio": save_audio,
            "save_history": save_history,
            "privacy_mode": not (save_audio or save_history),
            "appearance_mode": ui_theme.APPEARANCE_MODES[appearance_idx],
            HOTKEY_MODE_CONFIG_KEY: HOTKEY_MODES[
                self.hotkey_mode_popup.indexOfSelectedItem()
            ],
            **hotkey_to_config(self.hotkey_binding, label=self.hotkey_label),
            # Empty when the popup never moved: a language is only ever written
            # because the user picked one.
            **self.language_section.changes_for_index(
                self.language_popup.indexOfSelectedItem()
            ),
            **self.vocabulary_section.to_config(),
        }
        # Mutated in place: EngineSectionModel holds this same dict.
        self.config.update(update_config(changes))
        ui_theme.set_appearance_mode(self.config["appearance_mode"])

        app = _murmur_app_instance()
        if app is not None:
            app.reload_hotkey(prompt=False)

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
