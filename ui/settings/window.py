#!/usr/bin/env python3
"""The tabbed Settings window.

An ``NSTabView`` shell and nothing else: it loads the config, builds one
:class:`~ui.settings.base.TabContext`, and hands it to whichever tabs
registered. Which controls exist, and what they write, is each tab's business.

The window has no Save button. Tabs persist a change the moment it is made,
through the ``save`` callable in the context, which merges the changed keys
into the live config and writes the file. The tab the user was last on is
remembered in ``settings_last_tab``.

AppKit is imported inside the methods that need it, so the module — and
``initial_tab`` with it — stays importable headlessly.

``services`` is the dict forwarded, unchanged, into every tab's
:class:`~ui.settings.base.TabContext`. The keys a tab may look for:

- ``usage``: usage/quota provider (Smart tab).
- ``license``: the Wave 4 licence service (Account tab).
- ``pro_gate``: ``is_pro_feature_enabled(feature) -> bool``, the one question
  the UI asks about entitlement. Absent means every Pro feature is off.
- ``keychain``: a ``SecretStore`` for own-key credentials (Account tab).
- ``scheduler``: schedules a delayed callback, e.g. for device-link polling
  (Account tab).
- ``version``: the app version string (Account tab).
- ``build_info``: the build metadata dict (Account tab).
- ``persistence``: the ``PersistenceService`` (Privacy tab).
- ``audio_dir``: where recorded audio is stored (Privacy tab).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from services.persistence_service import (
    DEFAULT_CONFIG,
    PersistencePaths,
    PersistenceService,
)
from ui.settings import TAB_ORDER, load_tabs
from ui.settings.base import TabContext

logger = logging.getLogger(__name__)

WINDOW_TITLE = "Murmur Settings"
SETTINGS_WIDTH = 520
SETTINGS_HEIGHT = 660

#: Remembers which tab was open, so Settings reopens where the user left it.
CONFIG_LAST_TAB = "settings_last_tab"

CONFIG_FILE = os.path.expanduser("~/.murmur_config.json")
HISTORY_FILE = os.path.expanduser("~/.murmur_history.json")

PERSISTENCE = PersistenceService(
    PersistencePaths(config_file=CONFIG_FILE, history_file=HISTORY_FILE),
    logger=logger,
)


def load_config() -> dict:
    """The stored config, with every default filled in."""
    return PERSISTENCE.load_config(dict(DEFAULT_CONFIG))


def initial_tab(
    config: dict,
    identifiers: tuple[str, ...],
    requested: str | None = None,
) -> str | None:
    """Which tab opens: the one asked for, else the remembered one, else the first.

    A remembered tab whose module is missing (Wave 3 writes the five in
    parallel) falls back to the first available rather than opening nothing.
    """
    assert config is not None, "config is required"
    if not identifiers:
        return None
    if requested is not None and requested in identifiers:
        return requested
    remembered = config.get(CONFIG_LAST_TAB)
    if remembered in identifiers:
        return remembered
    return identifiers[0]


def murmur_app_instance() -> Any | None:
    """The running menu bar app, or ``None`` when Settings runs standalone."""
    for module_name in ("murmur", "__main__"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        app = getattr(module, "APP_INSTANCE", None)
        if app is not None:
            return app
    return None


def engine_info_for(app: Any | None) -> Any | None:
    """The loaded engine's ``EngineInfo``, or ``None``.

    The only place the window touches an engine; tabs read it off the context.
    """
    engine = getattr(app, "engine", None) if app is not None else None
    if engine is None:
        return None
    try:
        return engine.info()
    except Exception as error:  # pragma: no cover - defensive around engine code
        logger.warning("Could not read engine info: %s", error)
        return None


class SettingsWindowController:
    """Owns the window, the config, and the tab instances.

    Plain Python rather than an ``NSObject`` subclass so the module imports
    without AppKit; the one place that needs an Objective-C object — the tab
    view's delegate — makes one on demand.
    """

    def __init__(
        self,
        *,
        app: Any | None = None,
        config: dict | None = None,
        save: Any | None = None,
        theme: Any | None = None,
        services: dict | None = None,
        engine_info: Any | None = None,
    ) -> None:
        self.app = app if app is not None else murmur_app_instance()
        self.config = config if config is not None else load_config()
        self._save_config = save if save is not None else PERSISTENCE.save_config
        self._theme = theme
        self._services = services or {}
        self._engine_info = engine_info if engine_info is not None else engine_info_for(self.app)
        self.tabs: dict[str, Any] = {}
        self.identifiers: tuple[str, ...] = ()
        self.window = None
        self.tab_view = None
        self._delegate = None
        self._window_delegate = None
        #: The one context the tabs were built with, kept so a live engine swap
        #: can update it in place — the tabs hold this same object.
        self._live_context: TabContext | None = None

    # -- config ----------------------------------------------------------

    def save(self, changed: dict) -> None:
        """Merge a tab's changed keys into the live config and persist it."""
        assert changed is not None, "changed is required"
        if not changed:
            return
        self.config.update(changed)
        self._save_config(self.config)

    @property
    def theme(self) -> Any:
        """``ui_theme``, imported on first use so the module stays headless."""
        if self._theme is None:
            import ui_theme

            self._theme = ui_theme
        return self._theme

    def context(self) -> TabContext:
        """The single context every tab is built with."""
        return TabContext(
            config=self.config,
            save=self.save,
            app=self.app,
            theme=self.theme,
            engine_info=self._engine_info,
            services=self._services,
        )

    # -- tabs ------------------------------------------------------------

    def build_tabs(self) -> tuple[str, ...]:
        """Instantiate every registered tab in display order."""
        self.tabs = {}
        for tab_class in load_tabs():
            self.tabs[tab_class.identifier] = tab_class()
        self.identifiers = tuple(
            key for key in TAB_ORDER if key in self.tabs
        )
        if not self.identifiers:
            logger.warning("No Settings tabs registered; the window will be empty")
        return self.identifiers

    @property
    def selected_tab(self) -> str | None:
        """The identifier of the tab on screen, or ``None`` before ``show``."""
        if self.tab_view is None:
            return None
        item = self.tab_view.selectedTabViewItem()
        if item is None:
            return None
        return str(item.identifier())

    def select_tab(self, identifier: str | None) -> None:
        """Show one tab and remember it. Unknown identifiers are ignored."""
        if identifier is None or self.tab_view is None:
            return
        if identifier not in self.identifiers:
            logger.info("Settings tab %r is not available", identifier)
            return
        self.tab_view.selectTabViewItemWithIdentifier_(identifier)
        self.remember_tab(identifier)

    def remember_tab(self, identifier: str | None) -> None:
        """Persist ``settings_last_tab`` when it actually moved."""
        if identifier is None or self.config.get(CONFIG_LAST_TAB) == identifier:
            return
        self.save({CONFIG_LAST_TAB: identifier})

    # -- window ----------------------------------------------------------

    def show(self, tab: str | None = None) -> Any:
        """Create the window if needed, select a tab, and bring it forward."""
        from Cocoa import NSApp

        if self.window is None:
            self.create_window()
        self.select_tab(initial_tab(self.config, self.identifiers, tab))
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        return self.window

    def create_window(self) -> Any:
        """Build the window and fill its tab view. Called once."""
        from Cocoa import (
            NSBackingStoreBuffered,
            NSMakeRect,
            NSTabView,
            NSTabViewItem,
            NSWindow,
            NSWindowStyleMaskClosable,
            NSWindowStyleMaskTitled,
        )

        theme = self.theme
        theme.set_appearance_mode(self.config.get("appearance_mode", "system"))

        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, SETTINGS_WIDTH, SETTINGS_HEIGHT),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_(WINDOW_TITLE)
        self.window.center()
        self.window.setReleasedWhenClosed_(False)
        self.window.setContentMinSize_((SETTINGS_WIDTH, SETTINGS_HEIGHT))
        theme.apply_window_theme(self.window)

        content = self.window.contentView()
        frame = content.frame()
        self.tab_view = NSTabView.alloc().initWithFrame_(
            NSMakeRect(0, 0, frame.size.width, frame.size.height)
        )
        self.tab_view.setAutoresizingMask_(2 | 16)  # width | height sizable
        self.tab_view.setAppearance_(theme.control_appearance())

        context = self.context()
        self._live_context = context
        self.build_tabs()
        for identifier in self.identifiers:
            tab = self.tabs[identifier]
            item = NSTabViewItem.alloc().initWithIdentifier_(identifier)
            item.setLabel_(tab.title)
            item.setView_(tab.build(context))
            self.tab_view.addTabViewItem_(item)

        self._delegate = _make_tab_delegate(self.remember_tab)
        self.tab_view.setDelegate_(self._delegate)
        self._window_delegate = _make_window_delegate(self.window_will_close)
        self.window.setDelegate_(self._window_delegate)
        content.addSubview_(self.tab_view)
        return self.window

    # -- the running app talking back --------------------------------------

    def engine_reloaded(self, info: Any) -> None:
        """The app finished swapping engines: tell the tabs about the new one.

        The window reads the engine once, when it opens, and hands that
        ``EngineInfo`` to every tab in the one context they share. A background
        swap makes it stale — the Language popup would go on offering the old
        engine's languages — so the context is updated in place and each tab is
        asked to re-read it. Called on the main thread by the app.
        """
        self._engine_info = info
        if self._live_context is not None:
            self._live_context.engine_info = info
        for identifier, tab in self.tabs.items():
            refresh = getattr(tab, "refresh", None)
            if refresh is None:
                continue
            try:
                refresh()
            except Exception as error:  # noqa: BLE001 - one tab must not stop the rest
                logger.warning(
                    "Settings tab %r could not follow the engine swap: %s", identifier, error
                )

    # -- closing ---------------------------------------------------------

    def close_tabs(self) -> None:
        """Give back everything the tabs hold open.

        The Account tab is polling Boske on a timer and the General tab may be
        holding a keyboard event monitor; both outlive the view, and a timer
        left armed keeps refreshing a window nobody can see. ``close`` is
        optional to write — a tab holding nothing needs none — so it is asked
        for rather than assumed, and a tab that fails to close is logged rather
        than allowed to keep the window open.
        """
        for identifier, tab in self.tabs.items():
            close = getattr(tab, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as error:  # noqa: BLE001 - closing must always finish
                logger.warning("Settings tab %r did not close cleanly: %s", identifier, error)

    def window_will_close(self) -> None:
        """The window is going away, by whatever route brought it here.

        Called by the window's own delegate, so closing from the title bar —
        or from anywhere that closes the ``NSWindow`` directly — tears the tabs
        down exactly as :meth:`close` does.
        """
        self.remember_tab(self.selected_tab)
        self.close_tabs()

    def close(self) -> None:
        """Remember the open tab, tear the tabs down, close the window.

        The object survives: reopening rebuilds nothing but re-shows the same
        window, and the tabs come back in whatever state ``close`` left them.
        """
        self.window_will_close()
        if self.window is not None:
            self.window.close()


_TAB_DELEGATE_CLASS: Any = None


def _make_tab_delegate(on_select) -> Any:
    """An ``NSTabView`` delegate that reports the newly selected identifier."""
    global _TAB_DELEGATE_CLASS
    if _TAB_DELEGATE_CLASS is None:
        import objc
        from Foundation import NSObject

        class MurmurSettingsTabDelegate(NSObject):
            @objc.python_method
            def setCallback_(self, callback):
                self._callback = callback

            def tabView_didSelectTabViewItem_(self, tab_view, item):
                if item is None:
                    return
                self._callback(str(item.identifier()))

        _TAB_DELEGATE_CLASS = MurmurSettingsTabDelegate

    delegate = _TAB_DELEGATE_CLASS.alloc().init()
    delegate.setCallback_(on_select)
    return delegate


_WINDOW_DELEGATE_CLASS: Any = None


def _make_window_delegate(on_will_close) -> Any:
    """An ``NSWindow`` delegate that reports the window closing.

    The red button in the title bar closes the window without going through
    :meth:`SettingsWindowController.close`, and so does anything that calls
    ``window.close()`` directly. Without this, a tab's timers and event
    monitors would simply keep running.
    """
    global _WINDOW_DELEGATE_CLASS
    if _WINDOW_DELEGATE_CLASS is None:
        import objc
        from Foundation import NSObject

        class MurmurSettingsWindowDelegate(NSObject):
            @objc.python_method
            def setCallback_(self, callback):
                self._callback = callback

            def windowWillClose_(self, _notification):
                self._callback()

        _WINDOW_DELEGATE_CLASS = MurmurSettingsWindowDelegate

    delegate = _WINDOW_DELEGATE_CLASS.alloc().init()
    delegate.setCallback_(on_will_close)
    return delegate


#: The one open Settings window, so the menu item reuses it instead of
#: stacking a second copy on top of the first.
_CONTROLLER: SettingsWindowController | None = None


def open_settings(
    app: Any = None,
    tab: str | None = None,
    services: dict | None = None,
) -> SettingsWindowController:
    """Open (or raise) the Settings window, optionally on a given tab.

    ``services`` reaches every tab's :class:`~ui.settings.base.TabContext`;
    see the module docstring for the keys tabs look for.
    """
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = SettingsWindowController(app=app, services=services)
    elif app is not None:
        _CONTROLLER.app = app
    _CONTROLLER.show(tab)
    return _CONTROLLER


def current_controller() -> SettingsWindowController | None:
    """The Settings window's controller, or ``None`` before it is first opened.

    How the running app reaches an open window — to tell it the engine was
    swapped, say — without importing the window's private state.
    """
    return _CONTROLLER
