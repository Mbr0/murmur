"""The window controllers the menu bar opens, and the theme they are built with.

Every window module is imported inside the function that opens it: PyObjC
classes cannot be redefined once loaded, and a menu bar that never opens History
should not pay for its class definitions. So this module keeps its own imports
headless, and the windows arrive on first use.
"""

import sys

from services.persistence_service import DEFAULT_CONFIG

from app import config as app_config
from app.config import PERSISTENCE, logger

# Store window controller references to prevent garbage collection
_window_controllers = []
_history_module = None


def _get_history_module():
    """Load the history window module.

    A plain import since Wave 5: ``history_window.py`` moved into the ``ui``
    package, so Python's own module cache gives what the hand-rolled loader gave
    — one instance of every PyObjC class, for the life of the process — and
    PyInstaller sweeps it as a module rather than shipping it as a data file.
    """
    global _history_module

    from ui import history_window

    _history_module = history_window
    return _history_module


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

    if "ui.theme" in sys.modules:
        importlib.reload(sys.modules["ui.theme"])
    from ui import theme as ui_theme
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

def show_settings_window_direct(tab=None):
    """Show the tabbed Settings window in this process, optionally on one tab.

    The window is a singleton owned by :mod:`ui.settings.window`: reopening
    raises the one that exists rather than stacking a second copy, so there is
    nothing here to cache or to close first.
    """
    logger.info("show_settings_window_direct called (tab=%s)", tab)
    _reload_ui_theme()

    from ui.settings.window import open_settings

    app = app_config.APP_INSTANCE
    services = app._settings_services() if app is not None else None
    return open_settings(app, tab=tab, services=services)
