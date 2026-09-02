#!/usr/bin/env python3
"""Native alerts with Murmur branding (avoids the default Python icon)."""

import os
import sys

from Cocoa import (
    NSAlert,
    NSAlertFirstButtonReturn,
    NSAlertSecondButtonReturn,
    NSImage,
    NSInformationalAlertStyle,
    NSWarningAlertStyle,
)

ICON_CANDIDATES = (
    "assets/logos/logo_rounded.png",
    "assets/logos/logo_dock.png",
    "assets/logos/logo.png",
)
_icon_cache = None

#: The checkout this file lives in. Wave 5 moved the module from the repo root
#: into ``ui/``, so its own directory is one level too deep for ``assets/``: a
#: source run would have looked in ``ui/assets/logos/`` and quietly shown every
#: alert with the default Python icon. The bundle was never affected — it reads
#: ``sys._MEIPASS`` — which is exactly why this would not have been noticed.
_SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(_SOURCE_ROOT, relative_path)


def app_icon_image():
    """Load Murmur icon once for alert dialogs."""
    global _icon_cache
    if _icon_cache is not None:
        return _icon_cache

    for name in ICON_CANDIDATES:
        path = resource_path(name)
        if not os.path.exists(path):
            continue
        image = NSImage.alloc().initByReferencingFile_(os.path.abspath(path))
        if image is None or image.size().width <= 0:
            continue
        _icon_cache = image
        return _icon_cache
    return None


def configure_alert(alert):
    icon = app_icon_image()
    if icon is not None:
        alert.setIcon_(icon)


def show_alert(title, message, *, style=NSInformationalAlertStyle):
    alert = NSAlert.alloc().init()
    alert.setMessageText_(title)
    alert.setInformativeText_(message)
    alert.setAlertStyle_(style)
    configure_alert(alert)
    alert.runModal()


def show_confirm(title, message, ok="OK", cancel="Cancel", *, style=NSWarningAlertStyle):
    """Return True when the primary (OK) button is clicked."""
    alert = NSAlert.alloc().init()
    alert.setMessageText_(title)
    alert.setInformativeText_(message)
    alert.setAlertStyle_(style)
    alert.addButtonWithTitle_(ok)
    alert.addButtonWithTitle_(cancel)
    configure_alert(alert)
    return alert.runModal() == NSAlertFirstButtonReturn
