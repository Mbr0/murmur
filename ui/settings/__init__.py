#!/usr/bin/env python3
"""The tabbed Settings window: a registry of tabs, one module each.

A tab module owns one file, defines one class satisfying
:class:`~ui.settings.base.SettingsTab`, and calls :func:`register_tab` on it at
import time. The window imports the modules named in :data:`TAB_MODULES` and
shows whatever registered, in :data:`~ui.settings.base.TAB_ORDER` — so a tab
whose module does not exist yet is skipped with a log line rather than taking
the window down with it.

Nothing here imports AppKit.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

from ui.settings.base import (
    TAB_ACCOUNT,
    TAB_ENGINE,
    TAB_GENERAL,
    TAB_ORDER,
    TAB_PRIVACY,
    TAB_SMART,
    SettingsTab,
    TabContext,
    TabLifecycle,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SettingsTab",
    "TabContext",
    "TabLifecycle",
    "TABS",
    "TAB_MODULES",
    "TAB_ORDER",
    "clear_tabs",
    "load_tabs",
    "register_tab",
    "registered_tabs",
]

#: The module that owns each tab. Import order does not matter;
#: :func:`registered_tabs` sorts by ``TAB_ORDER``.
TAB_MODULES: dict[str, str] = {
    TAB_GENERAL: "ui.settings.general_tab",
    TAB_ENGINE: "ui.settings.engine_tab",
    TAB_SMART: "ui.settings.smart_tab",
    TAB_PRIVACY: "ui.settings.privacy_tab",
    TAB_ACCOUNT: "ui.settings.account_tab",
}

#: Registered tab classes, in registration order.
TABS: list[type] = []


def register_tab(cls: type) -> type:
    """Register a tab class. Returns it, so it can be used as a decorator.

    Registering the same identifier twice replaces the earlier class instead
    of stacking a duplicate: a module reloaded in a test must not leave two
    "General" tabs behind.
    """
    assert cls is not None, "a tab class is required"
    identifier = getattr(cls, "identifier", None)
    assert identifier in TAB_ORDER, (
        f"Unknown tab identifier {identifier!r}; expected one of {', '.join(TAB_ORDER)}"
    )
    assert getattr(cls, "title", None), f"tab {identifier!r} needs a title"
    for index, existing in enumerate(TABS):
        if getattr(existing, "identifier", None) == identifier:
            TABS[index] = cls
            return cls
    TABS.append(cls)
    return cls


def registered_tabs() -> tuple[type, ...]:
    """The registered tabs in display order, whatever order they registered in."""
    by_identifier = {cls.identifier: cls for cls in TABS}
    return tuple(by_identifier[key] for key in TAB_ORDER if key in by_identifier)


def clear_tabs() -> None:
    """Empty the registry. For tests that register stand-ins."""
    TABS.clear()


def load_tabs(
    modules: dict[str, str] | None = None,
    *,
    importer: Callable[[str], Any] | None = None,
) -> tuple[type, ...]:
    """Import every tab module, then return the registry in display order.

    A module that is not there yet — Wave 3 writes the five in parallel — is
    logged and skipped, so the window opens with the tabs that do exist. Any
    other import error is left to propagate: a tab that exists but is broken
    is a bug to see, not a tab to quietly drop.

    That distinction is the reason for :func:`_is_missing_tab_module`. A
    ``ModuleNotFoundError`` raised by the tab module's *own* imports looks
    identical from here, and swallowing it would hide a whole working tab
    behind a typo'd import.
    """
    import_module = importer or importlib.import_module
    for identifier, module_name in (modules or TAB_MODULES).items():
        try:
            import_module(module_name)
        except ModuleNotFoundError as error:
            if not _is_missing_tab_module(error, module_name):
                raise
            logger.info(
                "Settings tab %r skipped: %s is not available (%s)",
                identifier,
                module_name,
                error,
            )
    return registered_tabs()


def _is_missing_tab_module(error: ModuleNotFoundError, module_name: str) -> bool:
    """Whether ``error`` says *this tab module* is absent, rather than one of
    the modules it imports.

    ``ModuleNotFoundError.name`` is set by the import machinery to the module
    that could not be found. An error carrying no name proves nothing, so it
    is treated as a real failure and re-raised.
    """
    name = getattr(error, "name", None)
    if not name:
        return False
    return name == module_name or name.startswith(f"{module_name}.")
