#!/usr/bin/env python3
"""The contract every Settings tab is written against, plus shared controls.

Two halves:

* :class:`TabContext` and :class:`SettingsTab` — what a tab is given and what
  it must expose. The window builds one context and hands the same one to
  every tab, so no tab reaches for the config, the app or the theme itself.
* ``make_*`` / ``stack_vertical`` — thin wrappers over the AppKit calls
  ``settings_window.py`` used to make inline, so five tabs written by five
  hands still look like one window.

AppKit is imported inside the functions that need it, never at module scope:
the models behind the tabs are plain Python and must stay importable (and
testable) without a window server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

#: Tab identifiers, in the order the window shows them.
TAB_GENERAL = "general"
TAB_ENGINE = "engine"
TAB_SMART = "smart"
TAB_PRIVACY = "privacy"
TAB_ACCOUNT = "account"

TAB_ORDER: tuple[str, ...] = (
    TAB_GENERAL,
    TAB_ENGINE,
    TAB_SMART,
    TAB_PRIVACY,
    TAB_ACCOUNT,
)

#: Padding around a tab's content, matching the old single-page window.
CONTENT_MARGIN = 20

#: Vertical gap between two rows of a tab.
ROW_SPACING = 10


@dataclass
class TabContext:
    """Everything a tab is allowed to reach for.

    ``config`` is the live dict the window loaded; a tab reads it directly and
    writes through ``save``, which merges the changed keys and persists them.
    Handing tabs a writer rather than the file keeps "when is this written"
    one decision, made by the window, not five.

    ``app`` is the running :class:`~murmur.MurmurApp` when Settings was opened
    from the menu bar, and ``None`` when the window runs standalone or under
    test — every use of it is guarded.
    """

    config: dict
    save: Callable[[dict], None]
    app: Any | None
    theme: Any
    engine_info: Any | None = None
    services: dict = field(default_factory=dict)

    def service(self, name: str, default: Any = None) -> Any:
        """One injected provider by name, or ``default`` when absent.

        Providers are optional by design: the account tab has no license
        service under test, and asking for one must not be a crash.
        """
        assert name, "name is required"
        return self.services.get(name, default)

    def app_call(self, method_name: str, *args, **kwargs) -> Any:
        """Call a method on the running app, or do nothing when there is none.

        The single place the tabs are allowed to poke the app from, so "is
        Murmur actually running" is asked once rather than in every handler.
        Returns ``None`` when there is no app or no such method.
        """
        assert method_name, "method_name is required"
        if self.app is None:
            return None
        method = getattr(self.app, method_name, None)
        if method is None:
            return None
        return method(*args, **kwargs)


@runtime_checkable
class SettingsTab(Protocol):
    """One tab of the Settings window.

    ``identifier`` must be one of :data:`TAB_ORDER`; ``title`` is what the tab
    bar shows. ``build`` returns the ``NSView`` for the tab's content and is
    called once; ``refresh`` re-reads config into the controls it created.

    ``close`` is called when the window goes away and is where a tab gives back
    anything that outlives its view: a poll timer, an event monitor, a thread.
    It may be called more than once — closing the window reaches it both
    directly and through the window's delegate — so teardown must be
    idempotent. A tab that holds nothing inherits the no-op from
    :class:`TabLifecycle`; the window tolerates a tab without ``close`` at all,
    so the method is optional to *write* but never optional to *call*.
    """

    identifier: str
    title: str

    def build(self, context: TabContext) -> Any:
        ...

    def refresh(self) -> None:
        ...

    def close(self) -> None:
        ...


class TabLifecycle:
    """The default ``close`` for a tab with nothing to tear down.

    Mixed into a tab class so "this tab holds nothing live" is stated rather
    than left to the reader of a missing method.
    """

    def close(self) -> None:
        """Nothing to give back."""
        return None


# -- Shared controls ---------------------------------------------------------
#
# Every helper takes the theme module (``ui.theme``) rather than importing it,
# so a tab can be rendered against a stub. ``theme=None`` means "the real one".


def _theme_or_default(theme: Any) -> Any:
    """The passed theme, or ``ui.theme`` imported on demand."""
    if theme is not None:
        return theme
    from ui import theme as ui_theme

    return ui_theme


def make_action_target(callback: Callable[[Any], None]) -> Any:
    """An Objective-C target whose action calls ``callback(sender)``.

    AppKit holds targets weakly, so whoever wires this up must keep the
    returned object alive; :func:`make_popup`, :func:`make_checkbox` and
    :func:`make_button` do it for you by associating it with the control.
    """
    assert callable(callback), "callback must be callable"
    target = _action_target_class().alloc().init()
    target.setCallback_(callback)
    return target


_ACTION_TARGET_CLASS: Any = None


def _action_target_class() -> Any:
    """Define (once) the NSObject subclass that forwards actions to Python."""
    global _ACTION_TARGET_CLASS
    if _ACTION_TARGET_CLASS is not None:
        return _ACTION_TARGET_CLASS

    import objc
    from Foundation import NSObject

    class MurmurSettingsActionTarget(NSObject):
        """Forwards ``perform:`` to a Python callable."""

        @objc.python_method
        def setCallback_(self, callback):
            self._callback = callback

        def perform_(self, sender):
            self._callback(sender)

    _ACTION_TARGET_CLASS = MurmurSettingsActionTarget
    return _ACTION_TARGET_CLASS


def bind_action(control: Any, action: Callable[[Any], None] | None) -> Any:
    """Wire ``action`` to ``control`` and keep the target alive with it.

    ``action`` is a plain Python callable taking the sender. Returns the
    target so a caller that prefers to own the reference can hold it.
    """
    if action is None:
        return None
    import objc

    target = make_action_target(action)
    control.setTarget_(target)
    control.setAction_(objc.selector(target.perform_, signature=b"v@:@"))
    # setTarget_ does not retain; the association makes the target live
    # exactly as long as the control that calls it.
    objc.setAssociatedObject(
        control, b"murmur_action_target", target, objc.OBJC_ASSOCIATION_RETAIN
    )
    return target


def make_label(
    text: str,
    theme: Any = None,
    *,
    size: int = 12,
    bold: bool = False,
    color: Any = None,
    wraps: bool = False,
) -> Any:
    """A non-editable, non-bezeled ``NSTextField`` — the window's only label."""
    assert text is not None, "text is required"
    palette = _theme_or_default(theme)
    from Cocoa import NSFont, NSTextField

    label = NSTextField.alloc().init()
    label.setStringValue_(str(text))
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    if bold:
        label.setFont_(NSFont.systemFontOfSize_weight_(size, 0.3))
    else:
        label.setFont_(NSFont.systemFontOfSize_(size))
    label.setTextColor_(color if color is not None else palette.primary_text_color())
    if wraps:
        label.setLineBreakMode_(0)  # NSLineBreakByWordWrapping
        label.cell().setWraps_(True)
    label.sizeToFit()
    return label


def make_section_title(text: str, theme: Any = None) -> Any:
    """The bold 13 pt heading that opens a group of settings."""
    assert text, "text is required"
    return make_label(text, theme, size=13, bold=True)


def make_hint(text: str, theme: Any = None) -> Any:
    """The small muted line under a control that explains what it does."""
    assert text, "text is required"
    palette = _theme_or_default(theme)
    return make_label(text, palette, size=11, color=palette.muted_text_color(), wraps=True)


def make_popup(
    items: list[str] | tuple[str, ...],
    selected: int | str = 0,
    theme: Any = None,
    action: Callable[[Any], None] | None = None,
    *,
    width: int = 220,
) -> Any:
    """A themed ``NSPopUpButton`` holding ``items``.

    ``selected`` is either the index to highlight or one of ``items``; an
    index outside the list is a programming error, not something to clamp.
    ``action`` is a Python callable taking the sender.
    """
    assert items, "a popup needs at least one item"
    palette = _theme_or_default(theme)
    from Cocoa import NSMakeRect, NSPopUpButton

    titles = [str(item) for item in items]
    if isinstance(selected, str):
        assert selected in titles, f"{selected!r} is not one of the popup items"
        index = titles.index(selected)
    else:
        index = int(selected)
        assert 0 <= index < len(titles), f"selected index {index} is out of range"

    popup = NSPopUpButton.alloc().initWithFrame_(NSMakeRect(0, 0, width, 26))
    for title in titles:
        popup.addItemWithTitle_(title)
    popup.selectItemAtIndex_(index)
    palette.style_popup_on_dark(popup)
    bind_action(popup, action)
    return popup


def make_checkbox(
    title: str,
    on: bool = False,
    theme: Any = None,
    action: Callable[[Any], None] | None = None,
) -> Any:
    """A macOS switch-style checkbox. ``on`` is its initial state."""
    assert title, "title is required"
    palette = _theme_or_default(theme)
    from Cocoa import NSMakeRect, NSOffState, NSOnState, NSButton

    checkbox = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 24))
    checkbox.setButtonType_(3)  # NSSwitchButton
    checkbox.setTitle_(str(title))
    checkbox.setState_(NSOnState if on else NSOffState)
    checkbox.setAppearance_(palette.control_appearance())
    checkbox.sizeToFit()
    bind_action(checkbox, action)
    return checkbox


def checkbox_is_on(checkbox: Any) -> bool:
    """Read a checkbox made by :func:`make_checkbox` without importing AppKit."""
    from Cocoa import NSOnState

    return checkbox.state() == NSOnState


def make_button(
    title: str,
    theme: Any = None,
    action: Callable[[Any], None] | None = None,
    *,
    primary: bool = False,
    width: int = 180,
) -> Any:
    """A themed push button. ``primary`` gives it the brand green fill."""
    assert title, "title is required"
    palette = _theme_or_default(theme)
    from Cocoa import NSButton, NSMakeRect

    button = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, width, 28))
    button.setTitle_(str(title))
    if primary:
        palette.style_primary_button(button)
    else:
        palette.style_dark_button(button)
    bind_action(button, action)
    return button


def stack_vertical(views: list[Any], spacing: int = ROW_SPACING) -> Any:
    """A leading-aligned vertical ``NSStackView`` holding ``views`` in order."""
    return _stack(views, spacing, vertical=True)


def stack_horizontal(views: list[Any], spacing: int = 8) -> Any:
    """A vertically centred horizontal ``NSStackView``: a row of controls."""
    return _stack(views, spacing, vertical=False)


def _stack(views: list[Any], spacing: int, *, vertical: bool) -> Any:
    from AppKit import (
        NSLayoutAttributeCenterY,
        NSLayoutAttributeLeading,
        NSStackView,
        NSUserInterfaceLayoutOrientationHorizontal,
        NSUserInterfaceLayoutOrientationVertical,
    )

    assert views is not None, "views is required"
    stack = NSStackView.alloc().init()
    stack.setOrientation_(
        NSUserInterfaceLayoutOrientationVertical
        if vertical
        else NSUserInterfaceLayoutOrientationHorizontal
    )
    stack.setAlignment_(NSLayoutAttributeLeading if vertical else NSLayoutAttributeCenterY)
    stack.setSpacing_(float(spacing))
    stack.setTranslatesAutoresizingMaskIntoConstraints_(False)
    for view in views:
        if view is None:
            continue
        stack.addArrangedSubview_(view)
    return stack
