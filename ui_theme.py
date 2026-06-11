#!/usr/bin/env python3
"""Shared Murmur UI theme — brand green palette with light/dark/system modes."""

from Foundation import NSAttributedString
from AppKit import NSFontAttributeName, NSForegroundColorAttributeName
from Cocoa import NSAppearance, NSApp, NSColor, NSFont, NSMakeRect, NSRoundedBezelStyle, NSView
from Quartz import CGColorCreateGenericRGB

THEME_VERSION = "murmur-green-2"
APPEARANCE_MODES = ("system", "dark", "light")

MARGIN = 20
FOOTER_HEIGHT = 56

BRAND_GREEN = (0.0, 0.50, 0.30)
BRAND_GREEN_BRIGHT = (0.0, 0.63, 0.38)
BRAND_GREEN_DARK = (0.0, 0.20, 0.12)

_requested_mode = "system"
_effective_mode = "dark"


def _rgb(r, g, b, a=1.0):
    return NSColor.colorWithRed_green_blue_alpha_(r, g, b, a)


def _cg(r, g, b, a=1.0):
    return CGColorCreateGenericRGB(r, g, b, a)


def set_appearance_mode(mode):
    global _requested_mode, _effective_mode
    if mode not in APPEARANCE_MODES:
        mode = "system"
    _requested_mode = mode
    _effective_mode = _resolve_effective_mode(mode)


def appearance_mode():
    return _requested_mode


def _resolve_effective_mode(mode):
    if mode in ("dark", "light"):
        return mode
    try:
        name = str(NSApp.effectiveAppearance().name())
        if "Dark" in name:
            return "dark"
    except Exception:
        pass
    return "light"


def _is_dark():
    return _effective_mode == "dark"


def primary_text_color():
    if _is_dark():
        return _rgb(0.93, 0.96, 0.94)
    return _rgb(0.10, 0.12, 0.11)


def muted_text_color():
    if _is_dark():
        return _rgb(0.62, 0.70, 0.65)
    return _rgb(0.38, 0.42, 0.40)


def subtle_text_color():
    if _is_dark():
        return _rgb(0.45, 0.52, 0.48)
    return _rgb(0.55, 0.58, 0.56)


def panel_background_color():
    if _is_dark():
        return _rgb(0.05, 0.07, 0.06)
    return _rgb(0.96, 0.97, 0.96)


def sidebar_background_color():
    if _is_dark():
        return _cg(0.06, 0.09, 0.07)
    return _cg(0.92, 0.94, 0.92)


def card_background_color():
    if _is_dark():
        return _cg(0.09, 0.12, 0.10)
    return _cg(0.99, 1.0, 0.99)


def row_background_color():
    if _is_dark():
        return _cg(0.05, 0.08, 0.06)
    return _cg(0.94, 0.96, 0.94)


def selected_row_background_color():
    if _is_dark():
        return _cg(0.08, 0.18, 0.12)
    return _cg(0.82, 0.92, 0.86)


def separator_color():
    if _is_dark():
        return _cg(0.14, 0.20, 0.16)
    return _cg(0.82, 0.86, 0.83)


def accent_bar_color():
    return _cg(*BRAND_GREEN_BRIGHT)


def footer_background_color():
    if _is_dark():
        return _cg(0.06, 0.09, 0.07)
    return _cg(0.92, 0.94, 0.92)


def brand_accent_color():
    return _rgb(*BRAND_GREEN_BRIGHT)


def control_appearance():
    if _is_dark():
        return NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua")
    return NSAppearance.appearanceNamed_("NSAppearanceNameAqua")


def apply_window_theme(window):
    global _effective_mode
    _effective_mode = _resolve_effective_mode(_requested_mode)
    appearance = control_appearance()
    if _requested_mode == "system":
        window.setAppearance_(None)
    else:
        window.setAppearance_(appearance)
    window.setBackgroundColor_(panel_background_color())


def _button_title(button, text, color, *, size=13, weight=0.0):
    font = NSFont.systemFontOfSize_weight_(size, weight)
    attrs = {
        NSForegroundColorAttributeName: color,
        NSFontAttributeName: font,
    }
    button.setAttributedTitle_(NSAttributedString.alloc().initWithString_attributes_(text, attrs))


def style_dark_button(button, *, primary=False, font_size=13):
    title = str(button.title() or "")
    button.setBezelStyle_(NSRoundedBezelStyle)
    button.setBordered_(False)
    button.setWantsLayer_(True)
    button.setAppearance_(control_appearance())
    layer = button.layer()
    layer.setCornerRadius_(7)

    if primary:
        layer.setBackgroundColor_(_cg(*BRAND_GREEN))
        layer.setBorderWidth_(0)
        _button_title(button, title, NSColor.whiteColor(), size=font_size, weight=0.2)
    elif _is_dark():
        layer.setBackgroundColor_(_cg(0.10, 0.14, 0.11))
        layer.setBorderWidth_(1)
        layer.setBorderColor_(_cg(*BRAND_GREEN_DARK, 0.9))
        _button_title(button, title, primary_text_color(), size=font_size)
    else:
        layer.setBackgroundColor_(_cg(0.90, 0.93, 0.91))
        layer.setBorderWidth_(1)
        layer.setBorderColor_(_cg(*BRAND_GREEN_DARK, 0.35))
        _button_title(button, title, primary_text_color(), size=font_size)


def style_primary_button(button, *, font_size=13):
    style_dark_button(button, primary=True, font_size=font_size)


def style_popup_on_dark(popup):
    popup.setAppearance_(control_appearance())


def add_horizontal_rule(parent, x, y, width):
    rule = NSView.alloc().initWithFrame_(NSMakeRect(x, y, width, 1))
    rule.setWantsLayer_(True)
    rule.layer().setBackgroundColor_(separator_color())
    parent.addSubview_(rule)
    return rule


def add_footer_bar(parent, width, footer_height=FOOTER_HEIGHT):
    footer = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, footer_height))
    footer.setWantsLayer_(True)
    footer.layer().setBackgroundColor_(footer_background_color())
    parent.addSubview_(footer)
    add_horizontal_rule(parent, 0, footer_height - 1, width)
    return footer
