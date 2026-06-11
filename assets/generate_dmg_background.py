#!/usr/bin/env python3
"""Generate the Murmur DMG background with drag-to-Applications artwork."""

from __future__ import annotations

import math
import os
import sys

from AppKit import (
    NSBezierPath,
    NSBitmapImageRep,
    NSColor,
    NSCompositingOperationSourceOver,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSImage,
    NSMakePoint,
    NSMakeRect,
    NSMakeSize,
    NSMutableDictionary,
    NSOffState,
    NSPNGFileType,
    NSString,
)
from Foundation import NSAttributedString

WIDTH = 660
HEIGHT = 400
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dmg_background.png")

BRAND = (0.0, 0.50, 0.30, 1.0)
BRAND_BRIGHT = (0.0, 0.63, 0.38, 1.0)
BG_TOP = (0.96, 0.98, 0.97, 1.0)
BG_BOTTOM = (0.90, 0.95, 0.92, 1.0)
TEXT = (0.12, 0.16, 0.14, 1.0)
MUTED = (0.40, 0.46, 0.42, 1.0)


def _color(rgba):
    return NSColor.colorWithRed_green_blue_alpha_(*rgba)


def _draw_gradient(image):
    image.lockFocus()
    for y in range(HEIGHT):
        t = y / max(HEIGHT - 1, 1)
        r = BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t
        g = BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t
        b = BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t
        _color((r, g, b, 1.0)).setFill()
        NSBezierPath.fillRect_(NSMakeRect(0, y, WIDTH, 1))
    image.unlockFocus()


def _draw_arrow(image):
    image.lockFocus()
    start = NSMakePoint(250, 205)
    end = NSMakePoint(410, 205)
    path = NSBezierPath.bezierPath()
    path.setLineWidth_(4.0)
    _color(BRAND).setStroke()
    path.moveToPoint_(start)
    path.lineToPoint_(end)
    path.stroke()

    angle = math.atan2(end.y - start.y, end.x - start.x)
    head_len = 16
    head_angle = math.pi / 7
    for sign in (1, -1):
        wing = NSMakePoint(
            end.x - head_len * math.cos(angle - sign * head_angle),
            end.y - head_len * math.sin(angle - sign * head_angle),
        )
        head = NSBezierPath.bezierPath()
        head.setLineWidth_(4.0)
        _color(BRAND_BRIGHT).setStroke()
        head.moveToPoint_(end)
        head.lineToPoint_(wing)
        head.stroke()
    image.unlockFocus()


def _draw_text(image):
    image.lockFocus()
    title_attrs = NSMutableDictionary.dictionary()
    title_attrs[NSFontAttributeName] = NSFont.systemFontOfSize_weight_(24, 0.3)
    title_attrs[NSForegroundColorAttributeName] = _color(TEXT)
    title = NSAttributedString.alloc().initWithString_attributes_(
        "Drag Murmur to Applications",
        title_attrs,
    )
    title_size = title.size()
    title.drawAtPoint_(
        NSMakePoint((WIDTH - title_size.width) / 2, HEIGHT - 72)
    )

    subtitle_attrs = NSMutableDictionary.dictionary()
    subtitle_attrs[NSFontAttributeName] = NSFont.systemFontOfSize_(13)
    subtitle_attrs[NSForegroundColorAttributeName] = _color(MUTED)
    subtitle = NSAttributedString.alloc().initWithString_attributes_(
        "Then open Murmur from Applications and grant microphone access.",
        subtitle_attrs,
    )
    subtitle_size = subtitle.size()
    subtitle.drawAtPoint_(
        NSMakePoint((WIDTH - subtitle_size.width) / 2, HEIGHT - 98)
    )
    image.unlockFocus()


def main() -> int:
    image = NSImage.alloc().initWithSize_(NSMakeSize(WIDTH, HEIGHT))
    _draw_gradient(image)
    _draw_arrow(image)
    _draw_text(image)

    rep = NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
    if rep is None:
        print("Failed to encode DMG background PNG.", file=sys.stderr)
        return 1

    png_data = rep.representationUsingType_properties_(NSPNGFileType, {})
    if png_data is None:
        print("Failed to encode DMG background PNG.", file=sys.stderr)
        return 1

    with open(OUTPUT, "wb") as handle:
        handle.write(png_data.bytes())
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
