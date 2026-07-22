# Murmur — Design Manifest

*App visual overlay. Studio charter: [STUDIO_DESIGN_MANIFEST.md](../../../docs/STUDIO_DESIGN_MANIFEST.md)*

> **Grove register** — macOS menu-bar utility. Local Whisper, forest green chrome.

---

## Register

**Grove**

## Palette (live)

Source: [`ui_theme.py`](../ui_theme.py)

| Role | Light (RGB 0–1) | Target hex |
|------|-----------------|------------|
| Panel bg | ~0.96, 0.97, 0.96 | align `paper-50` |
| Primary CTA | 0, 0.50, 0.30 | migrate toward `#387033` |
| Primary bright | 0, 0.63, 0.38 | accent hover |
| Dark panel | ~0.05, 0.07, 0.06 | align `pine-950` |

Current brand green is slightly brighter than Boske moss — low-priority harmonization to `#387033`.

## Mood

- Menu bar utility — minimal chrome, clear permission errors
- No engagement features; user invokes dictation intentionally

## For agents

AppKit/Cocoa styling only. Keep local-only trust copy aligned with `MANIFEST.md`. Theme constants live in `ui_theme.py`.
