# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Murmur.app"""

import os
import shutil

from PyInstaller.utils.hooks import collect_all

import whisper

whisper_assets_path = os.path.join(os.path.dirname(whisper.__file__), "assets")
ffmpeg_path = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
ffprobe_path = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

datas = [
    ("assets/icons/logo_menu_template.png", "assets/icons"),
    ("assets/icons/logo_menu_white.png", "assets/icons"),
    ("assets/logos/logo_rounded.png", "assets/logos"),
    ("assets/logos/logo_dock.png", "assets/logos"),
    ("assets/icons/icon_recording.png", "assets/icons"),
    ("assets/icons/icon_processing.png", "assets/icons"),
    ("assets/icons/icon_error.png", "assets/icons"),
    ("history_window.py", "."),
    ("settings_window.py", "."),
    ("ui_theme.py", "."),
    ("ui_alerts.py", "."),
    ("transcription_filters.py", "."),
    (whisper_assets_path, "whisper/assets"),
]

binaries = []
if os.path.isfile(ffmpeg_path):
    binaries.append((ffmpeg_path, "."))
if os.path.isfile(ffprobe_path):
    binaries.append((ffprobe_path, "."))

hiddenimports = [
    "rumps",
    "whisper",
    "torch",
    "sounddevice",
    "scipy",
    "numpy",
    "pyperclip",
    "quickmachotkey",
    "quickmachotkey._MinimalHIToolbox",
    "quickmachotkey.constants",
    "PyObjCTools",
    "PyObjCTools.AppHelper",
    "objc",
    "Cocoa",
    "AppKit",
    "Foundation",
    "Quartz",
    "ui_theme",
    "ui_alerts",
    "transcription_filters",
    "services",
    "services.audio_capture_service",
    "services.hotkey_service",
    "ApplicationServices",
    "Foundation",
    "services.model_profile_service",
    "services.persistence_service",
    "services.text_insertion_service",
    "services.transcription_service",
]

for package in ("whisper", "torch", "quickmachotkey"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    ["murmur.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Murmur",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["assets/Murmur.icns"],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Murmur",
)
app = BUNDLE(
    coll,
    name="Murmur.app",
    icon="assets/Murmur.icns",
    bundle_identifier="com.canopystudio.murmur",
    info_plist={
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
        "LSUIElement": True,
        "LSBackgroundOnly": False,
        "NSMicrophoneUsageDescription": "Murmur needs microphone access to transcribe your speech locally.",
        "NSAccessibilityUsageDescription": "Murmur needs Accessibility access to paste transcribed text at your cursor.",
    },
)
