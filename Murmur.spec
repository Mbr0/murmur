# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import shutil
import os

# Find ffmpeg path
ffmpeg_path = shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg'
ffprobe_path = shutil.which('ffprobe') or '/opt/homebrew/bin/ffprobe'

# Find whisper assets path dynamically
import whisper
whisper_assets_path = os.path.join(os.path.dirname(whisper.__file__), 'assets')

datas = [('logo_menu_white.png', '.'), ('logo_rounded.png', '.'), ('icon_recording.png', '.'), ('icon_processing.png', '.'), ('icon_error.png', '.'), ('history_window.py', '.'), ('settings_window.py', '.'), (whisper_assets_path, 'whisper/assets')]
binaries = [(ffmpeg_path, '.'), (ffprobe_path, '.')]
hiddenimports = ['rumps', 'whisper', 'torch', 'sounddevice', 'scipy', 'numpy', 'pynput', 'PyObjCTools', 'PyObjCTools.AppHelper']
tmp_ret = collect_all('whisper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['murmur.py'],
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
    name='Murmur',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['Murmur.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Murmur',
)
app = BUNDLE(
    coll,
    name='Murmur.app',
    icon='Murmur.icns',
    bundle_identifier='com.canopystudio.murmur',
    info_plist={
        'LSUIElement': True,  # Make it a menu bar app (no dock icon)
        'LSBackgroundOnly': False,
        'NSMicrophoneUsageDescription': 'Murmur needs microphone access to transcribe speech.',
        'NSAppleEventsUsageDescription': 'Murmur needs to simulate keyboard input to paste transcribed text.',
    },
)
