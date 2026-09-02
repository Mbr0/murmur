# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Murmur.app

Wave 1d changes:

- ``torch`` and ``openai-whisper`` are gone. Every candidate for decision D1
  drops them, so they are excluded outright rather than merely not collected —
  ``mlx_audio`` carries TTS modules that import torch, and a torch left in the
  build environment would otherwise be dragged back in through them.
- ``ffmpeg`` and ``ffprobe`` are gone with them: they were bundled only for
  that adapter, and nothing in the app shells out to either any more.
- The whisper.cpp ``whisper-server`` binary is bundled as ``bin/whisper-server``
  (decision D2). ``engines.whispercpp.resolve_whisper_server_binary()`` looks
  for it at ``<sys._MEIPASS>/bin/whisper-server`` when frozen.
- MLX wheels are collected on Apple Silicon only (decision D7: Intel Macs run
  whisper.cpp).
- First-party packages (``services``, ``engines``, ``ui``, ``cleanup``) are
  enumerated from disk, so a package another Wave 1 agent lands does not need a
  spec edit, and one that does not exist yet costs nothing.
"""

import importlib.util
import os
import platform

from PyInstaller.utils.hooks import collect_all, collect_submodules

try:
    ROOT = os.path.abspath(SPECPATH)  # noqa: F821 - injected by PyInstaller
except NameError:  # pragma: no cover - only when the spec is read outside PyInstaller
    ROOT = os.path.abspath(os.getcwd())

IS_ARM64 = platform.machine() == "arm64"

# --- bundled whisper.cpp server (decision D2) -------------------------------

WHISPER_SERVER = os.path.join(ROOT, "vendor", "whispercpp", "whisper-server")
if not os.path.isfile(WHISPER_SERVER):
    raise SystemExit(
        "Murmur.spec: vendor/whispercpp/whisper-server is missing.\n"
        "  Build it first:  bash scripts/tools/fetch_whispercpp.sh\n"
        "  (decision D2 — the app talks HTTP to a bundled whisper-server child\n"
        "   process; without the binary the default engine cannot start.)"
    )

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
]

# Destination "bin" puts it at <sys._MEIPASS>/bin/whisper-server.
# ffmpeg and ffprobe used to sit beside it for the openai-whisper adapter. That
# adapter is archived, nothing left in the app shells out to either binary, and
# whisper.cpp reads the WAV the recorder already writes — so they are gone, and
# with them two Homebrew binaries of someone else's provenance in a signed bundle.
binaries = [(WHISPER_SERVER, "bin")]

hiddenimports = [
    "rumps",
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
    "ApplicationServices",
    "ui_theme",
    "ui_alerts",
    "transcription_filters",
]


def local_package(name):
    """Hidden imports for a first-party package that may not exist yet.

    ``engines`` and ``services`` import their members dynamically, so every
    submodule has to be named. ``ui`` and ``cleanup`` arrive later in Wave 1
    and Wave 2; until then this returns nothing instead of a build warning.
    """
    if not os.path.isfile(os.path.join(ROOT, name, "__init__.py")):
        return []
    return [name, *collect_submodules(name)]


for package in ("services", "engines", "ui", "cleanup"):
    hiddenimports += local_package(package)

# --- MLX, Apple Silicon only (decisions D1 and D7) --------------------------

MLX_REQUIRED = ("mlx", "mlx_audio")
#: Runtime companions of mlx-audio. sentencepiece is listed because some
#: tokenizers need it; it is skipped when the environment does not have it.
MLX_OPTIONAL = ("tokenizers", "safetensors", "sentencepiece", "huggingface_hub", "miniaudio")

collect_packages = ["quickmachotkey"]

if IS_ARM64:
    missing = [name for name in MLX_REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "Murmur.spec: missing Apple Silicon speech dependencies: "
            + ", ".join(missing)
            + "\n  Install them first:  pip install -r requirements.txt"
        )
    collect_packages += list(MLX_REQUIRED)
    collect_packages += [name for name in MLX_OPTIONAL if importlib.util.find_spec(name) is not None]
    # transformers is pulled in by mlx-audio; hooks-contrib ships a hook for it,
    # so it is left to static analysis rather than collected wholesale (its
    # module tree is thousands of model files we never load).
    hiddenimports.append("transformers")

for package in collect_packages:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# Torch and openai-whisper must not come back through a transitive import.
excludes = [
    "torch",
    "torchaudio",
    "torchvision",
    "whisper",
    "tensorflow",
    "numba",
    "llvmlite",
    "tiktoken",
]

a = Analysis(
    ["murmur.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

def drop_torch_metadata(toc):
    """Remove torch ``.dist-info`` folders from the collected data files.

    hooks-contrib's transformers hook copies the metadata of every dependency
    it finds in the build environment, torch included when a developer still
    has it installed for the Wave 0 bake-off. The package itself is excluded,
    so the metadata is only misleading — it makes the bundle look like it
    carries torch when it does not.
    """
    kept = []
    for entry in toc:
        top = str(entry[0]).replace("\\", "/").split("/")[0]
        if top.endswith(".dist-info") and top.split("-")[0] in ("torch", "torchvision", "torchaudio"):
            continue
        kept.append(entry)
    return kept


a.datas = drop_torch_metadata(a.datas)

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
