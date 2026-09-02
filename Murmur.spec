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
- First-party packages (``app``, ``services``, ``engines``, ``ui``, ``cleanup``)
  are enumerated from disk, so a package another agent lands does not need a
  spec edit, and one that does not exist yet costs nothing.

Wave 2 changes:

- The llama.cpp ``llama-server`` binary is bundled as ``bin/llama-server``
  beside ``whisper-server`` (decision D3).
  ``cleanup.llama_server.resolve_llama_server_binary()`` looks for it at
  ``<sys._MEIPASS>/bin/llama-server`` when frozen.
- ``cleanup/prompts/`` is collected as data. ``cleanup.modes`` reads those files
  from disk at import time (``MODES``/``TONES`` load with the module, so a missing
  manifest fails app launch) (``PROMPTS_DIR = Path(__file__).parent / "prompts"``),
  and a frozen module's ``__file__`` is ``<sys._MEIPASS>/cleanup/modes.py`` — so
  the destination has to be ``cleanup/prompts`` exactly, or every mode raises
  ``PromptFileMissingError`` in the bundle and nowhere else.

Wave 5 changes:

- ``murmur.py`` is a 40-line entry point; the app lives in the ``app`` package,
  which is added to the first-party sweep below. Nothing about the analysis
  changes: ``Analysis(["murmur.py"])`` still starts there, and the sweep is what
  guarantees a module reached only through a mixin is collected.
- ``history_window.py``, ``ui_theme.py``, ``ui_alerts.py`` and
  ``transcription_filters.py`` moved into ``ui/`` and ``cleanup/``. They were
  listed twice here — once as data files, because the history window was loaded
  by path, and once as hidden imports, because a root module is invisible to the
  package sweep. Both lists lose them: the sweep collects them as ordinary
  submodules now, and ``app.windows._get_history_module`` imports rather than
  reads them.

Wave 4 changes:

- ``cryptography`` is collected whole. It verifies the Ed25519 signature on a
  Boske lease, and its primitives live in a compiled Rust extension reached
  through ``_cffi_backend``; static analysis finds neither, so a bundle built
  without ``collect_all`` imports ``services.license_service`` happily and then
  fails at the first lease.
- ``Security`` is a hidden import: ``services/keychain.py`` reaches the macOS
  Keychain through PyObjC's Security bindings when they are installed
  (``pyobjc-framework-Security``, see requirements.txt) and through ctypes when
  they are not. Only the PyObjC path needs naming here.
- ``engines/factory.py`` and ``services/engine_router.py`` need no spec edit:
  the first-party sweep below enumerates every submodule of ``engines`` and
  ``services`` from disk.
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

# --- bundled llama.cpp cleanup server (decision D3) -------------------------

LLAMA_SERVER = os.path.join(ROOT, "vendor", "llamacpp", "llama-server")
if not os.path.isfile(LLAMA_SERVER):
    raise SystemExit(
        "Murmur.spec: vendor/llamacpp/llama-server is missing.\n"
        "  Build it first:  bash scripts/tools/fetch_llama.sh\n"
        "  (decision D3 — cleanup runs on a bundled llama-server child process;\n"
        "   without the binary every non-dictation mode skips cleanup.)"
    )

# Prompt templates and manifests are read from disk at runtime by
# cleanup/modes.py, so they must exist as files inside the bundle. The
# destination mirrors the source path exactly; see the module docstring.
PROMPTS_DIR = os.path.join(ROOT, "cleanup", "prompts")
if not os.path.isdir(PROMPTS_DIR):
    raise SystemExit(
        "Murmur.spec: cleanup/prompts is missing — cleanup modes cannot render."
    )

datas = [
    ("cleanup/prompts", "cleanup/prompts"),
    ("assets/icons/logo_menu_template.png", "assets/icons"),
    ("assets/icons/logo_menu_white.png", "assets/icons"),
    ("assets/logos/logo_rounded.png", "assets/logos"),
    ("assets/logos/logo_dock.png", "assets/logos"),
    ("assets/icons/icon_recording.png", "assets/icons"),
    ("assets/icons/icon_processing.png", "assets/icons"),
    ("assets/icons/icon_error.png", "assets/icons"),
]

# Destination "bin" puts them at <sys._MEIPASS>/bin/<name>, which is where both
# resolvers look when frozen.
# ffmpeg and ffprobe used to sit beside whisper-server for the openai-whisper
# adapter. That adapter is archived, nothing left in the app shells out to
# either binary, and whisper.cpp reads the WAV the recorder already writes — so
# they are gone, and with them two Homebrew binaries of someone else's
# provenance in a signed bundle.
binaries = [(WHISPER_SERVER, "bin"), (LLAMA_SERVER, "bin")]

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
    # Launch at login. ``app.decisions.login_item_service`` imports SMAppService inside
    # a function — it does not exist before macOS 13 — and PyInstaller cannot
    # see an import made there, so the framework has to be named here or the
    # checkbox reads "Not available in this build" in the bundle only.
    "ServiceManagement",
    # Wave 4: the Keychain store's PyObjC backend. Named unconditionally on
    # purpose — a "hidden import not found" warning here is the signal that the
    # build machine skipped `pip install -r requirements.txt` and the bundle is
    # about to ship on the ctypes fallback. It is a warning, not an error,
    # because that fallback does work.
    "Security",
    # cryptography's CFFI bridge. collect_all below brings the package and its
    # Rust extension; the backend module is a separate top-level import.
    "_cffi_backend",
]


def local_package(name):
    """Hidden imports for a first-party package that may not exist yet.

    ``engines`` and ``services`` import their members dynamically, so every
    submodule has to be named. ``ui`` and ``cleanup`` arrived in Wave 1 and
    Wave 2 and ``app`` in Wave 5; a package that does not exist yet returns
    nothing instead of a build warning.
    """
    if not os.path.isfile(os.path.join(ROOT, name, "__init__.py")):
        return []
    return [name, *collect_submodules(name)]


for package in ("app", "services", "engines", "ui", "cleanup"):
    hiddenimports += local_package(package)

# --- MLX, Apple Silicon only (decisions D1 and D7) --------------------------

MLX_REQUIRED = ("mlx", "mlx_audio")
#: Runtime companions of mlx-audio. sentencepiece is listed because some
#: tokenizers need it; it is skipped when the environment does not have it.
MLX_OPTIONAL = ("tokenizers", "safetensors", "sentencepiece", "huggingface_hub", "miniaudio")

# cryptography is collected on every architecture: lease verification is not a
# macOS-only or Apple-Silicon-only concern, and its Rust extension does not
# survive static analysis.
collect_packages = ["quickmachotkey", "cryptography"]

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
