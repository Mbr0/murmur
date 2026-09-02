"""Paths, constants, logging and the singletons the whole app shares.

The first module every other one imports, and the only one that does work when
it is imported: it configures logging, migrates the pre-Murmur data files and
makes sure the audio directory exists — exactly what ``murmur.py`` did at import
before Wave 5 split it up.

No AppKit here, deliberately. This module is imported by tests that never open a
window, and by :mod:`app.decisions`, which has to stay pure.
"""

import logging
import os
import shutil
import sys

from cleanup.llama_server import CLEANUP_MODEL_SPEC
from engines.model_store import CATALOG, ModelStore
from services.persistence_service import PersistencePaths, PersistenceService

#: The checkout (or the unpacked bundle) this file lives in. ``resource_path``
#: resolves assets against it, so moving this module into ``app/`` does not move
#: the assets: ``__file__`` is now one directory deeper than ``murmur.py`` was.
_SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _configure_logging() -> logging.Logger:
    """Configure production-safe logging (no transcription text in logs)."""
    is_bundled = hasattr(sys, "_MEIPASS")
    debug_flag_file = os.path.expanduser("~/.murmur_debug")
    debug_enabled = (
        os.environ.get("MURMUR_DEBUG", "").lower() in ("1", "true", "yes")
        or os.path.isfile(debug_flag_file)
    )
    level = logging.DEBUG if debug_enabled else logging.WARNING

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if debug_enabled and not is_bundled:
        handlers.append(logging.FileHandler("/tmp/murmur_debug.log"))
    else:
        log_dir = os.path.expanduser("~/Library/Logs/Murmur")
        try:
            os.makedirs(log_dir, mode=0o700, exist_ok=True)
            log_path = os.path.join(log_dir, "murmur.log")
            handlers.append(logging.FileHandler(log_path))
        except OSError:
            pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    # Named, not ``__name__``: the records used to be tagged "murmur" when the
    # app was imported and "__main__" when it was run. One name for both, and
    # the same one every log filter and test already expects.
    app_logger = logging.getLogger("murmur")

    for handler in handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        log_path = handler.baseFilename
        try:
            os.chmod(log_path, 0o600)
        except OSError:
            pass
        log_dir = os.path.dirname(log_path)
        if os.path.isdir(log_dir):
            try:
                os.chmod(log_dir, 0o700)
            except OSError:
                pass

    return app_logger

logger = _configure_logging()

if hasattr(sys, '_MEIPASS'):
    logger.info(f"Added bundled resources to PATH: {sys._MEIPASS}")

# Settings
SAMPLE_RATE = 16000
APP_NAME = "Murmur"
APP_VERSION = "1.0.0"

#: Length of the wizard's "Try it" recording, in seconds.
ONBOARDING_TEST_SECONDS = 4.0

#: How long the batch path waits for the live decoder to finish its last
#: partial before giving up on it and transcribing the recorded file instead.
STREAM_JOIN_TIMEOUT_S = 10.0

# Config file for settings
CONFIG_FILE = os.path.expanduser("~/.murmur_config.json")

# History and audio storage
HISTORY_FILE = os.path.expanduser("~/.murmur_history.json")
AUDIO_DIR = os.path.expanduser("~/.murmur_audio")
LEGACY_CONFIG_FILE = os.path.expanduser("~/.mywhisper_config.json")
LEGACY_HISTORY_FILE = os.path.expanduser("~/.mywhisper_history.json")
LEGACY_AUDIO_DIR = os.path.expanduser("~/.mywhisper_audio")
PERSISTENCE_PATHS = PersistencePaths(config_file=CONFIG_FILE, history_file=HISTORY_FILE)
PERSISTENCE = PersistenceService(paths=PERSISTENCE_PATHS, logger=logger)
APP_INSTANCE = None


def migrate_legacy_data():
    """Migrate legacy MyWhisper local data to Murmur paths once."""
    migrations = [
        (LEGACY_CONFIG_FILE, CONFIG_FILE, False),
        (LEGACY_HISTORY_FILE, HISTORY_FILE, False),
        (LEGACY_AUDIO_DIR, AUDIO_DIR, True),
    ]
    for legacy_path, new_path, is_directory in migrations:
        if os.path.exists(new_path) or not os.path.exists(legacy_path):
            continue
        try:
            if is_directory:
                shutil.move(legacy_path, new_path)
            else:
                shutil.copy2(legacy_path, new_path)
            logger.info(f"Migrated local data from {legacy_path} to {new_path}")
        except OSError as error:
            logger.error(f"Failed to migrate local data from {legacy_path} to {new_path}: {error}")


migrate_legacy_data()

# The engine and model come from config at load time (see MurmurApp.load_model);
# nothing about the speech engine is decided at import.
PERSISTENCE.ensure_audio_dir(AUDIO_DIR)

# Get resource path (works for both dev and PyInstaller bundle)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        # Running in PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    # Running in normal Python environment
    return os.path.join(_SOURCE_ROOT, relative_path)


ICON_PATH = resource_path("assets/icons/logo_menu_template.png")
ICON_RECORDING = resource_path("assets/icons/icon_recording.png")
ICON_PROCESSING = resource_path("assets/icons/icon_processing.png")
ICON_ERROR = resource_path("assets/icons/icon_error.png")
STATE_ICON_PATHS = {
    "ready": ICON_PATH,
    "recording": ICON_RECORDING,
    "processing": ICON_PROCESSING,
}
if os.path.exists(ICON_ERROR):
    STATE_ICON_PATHS["error"] = ICON_ERROR
BUNDLE_ID = "com.canopystudio.murmur"


#: Everything Murmur can download: the speech models plus the cleanup GGUF.
#:
#: The cleanup model is not a speech engine, so ``engines.model_store.CATALOG``
#: deliberately does not carry it and the app composes the two here instead.
#: ``ui.download_sheet.EngineSectionModel`` filters back down to
#: ``engines.ENGINE_IDS``, so the Settings popup never offers a chat model as a
#: transcriber — but the same store, downloader and integrity checks serve both.
APP_CATALOG = CATALOG + (CLEANUP_MODEL_SPEC,)


def app_model_store() -> ModelStore:
    """The store the app uses everywhere: speech models and the cleanup model."""
    return ModelStore(catalog=APP_CATALOG)
