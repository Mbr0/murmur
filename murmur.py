#!/usr/bin/env python3
"""Murmur - A simple local speech-to-text menu bar app

Shortcut: Option+Space to start/stop recording.

The entry point, and nothing else. Wave 5 moved the app into ``app/``:

* ``app.config``    paths, constants, logging, the persistence singleton
* ``app.decisions`` every pure decision, by its original name
* ``app.services``  the licence, the usage counters and the Keychain
* ``app.pipeline``  record → route → transcribe → clean up → paste
* ``app.menu``      the menu bar and the windows it opens
* ``app.lifecycle`` ``MurmurApp`` itself: startup, shortcut, wizard, quit

What stays here is what has to run before any of that is imported: the bundled
resources go on PATH first, because ``app.pipeline`` and ``engines.whispercpp``
resolve a bare ``whisper-server`` through it.
"""

import os
import sys

# Put the bundled resources directory on PATH before anything that shells out.
# The bundled `whisper-server` lives there, so a plain command name resolves to
# the bundled copy without any monkey-patching of subprocess.
if hasattr(sys, '_MEIPASS'):
    os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ.get('PATH', '')

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

from app.lifecycle import MurmurApp, ensure_single_instance


def main() -> None:
    """Run the menu bar app: one instance, no Dock icon, until it is quit."""
    ensure_single_instance()
    ns_app = NSApplication.sharedApplication()
    ns_app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    MurmurApp().run()


if __name__ == "__main__":
    main()
