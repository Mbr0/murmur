"""The Murmur app itself: config, decisions, services, pipeline, menu, lifecycle.

Wave 5 split the 4,684-line ``murmur.py`` into this package. What went where:

* :mod:`app.config` — logging, paths, constants, the persistence singleton and
  the model catalog. Imported first by everything else, and the one module that
  does work at import (the legacy-data migration, exactly as ``murmur.py`` did).
* :mod:`app.decisions` — every pure decision the app takes, unchanged and under
  its original name. No AppKit, no ``self``, no I/O beyond the licence transport.
* :mod:`app.services` — the licence, usage and Keychain composition.
* :mod:`app.pipeline` — recording, transcription, routing, cleanup and paste.
* :mod:`app.menu` — the menu bar and the windows it opens.
* :mod:`app.windows` — the window controllers the menu bar keeps alive.
* :mod:`app.lifecycle` — :class:`~app.lifecycle.MurmurApp` itself, composed from
  the three mixins above, plus startup, hotkeys, onboarding, updates and quit.

Nothing is imported here on purpose. ``app.config`` migrates legacy data at
import, and a package that pulled it in would run that for anyone who merely
touched ``app``.
"""
