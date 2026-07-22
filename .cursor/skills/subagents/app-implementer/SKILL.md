---
name: app-implementer
description: >-
  Companion skill for /app-implementer. Edits Murmur app code only.
  TDD, minimal diff, repo conventions from AGENTS.md.
paths:
  - services/**
  - tests/**
  - scripts/**
  - assets/**
  - murmur.py
  - settings_window.py
  - history_window.py
  - ui_theme.py
  - ui_alerts.py
  - transcription_filters.py
  - Murmur.spec
  - requirements.txt
  - entitlements.plist
---

# app-implementer

**Subagent:** `/app-implementer` · **Scope:** app code (excluding `docs/**`, `.cursor/**`)

## Before coding

1. `@murmur-implementation` + work folder skill + plan wave task
2. [`AGENTS.md`](../../../../AGENTS.md)

## Conventions

- Service layer in `services/`; UI modules at repo root
- After `requirements.txt` changes: reinstall in venv and verify PyInstaller bundle if shipping
- Follow patterns in owned paths; no drive-by refactors outside scope
- TDD — failing test first where tests exist
- Do not log transcription text in production logs

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Report files changed + test output.
