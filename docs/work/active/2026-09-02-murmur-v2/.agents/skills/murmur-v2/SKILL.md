---
name: murmur-v2
description: >-
  Work folder skill for Murmur v2: engine layer (Voxtral MLX + whisper.cpp),
  push-to-talk, vocabulary, downloads, onboarding, signing and updater, local
  cleanup with modes, live pill, tabbed Settings, license and cloud clients.
  Use when executing MASTER.md or editing listed paths.
paths:
  - docs/work/active/2026-09-02-murmur-v2/**
  - murmur.py
  - Murmur.spec
  - requirements.txt
  - settings_window.py
  - history_window.py
  - app/**
  - engines/**
  - cleanup/**
  - ui/**
  - services/**
  - scripts/**
  - .github/workflows/release.yml
  - tests/**
---

# Work folder: Murmur v2

**Plan:** [MASTER.md](../../../MASTER.md) · **Decisions:** [decisions.md](../../../decisions.md)
**Status:** (from README.md)

Load **@murmur-implementation** (extends global `@plan-first-implementation`) + this skill.

## Scope

Six waves, each its own PR. Wave 0 must confirm decision D1 with bake-off data before Wave 1d removes torch. Boske repository is out of scope; the cloud client in Wave 4 is tested against recorded fixtures.

## Hot files (never parallel-edit)

- `murmur.py`
- `Murmur.spec`
- `requirements.txt`
- `services/hotkey_service.py`
- `services/text_insertion_service.py`
- `.github/workflows/release.yml`

## Rules specific to this folder

- New code goes in `engines/`, `cleanup/`, `ui/`, `app/` packages. Do not grow `murmur.py`; Wave 5 shrinks it.
- Every engine implements `engines/base.Engine`. No engine-specific branches in UI code.
- One Pro gate function. UI asks `is_pro_feature_enabled(feature)`, nothing else.
- The only allowed silent-looking fallback is cloud → local at the allowance, and it shows a one-time notice.
- No transcript text in logs, ever. Fixture audio is ours, not third-party.

## Phase V

`/verifier` → `/review-bugbot`; `/review-security` for Waves 1, 3 and 4.
