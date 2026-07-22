# Plan: Audit reliability fixes

**Status:** Approved

## Goal

Ship a focused reliability wave that closes confirmed Critical/High bugs from the deep audit without drive-by refactors. Preserve local-only privacy posture; improve delete-all completeness for legacy MyWhisper paths.

## Out of scope (follow-up folders)

- Split `murmur.py` / large architecture refactor
- MPS/fp16 performance work
- VoiceOver / a11y pass
- Real update checker / Sparkle
- Developer ID signing ops (secrets/process)
- Clipboard save/restore UX redesign

## Owned paths

| Area | Paths |
|------|-------|
| App orchestrator (hot — serial) | `murmur.py` |
| Settings UI | `settings_window.py` |
| Persistence / privacy wipe | `services/persistence_service.py` |
| Tests | `tests/test_persistence_service.py`, `tests/test_app_state.py` (new if needed) |
| Docs | this folder, `TODO.md`, `docs/work/skills-registry.yaml` |

Hot files (never parallel-edit): `murmur.py`, `services/hotkey_service.py`, `services/text_insertion_service.py`.

## Work items

### Wave 1 — Single pass (serial `/app-implementer`)

1. **Model load brick (Critical)** — On `load_model` failure set `loading=False`, show user-visible error, prefer error menu-bar state if `icon_error` is already bundled.
2. **Processing guard (High)** — Add `is_processing` (or equivalent); reject hotkey/menu toggle while processing; `_reset_menu_state` must not clobber an active recording.
3. **Serialize Whisper (High)** — Mutex/lock around live + file transcription so shared `self.model` is never concurrent.
4. **`format_hotkey` import (High)** — Import and use correctly in `reload_hotkey` deferral path.
5. **Settings `APP_NAME` (High)** — Define/import so privacy button cannot `NameError`.
6. **File transcription `audio_path` (Medium)** — Guard `os.path.exists` when `save_audio` is off.
7. **Mic start failure menu (Medium)** — Call `_reset_menu_state()` (or equivalent) in `start_recording` except.
8. **Legacy wipe (Medium / privacy)** — `clear_all_local_data` also removes `~/.mywhisper_*` leftovers; unit test.
9. **Temp WAV cleanup (Low–Med)** — `try/finally` delete temp wav on transcription exception when not retained.
10. **TDD** — Failing tests first for persistence legacy wipe + any extractable state guards; full suite green.

### Optional same-wave polish (only if cheap)

- Calm notification when paste fails for Accessibility (no silent fail).
- Align `MANIFEST.md` default shortcut to ⌥ Space if that file is editable in this repo checkout.

## Done when

- [x] Model load failure does not leave the app permanently unable to record
- [x] Hotkey cannot start a second recording while transcription is in flight
- [x] Whisper transcription is serialized
- [x] Settings privacy button does not crash
- [x] Hotkey deferral path does not raise `NameError`
- [x] Empty file transcription with `save_audio=False` does not raise
- [x] Mic start failure restores menu callbacks/title
- [x] Delete All Local Data removes legacy MyWhisper history/audio (and config if present)
- [x] Temp WAV cleaned up on exception when not retained
- [x] `python -m unittest discover -s tests -p "test_*.py" -v` passes
- [x] Phase V: `/verifier` → `/review-bugbot` → `/review-security` (touches hotkeys, local data paths)

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Approval gate

Reply **approve** (or set README/plan status to **Approved**) to start Phase R → X → E → V.
Leave draft if you want scope changes first.
