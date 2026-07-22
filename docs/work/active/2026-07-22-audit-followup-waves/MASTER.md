# MASTER: Audit follow-up waves

**Status:** Approved  
**Base branch:** `fix/audit-reliability-wave1` (stacks on Wave 1)

## Goal

Ship remaining audit improvements without drive-by architecture rewrite. Skip Developer ID/signing ops (credentials). Skip full `murmur.py` split (separate effort).

## Hot files (serial-only; ≤1 agent each)

- `murmur.py`
- `Murmur.spec`
- `requirements.txt`
- `services/hotkey_service.py`
- `services/text_insertion_service.py`

## Wave 2 — PARALLEL (disjoint owned_paths)

| Agent | owned_paths | Tasks |
|-------|-------------|--------|
| E2a | `services/hotkey_service.py`, `tests/test_hotkey_service.py` | Carbon path: enforce or reject `fn`; document/test |
| E2b | `Murmur.spec` | Intel ffmpeg fallback `/usr/local/bin`; fail clearly if missing when expected |
| E2c | `services/persistence_service.py`, `tests/test_persistence_service.py` | Atomic `0o600` create for JSON saves |
| E2d | `settings_window.py` | Replace `logger=print` with real logger; no APP_NAME regression |
| E2e | `history_window.py` | VoiceOver labels on list/actions; keep path guard; minimal |

## Wave 3 — SERIAL (`murmur.py` hot)

| Agent | owned_paths | Tasks |
|-------|-------------|--------|
| E3 | `murmur.py`, `services/persistence_service.py` (mic key only if needed), `tests/test_app_state.py`, `tests/test_persistence_service.py` | Calm notify: mic fail, no-speech/skip, short audio; persist mic device in config; remove dead unused methods if safe; real GitHub Releases update check (version metadata only, no audio/text egress) |

## Wave 4 — SERIAL (text_insertion hot)

| Agent | owned_paths | Tasks |
|-------|-------------|--------|
| E4 | `services/text_insertion_service.py`, `murmur.py` (drop redundant pre-copy if safe), `tests/test_text_insertion_service.py` (new) | Save/restore clipboard around paste |

## Wave 5 — SERIAL (perf)

| Agent | owned_paths | Tasks |
|-------|-------------|--------|
| E5 | `services/transcription_service.py`, `murmur.py` (device/fp16 wire only), `tests/test_transcription_service.py` | Apple Silicon MPS / fp16 when safe; CPU fallback; no new deps |

## Out of scope

- Full god-file split / package restructure
- Signing/notarization credential setup
- Sparkle auto-update framework
- Broad Dynamic Type / full a11y redesign

## Done when

- [x] Wave 2 shipped + Phase V (43 tests; bugbot clean; security no medium+)
- [x] Wave 3 shipped + Bugbot fixes + Phase V (60 tests; security no medium+)
- [x] Wave 4 clipboard restore + Bugbot delay/finally fixes (65 tests)
- [x] Wave 5 MPS + CUDA fp16 + CPU fallback on MPS fail (76 tests)
- [x] Full unittest suite green
- [x] Phase V per waves (bugbot follow-ups applied; security no medium+)

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
