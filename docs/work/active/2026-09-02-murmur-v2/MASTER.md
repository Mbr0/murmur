# MASTER: Murmur v2

**Status:** ✅ approved 2026-09-02 · Wave 0 in PR #4 · Wave 1 next
**Base branch:** `main` (after PR #2)
**Source:** study in `../2026-09-02-competitive-analysis/`, Part II "What should change"

## Goal

Ship, in order: an engine layer that no longer depends on torch, the foundation needed to charge money (push-to-talk, language, vocabulary, downloads, onboarding, signing, updater), the smart layer that Pro sells (local cleanup, modes, context, live pill), a Settings redesign, and the license and cloud clients. Each wave ships on its own PR with the suite green. No Boske repository changes in this folder.

## Principles carried over

- Fail fast, no silent fallbacks, except the one deliberate product fallback: cloud allowance exhausted → local engine, with a visible notice.
- No transcript text in logs. No audio or text leaves the Mac unless the user chose Cloud or Own key.
- Plan first, small PRs, archive rather than delete.

## Decisions to take before Wave 1 (recorded in `decisions.md` of this folder)

| # | Decision | Default |
|---|----------|---------|
| D1 | Primary local engine | Voxtral Mini 4B Realtime, 4-bit, via `mlx-audio`, on Apple Silicon with ≥16 GB. whisper.cpp large-v3-turbo otherwise. Confirmed by Wave 0 bake-off, not assumed. |
| D2 | whisper.cpp integration | Bundle the `whisper-server` binary and talk HTTP (OpenAI-compatible `/v1/audio/transcriptions`), the same contract Boske uses. Not a Python binding. |
| D3 | Local cleanup runtime | Bundle `llama-server` and a ~3B GGUF (Ministral 3B or equivalent, Apache 2.0). Same HTTP pattern as D2. |
| D4 | Updater | Sparkle 2 through PyObjC if it loads cleanly in the PyInstaller bundle; otherwise a minimal signed-DMG updater that verifies the Developer ID signature before install. Decide in Wave 1d. |
| D5 | Pro gating | Pro features live in this repo but are enabled by a verified license. The repo stays MIT. Revisit source-available only if forks with Pro unlocked become a problem. |
| D6 | Cloud auth | Murmur validates Boske lease tokens (Ed25519 JWT, `boske-llm-proxy` audience) and obtains them through the same device-linking flow as the Boske desktop app. No key pasting for Murmur Cloud. Own key (BYOK) is separate and stored in Keychain. |
| D7 | Intel Macs | Supported through whisper.cpp only. Voxtral is Apple Silicon only. |

## Hot files (serial-only; ≤1 agent each)

- `murmur.py`
- `Murmur.spec`
- `requirements.txt`
- `services/hotkey_service.py`
- `services/text_insertion_service.py`
- `.github/workflows/release.yml`

## Target layout after Wave 5

```
murmur.py                     thin entry point
app/                          state machine, menu, lifecycle (from murmur.py)
engines/                      base.py, whispercpp.py, voxtral_mlx.py, cloud.py, byok.py, model_store.py
cleanup/                      llama_server.py, modes.py, context.py, vocabulary.py
ui/                           settings/, onboarding_window.py, pill_window.py, history_window.py, alerts.py, theme.py
services/                     audio_capture, hotkey, text_insertion, persistence, license, usage
tests/                        unit + integration (fixture audio EN/FR/NL/DE)
```

Moves happen only in Wave 5. Waves 0–4 add new packages next to the existing files and touch `murmur.py` minimally.

---

## Wave 0 — Engine layer and bake-off (SERIAL on `murmur.py`, otherwise parallel)

| Agent | owned_paths | Tasks |
|-------|-------------|--------|
| E0a | `engines/base.py`, `engines/__init__.py`, `tests/test_engines_base.py` | `Engine` protocol: `load()`, `unload()`, `transcribe(wav_path, language, hints) -> Transcript`, `stream(chunks) -> Iterator[Partial]`, `supports_streaming`, `supports_hints`, `info()` (name, size, languages). `Transcript` and `Partial` dataclasses. Engine selection by chip and RAM (`model_profile_service` extended). |
| E0b | `engines/whispercpp.py`, `scripts/tools/fetch_whispercpp.sh`, `tests/test_engine_whispercpp.py` | Manage a bundled `whisper-server` child process (port from a free-port probe, health check, shutdown). Client for `/v1/audio/transcriptions` with `initial_prompt` from hints. Tests use a fake HTTP server. |
| E0c | `engines/voxtral_mlx.py`, `tests/test_engine_voxtral.py` | `mlx-audio` Voxtral Mini 4B Realtime 4-bit: load from model store, batch transcribe, streaming partials, context biasing from hints. Guard import so the app runs where MLX is absent. Tests mock the runtime. |
| E0d | `engines/model_store.py`, `tests/test_model_store.py` | Model catalog (id, engine, size bytes, sha256, HF source), download with resumable progress callbacks, verify, delete. Location `~/Library/Application Support/Murmur/models/`. |
| E0e | `scripts/tools/bakeoff.py`, `tests/fixtures/audio/README.md` | Bake-off harness: fixture clips in EN/FR/NL/DE (10 each, dictation style, recorded by us, no third-party audio), WER via `jiwer`, latency, RAM. Output a markdown table. Fixture audio is not committed if over 5 MB; script documents how to regenerate. |
| E0f (serial) | `murmur.py`, `services/transcription_service.py` | Route existing transcription through `Engine`. Keep `openai-whisper` behind an `engines/whisper_openai.py` adapter for this wave only, so behaviour is unchanged until D1 is confirmed. |

**Done when**
- [ ] Bake-off table in `decisions.md` with WER and latency for whisper.cpp turbo, Voxtral Realtime 4-bit, and current openai-whisper on the same clips (harness landed; needs real recordings, see `tests/fixtures/audio/README.md`)
- [ ] D1 recorded with data (blocked on the recordings above)
- [x] App runs unchanged for users through the adapter
- [x] Suite green (249 tests, 2026-09-02)

## Wave 1 — Foundation, needed before selling (PARALLEL except 1a and 1d)

| Agent | owned_paths | Tasks |
|-------|-------------|--------|
| E1a (serial) | `services/hotkey_service.py`, `murmur.py`, `tests/test_hotkey_service.py`, `tests/test_app_state.py` | Push-to-talk: key-down starts, key-up stops, with a hold threshold (300 ms) below which the press toggles. Config `hotkey_mode: toggle | hold | auto`. Both Carbon and NSEvent paths. |
| E1b | `services/persistence_service.py`, `settings_window.py`, `cleanup/vocabulary.py`, `tests/test_vocabulary.py` | Language picker (auto + list from engine `info()`), remembered per front app bundle id. Vocabulary: list of terms and a replacements table (`from`, `to`, `match_case`), CSV import/export. Hints passed to engines. |
| E1c | `settings_window.py`, `engines/model_store.py`, `ui/download_sheet.py` | "Speech engine" section replaces "Whisper Model": engine and model popup with size, download button with progress sheet, delete. Hot-swap: switching models reloads the engine in the background; no restart alert. |
| E1d (serial) | `Murmur.spec`, `requirements.txt`, `scripts/build_pyinstaller.sh`, `scripts/release.sh`, `.github/workflows/release.yml`, `services/update_service.py`, `tests/test_update_service.py` | Remove torch and openai-whisper once D1 is confirmed. Bundle `whisper-server` and, on Apple Silicon, MLX wheels. Developer ID signing and notarization become the default CI path; ad-hoc builds are labelled "internal" in the About text. Updater per D4. Homebrew cask formula in `scripts/homebrew/`. |
| E1e | `ui/onboarding_window.py`, `murmur.py` (one hook, after E1a lands), `tests/test_onboarding_state.py` | First-run wizard: microphone, Accessibility, engine download with progress, a test sentence typed into a field inside the wizard. Skippable. Shown once, re-openable from the menu. |
| E1f | `settings_window.py` (strings only), `README.md`, `docs/design-manifest.md` | Copy: "Whisper Model" → "Speech engine"; README engine lines; design manifest register line. Keep the "OpenAI Whisper" credit in the licenses section. |

**Done when**
- [ ] Hold and toggle both work with Carbon and NSEvent paths; tests cover the 300 ms threshold
- [ ] Language and vocabulary hints reach both local engines; CSV round-trips
- [ ] Model switch without restart; download progress visible; sha256 verified
- [ ] CI produces a signed, notarized DMG when secrets exist; updater installs a signed build; bundle no longer contains torch
- [ ] Wizard completes on a clean macOS user account
- [ ] Suite green

## Wave 2 — Smart layer, what Pro sells (PARALLEL, then serial wiring)

| Agent | owned_paths | Tasks |
|-------|-------------|--------|
| E2a | `cleanup/llama_server.py`, `scripts/tools/fetch_llama.sh`, `tests/test_llama_server.py` | Manage a bundled `llama-server` with a ~3B GGUF from the model store. `/v1/chat/completions` client with a strict timeout (2 s per 100 words) and a raw-text fallback with a visible "cleanup skipped" notice, never a silent drop. |
| E2b | `cleanup/modes.py`, `tests/test_modes.py` | Modes: Dictation (verbatim), Message, Mail, Notes, Code. Each a prompt template plus tone (neutral, warm, formal, terse). Prompts stored as data, not code. Golden tests on fixed transcripts. |
| E2c | `cleanup/context.py`, `tests/test_context.py` | Front app bundle id and window title via `NSWorkspace`; selected text via Accessibility when granted. Map bundle ids to default modes (Mail → Mail, Terminal/Ghostty/iTerm → Code, Messages/Slack → Message). User overrides persisted. |
| E2d | `ui/pill_window.py`, `tests/test_pill_state.py` | Floating pill near the cursor: state (listening, working, done), live partial text from `stream()`, one line, fades out. No animation by default. VoiceOver label. |
| E2e | `cleanup/coding_mode.py`, `tests/test_coding_mode.py` | Spoken punctuation and flags ("dash dash force" → `--force`, "open paren"), camelCase and snake_case commands. Target detection for Claude Code and terminals reuses E2c. |
| E2f (serial) | `murmur.py` | Wire: after transcription, run cleanup when Pro is active and the mode is not Dictation; show the pill; menu gets a "Mode" submenu. |

**Done when**
- [ ] Cleanup runs locally within the timeout on a 16 GB M-series Mac; fallback notice tested
- [ ] Mode auto-selection matches the bundle-id table; override sticks per app
- [ ] Pill shows partials from Voxtral streaming; whisper.cpp shows state only
- [ ] Golden tests for all five modes and four tones
- [ ] Suite green

## Wave 3 — Settings redesign and privacy surface (PARALLEL by tab)

| Agent | owned_paths | Tasks |
|-------|-------------|--------|
| E3a | `ui/settings/window.py`, `ui/settings/general_tab.py` | `NSTabView` shell replacing `settings_window.py`; General: shortcut, mode (hold/toggle/auto), language, appearance, launch at login. |
| E3b | `ui/settings/engine_tab.py` | Local engine and model with download state; Cloud: Off / Murmur Cloud / Own key; usage this month (minutes and words). |
| E3c | `ui/settings/smart_tab.py` | Cleanup on/off, modes and tones, context awareness toggle, vocabulary and replacements editor, snippets. |
| E3d | `ui/settings/privacy_tab.py`, `services/persistence_service.py` | History and audio toggles, delete all data, and a generated plain-language list of what leaves the Mac for the current configuration. History entries gain `origin: local | cloud | byok`. |
| E3e | `ui/settings/account_tab.py`, `services/keychain.py`, `tests/test_keychain.py` | Pro license status, Boske ID sign-in, own-key entry stored in Keychain (`security` framework via PyObjC), version and update channel. |
| E3f (serial) | `murmur.py`, `settings_window.py` → `_archive/` | Swap the window; archive the old file with `git mv`. |

**Done when**
- [ ] All twelve existing config keys still round-trip; new keys documented in `persistence_service.py`
- [ ] "What leaves the Mac" text matches the engine actually selected (tested per configuration)
- [ ] No secret in JSON; Keychain read/write tested
- [ ] Suite green

## Wave 4 — License and cloud clients (PARALLEL, then serial wiring)

| Agent | owned_paths | Tasks |
|-------|-------------|--------|
| E4a | `services/license_service.py`, `tests/test_license_service.py` | Verify Ed25519 lease JWTs offline against the embedded Boske public key; entitlement flags (`pro`, `cloud_voice`, `msm` minutes); expiry and grace period; device-linking flow mirrored from the Boske desktop app; refresh in the background. Test vectors generated with a throwaway key in tests. |
| E4b | `engines/cloud.py`, `services/usage_service.py`, `tests/test_engine_cloud.py` | Boske proxy client: multipart upload to `/v1/audio/transcriptions` with the lease as bearer, language, 60-minute cap awareness, retries with backoff, and `GET /v1/voice/usage`. Soft limit: at 95% of the allowance switch to local with a one-time notice; never surface a 429 to the user. |
| E4c | `engines/byok.py`, `tests/test_engine_byok.py` | Own-key engine for Mistral and OpenAI transcription endpoints; key from Keychain; no metering. |
| E4d | `cleanup/cloud_cleanup.py` | Cloud cleanup through the proxy chat endpoint when Murmur Cloud is active; same mode prompts as Wave 2. |
| E4e (serial) | `murmur.py` | Engine routing: Cloud when entitled and under allowance, else local. Pro gate on cleanup, modes, context, vocabulary beyond 20 terms, snippets, coding mode. Free tier one-time 60-minute cloud trial counter. |

**Done when**
- [ ] License verification passes test vectors and rejects tampered, expired and wrong-audience tokens
- [ ] Cloud engine works against a recorded fixture of the proxy; fallback triggers at 95% in tests
- [ ] Pro gate is a single function, tested, with no feature check scattered in UI code
- [ ] Suite green

## Wave 5 — Split and harden (SERIAL)

| Agent | owned_paths | Tasks |
|-------|-------------|--------|
| E5 | `murmur.py`, `app/**`, `ui/**`, `services/**`, `tests/**` | Move the state machine, menu and lifecycle out of `murmur.py` into `app/`. `git mv` old UI files into `ui/`. Integration tests: fixture audio through each local engine end to end (skipped when the runtime is absent), paste flow with a fake pasteboard. CI runs unit on Linux and integration on macOS. |

**Done when**
- [ ] `murmur.py` under 100 lines
- [ ] Integration suite green on the macOS runner
- [ ] `pip-audit` clean

---

## Out of scope (this folder)

- Boske repository changes (sellable add-on, API keys, per-modality cost pin, usage UI in Boske). Tracked in Boske.
- Dictate-anywhere inside Boske Electron, "powered by Murmur".
- Meeting mode, iOS companion, Windows build.
- Realtime cloud transcription.
- Pricing page and website copy (canopystudio-website).

## Risks

| Risk | Mitigation |
|------|------------|
| `mlx-audio` Voxtral support immature or slow inside a PyInstaller bundle | Wave 0 bake-off before any user-facing change; whisper.cpp remains the default until D1 is confirmed |
| Bundle size with MLX wheels and two model runtimes | Runtimes bundled, models downloaded on demand; measure DMG size in CI and fail above 400 MB |
| Sparkle through PyObjC | D4 fallback updater; both verify signatures |
| Pro features forkable from MIT source | Accepted per D5; value is the signed build, models, cloud and support |
| Boske-side gaps delay Wave 4 | Wave 4 tests against recorded fixtures; the client ships dark and lights up when the proxy is ready |
| Cleanup latency on 8 GB Macs | Cleanup off by default below 16 GB; user can enable with a warning |

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Phase V per wave

`/verifier` → `/review-bugbot`; `/review-security` for Waves 1, 3 and 4 (hotkeys, paste, Keychain, license, network).
