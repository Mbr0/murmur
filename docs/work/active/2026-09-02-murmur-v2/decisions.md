# Decisions: Murmur v2

Record each decision from MASTER.md with the data that settled it.

| # | Decision | Status | Data |
|---|----------|--------|------|
| D1 | Primary local engine | Provisional 2026-09-02 (synthetic clips) — whisper.cpp large-v3-turbo q5_0 default on all Macs; Voxtral Mini 4B Realtime opt-in on Apple Silicon ≥16 GB | Bake-off below on 40 `say`-generated clips (10 per language, 8–12 s). whisper.cpp: 0.70 s median per 10 s clip, 0.66 GB, WER on par (better FR, slightly worse EN/NL). Voxtral: 4.79 s batch latency, 1.81 GB, no language or vocabulary hints in mlx-audio 0.5.1; its value is streaming partials for the live pill. openai-whisper medium (the retired baseline): 1.54 s, 1.77 GB. Re-run `scripts/tools/bakeoff.py` on real EN/FR/NL/DE recordings before Wave 5 ships; FR WER is inflated for every engine by the synthetic voice. |
| D2 | whisper.cpp via bundled `whisper-server` over HTTP | Amended 2026-09-02 | whisper-server v1.7.5 (pinned) exposes no `/v1/audio/transcriptions`; the client uses native `POST /inference` (multipart `file`, `language`, `prompt`, `response_format=verbose_json`) and `GET /health`. Same child-process-over-HTTP pattern as Boske, not the same route. Server is started with `-l auto` because its default is English. |
| D3 | Cleanup via bundled `llama-server` and a ~3B GGUF | Confirmed 2026-09-02 | Model `mistralai/Ministral-3-3B-Instruct-2512-GGUF` Q4_K_M (2.1 GB, Apache 2.0, licence verified via the HF API); llama.cpp pinned `v0.3.0`, Metal. Smoke test on this M-series 24 GB Mac: server ready in 16.4 s cold; cleanup of 41/93/113-word dictations in 0.84/1.45/1.68 s against a 3.0 s budget (policy: 2 s per 100 words, min 3 s, cap 20 s); message, mail and notes modes produced usable text with fillers removed and vocabulary terms preserved. |
| D4 | Updater | Decided 2026-09-02 — signed-build updater | No Sparkle framework is vendored and Sparkle-through-PyObjC inside a PyInstaller bundle could not be verified, so `services/update_service.py` ships the fallback: GitHub releases feed, SemVer compare, DMG download, then `codesign --verify --deep --strict` + `spctl --assess --type open --context context:primary-signature` + Team ID match against `MURMUR_EXPECTED_TEAM_ID`/`build_info.json` before `hdiutil` mount, rename-swap and relaunch. Ad-hoc or unknown-team builds are refused. |
| D5 | Pro gated by license, repo stays MIT | Proposed | |
| D6 | Cloud auth through Boske lease tokens and device linking | Proposed | Requires Boske to expose the linking flow to Murmur. |
| D7 | Intel Macs on whisper.cpp only | Confirmed by code | `select_engine_id()` returns whisper.cpp for Intel or under 16 GB; Voxtral engine refuses to load off arm64. |

## Bake-off results (Wave 0)

_To be filled by `scripts/tools/bakeoff.py` once real EN/FR/NL/DE clips exist (harness landed in Wave 0; `scripts/tools/make_synthetic_fixtures.sh` produces `say`-based clips for smoke-testing the harness only, never for deciding D1)._

**Run 2026-09-02, M-series 24 GB, macOS 27, synthetic `say` fixtures (40 clips), `--runs 1`, one engine per process.** whisper.cpp v1.7.5 Metal; mlx-audio 0.5.1; openai-whisper 20250625 on MPS.

| Engine | Model | EN WER | FR WER | NL WER | DE WER | Median latency (10 s clip) | Peak RAM |
|--------|-------|--------|--------|--------|--------|----------------------------|----------|
| whispercpp | whispercpp-large-v3-turbo-q5_0/ggml-large-v3-turbo-q5_0.bin | 10.0% | 42.6% | 14.7% | 13.8% | 0.70s | 0.66 GB |
| voxtral_mlx | voxtral-mini-4b-realtime-4bit | 8.6% | 53.4% | 9.1% | 13.0% | 4.79s | 1.81 GB |
| whisper_openai | medium | 9.0% | 53.7% | 9.5% | 13.0% | 1.54s | 1.77 GB |

## Wave 0 findings (2026-09-02)

- Engine contract in `engines/base.py` (`Engine` ABC, `Transcript`, `Partial`, `Hints`, `EngineInfo`); registry `engines.create_engine()` imports engines lazily so the app runs without MLX or a whisper-server binary.
- App routes through `engines/whisper_openai.py`, an adapter around the existing openai-whisper path; behaviour unchanged for users. `Murmur.spec` gained `engines.*` hidden imports because the registry imports dynamically.
- Model catalog (`engines/model_store.py`) carries real sizes and sha256 for whisper.cpp large-v3-turbo (q5_0 and f16) and `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`; metadata pinned to `main` as of 2026-09-02.
- Voxtral hints gap (see D1) affects Wave 1b: hints reach whisper.cpp; the Voxtral engine reports `hints_applied=False` so the UI can say so.
- Bake-off WER is computed with a stdlib word-level Levenshtein instead of `jiwer`, to keep `requirements.txt` untouched until Wave 1d. Unit-tested against hand-computed cases.

## Wave 2 findings (2026-09-02)

- **Cleanup model download surface (E2f).** The download *sheet* is an `NSWindow` sheet owned by the Settings window, and the cleanup pass runs with no window attached. So the "cleanup model is missing" path reuses the same `ui.download_sheet.DownloadController`/`DownloadSheetState` and shows its status line in the menu bar, exactly as an app update does, rather than opening a second sheet host. One offer per session; declining pastes the raw text with a visible "cleanup skipped" notice. Wave 3's Smart tab gets the real sheet.
- **One store, two catalogs.** The app composes `ModelStore(catalog=CATALOG + (CLEANUP_MODEL_SPEC,))` (`murmur.app_model_store`), so the cleanup GGUF gets the same resume, sha256 and delete code as a speech model, while `engines.model_store.CATALOG` stays speech-only and `EngineSectionModel` filters to `engines.ENGINE_IDS`.
- **Pro gate placeholder.** `murmur.pro_enabled(feature, config)` reads the hidden `pro_override_for_dev` key and is the single call site the whole smart layer gates on. Wave 4e replaces its body with `is_pro_feature_enabled`; no other file learns what "Pro" means.
- **E2f smoke, 2026-09-02, M-series 24 GB:** 174-char filler-heavy sentence, mode `message`, tone `neutral` through `cleanup_plan` → `run_cleanup` → `CleanupRuntime`: 15.15 s for the first call including the cold server start, 0.29 s warm; both `ran=True`, no skip; server stopped cleanly afterwards.
