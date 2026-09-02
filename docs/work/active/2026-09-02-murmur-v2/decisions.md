# Decisions: Murmur v2

Record each decision from MASTER.md with the data that settled it.

| # | Decision | Status | Data |
|---|----------|--------|------|
| D1 | Primary local engine | Open — harness ready, awaiting real recordings | Candidates: Voxtral Mini 4B Realtime 4-bit (MLX), whisper.cpp large-v3-turbo, current openai-whisper. Clips: EN/FR/NL/DE dictation, 10 each, recorded by us (`tests/fixtures/audio/README.md`). Run `scripts/tools/bakeoff.py`. Caveat found in Wave 0: Voxtral Realtime through mlx-audio 0.5.1 accepts no language and no vocabulary/prompt parameter, so hints cannot bias it; whisper.cpp accepts both. |
| D2 | whisper.cpp via bundled `whisper-server` over HTTP | Amended 2026-09-02 | whisper-server v1.7.5 (pinned) exposes no `/v1/audio/transcriptions`; the client uses native `POST /inference` (multipart `file`, `language`, `prompt`, `response_format=verbose_json`) and `GET /health`. Same child-process-over-HTTP pattern as Boske, not the same route. Server is started with `-l auto` because its default is English. |
| D3 | Cleanup via bundled `llama-server` and a ~3B GGUF | Proposed | Apache 2.0 model required. |
| D4 | Updater | Open, Wave 1d | Sparkle 2 via PyObjC vs signed-DMG updater. |
| D5 | Pro gated by license, repo stays MIT | Proposed | |
| D6 | Cloud auth through Boske lease tokens and device linking | Proposed | Requires Boske to expose the linking flow to Murmur. |
| D7 | Intel Macs on whisper.cpp only | Confirmed by code | `select_engine_id()` returns whisper.cpp for Intel or under 16 GB; Voxtral engine refuses to load off arm64. |

## Bake-off results (Wave 0)

_To be filled by `scripts/tools/bakeoff.py` once real EN/FR/NL/DE clips exist (harness landed in Wave 0; `scripts/tools/make_synthetic_fixtures.sh` produces `say`-based clips for smoke-testing the harness only, never for deciding D1)._

| Engine | Model | EN WER | FR WER | NL WER | DE WER | Median latency (10 s clip) | Peak RAM |
|--------|-------|--------|--------|--------|--------|----------------------------|----------|
| | | | | | | | |

## Wave 0 findings (2026-09-02)

- Engine contract in `engines/base.py` (`Engine` ABC, `Transcript`, `Partial`, `Hints`, `EngineInfo`); registry `engines.create_engine()` imports engines lazily so the app runs without MLX or a whisper-server binary.
- App routes through `engines/whisper_openai.py`, an adapter around the existing openai-whisper path; behaviour unchanged for users. `Murmur.spec` gained `engines.*` hidden imports because the registry imports dynamically.
- Model catalog (`engines/model_store.py`) carries real sizes and sha256 for whisper.cpp large-v3-turbo (q5_0 and f16) and `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`; metadata pinned to `main` as of 2026-09-02.
- Voxtral hints gap (see D1) affects Wave 1b: hints reach whisper.cpp; the Voxtral engine reports `hints_applied=False` so the UI can say so.
- Bake-off WER is computed with a stdlib word-level Levenshtein instead of `jiwer`, to keep `requirements.txt` untouched until Wave 1d. Unit-tested against hand-computed cases.
