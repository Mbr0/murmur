# Decisions: Murmur v2

Record each decision from MASTER.md with the data that settled it.

| # | Decision | Status | Data |
|---|----------|--------|------|
| D1 | Primary local engine | Open, awaiting Wave 0 bake-off | Candidates: Voxtral Mini 4B Realtime 4-bit (MLX), whisper.cpp large-v3-turbo, current openai-whisper. Clips: EN/FR/NL/DE dictation, 10 each. |
| D2 | whisper.cpp via bundled `whisper-server` over HTTP | Proposed | Same contract as Boske's whisper-server. |
| D3 | Cleanup via bundled `llama-server` and a ~3B GGUF | Proposed | Apache 2.0 model required. |
| D4 | Updater | Open, Wave 1d | Sparkle 2 via PyObjC vs signed-DMG updater. |
| D5 | Pro gated by license, repo stays MIT | Proposed | |
| D6 | Cloud auth through Boske lease tokens and device linking | Proposed | Requires Boske to expose the linking flow to Murmur. |
| D7 | Intel Macs on whisper.cpp only | Proposed | Voxtral requires Apple Silicon. |

## Bake-off results (Wave 0)

_To be filled by `scripts/tools/bakeoff.py`._

| Engine | Model | EN WER | FR WER | NL WER | DE WER | Median latency (10 s clip) | Peak RAM |
|--------|-------|--------|--------|--------|--------|----------------------------|----------|
| | | | | | | | |
