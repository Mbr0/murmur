# Audit reliability fixes

**Status:** Ready for PR  


**Plan:** [plan.md](./plan.md) · **Skill:** `@audit-reliability-fixes`

## What

Fix Critical/High reliability and privacy bugs found in the Jul 22, 2026 deep audit: model-load brick, processing/hotkey race, NameError crash paths, file-transcription TypeError, mic menu desync, Whisper serialization, and legacy data wipe on delete-all.

## Cursor

| Layer | Invoke |
|-------|--------|
| Repo | `@murmur-implementation` |
| This folder | `@audit-reliability-fixes` → [`.agents/skills/audit-reliability-fixes/SKILL.md`](./.agents/skills/audit-reliability-fixes/SKILL.md) |

## Links

- Audit canvas: workspace canvases `murmur-deep-audit.canvas.tsx`
- Privacy defaults: [`services/persistence_service.py`](../../../../services/persistence_service.py)
- Orchestrator: [`murmur.py`](../../../../murmur.py)
