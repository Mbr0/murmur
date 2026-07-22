---
name: audit-reliability-fixes
description: >-
  Work folder skill for Jul 2026 audit reliability fixes. Use when executing
  this folder's plan or editing listed paths.
paths:
  - docs/work/active/2026-07-22-audit-reliability-fixes/**
  - murmur.py
  - settings_window.py
  - services/persistence_service.py
  - tests/test_persistence_service.py
  - tests/test_app_state.py
---

# Work folder: Audit reliability fixes

**Plan:** [plan.md](../../../plan.md)  
**Status:** (from README.md)

Load **@murmur-implementation** (extends global `@plan-first-implementation`) + this skill.

## Scope

Critical/High reliability bugs + legacy delete-all privacy gap from the deep audit. Single serial wave; `murmur.py` is hot.

## Owned paths (for subagents)

| Area | Glob |
|------|------|
| Orchestrator (serial / hot) | `murmur.py` |
| Settings | `settings_window.py` |
| Persistence | `services/persistence_service.py` |
| Tests | `tests/test_persistence_service.py`, `tests/test_app_state.py` |
| Work docs | `docs/work/active/2026-07-22-audit-reliability-fixes/**` |

## Hot files (never parallel-edit)

- `murmur.py`
- `services/hotkey_service.py` (touch only if plan expands)
- `services/text_insertion_service.py` (touch only if paste notification needs it)

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Phase V (before PR)

1. `/verifier` — run tests above; confirm diff scope
2. `/review-bugbot` — usage-based review gate
3. `/review-security` — required (local data wipe + hotkey path)

## Subagent waves

| Phase | Agent | Notes |
|-------|-------|-------|
| R | `/research-readonly` | Confirm line-level fix sites |
| X | `/overlap-auditor` | Expect serial: murmur.py hot |
| E/S | `/app-implementer` | Single owner for all code paths |
| V | `/verifier` → bugbot → security | After green tests |
