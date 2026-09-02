---
name: murmur-implementation
description: >-
  Execute approved Murmur work folders with parallel subagents and file
  ownership. Extends @plan-first-implementation with repo paths, registry, and hot files.
---

# Murmur implementation

**Base workflow:** `@plan-first-implementation` (global — phases, approval gates, parallel rules).

This skill adds Murmur-specific resolution, hot files, and conventions.

## 1. Resolve the work folder

**Priority order:**

1. User names folder or path in prompt
2. [`docs/work/skills-registry.yaml`](../../../docs/work/skills-registry.yaml) — folder slug → skill + plan file
3. [`TODO.md`](../../../TODO.md) — backlog index
4. Glob: `docs/work/active/<folder>/.agents/skills/*/SKILL.md`

**Plan file:** `MASTER.md` (large efforts) · `plan.md` (standard) · `README.md` (status always).

**Before coding:** `README.md` status must be **approved** or **✅ approved**. If draft → stop at Phase R/P only.

## 2. Skill stack

```
@plan-first-implementation               ← global phases
        +
@murmur-implementation                   ← you are here (repo paths + hot files)
        +
@<work-folder-skill>                     ← docs/work/active/<folder>/.agents/skills/<name>/
        +
@<subagent-skill>                        ← global or .cursor/skills/subagents/<name>/
```

**If work folder has no skill yet:** use [`docs/work/_templates/work-folder/.agents/skills/_default/SKILL.md`](../../../docs/work/_templates/work-folder/.agents/skills/_default/SKILL.md). Add skill before multi-agent waves.

## 3. Phase E implementers (Murmur)

| Subagent | Scope |
|----------|-------|
| `/app-implementer` | `app/**`, `ui/**`, `cleanup/**`, `engines/**`, `services/**`, root `*.py`, `tests/**`, `scripts/**`, `assets/**`, `Murmur.spec`, `requirements.txt`, `entitlements.plist` |

Generic phases R, X, V use global `/research-readonly`, `/overlap-auditor`, `/verifier`.

**Phase V ship gate (after tests pass):** `/review-bugbot` on every code wave; add `/review-security` when the wave touches microphone, accessibility, hotkeys, local data paths (`~/.murmur_*`), or PyInstaller bundle config.

## 4. Repo-wide hot files (never parallel-edit)

- `app/lifecycle.py` — app startup/teardown and menu bar orchestration
- `app/pipeline.py` — record → transcribe → cleanup → paste pipeline
- `app/services.py` — service wiring shared across the app
- `Murmur.spec` — PyInstaller bundle definition
- `requirements.txt` — pinned Python dependencies
- `services/hotkey_service.py` — global shortcut registration
- `services/text_insertion_service.py` — Accessibility paste-at-cursor
- `.github/workflows/release.yml` — signed/notarized release CI

Folder `MASTER.md` §7 or work folder skill may add more.

## 5. Murmur conventions

- [`AGENTS.md`](../../../AGENTS.md) first
- **UI:** [STUDIO_DESIGN_MANIFEST.md](../../../docs/STUDIO_DESIGN_MANIFEST.md) + [`docs/design-manifest.md`](../../../docs/design-manifest.md)
- App orchestration in `app/`; UI modules in `ui/` (`ui/settings/`, `ui/history_window.py`, etc.); `murmur.py` stays a thin entry point
- macOS menu bar app (rumps); local by default — cloud only when the user chooses Murmur Cloud or Own key
- PyInstaller bundle via `Murmur.spec`; build scripts in `scripts/`
- TDD with `unittest` in `tests/`; minimal diff; fail fast

## 6. Parent prompt template

```text
@murmur-implementation @<work-folder-skill>

Work folder: docs/work/active/<folder>/
Plan: <MASTER.md | plan.md>
Wave: N (or "single pass")

Phase R: /research-readonly
Phase X: /overlap-auditor
Phase E: /app-implementer (disjoint owned_paths)
Phase V: /verifier → /review-bugbot → /review-security (if security-sensitive paths)
```

## 7. New work folder

1. `docs/work/active/<yyyy-mm-dd>-<slug>/` with `README.md` + `plan.md`
2. `.agents/skills/<slug>/SKILL.md` — copy from [`docs/work/_templates/work-folder/`](../../../docs/work/_templates/work-folder/)
3. Add row to [`docs/work/skills-registry.yaml`](../../../docs/work/skills-registry.yaml)
4. Link from [`TODO.md`](../../../TODO.md)
5. Plan approval → Phase E

See [`docs/work/skills-registry.yaml`](../../../docs/work/skills-registry.yaml) for active folders.
