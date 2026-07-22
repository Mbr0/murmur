---
name: _default-work-folder
description: >-
  Fallback pattern for work folders without a custom skill yet. Copy to
  docs/work/active/<folder>/.agents/skills/<slug>/SKILL.md and customize.
disable-model-invocation: true
---

# Default work folder skill (copy & customize)

Copy this folder to:

```text
docs/work/active/<yyyy-mm-dd>-<slug>/.agents/skills/<slug>/SKILL.md
```

Replace frontmatter `name`, `description`, and `paths`. Add row to `docs/work/skills-registry.yaml`.

```yaml
---
name: <slug>
description: Work folder skill for <title>. Use when executing this folder's plan or editing listed paths.
paths:
  - docs/work/active/<folder>/**
  # List every code path this work touches:
  - ...
---
```

# Work folder: <title>

**Plan:** [plan.md](../plan.md) or [MASTER.md](../MASTER.md)  
**Status:** (from README.md)

Load **@murmur-implementation** (extends global `@plan-first-implementation`) + this skill.

## Scope

(one paragraph from README)

## Owned paths (for subagents)

| Area | Glob |
|------|------|
| | |

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Phase V (before PR)

1. `/verifier` — run tests above; confirm diff scope
2. `/review-bugbot` — usage-based review gate
3. `/review-security` — when wave touches microphone, accessibility, hotkeys, local data paths, or PyInstaller bundle config

## Subagent waves

Optional — add SA-R/E tables when effort is large. Until then, parent agent assigns owned_paths from plan sections.
