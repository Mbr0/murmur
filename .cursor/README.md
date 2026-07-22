# Cursor agent config (Murmur)

Tracked in git (see `.gitignore` exceptions for `.cursor/skills`, `.cursor/agents`).

## Skill stack

```
@plan-first-implementation               ← global (~/.cursor/skills/)
        +
@murmur-implementation                   ← repo paths, hot files, registry
        +
@<work-folder-skill>                     ← docs/work/active/<folder>/.agents/skills/
        +
@<subagent-skill>                        ← global or repo implementer skills
```

**Registry:** [`docs/work/skills-registry.yaml`](../docs/work/skills-registry.yaml)

| Layer | Path | Invoke |
|-------|------|--------|
| **Global** | `~/.cursor/skills/plan-first-implementation/` | `@plan-first-implementation` |
| **Repo** | `.cursor/skills/murmur-implementation/` | `@murmur-implementation` |
| **Work folder** | `docs/work/active/<folder>/.agents/skills/<slug>/` | `@<slug>` |
| **Implementer** | `.cursor/skills/subagents/<name>/` | with `/…-implementer` |
| **Generic subagents** | `~/.cursor/skills/subagents/` + `~/.cursor/agents/` | `/research-readonly`, etc. |

Project subagents override global subagents when names match.

## Subagents

| Subagent | Scope | Skill |
|----------|-------|-------|
| `/research-readonly` | Phase R | global `@research-readonly` |
| `/overlap-auditor` | Phase X | global `@overlap-auditor` |
| `/verifier` | Phase V (tests → reviews) | global `@verifier` |
| `/review-bugbot` | Phase V gate | `@review-bugbot` — usage-based |
| `/review-security` | Phase V gate (security-sensitive) | `@review-security` — usage-based |
| `/app-implementer` | `services/**`, root `*.py`, `tests/**`, `scripts/**`, `assets/**`, build config | `@app-implementer` |

## New work folder

Copy from [`docs/work/_templates/work-folder/`](../docs/work/_templates/work-folder/). Add registry row in [`docs/work/skills-registry.yaml`](../docs/work/skills-registry.yaml).

## Reload

After pull: **Developer: Reload Window**. Check **User** + **Project** tabs in Settings → Rules, Skills, Subagents.

## Also

- `.agents/skills/` at repo root — cross-tool mirror (Claude/Codex)
- Global user config: `~/.cursor/README.md`
- Bootstrap this stack: `@repo-cursor-bootstrap`
