# Murmur v2

**Status:** ✅ approved 2026-09-02 · Waves 0–4 merged (PRs #4–#8) · Wave 5 in progress
**Plan:** [MASTER.md](./MASTER.md) · **Skill:** `@murmur-v2`
**Study:** [`../2026-09-02-competitive-analysis/`](../2026-09-02-competitive-analysis/) (Part II is the source of this plan)

## What

Take Murmur from a bare local Whisper utility to a sellable local-first dictation app: a pluggable engine layer with a European streaming engine (Voxtral Mini 4B Realtime on MLX) and whisper.cpp as fallback, push-to-talk, language and vocabulary, in-app model downloads, onboarding, signed and auto-updating builds, a local AI cleanup layer with modes and tone, a floating live-text pill, a five-tab Settings window, a Pro license client, and a cloud engine that talks to the Boske proxy with automatic local fallback. Murmur repository only; Boske-side work is tracked in Boske.

## Cursor

| Layer | Invoke |
|-------|--------|
| Repo | `@murmur-implementation` |
| This folder | `@murmur-v2` → [`.agents/skills/murmur-v2/SKILL.md`](./.agents/skills/murmur-v2/SKILL.md) |

## Links

- Study: [`../2026-09-02-competitive-analysis/README.md`](../2026-09-02-competitive-analysis/README.md)
- Prior waves: [`../2026-07-22-audit-followup-waves/`](../2026-07-22-audit-followup-waves/) (PR #2)
- Design: [`../../design-manifest.md`](../../design-manifest.md)
