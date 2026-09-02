# Canopy Studio: portfolio overview

*Standalone overview of the studio, its products, Boske, the shared engines and how they connect. Written 2 September 2026 from the repositories and canopystudio.eu. Safe to paste elsewhere.*

---

## 1. The studio

**Canopy Studio** · canopystudio.eu · *"Thoughtful software for high-trust moments."*

We build calm, privacy-first products for the moments that matter: at work, in memory, and in care. Every app commits to the public product manifest at canopystudio.eu/manifest:

| Principle | Meaning |
|-----------|---------|
| Privacy by design | Data stays yours. Local-first where it fits, encrypted by default, never trained on your content |
| Calm software | No dark patterns, no attention traps. Tools, not dopamine machines |
| High-trust moments | Work decisions, travel memories, pregnancy. Software that respects the weight of the moment |
| EU-rooted | Built in Europe, hosted in Europe. GDPR is the baseline, not the ceiling |

Consumer apps keep sensitive data on the device unless the user exports or syncs. Boske may run locally or in EU cloud under the same rules: no training on tenant content, export and deletion, audit where teams need it.

Site: Next.js 16, Tailwind 4, next-intl in English, French and Dutch. Studio brand copy syncs from a `packages/brand` package in the studio monorepo.

---

## 2. Product catalog

| Product | One line | Platform and stack | Distribution | Status | Repo |
|---------|----------|--------------------|--------------|--------|------|
| **Boske** | Private AI workspace for teams, local or EU cloud | Electron desktop, web, Capacitor mobile; Node backend, llama.cpp, whisper.cpp, Piper | boske.dev, seat licenses | Live | boske-ai/boske (private) |
| **Grove Fit** | Can this machine run this model? Hardware fit for local AI | TypeScript, Bun; Tauri desktop, Capacitor mobile, CLI; built on llmfit | boske.dev/fit, open source | Live | boske-ai/grove-fit |
| **Murmur** | Dictate anywhere on macOS, 100% local | Python, PyObjC, Whisper; menu bar app | canopystudio.eu/murmur, DMG on GitHub Releases, MIT | Live v1.0.0 (June 2026) | Mbr0/murmur |
| **Little Bean** | Calm pregnancy companion | Expo, React Native, on-device encryption | App Store, Google Play | Live | Mbr0/little-bean (private) + little-bean-data (public safety-alert feed) |
| **BearBell** | Motion-triggered bell sounds, fully offline | Expo, React Native, native background audio | App Store, Google Play | Live | Mbr0/bearbell (private) |
| **Vardn** | Local-first tasks and notes with peer sync on your own network | Flutter, Dart relay, mDNS and Tailscale | Coming soon | In progress | Mbr0/vardn |
| **Minne** | A notebook for a whole life: routines, family, health, meals, the years ahead | Not read this session | Coming soon | Announced | Mbr0/minne (private) |
| **Carnet** | Travel journal that rebuilds past trips from your photos, no cloud | Not read this session | Coming soon | Announced | Mbr0/carnet (private) |

Boske-side open tools, under the boske-ai GitHub organisation:

| Tool | Purpose | Stack |
|------|---------|-------|
| **Grove Port** | Open standard and CLI to export and import a whole AI workspace (not just chats) into Boske | TypeScript, Bun; schema, CLI, browser converter; MIT |
| **Boske Pulse** | Menu bar and widget monitor for Boske infrastructure (Hetzner, Coolify, Tailscale) | Swift, SwiftUI, WidgetKit |

Naming registers: forest words for Boske and its engines (Grove, Seed, Branch, Canopy, Forest, Ancient), Nordic for Vardn and Minne, French for Carnet, Savvo, Grand Livre and Salad. Each app has its own look and job; the rules are the same.

---

## 3. Boske in depth

**Tagline:** *"Your AI, on your terms."* **One-liner:** private AI workspace for teams who care where their data lives. The name comes from *bosquet*, a small cultivated grove: intimate, self-sufficient, yours.

### Brand architecture

- **Boske**, the product: Local, Cloud, Team, Enterprise, On-Premise.
- **Boske Labs**, R&D: custom models, efficiency research, privacy tech, benchmarks. Tagline *"Building AI that runs where you do."*
- **Boske Community**: open source, docs, Discord, plugins.

### Tiers and pricing

| Tier | Per seat per month | What runs where |
|------|--------------------|-----------------|
| Local | €19 | Everything on the user's machine: llama.cpp, whisper.cpp, Piper, embedded SQLite, Redis, Meilisearch, pgvector. Works offline 30+ days on signed lease tokens |
| Cloud | €49 in the brand document, **€39 in the website code** | Mistral API, Voxtral cloud speech, managed stores, 10 GB storage. Most popular |
| Team | €59 | Cloud plus browser access |
| Enterprise / Private Cloud | €65 and up | Single tenant, white label, 99.9% SLA |
| On-Premise | €35 | Customer servers, browser only for now |

Annual discount 10%, 15% at 26 seats and more. Add-ons: Automations (n8n, 1,300+ integrations) and Agents. Cloud voice is an add-on id in the license model but is not yet sellable. **The two price lists must be reconciled before any public update.**

### Model tiers, the growth metaphor

| Tier | Model | RAM | Character |
|------|-------|-----|-----------|
| Seed | Ministral 3B | 6 GB | Quick, light, always ready |
| Branch | Ministral 8B | 12 GB | Daily companion |
| Canopy | Ministral 14B | 16 GB | Serious local power |
| Forest | Mistral Small 24B | 24 GB | Near-cloud quality, still local |
| Ancient | Mistral Large, cloud | — | Cloud-scale wisdom |

Each has a Think reasoning variant. Today these are stock Mistral weights served by llama.cpp under Boske names; no Boske Labs fine-tune has shipped yet.

### Architecture

| Layer | What it is |
|-------|------------|
| Desktop | Electron 39, electron-builder, DMG and offline installer. A `ServiceSupervisor` manages every local binary: llama-server, whisper-server, Piper, Redis, Meilisearch, Postgres, rag-api |
| Web | React frontend with chat, voice composer, transcript editor, settings |
| Mobile | Capacitor shell for cloud accounts |
| Backend | Node. Files and audio services, speech routes, STT entitlement gate, rate limits, license service |
| llm-proxy | Cloud gateway: routes chat to Mistral, speech to Voxtral, meters tokens and audio seconds per seat, enforces 240 speech minutes per seat per month |
| rag-api | FastAPI, LangChain, Postgres with pgvector, dense plus lexical retrieval. Surfaces in the product as **Sources** |
| Licensing | Ed25519 lease tokens issued by boske.ai, verified offline, device linking. The unified identity plan names boske.ai as the single account authority |
| Companion | Quick-chat tray window with global hotkeys, clipboard capture, Accessibility handling |

### Visual identity

Deep Forest `#1a2f23`, Canopy Green `#2d5a3d`, Moss `#4a7c59` with `#387033` as the primary action colour, Bark `#8b7355`, Morning Light `#f4f1e8`. Light tokens are Birch, dark tokens are Pine. Gold `#ECB03F` is reserved for premium. Fonts: Inter or DM Sans for headlines, Inter for body, JetBrains Mono for code. Voice of the design: *a quiet, cultivated grove, warm paper, deep green, nothing shouting.* No vendor names in UI copy.

---

## 4. The engines

Five things the studio owns that can carry a "powered by" badge. Two are unnamed today, one carries the Murmur name, one is a naming convention, one is an open standard.

| Engine | What it is | Powers today | Could power |
|--------|------------|--------------|-------------|
| **Grove models** (Boske Labs) | Seed to Forest size tiers on llama.cpp, Ancient in the cloud | Boske chat and agents, Grove Fit's catalog | Murmur Pro cleanup, Savvo recipes |
| **Murmur engine** | Local speech to text with dictate-anywhere flow; today OpenAI Whisper, planned Voxtral Mini 4B Realtime (Mistral, Apache 2.0) with whisper.cpp fallback | The Murmur app | Boske Voice local mode, voice notes in Minne, Carnet, Little Bean, Vardn, Savvo |
| **Boske Sources** | Grounded answers over your documents, rag-api | Boske | Boske |
| **Boske ID** | One account, one key, offline leases | Boske seats | Murmur Pro licenses, Minne and Carnet accounts |
| **Grove Port** | Open workspace import and export format | Boske import | Boske |

"Powered by" lines that hold today: *Grove Fit, powered by llmfit, curated by Boske Labs* and *Import into Boske, Grove Port standard*. Proposed: *Boske Voice, powered by Murmur*; *Murmur Pro cleanup by Grove Seed*; *voice notes powered by Murmur* in the consumer apps. Say *powered by Grove models* only once Boske Labs ships a tuned or quantised model; until then the tier names are size labels on Mistral weights.

---

## 5. Outside the studio brand

| Project | What | Overlap |
|---------|------|---------|
| Savvo | Pantry and food assistant, Expo, paused June 2026 | Already runs on-device Whisper (whisper.rn), multi-provider LLM chat, Clerk auth. First candidate for the Murmur engine and Grove models |
| Homeapp | Vegan meal planning on a home NAS, vanilla JS and Node | Mistral chat and OCR proxy. Its pantry model was reused by M-M Cockpit |
| M-M Cockpit | Family finance and kitchen cockpit, Vite, Bun, Hono, Postgres | Imports Homeapp's data model, Ollama stub |
| Grand Livre | French public-data research engine, Python, DuckDB, Astro | Governed, cached, cost-capped LLM enrichment; no LLM ever produces a number |
| Salad | Evidence and decision platform, Next.js, Postgres, PostGIS; deliberately unbranded | Own MCP server and licence-gated API, the most agent-ready surface in the portfolio |

---

## 6. How it fits together

```
Canopy Studio ─ thoughtful software for high-trust moments
│
├── Boske ─────────── teams ── Local · Cloud · Team · Enterprise · On-Prem
│     ├── Boske Labs ── Grove models (Seed → Forest → Ancient), Grove Fit
│     └── Community ─── Grove Port, Boske Pulse, docs, plugins
│
├── Murmur ────────── macOS dictation, local, free, MIT ── the voice engine
├── Little Bean ───── pregnancy, stores
├── BearBell ──────── motion bell, stores, offline
├── Vardn ─────────── local-first tasks, peer sync            (soon)
├── Minne ─────────── life notebook                            (soon)
└── Carnet ────────── travel journal from photos               (soon)

Shared engines:  Grove models · Murmur engine · Boske Sources · Boske ID · Grove Port
```

Sequencing that the repositories already imply: Boske v1.0 first, then Little Bean and the other consumer apps on Boske's cross-app encrypted export groundwork, with Boske ID as the common account.

---

## 7. Open points

- Reconcile Boske seat prices between the brand document and the website code.
- Decide whether Murmur Pro is included in every Boske seat, and issue its license from Boske's license service.
- Confirm Mistral data residency and zero-retention terms before labelling cloud speech as European.
- Boske Labs has no shipped model yet; either ship one or keep the tier names as size labels.
- Minne and Carnet were not readable this session and are described from the website only.
