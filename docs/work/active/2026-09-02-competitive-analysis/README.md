# Competitive analysis, pricing and Boske reuse

**Status:** Research — no code changes. Date: 2026-09-02.

**Purpose:** Decide whether and how to price Murmur, what to build to reach parity with Superwhisper and the 2026 dictation market, and how Murmur fits into Boske.

Sources are listed at the end. Items marked *(unverified)* could not be checked against a primary source from this environment (superwhisper.com is not reachable from the research sandbox).

---

## 1. Summary

- Murmur 1.0.0 (June 2026) is a clean, private, single-purpose dictation utility. It has no AI cleanup, no vocabulary, no modes, no push-to-talk, no streaming, no auto-update, no licensing. It runs on `openai-whisper` + torch, which makes the bundle heavy and cold start slow.
- The market has split in two. Cloud apps (Wispr Flow, Willow, Aqua, Typeless, Monologue) charge $12–15/month, cap free use at 1–8k words/week, and compete on AI formatting and context. Local apps (Superwhisper, MacWhisper, VoiceInk, BetterDictation, Voibe, Spokenly, Handy) sell one-time licenses of $25–149 or cheap subscriptions of $7–10/month, and compete on privacy.
- Superwhisper is the benchmark for a local-first app. Its summer 2026 releases added on-device LLM cleanup with tone control (S1 / S1-mini), Parakeet models, Cohere Transcribe, coding-agent integration (Claude Code, OpenCode, Grok), CSV vocabulary import and a redesigned modes UI. Pricing: free local tier, Pro $8.49/month or $84.99/year, lifetime $249.99 *(unverified, some sources cite a rise to ~$849)*.
- Apple's on-device SpeechAnalyzer in macOS 26 and Windows Voice Access make raw dictation free. Paid value now sits in AI cleanup, context awareness, vocabulary, modes and agent integration, not in transcription itself.
- Recommended pricing: keep local dictation free and unlimited, sell a one-time Pro license around $49 for the smart layer, and include Pro in every Boske seat. Word caps make no sense for a local app.
- Recommended reuse: bundle and cross-license Murmur with Boske first (low effort), point Murmur at Boske's existing whisper-server second, and defer a full port into the Electron desktop app.

---

## 2. Murmur today

### Implemented (v1.0.0 plus PR #1 and #2)

| Area | State |
|------|-------|
| Trigger | Toggle only (⌥Space default, configurable). Carbon hotkey first, NSEvent fallback. No push-to-talk. |
| Engine | `openai-whisper` on MPS or CPU, tiny→large, model picked by RAM. Model change needs restart. |
| Language | Auto-detect only, no picker, no `initial_prompt`. |
| Insertion | Copy → synthetic ⌘V → restore clipboard after 0.4s. |
| Post-processing | Rule-based only: skip clips < 1s or near-silent, drop known Whisper hallucination strings. |
| History / audio | Optional local JSON history (100 entries) and WAV retention, both off by default. History window with playback. |
| Extras | File transcription (clipboard only), mic picker, single-instance lock, manual update check against GitHub Releases, dark/light/system. |
| Privacy | No telemetry, no cloud, no transcript logging. Files 0600/0700. |
| Packaging | PyInstaller + DMG. Ad-hoc signed by default, Developer ID + notarization only when CI secrets exist. No Sparkle. |
| Tests | ~93 unit tests on pure logic. No UI or end-to-end tests. |
| Monetization | None. MIT license. No account, license, or feature-flag code. |

### Gaps versus the market

1. No AI cleanup or formatting (every competitor above the free tier has it).
2. No custom vocabulary or replacements (cheapest possible win via Whisper `initial_prompt`).
3. No app-aware modes or context (selected text, active app).
4. No push-to-talk, no streaming or live text.
5. Cold start and bundle size from torch + openai-whisper. Competitors run whisper.cpp, Parakeet on MLX, or Apple SpeechAnalyzer.
6. No signed/notarized default build and no auto-update. Both are prerequisites for charging money.
7. No onboarding wizard for permissions.
8. No licensing or entitlement mechanism.

---

## 3. Superwhisper's 2026 update

Continuous 2.x point releases, not a single relaunch. Notable items May–Aug 2026:

| Date | Change |
|------|--------|
| ~Apr 2026 | Claude Code / OpenCode integration: voice piped into agent sessions. |
| Jul 29 (2.17.0) | Cohere Transcribe as cloud ASR. Grok added as a coding agent. |
| Aug 4 | Auto language detection for Cohere. Rebrand to "Superwhisper". |
| Aug 6 | "S1" model family announced. Short silent clips skipped before transcription. |
| Aug 17 (2.17.3) | Tone control for Message / Mail / Notes modes. Redesigned model sheet with download sizes. |
| Aug 19 (2.18.0) | S1-mini: on-device LLM with tone control, no network. Local Mode auto-suggested for English keyboards. |
| Undated 2026 | Redesigned Modes UI, GPT-5.5 BYOK, CSV vocabulary import, live waveform gating, Shortcuts deep links for start/stop. |

Feature set: local Whisper (Fast/Nano/Standard/Pro/Ultra), local Parakeet V2/V3, cloud ASR (Cohere, Deepgram, ElevenLabs Scribe); modes with LLM rewrite (local S1 or cloud BYOK); context awareness from selected text, active window and clipboard; vocabulary; meeting mode; file transcription; Windows and iOS companions.

Pricing: Free (small local models, 3 custom modes, meeting recording, 15-minute Pro trial), Pro $8.49/month or $84.99/year, lifetime $249.99 *(unverified)*, enterprise custom, 40% student discount, 30-day refund.

Criticisms: proper-noun accuracy, audio saved by default, API keys in plaintext JSON, overwhelming settings, expensive lifetime tier, Windows and iOS lag behind Mac.

---

## 4. Competitor landscape

| App | Processing | Engine | AI cleanup | Free tier | Paid |
|-----|-----------|--------|------------|-----------|------|
| Superwhisper | Local + optional cloud | whisper.cpp, Parakeet, Cohere, Deepgram | Yes, local S1 or cloud | Unlimited small local models | $8.49/mo, $84.99/yr, lifetime $249.99 *(unverified)* |
| Wispr Flow | Cloud | Proprietary | Yes, context aware | 2,000 words/wk | $15/mo, $144/yr. Raised $280M at $2B (Aug 2026). |
| Willow Voice | Cloud | Proprietary | Yes | 2,000 words/wk | $15/mo, $144/yr, team $10/user/mo |
| Aqua Voice | Cloud | Proprietary | Yes, voice commands | 1,000 words | Pro $8/mo annual, Max $24/mo annual |
| Typeless | Cloud | Proprietary | Yes | 8,000 words/wk | $30/mo or $144/yr |
| Monologue (Every) | Hybrid | Local + cloud | Yes | 1,000 words once | $15/mo, $144/yr |
| MacWhisper | Local | whisper.cpp | Optional (Pro) | tiny/base | €59 lifetime (Gumroad); $6.99/mo, $29.99/yr, $99.99 lifetime (App Store) |
| VoiceInk | Local, GPL | whisper.cpp | Optional local LLM | Free from source | $25 / $39 / $49 one-time by device count |
| BetterDictation | Local | Whisper on ANE | Pro tier | Limited | $39 lifetime, $49 + $2/mo Pro, $149 studio |
| Voibe | Local or zero-retention cloud | Undisclosed | Limited | Refund only | $7.50/mo, $59/yr, $149 lifetime |
| Spokenly | Local free, cloud Pro | Whisper, Parakeet, GPT-4o, Deepgram | Yes (Pro) | Unlimited local | $9.99/mo, BYOK |
| Handy (open source) | Local | Whisper, Parakeet, Moonshine | No | Free | Donations |
| Apple Dictation (macOS 26) | Local | SpeechAnalyzer | Apple Intelligence | Free | Free |

### Pricing patterns

- Cloud apps converge on $15/month or $144/year with 1–2k free words per week.
- Local apps converge on $25–60 one-time for the basic app, $100–250 for premium lifetime, or $7–10/month when a subscription exists.
- Free local tiers are unlimited. Word caps are only used where inference costs the vendor money.
- Privacy is the top complaint category (Wispr Flow screenshot controversy), which is Murmur's strongest existing asset.
- The built-in dictation from Apple and Microsoft is now good enough that "raw transcription" cannot be the paid feature.

---

## 5. Pricing proposal for Murmur

### Constraints

- Murmur is MIT. Anyone can fork the whole app. Feature gating only works if the gated code is not in the public repo or if the value is in signed binaries, models, services and support rather than code. VoiceInk (GPL + paid packaged builds) and MacWhisper prove that people pay for the convenient build even when source is open.
- Local inference costs nothing per user, so a usage cap would only annoy people and push them to Handy or Apple Dictation.
- Boske already prices per seat at €19–65/month and markets "Voice-ready". Murmur is a cheap, credible way to make that claim concrete.

### Proposed ladder

| Tier | Price | Contents |
|------|-------|----------|
| Free | $0, unlimited | Local Whisper/Parakeet dictation, toggle and push-to-talk, history, file transcription, vocabulary (basic), signed and notarized build, auto-update. |
| Pro | $49 one-time per person, or included in any Boske seat | Local AI cleanup with modes and tone (Message, Mail, Notes, Code), context awareness, unlimited custom vocabulary and replacements with CSV import, snippets, coding-agent mode, priority support. Family/team pack at $99 for 3 Macs. |
| Cloud add-on | $6/month or BYOK, billed through Boske | Cloud LLM cleanup and cloud ASR via Boske's llm-proxy for people who want the best accuracy. Only tier with per-use cost, so only tier that is metered. |

Positioning: cheaper than Superwhisper lifetime by 5x, in line with VoiceInk and BetterDictation, and the only one bundled into a private AI workspace. Optional launch discount and 40% student pricing to match Superwhisper.

Do not launch pricing before items 5, 6 and 8 in section 2 are done. An ad-hoc signed app with no auto-update cannot be sold.

---

## 6. Roadmap to reach parity

### Phase 0 — foundation (required before selling)

1. Replace `openai-whisper` + torch with whisper.cpp (matches Boske's existing `whisper-server`) or Parakeet via MLX. Cuts bundle size and cold start dramatically. Evaluate Apple SpeechAnalyzer on macOS 26 as a zero-download default.
2. Add push-to-talk alongside toggle.
3. Language picker and `initial_prompt` custom vocabulary. Small change, large accuracy gain on names.
4. Developer ID signing + notarization by default, Sparkle auto-update, first-run permission wizard.
5. Licensing: reuse Boske's `LicenseService` and Ed25519 lease tokens, offline-friendly.

### Phase 1 — the smart layer (Pro)

6. Local LLM cleanup: small GGUF model via llama.cpp (the same runtime Boske ships) with modes and tone control. This is Superwhisper's S1-mini equivalent. Keep it strictly offline by default.
7. Context awareness: active app bundle id, window title, selected text, clipboard, used to pick the mode automatically.
8. Vocabulary and replacements UI with CSV import. Snippets (voice command → text template).
9. Streaming or partial live text with whisper.cpp stream mode or Parakeet.
10. Coding-agent mode: dictation optimized for terminals and editors, and a Claude Code / Cursor integration.

### Phase 2 — reach

11. Meeting mode with system-audio capture and summary through Boske.
12. Shortcuts deep links, menu-bar quick actions.
13. iOS companion keyboard, Windows build (Boske desktop is Electron and cross-platform, which is another reason to converge).

---

## 7. Reuse in Boske

Facts established from the Boske repo:

- Boske desktop is Electron 39 with a `ServiceSupervisor` that already runs whisper.cpp `whisper-server` (10-model catalog, HF download, OpenAI-compatible `/v1/audio/transcriptions`), llama.cpp for LLMs, and Piper for TTS.
- The companion feature already has global hotkeys (`Alt+Space`, `Alt+Shift+Space`), a tray icon, clipboard save/restore and macOS Accessibility permission handling.
- The one primitive Boske lacks is injecting text into another app (Murmur's CGEvent ⌘V). Everything else Murmur does exists in some form.
- Boske's `AGENTS.md` forbids Python on the desktop, so Murmur cannot be embedded as-is.
- Both apps already share the Canopy Studio identity (`com.canopystudio.murmur`), and the design manifest targets Boske moss `#387033`.
- Boske backend has an STT entitlement gate, rate limits, licensing service and voice usage metering in `llm-proxy`.

Options, ranked by effort:

| Option | Effort | Value | Recommendation |
|--------|--------|-------|----------------|
| A. Bundle: Murmur Pro included in every Boske seat, license issued by Boske's `LicenseService` | Low | Marketing claim becomes real, upsell path from free Murmur to Boske | Do first |
| D. Shared account and licensing (Murmur validates the same lease tokens) | Low–medium | One purchase, one identity | Do with A |
| C. Murmur uses Boske's running `whisper-server` and `llama-server` when Boske is installed | Medium | No duplicate models or processes, shared AI cleanup | After A and D; needs port discovery |
| B. Port dictation-anywhere into Boske Electron (hotkey, tray, capture, whisper-server, text injection via nut-js or osascript, filters ported to TS) | High | Native cross-platform feature, no second app | Separate plan later |

Portable as specifications, not code: `transcription_filters.py`, `transcription_service.py` resolution logic, `persistence_service.py` schemas, `model_profile_service.py` RAM tiering. Not portable: hotkey, audio capture, text insertion, `murmur.py` orchestration.

---

## 8. Open questions

- Confirm Superwhisper's current lifetime price on superwhisper.com before quoting it anywhere.
- Decide whether Pro code stays out of the MIT repo (closed module) or the repo moves to a source-available license. This decides how feature gating is enforced.
- Decide the engine direction: whisper.cpp (aligns with Boske) versus Parakeet/MLX (fastest on Apple Silicon) versus Apple SpeechAnalyzer (no download, macOS 26 only).
- The gitignored maintainer files (`MANIFEST.md`, `SHIPPING.md`, `LAUNCH_READINESS.md`) were not available in this checkout and may already answer some of this.

---

## Sources

- Superwhisper changelog and docs: https://superwhisper.com/changelog · https://superwhisper.com/docs/models/voice · https://x.com/superwhisper/status/2041887677536952749 · https://news.ycombinator.com/item?id=47936169
- Superwhisper pricing (secondary): https://spokenly.app/blog/superwhisper-pricing · https://www.getvoibe.com/resources/superwhisper-pricing/ · https://usevoicy.com/blog/superwhisper-pricing
- Wispr Flow funding: https://techcrunch.com/2026/08/17/wispr-raises-280m-at-2b-valuation-as-it-looks-beyond-dictation/
- Wispr Flow / MacWhisper / VoiceInk / Dragon pricing: https://www.getvoibe.com/resources/wispr-flow-pricing/ · https://www.getvoibe.com/resources/macwhisper-pricing/ · https://www.getvoibe.com/resources/voiceink-pricing/ · https://www.getvoibe.com/resources/dragon-pricing/
- Willow / Aqua / Typeless: https://usevoicy.com/blog/willow-voice-pricing · https://aquavoice.com/pricing · https://usevoicy.com/blog/typeless-pricing
- Monologue: https://www.monologue.to/pricing · Voibe: https://www.getvoibe.com/pricing/ · Spokenly: https://spokenly.app/pricing
- Handy: https://github.com/OpenWhispr/openwhispr · VoiceInk: https://github.com/Beingpax/VoiceInk
- Apple SpeechAnalyzer: https://9to5mac.com/2025/06/18/apple-devices-offer-amazing-speech-to-text-transcription-in-developer-betas-shows-test/
- Nothing Essential Voice: https://techcrunch.com/2026/04/24/nothing-introduces-an-ai-powered-dictation-tool/

---

## 9. Portfolio leverage (added 2026-09-02, second pass)

Canopy Studio catalog today (canopystudio.eu): Boske, Grove Fit, Little Bean, Murmur, BearBell live; Minne, Carnet, Vardn coming soon. Boske-side open tools: Grove Port, Boske Pulse. Five more repositories (Savvo, Homeapp, Grand Livre, Salad, M-M Cockpit) sit outside the studio brand.

### Engines worth naming

| Engine | Backing | Today | Proposed reach |
|--------|---------|-------|----------------|
| Grove models (Boske Labs) | Seed 3B → Forest 24B on llama.cpp. Relabelled Mistral/Ministral weights, no tuned checkpoint yet | Boske chat and agents, Grove Fit catalog | Murmur Pro cleanup, Savvo recipes |
| Murmur engine | Local STT, whisper.cpp or Parakeet | The Murmur app | Boske Voice local mode ("powered by Murmur"), voice notes in Minne, Carnet, Little Bean, Vardn, Savvo |
| Boske Sources | rag-api, pgvector | Boske | Boske only |
| Boske ID | License service, Ed25519 offline leases, unified identity plan | Boske seats | Murmur Pro license, Minne, Carnet accounts |
| Grove Port | Open workspace import/export | Boske import | Boske only |

### Options

- **A. Engines inside, products outside (recommended).** Keep consumer brands, name the engines, ship Murmur Pro with every Boske seat, label Boske Voice local as powered by Murmur.
- **B. Fold Murmur into Boske.** Right for the desktop feature, wrong for the brand: loses the free funnel and the only public-repo audience. Do the port later under the Murmur name.
- **C. One studio membership.** Premature until Boske ID runs in two apps and Minne/Carnet ship.

### Order of work

1. Murmur to whisper.cpp, signing, auto-update, push-to-talk.
2. Murmur Pro licenses from Boske's license service, included in every seat.
3. Murmur Pro cleanup with Grove Seed, modes and tone.
4. Boske desktop dictate-anywhere powered by Murmur (separate plan).
5. Murmur engine as a mobile package for the consumer apps.

Caution: "powered by Grove models" is honest only once Boske Labs ships a tuned or quantised model. Until then, use the tier names as size tiers only.

---

## 10. Murmur Cloud: Voxtral through the Boske proxy (added 2026-09-02, third pass)

### Unit cost

- Voxtral Mini Transcribe 2: $0.003/min batch, $0.006/min realtime (Mistral, Feb 2026, via aggregators; mistral.ai unreachable from the sandbox).
- One hour of speech (~9,000 words): Groq Whisper $0.04, Voxtral batch $0.18, OpenAI gpt-4o-mini-transcribe $0.18, whisper-1 $0.36, Google Chirp 3 $0.96. Cleanup of that hour with Ministral 8B: under $0.01.
- Per user per month (batch + Ministral 8B cleanup, +20% overhead): light 0.5 h €0.11 · typical 3 h €0.66 · heavy 15 h €3.31 · 40 h €8.81.

### Competitor caps

Free tiers cap words: Wispr Flow and Willow 2,000/week (~1 h/month), Typeless 8,000/week (~3.9 h), Aqua and Monologue 1,000 words once. Paid tiers say unlimited and carry fair-use clauses (Superwhisper terms ban bulk/scripted use). Willow meters AI formatting separately on free. Spokenly and VoiceInk give BYOK away and charge only for hosted models.

### Boske today (from the repo)

- Pipeline: `apps/llm-proxy/src/proxy-stt.js` → `api.mistral.ai/v1/audio/transcriptions`, model alias `voxtral-mini-latest`, batch only, 60 min max per request, usage metered in seconds from Mistral's `prompt_audio_seconds` into `cloud_stt_usage`.
- Allowance: 240 min/seat/month on Cloud tiers (`license-entitlements.ts`), hard HTTP 429 beyond it. `GET /v1/voice/usage` exists, nothing calls it.
- Auth: short-lived Ed25519 lease JWT issued only to a logged-in website session with a claimed device. No API-key flow.
- `cloud_voice` add-on is a legacy id, not sellable, no Stripe price. One shared COGS knob across modalities. BEI-008 records $0.72 per fully used seat allowance.
- Website code prices seats €19 / €39 / €65; ABOUT_BOSKE says €19 / €49 / €59 / €65. Reconcile.

### Proposal

| Tier | Price | Cloud allowance | COGS |
|------|-------|-----------------|------|
| Murmur Free | €0 | One-time 60-minute cloud trial | €0.22 once |
| Murmur Pro | $49 once or any Boske seat | BYOK free (bypasses proxy) | €0 |
| **Murmur Cloud** | **€5/month or €48/year** | **15 h/month (~135k words), then automatic local fallback, no 429** | typical €0.66, worst €3.31 (34–87% margin) |
| Boske seats | included | Cloud 4 h (today), Team 10 h (proposed), top-up €5 per 15 h | €0.72 / €1.80 per seat |

Break-even for €5 is ~22 h of batch dictation. Realtime is not offered by the proxy and would double cost; give it its own tier if built. Market in words, meter in seconds.

### Gaps to ship

1. Promote `cloud_voice` to a sellable add-on with a Stripe price (low).
2. Auth path for the Mac app: reuse desktop device-linking flow or add a per-user API key (medium).
3. Soft limit: Murmur polls usage and falls back to local before the cap; optional top-up SKU (low).
4. Usage shown in Murmur Settings and on the Boske account page (low).
5. Per-modality COGS pin (medium).
6. Pin the Voxtral version, write the STT decision record, confirm Mistral data residency and zero retention (low).
