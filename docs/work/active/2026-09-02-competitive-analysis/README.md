# Murmur: market, product and pricing study

**Status:** Research, consolidated. Date: 2026-09-02. No code changes.
**Scope:** Where Murmur stands against the 2026 dictation market, which local engine should replace Whisper, what should change in the app, how it fits Boske and the Canopy Studio portfolio, and what to charge.

Every figure was checked against at least two sources. The verification table in Part V lists confidence per claim. superwhisper.com, mistral.ai, apps.apple.com and deepgram.com were unreachable from the research environment, so those figures rest on corroborating secondary sources.

---

## Summary

- **Murmur 1.0 is clean and bare.** Local Whisper on torch, toggle hotkey, clipboard paste, optional history. No AI cleanup, vocabulary, modes, push-to-talk, streaming, signed default build, auto-update or licensing.
- **Raw transcription is free now.** Apple ships on-device dictation, Handy is free and open, Spokenly gives local models away. Paid value sits above transcription: cleanup, context, vocabulary, agent integration.
- **Superwhisper set the local-first bar this summer** with an on-device cleanup model (S1-mini), Parakeet, Cohere Transcribe and Claude Code integration, at $8.49 a month.
- **A European engine exists and is better than Whisper for our languages.** Mistral's Voxtral Mini 4B Realtime is Apache 2.0, streams with sub-second latency, covers EN/FR/NL/DE, and runs 4-bit on a 16 GB Apple Silicon Mac through MLX. Keep whisper.cpp as the fallback for 8 GB machines and for Boske's Electron app.
- **Cloud dictation costs cents.** Voxtral through the Boske proxy costs €0.66 a month for a typical user. Boske already meters it per seat with a 240-minute allowance.
- **Pricing:** Free local unlimited. Pro $49 once, included with every Boske seat. Murmur Cloud €5 a month for 15 hours with automatic local fallback. Bring-your-own-key free.
- **Portfolio:** name the engines (Grove models, Murmur voice, Boske ID), keep the product brands, ship "Boske Voice, powered by Murmur" first, fold voice notes into the consumer apps later.

---

# Part I: Research

## 1. The dictation market in September 2026

| App | Processing | Engine | AI cleanup | Free tier | Paid | Confidence |
|-----|-----------|--------|------------|-----------|------|------------|
| Murmur | Local | openai-whisper | No | Everything | None | Repo |
| Superwhisper | Local + cloud | whisper.cpp, Parakeet V2/V3, Cohere, Deepgram, ElevenLabs | Yes, local S1-mini or cloud | Unlimited small local models, 15-min Pro trial | $8.49/mo, $84.99/yr, $249.99 lifetime, 40% student, 30-day refund | Medium |
| Wispr Flow | Cloud | Proprietary | Yes, context aware | 2,000 words/wk (1,000 on iPhone) | $15/mo, $144/yr. Raised $280M at $2B on 17 Aug 2026 | High |
| Willow Voice | Cloud | Proprietary | Yes | 2,000 words/wk | $15/mo, $144/yr. Team $12/seat monthly or $10 annual, 3-seat minimum | Medium |
| Aqua Voice | Cloud | Proprietary "Avalon" | Yes, voice commands | 1,000 words once | Pro $8/mo annual ($96), Max $24/mo annual, Team $12/seat | High |
| Typeless | Cloud | Proprietary | Yes | 8,000 words/wk | $30 monthly or $144/yr | High |
| Monologue (Every) | Hybrid | Local + cloud | Yes | 1,000 words once + 10 notes | $15/mo, $144/yr | High |
| MacWhisper | Local | whisper.cpp | Optional (Pro) | tiny/base | €59 once (Gumroad); $6.99/mo, $29.99/yr, $99.99 lifetime (App Store) | Medium |
| VoiceInk (GPL) | Local | whisper.cpp | Optional, your key | Free from source | $29 / $49 / $69 once since 1 Aug 2026 | Medium |
| BetterDictation | Local | Whisper on ANE | Pro add-on | Limited | $39 / $49 / $149 lifetime, Pro $2/mo | Medium |
| Voibe | Local or zero-retention cloud | Undisclosed | Limited | 7-day trial | $7.50/mo, $59/yr, $149 lifetime | Medium |
| Spokenly | Local free, cloud Pro | Whisper, Parakeet; GPT-4o, Deepgram, Groq | Yes (Pro) | Unlimited local, BYOK free | $9.99/mo | Medium |
| Handy (cjpais/Handy, MIT) | Local | Whisper, Parakeet, Moonshine | No | Everything | Free | High |
| Apple Dictation | Local | Apple Speech | Apple Intelligence | Everything | Free | High for "free, on-device"; language count unverified |

Patterns:

- Cloud apps converge on $15 a month or $144 a year with 1 to 2k free words a week. The cap exists because inference costs them money.
- Local apps converge on $25 to $70 one-time for the base app and $100 to $250 for premium lifetime. Free local tiers are never capped.
- Every "unlimited" paid tier carries a fair-use clause. Superwhisper's terms ban bulk and scripted use.
- Privacy is the top complaint in the category after Wispr Flow's screenshot controversy. Local-first apps are multiplying (Voibe, Spokenly, Handy, VoiceInk, localvoxtral).
- Newcomers in 2026 lead with on-device: Google's Eloquent, Nothing's Essential Voice, Dictato with three engines side by side.

## 2. Superwhisper's 2026 update

Continuous 2.x releases, not a relaunch:

| When | Change |
|------|--------|
| ~Apr 2026 | Claude Code and OpenCode integration, voice piped into agent sessions (needs v2.13+) |
| Jul 2026 (2.17.0) | Cohere Transcribe (2B, 14 languages) as cloud ASR, Grok as a coding agent |
| 6 Aug 2026 | S1 family announced: S1-Voice, S1-Language, S1-mini (462 MB open-weights text normaliser that turns raw ASR into clean written text) |
| Aug 2026 (2.17.3, 2.18.0) | Tone control for Message/Mail/Notes, redesigned model sheet, S1-mini shipped on-device, local mode auto-suggested for English |
| 2026, undated | Redesigned modes UI, GPT-5.5 BYOK, vocabulary import, waveform gating, Shortcuts deep links |

Criticised for proper-noun accuracy, audio saved by default, API keys in plaintext JSON, overwhelming settings, the lifetime price, and Windows/iOS lagging the Mac app.

## 3. Cloud tiers: caps and the meaning of unlimited

| App | Free cap | ≈ hours/month | Unlimited means | Cleanup metered separately | BYOK |
|-----|----------|---------------|-----------------|-----------------------------|------|
| Superwhisper | Unlimited local small models | local | Cloud and large models under fair use | No | Yes, on Pro |
| Wispr Flow | 2,000 words/wk | 1.0 | Unlimited dictation | No | No |
| Willow | 2,000 words/wk, 5-min sessions, 20 AI-format uses/wk | 1.0 | Unlimited words and formatting | Yes on free | No |
| Aqua | 1,000 words once | 0.1 total | Unlimited; Max adds realtime | No | No |
| Typeless | 8,000 words/wk | 3.9 | Unlimited | No | No |
| Monologue | 1,000 words once | 0.1 total | Unlimited | No | No |
| Spokenly | Unlimited local | local | Hosted cloud models | Depends on key | Yes, free |
| VoiceInk | Source build | local | No cloud of its own | Yes, your key | Required |

## 4. Unit costs of cloud speech and cleanup

Sixty minutes of audio, about 9,000 words, at list price:

| Provider | $/hour |
|----------|--------|
| Groq whisper-large-v3-turbo | 0.04 |
| Fireworks Whisper | 0.07 |
| Together Whisper | 0.09 |
| AssemblyAI Universal-2 | 0.15 |
| **Mistral Voxtral Mini Transcribe 2, batch** | **0.18** |
| OpenAI gpt-4o-mini-transcribe | 0.18 |
| AssemblyAI Universal-3 Pro | 0.21 |
| ElevenLabs Scribe v2 | 0.22 |
| Deepgram Nova-3 (aggregator figure) | 0.27 |
| **Voxtral Realtime** | **0.36** |
| OpenAI whisper-1, gpt-4o-transcribe | 0.36 |
| Google Chirp 3 | 0.96 |

Cleanup of that hour with Ministral 8B at $0.10 per million tokens each way costs under one cent. Mistral Small costs under six cents. Voxtral Transcribe 2 shipped 4 Feb 2026; the original Voxtral (Jul 2025) was priced from $0.001 a minute.

## 5. Local engines: a European replacement for Whisper

| Engine | Origin | License | Size | Mac runtime | Streaming | EN/FR/NL/DE | Notes |
|--------|--------|---------|------|-------------|-----------|-------------|-------|
| **Voxtral Mini 4B Realtime (2602)** | Mistral, France | Apache 2.0, open weights | 4.4B, 8.9 GB f16, ~2.5 GB 4-bit | MLX via mlx-audio (mlx-community 4-bit build), C and Rust community ports | Yes, 80 ms to 480 ms delay | 13 languages incl. all four | Runs on 16 GB Apple Silicon. Context biasing, word timestamps |
| Voxtral Mini 3B (2507) | Mistral | Apache 2.0 | 3B | MLX (mlx-voxtral), community GGUF | No | Strong, Mistral's own FLEURS numbers | Batch; prompt-based vocabulary biasing |
| Voxtral Small 24B | Mistral | Apache 2.0 | 24B | llama.cpp GGUF | No | Best of family | Needs 32 GB+ |
| Kyutai stt-1b-en_fr | Kyutai, Paris | CC-BY 4.0 | 1B | Native MLX | Yes, 0.5 s | EN and FR only | No DE/NL yet |
| whisper.cpp + large-v3-turbo | Gerganov (BG) on OpenAI weights | MIT | 809M | Native, CoreML/ANE | Experimental | Good, Whisper quality | What Boske's whisper-server runs today |
| faster-whisper | SYSTRAN, France, on OpenAI weights | MIT | Whisper sizes | CTranslate2 | No | Whisper quality | Python, 4x the reference speed |
| Distil-Whisper large-v3 | Hugging Face | MIT | 756M | whisper.cpp | No | English-leaning | Fewer hallucinations, weaker FR/NL/DE |
| Parakeet TDT 0.6B v3 | NVIDIA, US | CC-BY 4.0 | 0.6B | ONNX / sherpa-onnx | No | 25 European languages, 6.3% avg WER | CUDA-first tooling, Mac path less mature |
| Canary 1B v2 | NVIDIA | CC-BY 4.0 | 1B | Community GGUF | No | 25 languages, 7.2% avg WER | Same caveat |
| Moonshine v2 | Useful Sensors, US | Open | <1 GB | ONNX | Yes | English focus | Not for FR/NL/DE |
| Apple SpeechAnalyzer | Apple | Closed, macOS 26 API | — | Native Speech framework | Yes | Language list unverified | Developer API; whether system dictation uses it is unconfirmed |
| Granite Speech 4.1 2B | IBM, US | Apache 2.0 | 2B | mlx-audio | Partly | Multilingual, 5.3% Open ASR mean | Worth a bake-off |
| Qwen3-ASR | Alibaba, CN | Apache 2.0 | 0.6B / 1.7B | Native MLX | Unclear | 30 languages claimed | Not European |
| Gladia Solaria-3, Speechmatics | FR / UK | Cloud only | — | — | Yes | Tuned for European business audio | Not local |

**Recommendation.** Primary local engine: Voxtral Mini 4B Realtime, 4-bit, through mlx-audio, on Macs with 16 GB or more. It is European, Apache 2.0, streams, and covers the studio's languages. Fallback: whisper.cpp with large-v3-turbo for 8 GB Macs and for Boske's Electron app, which cannot ship Python. Independent WER replication of Mistral's claims is thin, so run a bake-off on FR/NL/DE dictation samples before committing. Watch Kyutai for DE/NL, and Parakeet v3 once a Mac runtime is proven.

**A warning sign.** An MIT Swift app called localvoxtral already does streaming local dictation with Voxtral Realtime on MLX, with LLM polishing and Claude Code session context. Fifty stars today. It is the exact roadmap below, built by someone else. Speed matters.

---

# Part II: The app

## 6. Murmur today

### Menu bar
Start/Stop Recording · Transcribe File · History · Settings · Microphone submenu · separator · "Model: Medium" (disabled) · "Murmur 1.0.0" (disabled) · Check for Updates… · Enable Shortcut Permission… · separator · Quit. Icons: template logo (ready), red (recording), spinner (processing, also used while the model loads), error.

### Settings window (520×660, Save/Cancel footer)
System Information (chip, RAM, recommended model) · **"Whisper Model"** popup tiny→large, changing it needs a restart · Privacy & Local Data: "Save audio recordings on this Mac", "Save transcription history on this Mac", both off by default, "Delete All Local Data" · Appearance: System/Dark/Light · Keyboard shortcut recorder with "Default (⌥ Space)" · "Open Privacy Settings" and a diagnostics label.

### History window (920×560)
Sidebar list with preview and date/time/duration, Copy and Clear History; detail pane with transcript and audio player when audio was kept.

### Behaviour
No onboarding. Model loads in a background thread with no progress UI; recording is refused with "Model is still loading…" until then. Language auto-detect only. Toggle only. Copy → synthetic ⌘V → clipboard restored after 0.4 s. Rule-based hallucination filter. Config in `~/.murmur_config.json` with twelve keys. No telemetry. Ad-hoc signing by default. About 93 unit tests on pure logic.

The only user-visible copy tied to the engine is the "Whisper Model" header and four README lines.

## 7. What should change

### Engine
1. Introduce an engine interface with three implementations: `VoxtralRealtimeEngine` (mlx-audio, streaming), `WhisperCppEngine` (whisper.cpp server, also the one Boske runs), `CloudEngine` (Boske proxy or own key). Pick by RAM: 16 GB and up gets Voxtral, below gets whisper.cpp turbo.
2. Drop torch and openai-whisper. Cold start and bundle size are the first thing a reviewer notices.
3. Download models in-app with progress and a size shown before download. Hot-swap without restart.
4. Streaming: show words as they are spoken. Voxtral Realtime and whisper.cpp stream mode both allow it.
5. Vocabulary: pass names and terms as context biasing (Voxtral) or initial prompt (Whisper). Replacements table with CSV import.

### Interaction
6. Push-to-talk alongside toggle. Hold the shortcut to stream, tap to toggle.
7. A small floating pill near the cursor showing state and live text, instead of relying on the menu bar icon. Calm: one line, no waveform animation by default.
8. Modes: Dictation (verbatim), Message, Mail, Notes, Code. Auto-picked from the front app's bundle id and window title, overridable in the menu. Tone control per mode.
9. Cleanup runs on a local small model by default (Grove Seed 3B on llama.cpp, or Superwhisper's approach with a tiny normaliser). Cloud cleanup only when Murmur Cloud or a key is set.
10. Coding mode: spoken punctuation and flags ("dash dash force"), and a Claude Code / terminal target.
11. Language picker with auto as default, remembering the last choice per app.

### Settings redesign
Replace the single scrolling window with five tabs:
- **General**: shortcut, hold or toggle, language, appearance, launch at login.
- **Engine**: local engine and model with download state; cloud: Off / Murmur Cloud / Own key; usage meter (minutes and words this month).
- **Smart**: cleanup on/off, modes and tones, context awareness, vocabulary and replacements, snippets.
- **Privacy**: history and audio toggles, delete all data, and a plain list of what leaves the Mac in each configuration.
- **Account**: Pro license, Boske ID sign-in, version and update channel.
Rename "Whisper Model" to "Speech engine". Remove the restart requirement.

### First run
12. Onboarding in four screens: microphone permission, Accessibility permission with the ad-hoc-signing caveat gone once notarized, engine download with progress, a test sentence pasted into a text field.

### Trust and distribution
13. Developer ID signing and notarization by default, Sparkle updates, Homebrew cask.
14. Keep audio off by default and say so on the website. Add "this recording stayed on your Mac" to history entries, and "sent to Murmur Cloud" when it did not.
15. Store keys in Keychain, not JSON.

### Code
16. Split the 1,362-line orchestrator into `app/` (state machine), `engines/`, `cleanup/`, `ui/`. Keep the service layer, which is already clean.
17. Add integration tests for the engine interface with fixture audio in EN/FR/NL/DE.

### Roadmap
- **Phase 0, foundation, needed before selling**: engine interface, Voxtral and whisper.cpp engines, push-to-talk, language and vocabulary, signing, updater, onboarding, licensing through Boske.
- **Phase 1, the smart layer, what Pro sells**: local cleanup with modes and tone, context awareness, vocabulary UI, streaming pill, coding mode.
- **Phase 2, reach**: Murmur Cloud, meeting mode via Boske, Shortcuts, iOS companion, Windows through the Boske port.

---

# Part III: Boske and the studio

## 8. What Boske already has

- Electron 39 desktop with a service supervisor running whisper.cpp's whisper-server (10-model catalog), llama.cpp, Piper TTS.
- Companion feature: global hotkeys on Alt+Space, tray icon, clipboard save/restore, Accessibility permission handling. Missing: injecting text into another app.
- Backend: STT entitlement gate, rate limits, license service with Ed25519 offline leases, voice usage metering in the proxy.
- Cloud STT: `voxtral-mini-latest` via `api.mistral.ai`, batch only, 60-minute max per request, usage in seconds from Mistral's own count, 240 minutes per seat per month, hard 429 beyond.
- `AGENTS.md` forbids Python on the desktop, so Murmur cannot be embedded as-is.
- Both apps are Canopy Studio products; Murmur's design manifest targets Boske moss.

## 9. Reuse options

| Option | Effort | Call |
|--------|--------|------|
| A. Murmur Pro included in every Boske seat, license from Boske's service | Low | First |
| D. Boske ID as the one account across Boske and Murmur Pro | Low to medium | With A |
| C. Murmur uses Boske's running whisper-server and llama-server when installed | Medium | After A and D |
| B. Dictate-anywhere inside Boske Electron, "powered by Murmur" | High | Own plan |

## 10. Portfolio

Canopy Studio: Boske, Grove Fit, Little Bean, Murmur, BearBell live; Minne, Carnet, Vardn coming. Boske open tools: Grove Port, Boske Pulse. Outside the brand: Savvo (paused, already uses whisper.rn and multi-provider LLM chat), Homeapp, Grand Livre, Salad, M-M Cockpit.

Engines worth naming and their reach:

| Engine | Today | Proposed |
|--------|-------|----------|
| Grove models (Boske Labs): Seed 3B → Forest 24B on llama.cpp. Relabelled Mistral weights, no tuned checkpoint yet | Boske chat, Grove Fit catalog | Murmur Pro cleanup, Savvo |
| Murmur engine: local STT | The Murmur app | Boske Voice local mode, voice notes in Minne, Carnet, Little Bean, Vardn, Savvo |
| Boske Sources: rag-api | Boske | Boske |
| Boske ID: license service, offline leases | Boske seats | Murmur Pro, Minne, Carnet |
| Grove Port: open import/export | Boske | Boske |

Arrangement: keep the consumer brands, name the engines, ship "Boske Voice, powered by Murmur" first. Folding Murmur into Boske loses the free funnel and the only public-repo audience. A studio-wide membership is premature. Say "powered by Grove models" only once Boske Labs ships a tuned or quantised model.

---

# Part IV: Pricing

## 11. Constraints

- Murmur is MIT. Gate Pro by keeping the smart layer out of the public repo, or by moving to source-available, and by selling the signed build, models and support. VoiceInk and MacWhisper prove people pay for the convenient build.
- Local inference costs nothing per user, so never cap local use.
- Cloud costs cents, so a cloud tier can be cheap and still carry margin.
- Website code prices Boske seats at €19 / €39 / €65; the brand document says €19 / €49 / €59 / €65. Reconcile before publishing any speech allowance.

## 12. The ladder

| Tier | Price | Contents | Cost to serve |
|------|-------|----------|---------------|
| Murmur Free | €0, unlimited | Local Voxtral or Whisper, toggle and push-to-talk, history, file transcription, basic vocabulary, signed auto-updating build, one-time 60-minute cloud trial | €0.22 once for the trial |
| Murmur Pro | $49 once, or any Boske seat; 3-Mac pack $99 | Local cleanup with modes and tone, context awareness, unlimited vocabulary and replacements with CSV, snippets, coding mode, bring-your-own-key cloud free, priority support | €0 |
| Murmur Cloud | €5/month or €48/year | Voxtral through the Boske proxy (European provider), cloud cleanup with Mistral models, 15 hours a month (~135,000 words), then automatic local fallback, usage meter | Typical €0.66, worst €3.31 |
| Boske seats | Included | Cloud seat 4 hours today, Team 10 hours proposed, top-up €5 per 15 hours on the same SKU | €0.72 / €1.80 per seat |

Per-user monthly cost at $0.003/min batch plus Ministral 8B cleanup plus 20% overhead: 0.5 h €0.11 · 3 h €0.66 · 15 h €3.31 · 40 h €8.81. A €5 plan breaks even near 22 hours. Realtime cloud would double the transcription line and needs its own tier if ever offered. Market in words, meter in seconds.

Positioning: five times cheaper than Superwhisper lifetime on Pro, under every cloud subscription on Cloud, the only dictation app bundled into a private AI workspace, and the only one that keeps working when the allowance ends.

## 13. What Boske must add to sell Murmur Cloud

1. Promote the legacy `cloud_voice` add-on to a sellable one with a Stripe price. Entitlement compute already supports it. Low.
2. An auth path for the Mac app: reuse the desktop device-linking lease flow or add a durable per-user API key. Medium.
3. Replace the hard 429 with local fallback driven by the existing `/v1/voice/usage` endpoint, plus an optional top-up SKU. Low.
4. Show usage in Murmur Settings and on the Boske account page. Low.
5. Per-modality cost pin instead of one shared knob across chat tokens and audio seconds. Medium.
6. Pin the Voxtral version, write the missing STT decision record, confirm Mistral's data residency and zero-retention terms. Low.

---

# Part V: Verification and sources

| Claim | Verdict | Confidence |
|-------|---------|------------|
| Superwhisper $8.49/mo, $84.99/yr, $249.99 lifetime, 15-min trial, 40% student, 30-day refund | Confirmed by three secondary sources; the ~$849 figure is an outlier | Medium |
| Superwhisper 2026 changes (Cohere, Grok, S1-mini, Claude Code, Parakeet) | Confirmed; exact version-to-date mapping and CSV import unconfirmed | Medium |
| Wispr Flow caps, price, $280M at $2B | Confirmed | High |
| Willow caps and prices | Confirmed; team $12 monthly or $10 annual; 20 Scribe uses and 5-min sessions unconfirmed | Medium |
| Aqua, Typeless, Monologue | Confirmed | High |
| MacWhisper prices | Confirmed | Medium |
| VoiceInk | Corrected: $29/$49/$69 since 1 Aug 2026 | Medium |
| BetterDictation, Spokenly, Voibe | Confirmed | Medium |
| Voxtral $0.003 batch, $0.006 realtime, released 4 Feb 2026 | Confirmed | High |
| OpenAI transcription prices | Confirmed | High |
| Apple SpeechAnalyzer | On-device confirmed; language count and whether system dictation uses it: unverified | Low |
| Handy identity | Corrected: cjpais/Handy, MIT, ~30k stars, Moonshine added Feb 2026 | High |
| Ministral 8B $0.10/M | Confirmed; Ministral 3B $0.04 unconfirmed, likely $0.10 | Medium |
| Voxtral Realtime open weights, Apache 2.0, MLX 4-bit on 16 GB | Confirmed | High |
| Voxtral WER superiority over Whisper on FR/NL/DE | Mistral's own benchmarks only | Low, run a bake-off |

Sources: Superwhisper blog (s1, cohere, claude-code) and terms; spokenly.app, getvoibe.com, usevoicy.com pricing pages; TechCrunch and Dealroom on Wispr; aquavoice.com/pricing; monologue.to/pricing; every.to; getvoibe.com VoiceInk pricing; lifetimo BetterDictation; MarkTechPost and The Decoder on Voxtral; OpenRouter model pages; costgoat and tokenmix on OpenAI; Hugging Face model cards (mistralai/Voxtral-Mini-3B-2507, mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit, kyutai/stt-1b-en_fr-mlx, nvidia/parakeet-tdt-0.6b-v3, nvidia/canary-1b-v2); Open ASR Leaderboard paper; github.com/cjpais/Handy; github.com/T0mSIlver/localvoxtral; Apple WWDC25 session 277; the murmur, boske and canopystudio-website repositories and twelve sibling repositories read directly.
