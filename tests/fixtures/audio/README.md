# Bake-off audio fixtures

Real dictation-style audio used by `scripts/tools/bakeoff.py` to compare
transcription engines for decision D1 (see
`docs/work/active/2026-09-02-murmur-v2/decisions.md`).

## What belongs here

- 10 short clips per language: **EN, FR, NL, DE** (40 total).
- Dictation style — the kind of sentence someone would actually speak into
  Murmur: messages, notes, short instructions. Not read-aloud prose, not
  scripted news copy.
- Recorded by us. Never third-party audio (no podcasts, no audiobooks, no
  scraped recordings) — we do not have redistribution rights for those, and
  D1 must be decided on audio that represents real Murmur usage.
- 16 kHz, mono, 16-bit PCM WAV, roughly 10 seconds each. 8-12 s is fine —
  the harness only takes the latency median over clips in that range.

## Layout

```
tests/fixtures/audio/
  manifest.json          <- see "Size budget" below for when this is committed
  en/001.wav ... en/010.wav
  fr/001.wav ... fr/010.wav
  nl/001.wav ... nl/010.wav
  de/001.wav ... de/010.wav
```

`manifest.example.json` in this directory is committed and shows the same
shape with a few entries and no audio, so the manifest-loading code (and its
tests) exercises the real format without needing recordings.

## Recording

Record on a Mac with the built-in mic or a headset, one sentence per clip,
in a quiet room. Any recorder that produces a mono file works — QuickTime
Player's "New Audio Recording", Voice Memos, etc.

Convert to the exact fixture format with `afconvert` (ships with macOS):

```bash
afconvert -f WAVE -d LEI16@16000 -c 1 input.m4a en/001.wav
```

Or with `ffmpeg`:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le en/001.wav
```

## Writing the manifest

`manifest.json` lists every clip with its exact reference transcript. The
harness normalises both sides before scoring (see `word_error_rate` in
`scripts/tools/bakeoff.py`: lowercase, punctuation stripped, whitespace
collapsed), so casing and punctuation in the manifest text don't affect WER
— write it the way you actually said it, for readability:

```json
{
  "clips": [
    {"path": "en/001.wav", "language": "en", "text": "Send the invoice to accounting by Friday."},
    {"path": "fr/001.wav", "language": "fr", "text": "Envoie la facture à la comptabilité avant vendredi."}
  ]
}
```

`path` is relative to this directory. `language` must be one of `en`, `fr`,
`nl`, `de` — `scripts/tools/bakeoff.py` rejects anything else when loading
the manifest.

## Size budget

At 16 kHz mono 16-bit, a 10 s clip is about 320 KB. Four languages times ten
clips each comes to roughly 12-13 MB, which is over this repo's 5 MB budget
for this directory. Do not commit the `.wav` files (or `manifest.json`,
since it's only useful with the audio it references) once the directory
would cross 5 MB — keep the real set local, share it out of band with
whoever runs the bake-off, and leave only `manifest.example.json` committed.
If you want a committed set, trim to fewer clips or ~5 s each so the total
stays under budget.

## Regenerating

There is no scripted way to regenerate real recordings — record clips
following the steps above, convert them with `afconvert`/`ffmpeg`, and
update `manifest.json`.

For a synthetic smoke test of the harness itself (not real dictation, not
useful for D1), run:

```bash
scripts/tools/make_synthetic_fixtures.sh
```

This uses macOS `say` to generate one clip per language and writes a
matching `manifest.json`. It exists only to exercise
`scripts/tools/bakeoff.py` end to end — **D1 must be decided from real
recordings**, never from `say` output.
