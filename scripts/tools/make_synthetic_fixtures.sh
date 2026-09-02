#!/usr/bin/env bash
# Synthetic smoke-test fixtures for scripts/tools/bakeoff.py.
#
# These are NOT the real fixtures for decision D1 (see
# tests/fixtures/audio/README.md). macOS `say` output does not represent
# real dictation and must never be used to decide the primary local engine.
# This script exists only so the bake-off harness itself can be exercised
# end to end without recording real audio first.
#
# Usage: scripts/tools/make_synthetic_fixtures.sh
# Requires macOS (`say`, `afconvert`) and python3.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURES_DIR="$REPO_ROOT/tests/fixtures/audio"

# Deliberately avoids bash 4 associative arrays (macOS /bin/bash is 3.2).
voice_for() {
  case "$1" in
    en) echo "Samantha" ;;
    fr) echo "Amelie" ;;
    nl) echo "Xander" ;;
    de) echo "Anna" ;;
    *) echo "unknown language: $1" >&2; exit 1 ;;
  esac
}

sentence_for() {
  case "$1" in
    en) echo "Please send the quarterly report to the finance team by Friday afternoon." ;;
    fr) echo "Merci d'envoyer le rapport trimestriel à l'équipe financière avant vendredi après-midi." ;;
    nl) echo "Stuur het kwartaalrapport voor vrijdagmiddag naar het financiële team." ;;
    de) echo "Bitte senden Sie den Quartalsbericht bis Freitagnachmittag an das Finanzteam." ;;
    *) echo "unknown language: $1" >&2; exit 1 ;;
  esac
}

mkdir -p "$FIXTURES_DIR"

for lang in en fr nl de; do
  voice="$(voice_for "$lang")"
  sentence="$(sentence_for "$lang")"

  lang_dir="$FIXTURES_DIR/$lang"
  mkdir -p "$lang_dir"

  aiff_path="$(mktemp -t "murmur_bakeoff_${lang}").aiff"
  wav_path="$lang_dir/synthetic.wav"

  say -v "$voice" -o "$aiff_path" "$sentence"
  afconvert -f WAVE -d LEI16@16000 -c 1 "$aiff_path" "$wav_path"
  rm -f "$aiff_path"

  echo "Wrote $wav_path"
done

python3 - "$FIXTURES_DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

fixtures_dir = Path(sys.argv[1])
sentences = {
    "en": "Please send the quarterly report to the finance team by Friday afternoon.",
    "fr": "Merci d'envoyer le rapport trimestriel à l'équipe financière avant vendredi après-midi.",
    "nl": "Stuur het kwartaalrapport voor vrijdagmiddag naar het financiële team.",
    "de": "Bitte senden Sie den Quartalsbericht bis Freitagnachmittag an das Finanzteam.",
}
clips = [
    {"path": f"{lang}/synthetic.wav", "language": lang, "text": text}
    for lang, text in sentences.items()
]
manifest_path = fixtures_dir / "manifest.json"
manifest_path.write_text(json.dumps({"clips": clips}, indent=2) + "\n")
print(f"Wrote {manifest_path}")
PYEOF

echo
echo "Synthetic fixtures are for smoke-testing scripts/tools/bakeoff.py only."
echo "D1 needs real dictation recordings; see tests/fixtures/audio/README.md."
