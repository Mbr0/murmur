#!/usr/bin/env bash
# Synthetic smoke-test fixtures for scripts/tools/bakeoff.py.
#
# These are NOT the real fixtures for decision D1 (see
# tests/fixtures/audio/README.md). macOS `say` output does not represent
# real dictation and must never be used to decide the primary local engine.
# This script exists only so the bake-off harness itself can be exercised
# end to end without recording real audio first.
#
# Generates ten synthetic dictation-style clips per language (en, fr, nl,
# de), each roughly 25-35 words so macOS `say` takes about 8-12 seconds to
# speak them -- the bake-off's latency median only counts clips in that
# window.
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

# Ten original, dictation-style sentences per language: the kind of thing
# someone might speak into a note, message, or email app. Each mentions
# Murmur, Boske, or Canopy Studio, and most carry a date or a number, to
# mimic real dictation content without reproducing any real dictation.
sentence_for() {
  local lang="$1" idx="$2"
  case "$lang" in
    en)
      case "$idx" in
        001) echo "Hi team, this is a quick voice note about the Murmur project -- please push the staging build before five o'clock today so Boske can start testing right away, thanks." ;;
        002) echo "Note to self: follow up with Canopy Studio about the September fourteenth invoice, and remember to attach the updated pricing sheet before sending it to finance this afternoon." ;;
        003) echo "Hey Marc, can you tell the Murmur team that the demo moved to ten thirty on Thursday, and we now have twelve people confirmed for the Boske walkthrough afterward?" ;;
        004) echo "Dear Elena, thank you for the call this morning about Canopy Studio's roadmap -- I have added three action items and will send the full summary by Monday." ;;
        005) echo "Quick reminder that the Murmur onboarding session starts at nine on October the second, and everyone should bring their laptop plus a working microphone for the recording test." ;;
        006) echo "Voice memo for the engineering log: Boske finished the transcription fix around four fifteen this afternoon, and the new build cut average latency by roughly eighteen percent overall, which is great news." ;;
        007) echo "Message for the Canopy Studio partners: the pilot with Murmur is going well, we have forty active users this week, and I would like to schedule a check-in for Friday." ;;
        008) echo "Note for Priya: please reorder the Murmur test microphones before the fourth of November, we only have six units left and the Boske pilot needs at least ten." ;;
        009) echo "Dictating this from the car: tell the Canopy Studio design team the new logo draft looks great, and ask them to send two more color options by Wednesday." ;;
        010) echo "Final note before the weekend: the Murmur bake-off results are in, Boske scored highest on accuracy, and I want to review the full report together on Monday morning." ;;
        *) echo "unknown index: $idx" >&2; exit 1 ;;
      esac
      ;;
    fr)
      case "$idx" in
        001) echo "Bonjour Julie, ceci est un petit message vocal à propos du projet Murmur : merci de pousser la version de test avant dix-sept heures pour que Boske puisse commencer les essais." ;;
        002) echo "Note pour moi-même : relancer Canopy Studio au sujet de la facture du quatorze septembre, et ne pas oublier de joindre la nouvelle grille tarifaire avant de l'envoyer à la comptabilité." ;;
        003) echo "Salut Marc, peux-tu dire à l'équipe Murmur que la démonstration est déplacée à dix heures trente jeudi, avec douze personnes confirmées pour la présentation de Boske ?" ;;
        004) echo "Chère Elena, merci pour l'appel de ce matin au sujet de la feuille de route de Canopy Studio ; j'ai noté trois actions et je t'enverrai le résumé complet lundi." ;;
        005) echo "Petit rappel : la session d'intégration de Murmur commence à neuf heures le deux octobre, et chacun doit apporter son ordinateur avec un microphone qui fonctionne." ;;
        006) echo "Mémo vocal pour le journal d'ingénierie : Boske a terminé la correction de la transcription vers seize heures, et la nouvelle version réduit la latence moyenne d'environ dix-huit pour cent." ;;
        007) echo "Message pour les partenaires de Canopy Studio : le pilote avec Murmur se passe bien, nous avons quarante utilisateurs actifs cette semaine, et j'aimerais programmer un point vendredi." ;;
        008) echo "Note pour Priya : merci de recommander les microphones de test Murmur avant le quatre novembre, il n'en reste que six et le pilote Boske en demande au moins dix." ;;
        009) echo "Je dicte ceci depuis la voiture : dis à l'équipe design de Canopy Studio que la nouvelle proposition de logo est très réussie, et demande deux autres options de couleur avant mercredi." ;;
        010) echo "Dernière note avant le week-end : les résultats du comparatif Murmur sont arrivés, Boske obtient le meilleur score en précision, et je veux revoir le rapport complet ensemble lundi matin." ;;
        *) echo "unknown index: $idx" >&2; exit 1 ;;
      esac
      ;;
    nl)
      case "$idx" in
        001) echo "Hoi team, dit is een korte spraaknotitie over het Murmur project -- kunnen jullie de teststaging voor vijf uur vandaag pushen, zodat Boske meteen kan beginnen met testen?" ;;
        002) echo "Notitie voor mezelf: contact opnemen met Canopy Studio over de factuur van veertien september, en niet vergeten het nieuwe prijsoverzicht toe te voegen voordat we het naar finance sturen." ;;
        003) echo "Hé Marc, kun je het Murmur team laten weten dat de demo is verplaatst naar half elf op donderdag, en dat er nu twaalf mensen bevestigd zijn voor de Boske sessie?" ;;
        004) echo "Beste Elena, bedankt voor het fijne gesprek vanmorgen over de roadmap van Canopy Studio; ik heb drie concrete actiepunten genoteerd en stuur je maandagochtend de volledige samenvatting per e-mail." ;;
        005) echo "Korte herinnering: de Murmur onboarding sessie begint om negen uur op twee oktober, en iedereen moet een laptop en een werkende microfoon meenemen voor de opnametest die middag." ;;
        006) echo "Spraakmemo voor het engineering logboek: Boske heeft de transcriptiefix rond kwart voor vijf afgerond, en de nieuwe build verlaagt de gemiddelde latency met ongeveer achttien procent, wat goed nieuws is voor het hele team." ;;
        007) echo "Bericht voor de partners van Canopy Studio: de pilot met Murmur verloopt goed, we hebben deze week veertig actieve gebruikers, en ik wil graag vrijdag een kort overleg inplannen." ;;
        008) echo "Notitie voor Priya: bestel de Murmur testmicrofoons opnieuw voor vier november, we hebben er nog maar zes en de Boske pilot heeft er minstens tien nodig voor de volgende testronde." ;;
        009) echo "Ik dicteer dit vanuit de auto: zeg tegen het designteam van Canopy Studio dat het nieuwe logo-ontwerp er goed uitziet, en vraag om twee extra kleuropties voor woensdag." ;;
        010) echo "Laatste notitie voor het weekend: de resultaten van de Murmur bake-off zijn binnen, Boske scoort het hoogst op nauwkeurigheid, en ik wil het volledige rapport maandagochtend samen doornemen." ;;
        *) echo "unknown index: $idx" >&2; exit 1 ;;
      esac
      ;;
    de)
      case "$idx" in
        001) echo "Hallo Team, das ist eine kurze Sprachnotiz zum Murmur Projekt -- bitte spielt den Staging Build vor siebzehn Uhr auf, damit Boske gleich mit dem Testen anfangen kann." ;;
        002) echo "Notiz an mich selbst: bei Canopy Studio wegen der Rechnung vom vierzehnten September nachfragen, und die neue Preisliste anhängen, bevor ich sie an die Buchhaltung schicke." ;;
        003) echo "Hallo Marc, kannst du dem Murmur Team sagen, dass die Demo auf halb elf am Donnerstag verschoben wurde, und dass jetzt zwölf Leute für die Boske Präsentation zugesagt haben?" ;;
        004) echo "Liebe Elena, danke für das Gespräch heute Morgen über die Roadmap von Canopy Studio; ich habe drei konkrete Punkte notiert und schicke dir am Montag die vollständige Zusammenfassung." ;;
        005) echo "Kurze Erinnerung: die Murmur Onboarding Sitzung beginnt um neun Uhr am zweiten Oktober, und jeder sollte einen Laptop und ein funktionierendes Mikrofon mitbringen." ;;
        006) echo "Sprachmemo für das Engineering Log: Boske hat den Transkriptionsfehler gegen Viertel vor fünf behoben, und der neue Build senkt die durchschnittliche Latenz um etwa achtzehn Prozent." ;;
        007) echo "Nachricht an die Partner von Canopy Studio: der Pilot mit Murmur läuft gut, wir haben diese Woche vierzig aktive Nutzer, und ich würde gern am Freitag telefonieren." ;;
        008) echo "Notiz für Priya: bitte die Murmur Testmikrofone vor dem vierten November nachbestellen, wir haben nur noch sechs Stück und der Boske Pilot braucht mindestens zehn davon." ;;
        009) echo "Ich diktiere das gerade aus dem Auto: sag dem Designteam von Canopy Studio, dass der neue Logo Entwurf gut aussieht, und frag nach zwei Farboptionen bis Mittwoch." ;;
        010) echo "Letzte Notiz vor dem Wochenende: die Ergebnisse vom Murmur Bake-off sind da, Boske hat bei der Genauigkeit am besten abgeschnitten, und ich möchte den Bericht besprechen." ;;
        *) echo "unknown index: $idx" >&2; exit 1 ;;
      esac
      ;;
    *) echo "unknown language: $1" >&2; exit 1 ;;
  esac
}

mkdir -p "$FIXTURES_DIR"

manifest_data="$(mktemp -t murmur_bakeoff_manifest)"

for lang in en fr nl de; do
  voice="$(voice_for "$lang")"

  lang_dir="$FIXTURES_DIR/$lang"
  mkdir -p "$lang_dir"

  for idx in 001 002 003 004 005 006 007 008 009 010; do
    sentence="$(sentence_for "$lang" "$idx")"
    relpath="$lang/$idx.wav"

    aiff_path="$(mktemp -t "murmur_bakeoff_${lang}_${idx}").aiff"
    wav_path="$lang_dir/$idx.wav"

    say -v "$voice" -o "$aiff_path" "$sentence"
    afconvert -f WAVE -d LEI16@16000 -c 1 "$aiff_path" "$wav_path"
    rm -f "$aiff_path"

    printf '%s\t%s\t%s\n' "$lang" "$relpath" "$sentence" >> "$manifest_data"

    echo "Wrote $wav_path"
  done
done

python3 - "$FIXTURES_DIR" "$manifest_data" <<'PYEOF'
import json
import sys
from pathlib import Path

fixtures_dir = Path(sys.argv[1])
data_path = Path(sys.argv[2])

clips = []
with data_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        lang, relpath, text = line.split("\t", 2)
        clips.append({"path": relpath, "language": lang, "text": text})

manifest_path = fixtures_dir / "manifest.json"
manifest_path.write_text(json.dumps({"clips": clips}, indent=2) + "\n")
print(f"Wrote {manifest_path}")
PYEOF

rm -f "$manifest_data"

echo
echo "Synthetic fixtures are for smoke-testing scripts/tools/bakeoff.py only."
echo "D1 needs real dictation recordings; see tests/fixtures/audio/README.md."
