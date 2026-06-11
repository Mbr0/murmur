<p align="center">
  <img src="assets/logos/logo_rounded.png" alt="Murmur" width="120">
</p>

<h1 align="center">Murmur</h1>

<p align="center">
  Local speech-to-text for macOS — press a hotkey, speak, text appears at your cursor.
</p>

<p align="center">
  <strong>100% on-device</strong> · Whisper · No cloud · Menu bar app
</p>

<br>

## Download

Get the latest **`.dmg`** from [GitHub Releases](https://github.com/Mbr0/murmur/releases).

1. Open the DMG and drag **Murmur** to **Applications**
2. Launch Murmur from Applications
3. Grant **Microphone** when prompted
4. For paste-at-cursor: menu bar → **Enable Shortcut Permission…** → allow **Accessibility**

## How it works

| Step | What to do |
|------|------------|
| 1 | Click where you want text |
| 2 | Press **⌥ Space** to start recording |
| 3 | Speak |
| 4 | Press **⌥ Space** again to stop |
| 5 | Text is pasted and copied to the clipboard |

Change the shortcut anytime in **Settings**.

## Permissions

| Permission | Required for |
|------------|--------------|
| **Microphone** | Recording your voice |
| **Accessibility** | Pasting text at the cursor |

The global shortcut does **not** require Input Monitoring.

## Settings

Open **Settings** from the menu bar:

- Whisper model (`tiny` → `large`)
- Custom keyboard shortcut
- Dark / light / system appearance
- Optional local history and audio retention
- Delete all local data

## Menu bar

| State | What you see |
|-------|----------------|
| Ready | Waveform icon |
| Recording | Red indicator |
| Working | Processing spinner |

## Privacy

All transcription runs **locally** on your Mac using [OpenAI Whisper](https://github.com/openai/whisper). No audio or text is sent to the cloud. Optional history and audio files are stored only on your machine under `~/.murmur_*`.

## Troubleshooting

<details>
<summary><strong>Shortcut not working</strong></summary>

Check the shortcut in **Settings** (default: **⌥ Space**). Quit Murmur completely and reopen it after changing the shortcut.
</details>

<details>
<summary><strong>Nothing pasted at the cursor</strong></summary>

Enable **Accessibility** for Murmur: menu bar → **Enable Shortcut Permission…**, then quit and reopen the app.
</details>

<details>
<summary><strong>No audio / model loading forever</strong></summary>

Allow **Microphone** access in System Settings. On first launch, Whisper downloads a model — this can take a minute depending on your connection and model size.
</details>

<details>
<summary><strong>Transcription is slow</strong></summary>

Use a smaller model in **Settings** (`tiny` or `base` for speed, `medium` or `large` for accuracy).
</details>

## Development

Requires macOS and Python 3.12+.

```bash
git clone https://github.com/Mbr0/murmur.git
cd murmur
chmod +x scripts/setup.sh scripts/run.sh
./scripts/setup.sh
./scripts/run.sh
```

Run tests:

```bash
source venv/bin/activate
python -m unittest discover -s tests -p "test_*.py"
```

## License

[MIT](LICENSE)
