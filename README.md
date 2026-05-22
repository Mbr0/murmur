# 🎤 Murmur

A simple, **100% local** speech-to-text app for macOS. Like SuperWhisper, but free!

## Features

- 🎯 **Simple**: Press hotkey → Speak → Text appears where your cursor is
- 🔒 **100% Local**: All processing on your Mac, nothing sent to the cloud
- ⚡ **Fast**: Uses OpenAI Whisper optimized for Apple Silicon
- 📍 **Menu Bar**: Lives in your menu bar next to battery/wifi icons

## Quick Start

### 1. Setup (one time)

```bash
cd murmur
chmod +x setup.sh run.sh
./setup.sh
```

### 2. Run

```bash
./run.sh
```

Or manually:
```bash
cd murmur
source venv/bin/activate
python murmur.py
```

## Usage

| Action | Shortcut |
|--------|----------|
| Start/Stop Recording | `⌘ + ⌥ + Space` |
| Cancel Recording | `Escape` |

1. Click where you want to type text
2. Press `⌘ + ⌥ + Space` to start recording
3. Speak clearly
4. Press `⌘ + ⌥ + Space` again to stop
5. Text is automatically typed and copied to clipboard!

## Menu Bar Icon States

| Icon | State |
|------|-------|
| 🎤 | Ready |
| ⏳ | Loading model / Transcribing |
| 🔴 | Recording |
| ❌ | Error |

## Permissions Required

Go to **System Settings → Privacy & Security** and grant access:

1. **Microphone** - To record your voice
2. **Accessibility** - To type text automatically  
3. **Input Monitoring** - For global keyboard shortcuts

## Configuration

Edit `murmur.py` to change:

```python
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
```

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39 MB | Fastest | Basic |
| base | 142 MB | Fast | Good |
| small | 466 MB | Medium | Better |
| medium | 1.5 GB | Slow | Great |
| large | 2.9 GB | Slowest | Best |

For Apple Silicon, `base` or `small` works great!

## Language

By default, it's set to English. To auto-detect language or use another:

```python
# In transcribe() function, change:
language="en"  # to your language code, or None for auto-detect
```

## Troubleshooting

### "Model is still loading..."
Wait a few seconds on first launch while the model downloads.

### Hotkey not working
Grant **Input Monitoring** permission in System Settings.

### No text typed
Grant **Accessibility** permission in System Settings.

### No audio recorded
Grant **Microphone** permission in System Settings.

### Slow transcription
Try `MODEL_SIZE = "tiny"` for faster (but less accurate) results.

## Tech Stack

- **rumps** - macOS menu bar app framework
- **openai-whisper** - OpenAI's speech recognition model
- **sounddevice** - Audio recording
- **pynput** - Global keyboard shortcuts

## License

MIT - Do whatever you want with it! 🎉

## Local-Private Launch Checklist

- [ ] Runtime checks pass (`/usr/bin/python3 -m unittest discover -s tests -p "test_*.py"`)
- [ ] App builds and signs locally (`./build_pyinstaller.sh`)
- [ ] Production signing identity configured for release builds (`CODE_SIGN_IDENTITY`)
- [ ] Optional notarization flow validated (`NOTARIZE=true` with Apple credentials)
- [ ] Privacy claims match implementation (local-only processing + local storage)
- [ ] Legacy migration works (`~/.mywhisper_*` migrates to `~/.murmur_*`)
- [ ] Murmur page on `canopystudio.eu/murmur` includes download link (when public)
