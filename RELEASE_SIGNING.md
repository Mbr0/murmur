# Murmur Release Signing & Notarization

This project now supports local builds and production-grade signing/notarization from the same script.

## Local Build (Ad-Hoc Signing)

```bash
./build_pyinstaller.sh
```

If `CODE_SIGN_IDENTITY` is not set, the script signs ad-hoc (`-s -`), suitable for local development only.

## Production Build (Developer ID + Notarization)

Set the required environment variables, then run the same script:

```bash
export CODE_SIGN_IDENTITY="Developer ID Application: Your Company (TEAMID)"
export BUNDLE_ID="com.canopystudio.murmur"
export NOTARIZE=true
export APPLE_ID="you@example.com"
export APPLE_TEAM_ID="TEAMID"
export APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"

./build_pyinstaller.sh
```

## What the script does

1. Builds the `.app` bundle using PyInstaller.
2. Sets required `Info.plist` flags for menu bar and microphone usage.
3. Signs with timestamp/runtime options when `CODE_SIGN_IDENTITY` is set.
4. If `NOTARIZE=true`, submits a zipped app via `xcrun notarytool`, waits for completion, then staples the ticket.

## Validation commands

```bash
codesign --verify --deep --strict --verbose=2 dist/Murmur.app
spctl --assess --type execute --verbose dist/Murmur.app
```
