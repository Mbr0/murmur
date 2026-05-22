#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
source venv/bin/activate

APP_NAME="Murmur"
APP_BUNDLE="dist/${APP_NAME}.app"
BUNDLE_ID="${BUNDLE_ID:-com.canopystudio.murmur}"
CODE_SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-}"
NOTARIZE="${NOTARIZE:-false}"
APPLE_ID="${APPLE_ID:-}"
APPLE_TEAM_ID="${APPLE_TEAM_ID:-}"
APPLE_APP_SPECIFIC_PASSWORD="${APPLE_APP_SPECIFIC_PASSWORD:-}"
ZIP_PATH="dist/${APP_NAME}.zip"

# Clean previous builds
rm -rf build dist "${APP_NAME}.spec"

# Build the app with PyInstaller
pyinstaller \
    --name "${APP_NAME}" \
    --windowed \
    --onedir \
    --osx-bundle-identifier "${BUNDLE_ID}" \
    --icon "Murmur.icns" \
    --add-data "logo_menu_white.png:." \
    --add-data "logo_rounded.png:." \
    --add-data "icon_recording.png:." \
    --add-data "icon_processing.png:." \
    --add-data "icon_error.png:." \
    --add-data "history_window.py:." \
    --add-data "settings_window.py:." \
    --add-data "venv/lib/python3.12/site-packages/whisper/assets:whisper/assets" \
    --hidden-import rumps \
    --hidden-import whisper \
    --hidden-import torch \
    --hidden-import sounddevice \
    --hidden-import scipy \
    --hidden-import numpy \
    --hidden-import pynput \
    --hidden-import PyObjCTools \
    --hidden-import PyObjCTools.AppHelper \
    --collect-all whisper \
    --collect-all torch \
    --noconfirm \
    murmur.py

# Check if .app bundle was created
if [ -d "${APP_BUNDLE}" ]; then
    # Set LSUIElement to hide from dock
    /usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "${APP_BUNDLE}/Contents/Info.plist" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Set :LSUIElement true" "${APP_BUNDLE}/Contents/Info.plist"

    # Add microphone usage description
    /usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string 'Murmur needs microphone access to transcribe speech.'" "${APP_BUNDLE}/Contents/Info.plist" 2>/dev/null

    # Clean extended attributes and sign
    xattr -cr "${APP_BUNDLE}"
    if [ -n "${CODE_SIGN_IDENTITY}" ]; then
        echo "Signing app with identity: ${CODE_SIGN_IDENTITY}"
        codesign --force --deep --timestamp --options runtime -s "${CODE_SIGN_IDENTITY}" "${APP_BUNDLE}"
    else
        echo "WARNING: CODE_SIGN_IDENTITY not set. Using ad-hoc signing for local-only builds."
        codesign --force --deep -s - "${APP_BUNDLE}"
    fi

    if [ "${NOTARIZE}" = "true" ]; then
        if [ -z "${CODE_SIGN_IDENTITY}" ] || [ -z "${APPLE_ID}" ] || [ -z "${APPLE_TEAM_ID}" ] || [ -z "${APPLE_APP_SPECIFIC_PASSWORD}" ]; then
            echo "NOTARIZE=true requires CODE_SIGN_IDENTITY, APPLE_ID, APPLE_TEAM_ID, APPLE_APP_SPECIFIC_PASSWORD."
            exit 1
        fi

        ditto -c -k --sequesterRsrc --keepParent "${APP_BUNDLE}" "${ZIP_PATH}"
        xcrun notarytool submit "${ZIP_PATH}" \
            --apple-id "${APPLE_ID}" \
            --team-id "${APPLE_TEAM_ID}" \
            --password "${APPLE_APP_SPECIFIC_PASSWORD}" \
            --wait
        xcrun stapler staple "${APP_BUNDLE}"
        echo "Notarization complete and ticket stapled."
    fi

    echo ""
    echo "======================================"
    echo "Build complete!"
    echo "Your app is at: ${APP_BUNDLE}"
    echo ""
    echo "To install, drag ${APP_NAME}.app to your Applications folder"
    echo "======================================"
else
    echo ""
    echo "======================================"
    echo "Build complete!"
    echo "Your app is at: dist/Murmur/"
    echo ""
    echo "Run with: ./dist/${APP_NAME}/${APP_NAME}"
    echo "======================================"
fi
