#!/bin/bash
# Build Murmur.app for macOS (local ad-hoc or signed/notarized release)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source venv/bin/activate

APP_NAME="Murmur"
APP_BUNDLE="dist/${APP_NAME}.app"
BUNDLE_ID="${BUNDLE_ID:-com.canopystudio.murmur}"
APP_VERSION="${APP_VERSION:-1.0.0}"
CODE_SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-}"
NOTARIZE="${NOTARIZE:-false}"
APPLE_ID="${APPLE_ID:-}"
APPLE_TEAM_ID="${APPLE_TEAM_ID:-}"
APPLE_APP_SPECIFIC_PASSWORD="${APPLE_APP_SPECIFIC_PASSWORD:-}"
ENTITLEMENTS="${ENTITLEMENTS:-entitlements.plist}"
ZIP_PATH="dist/${APP_NAME}.zip"
WHISPER_SERVER="vendor/whispercpp/whisper-server"
# "release" once a Developer ID identity signs the build, "internal" otherwise.
# The About text reads Contents/Resources/build_info.json to say which it is.
BUILD_CHANNEL="${BUILD_CHANNEL:-}"

if [ ! -f "${WHISPER_SERVER}" ]; then
    echo "ERROR: ${WHISPER_SERVER} is missing."
    echo "       Build it first: bash scripts/tools/fetch_whispercpp.sh"
    echo "       (decision D2 — the bundled whisper.cpp server is the default engine.)"
    exit 1
fi

if ! python -c "import PyInstaller" 2>/dev/null; then
    pip install pyinstaller
fi

echo "Running unit tests..."
python -m unittest discover -s tests -p "test_*.py" -q

echo "Building ${APP_NAME}.app (this may take several minutes)..."
rm -rf build dist
pyinstaller Murmur.spec --noconfirm

if [ ! -d "${APP_BUNDLE}" ]; then
    echo "ERROR: Expected ${APP_BUNDLE} was not created."
    exit 1
fi

# engines/whispercpp.py resolves <sys._MEIPASS>/bin/whisper-server at runtime.
if ! find "${APP_BUNDLE}/Contents" -type f -name whisper-server -path '*/bin/*' | grep -q .; then
    echo "ERROR: bin/whisper-server is not in ${APP_BUNDLE}; the default engine would not start."
    exit 1
fi

if find "${APP_BUNDLE}/Contents" \( -name torch -o -name 'libtorch*' \) -print -quit | grep -q .; then
    echo "ERROR: the bundle still contains torch. Check the excludes list in Murmur.spec."
    exit 1
fi

# Build marker read by the About text: ad-hoc builds label themselves internal.
RESOURCES_DIR="${APP_BUNDLE}/Contents/Resources"
mkdir -p "${RESOURCES_DIR}"
if [ -n "${CODE_SIGN_IDENTITY}" ]; then
    BUILD_SIGNED=true
    BUILD_CHANNEL="${BUILD_CHANNEL:-release}"
else
    BUILD_SIGNED=false
    BUILD_CHANNEL="${BUILD_CHANNEL:-internal}"
fi
if [ "${NOTARIZE}" = "true" ]; then
    BUILD_NOTARIZED=true
else
    BUILD_NOTARIZED=false
fi
cat > "${RESOURCES_DIR}/build_info.json" <<EOF
{
  "signed": ${BUILD_SIGNED},
  "channel": "${BUILD_CHANNEL}",
  "version": "${APP_VERSION}",
  "notarized": ${BUILD_NOTARIZED},
  "team_id": "${APPLE_TEAM_ID}",
  "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
echo "Build marker: channel=${BUILD_CHANNEL} signed=${BUILD_SIGNED}"

/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString ${APP_VERSION}" "${APP_BUNDLE}/Contents/Info.plist" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string ${APP_VERSION}" "${APP_BUNDLE}/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion ${APP_VERSION}" "${APP_BUNDLE}/Contents/Info.plist" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string ${APP_VERSION}" "${APP_BUNDLE}/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :LSUIElement true" "${APP_BUNDLE}/Contents/Info.plist" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "${APP_BUNDLE}/Contents/Info.plist"

xattr -cr "${APP_BUNDLE}"

# Inside-out signing (no --deep): nested Mach-O files first, then main executable, then bundle.
# --deep re-signs in an unpredictable order and can break nested signatures; Apple recommends
# signing each component individually from the inside out.
sign_macho() {
    local target="$1"
    local args=(--force --timestamp)
    if [ -n "${CODE_SIGN_IDENTITY}" ]; then
        args+=(--options runtime)
        if [ -f "${ENTITLEMENTS}" ]; then
            args+=(--entitlements "${ENTITLEMENTS}")
        fi
        args+=(-s "${CODE_SIGN_IDENTITY}")
    else
        args+=(-s -)
    fi
    codesign "${args[@]}" "${target}"
}

MAIN_EXECUTABLE="${APP_BUNDLE}/Contents/MacOS/${APP_NAME}"

if [ -n "${CODE_SIGN_IDENTITY}" ]; then
    echo "Signing with: ${CODE_SIGN_IDENTITY} (inside-out, no --deep)"
else
    echo "WARNING: CODE_SIGN_IDENTITY not set — ad-hoc signing for local use only."
    echo "Public release requires a Developer ID certificate and NOTARIZE=true."
    echo "NOTE: Ad-hoc DMG installs need a fresh Accessibility grant after each rebuild."
    echo "      Global shortcuts from source (python murmur.py) use the Python binary,"
    echo "      which is a different permission entry than /Applications/Murmur.app."
fi

while IFS= read -r -d '' item; do
    sign_macho "${item}"
done < <(find "${APP_BUNDLE}/Contents" -type f \( -name '*.dylib' -o -name '*.so' \) -print0)

while IFS= read -r -d '' item; do
    if [ "${item}" != "${MAIN_EXECUTABLE}" ] && file "${item}" | grep -q 'Mach-O'; then
        sign_macho "${item}"
    fi
done < <(find "${APP_BUNDLE}/Contents" -type f -perm +111 -print0)

sign_macho "${MAIN_EXECUTABLE}"
sign_macho "${APP_BUNDLE}"
codesign --verify --strict --verbose=2 "${APP_BUNDLE}"

if [ "${NOTARIZE}" = "true" ]; then
    # A stored keychain profile keeps the password out of the environment as
    # well as out of argv; without one, "@env:" at least keeps it out of argv,
    # where `ps` shows every argument to every user on the machine.
    if [ -n "${NOTARY_KEYCHAIN_PROFILE:-}" ]; then
        NOTARY_ARGS=(--keychain-profile "${NOTARY_KEYCHAIN_PROFILE}")
        if [ -z "${CODE_SIGN_IDENTITY}" ]; then
            echo "NOTARIZE=true requires CODE_SIGN_IDENTITY."
            exit 1
        fi
    else
        if [ -z "${CODE_SIGN_IDENTITY}" ] || [ -z "${APPLE_ID}" ] || [ -z "${APPLE_TEAM_ID}" ] || [ -z "${APPLE_APP_SPECIFIC_PASSWORD}" ]; then
            echo "NOTARIZE=true requires CODE_SIGN_IDENTITY, APPLE_ID, APPLE_TEAM_ID, APPLE_APP_SPECIFIC_PASSWORD"
            echo "(or NOTARY_KEYCHAIN_PROFILE — see RELEASE_SIGNING.md)."
            exit 1
        fi
        NOTARY_ARGS=(
            --apple-id "${APPLE_ID}"
            --team-id "${APPLE_TEAM_ID}"
            --password "@env:APPLE_APP_SPECIFIC_PASSWORD"
        )
        export APPLE_APP_SPECIFIC_PASSWORD
    fi
    ditto -c -k --sequesterRsrc --keepParent "${APP_BUNDLE}" "${ZIP_PATH}"
    xcrun notarytool submit "${ZIP_PATH}" "${NOTARY_ARGS[@]}" --wait
    xcrun stapler staple "${APP_BUNDLE}"
    echo "Notarization complete."
fi

echo ""
echo "======================================"
echo "Build complete: ${APP_BUNDLE}"
echo ""
echo "Install: drag Murmur.app to /Applications"
echo "First launch: grant Microphone; grant Accessibility for paste-at-cursor"
if [ "${NOTARIZE}" != "true" ]; then
    echo ""
    echo "NOTE: Public release requires NOTARIZE=true with Apple credentials."
    echo "      See RELEASE_SIGNING.md. Use ./create_dmg.sh to package and sign the DMG."
fi
echo "======================================"
