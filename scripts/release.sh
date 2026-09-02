#!/bin/bash
# Production release: signed/notarized Murmur.app + DMG

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

APP_VERSION="${APP_VERSION:-1.0.0}"
NOTARIZE="${NOTARIZE:-true}"

if [ -z "${CODE_SIGN_IDENTITY:-}" ]; then
    echo "ERROR: CODE_SIGN_IDENTITY is required for production release."
    echo "See RELEASE_SIGNING.md"
    exit 1
fi

for var in APPLE_ID APPLE_TEAM_ID APPLE_APP_SPECIFIC_PASSWORD; do
    if [ "${NOTARIZE}" = "true" ] && [ -z "${!var:-}" ]; then
        echo "ERROR: ${var} is required when NOTARIZE=true."
        echo "See RELEASE_SIGNING.md"
        exit 1
    fi
done

if [ ! -f "${ROOT}/vendor/whispercpp/whisper-server" ]; then
    echo "Building the bundled whisper.cpp server..."
    bash "${ROOT}/scripts/tools/fetch_whispercpp.sh"
fi

echo "Building Murmur ${APP_VERSION} (signed + notarized)..."
export NOTARIZE
"${ROOT}/scripts/build_pyinstaller.sh"

echo "Creating DMG..."
"${ROOT}/scripts/create_dmg.sh"

DMG_PATH="dist/Murmur-${APP_VERSION}.dmg"

# The app inside is already notarized and stapled; the image itself needs its
# own ticket so Gatekeeper clears it before the user ever opens it, and so the
# updater's `spctl --assess --type open` check passes offline.
if [ "${NOTARIZE}" = "true" ]; then
    echo "Notarizing the DMG..."
    xcrun notarytool submit "${DMG_PATH}" \
        --apple-id "${APPLE_ID}" \
        --team-id "${APPLE_TEAM_ID}" \
        --password "${APPLE_APP_SPECIFIC_PASSWORD}" \
        --wait
    xcrun stapler staple "${DMG_PATH}"
    xcrun stapler validate "${DMG_PATH}"
fi

# After stapling: the ticket changes the file, so the digest must come last.
DMG_SHA256="$(shasum -a 256 "${DMG_PATH}" | awk '{print $1}')"
echo "${DMG_SHA256}  $(basename "${DMG_PATH}")" > "${DMG_PATH}.sha256"

# Homebrew cask: the formula ships a placeholder, and this rewrites the version
# and sha256 lines in place after a successful release. Set UPDATE_CASK=false to
# release without touching it (for example when re-cutting the same version).
CASK="${ROOT}/scripts/homebrew/murmur.rb"
if [ "${UPDATE_CASK:-true}" = "true" ] && [ -f "${CASK}" ]; then
    /usr/bin/sed -i '' \
        -e "s/^  version \".*\"$/  version \"${APP_VERSION}\"/" \
        -e "s/^  sha256 \".*\"$/  sha256 \"${DMG_SHA256}\"/" \
        "${CASK}"
    echo "Updated ${CASK} (version ${APP_VERSION}, sha256 ${DMG_SHA256})"
fi

echo ""
echo "======================================"
echo "Production release ready"
echo ""
echo "  dist/Murmur.app"
echo "  ${DMG_PATH}"
echo "  ${DMG_PATH}.sha256  (${DMG_SHA256})"
echo ""
echo "Validate:"
echo "  codesign --verify --strict --verbose=2 dist/Murmur.app"
echo "  spctl --assess --type execute --verbose dist/Murmur.app"
echo "  spctl --assess --type open --context context:primary-signature ${DMG_PATH}"
echo "======================================"
