#!/bin/bash
# Install a locally built Murmur.app to /Applications

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
APP="dist/Murmur.app"
TARGET="/Applications/Murmur.app"

if [ ! -d "${APP}" ]; then
    echo "Build Murmur.app first: ./scripts/build_pyinstaller.sh"
    exit 1
fi

BUNDLE_ID="${BUNDLE_ID:-com.canopystudio.murmur}"

echo "Installing ${APP} -> ${TARGET}"
if [ -d "${TARGET}" ]; then
    rm -rf "${TARGET}"
fi

if [ -z "${CODE_SIGN_IDENTITY:-}" ]; then
    echo "Resetting Accessibility permission for ${BUNDLE_ID} (ad-hoc install)..."
    tccutil reset Accessibility "${BUNDLE_ID}" >/dev/null 2>&1 || true
fi

cp -R "${APP}" "${TARGET}"
xattr -cr "${TARGET}"

echo ""
echo "Installed: ${TARGET}"
echo ""
echo "Next steps:"
echo "  1. Open Murmur from /Applications"
echo "  2. Press Option+Space to test the shortcut"
echo "  3. For paste-at-cursor: Enable Shortcut Permission... -> allow Accessibility"
echo ""
if [ -z "${CODE_SIGN_IDENTITY:-}" ]; then
    echo "Tip: ad-hoc builds may need a fresh Accessibility grant after each reinstall."
    echo "     Use Developer ID signing for production (./release.sh)."
fi
