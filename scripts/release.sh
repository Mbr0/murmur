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

echo "Building Murmur ${APP_VERSION} (signed + notarized)..."
export NOTARIZE
"${ROOT}/scripts/build_pyinstaller.sh"

echo "Creating DMG..."
"${ROOT}/scripts/create_dmg.sh"

echo ""
echo "======================================"
echo "Production release ready"
echo ""
echo "  dist/Murmur.app"
echo "  dist/Murmur-${APP_VERSION}.dmg"
echo ""
echo "Validate:"
echo "  codesign --verify --strict --verbose=2 dist/Murmur.app"
echo "  spctl --assess --type execute --verbose dist/Murmur.app"
echo "======================================"
