#!/bin/bash
# Package dist/Murmur.app into a styled macOS disk image for distribution.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

APP_NAME="Murmur"
APP_BUNDLE="dist/${APP_NAME}.app"
APP_VERSION="${APP_VERSION:-1.0.0}"
DMG_PATH="dist/${APP_NAME}-${APP_VERSION}.dmg"
DMG_RW_PATH="dist/${APP_NAME}-${APP_VERSION}.rw.dmg"
STAGING="dist/dmg-staging"
VOLUME_NAME="${APP_NAME}"
BACKGROUND="assets/dmg_background.png"
CODE_SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-}"
RUN_BUILD=false

WINDOW_WIDTH=660
WINDOW_HEIGHT=400
ICON_SIZE=128
APP_ICON_X=180
APP_ICON_Y=185
APPS_LINK_X=480
APPS_LINK_Y=185

usage() {
    cat <<EOF
Usage: ./create_dmg.sh [--build]

Creates dist/${APP_NAME}-\${APP_VERSION}.dmg from dist/${APP_NAME}.app.

Options:
  --build   Run scripts/build_pyinstaller.sh first

Environment:
  APP_VERSION          Version label in the DMG filename (default: 1.0.0)
  CODE_SIGN_IDENTITY   If set, signs the DMG after creation
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --build)
            RUN_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ "${RUN_BUILD}" = true ]; then
    "${ROOT}/scripts/build_pyinstaller.sh"
fi

if [ ! -d "${APP_BUNDLE}" ]; then
    echo "ERROR: ${APP_BUNDLE} not found. Run ./build_pyinstaller.sh first."
    exit 1
fi

echo "Generating DMG background..."
source venv/bin/activate
python assets/generate_dmg_background.py

if [ ! -f "${BACKGROUND}" ]; then
    echo "ERROR: ${BACKGROUND} was not created."
    exit 1
fi

echo "Creating ${DMG_PATH}..."
rm -rf "${STAGING}" "${DMG_PATH}" "${DMG_RW_PATH}"
mkdir -p "${STAGING}/.background"

cat > "${STAGING}/INSTALL.txt" <<'EOF'
Murmur install notes
====================

1. Drag Murmur.app to Applications.
2. Eject this disk image.
3. Open Murmur from /Applications (not from the DMG).
4. Press Option+Space to start/stop recording.
5. For paste-at-cursor: menu bar -> Enable Shortcut Permission...
   then allow Murmur in System Settings -> Privacy & Security -> Accessibility.

The global shortcut works without extra permissions. Accessibility is only
needed to paste transcribed text where your cursor is.

For public distribution, use a Developer ID signed build (RELEASE_SIGNING.md).
EOF

cp -R "${APP_BUNDLE}" "${STAGING}/"
ln -s /Applications "${STAGING}/Applications"
cp "${BACKGROUND}" "${STAGING}/.background/background.png"

if mount | grep -q "/Volumes/${VOLUME_NAME} "; then
    hdiutil detach "/Volumes/${VOLUME_NAME}" >/dev/null 2>&1 || true
fi

hdiutil create \
    -volname "${VOLUME_NAME}" \
    -srcfolder "${STAGING}" \
    -format UDRW \
    -ov \
    "${DMG_RW_PATH}"

rm -rf "${STAGING}"

DEVICE="$(hdiutil attach -readwrite -noverify "${DMG_RW_PATH}" | awk '/^\/dev\// {print $1; exit}')"
MOUNT="/Volumes/${VOLUME_NAME}"

cleanup() {
    if [ -n "${DEVICE:-}" ]; then
        hdiutil detach "${DEVICE}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "Applying Finder layout (icons + drag arrow background)..."
/usr/bin/osascript <<APPLESCRIPT
tell application "Finder"
    tell disk "${VOLUME_NAME}"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {200, 120, $((200 + WINDOW_WIDTH)), $((120 + WINDOW_HEIGHT))}
        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to ${ICON_SIZE}
        set background picture of viewOptions to file ".background:background.png"
        set position of item "${APP_NAME}.app" of container window to {${APP_ICON_X}, ${APP_ICON_Y}}
        set position of item "Applications" of container window to {${APPS_LINK_X}, ${APPS_LINK_Y}}
        set position of item "INSTALL.txt" of container window to {330, 310}
        close
        open
        update without registering applications
        delay 2
        close
    end tell
end tell
APPLESCRIPT

sleep 2
chmod -Rf go-w "${MOUNT}" || true
sync

hdiutil detach "${DEVICE}"
DEVICE=""
trap - EXIT

hdiutil convert "${DMG_RW_PATH}" -format UDZO -imagekey zlib-level=9 -o "${DMG_PATH}"
rm -f "${DMG_RW_PATH}"

if [ -n "${CODE_SIGN_IDENTITY}" ]; then
    echo "Signing DMG with: ${CODE_SIGN_IDENTITY}"
    codesign --force --timestamp -s "${CODE_SIGN_IDENTITY}" "${DMG_PATH}"
    codesign --verify --verbose=2 "${DMG_PATH}"
fi

echo ""
echo "======================================"
echo "DMG ready: ${DMG_PATH}"
echo "Size: $(du -h "${DMG_PATH}" | awk '{print $1}')"
echo ""
echo "Open the DMG to see the drag-to-Applications layout."
echo "======================================"
