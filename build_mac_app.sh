#!/usr/bin/env bash
set -euo pipefail

# Go to the folder where this script lives
cd "$(dirname "$0")"

APP_NAME="FabStack"
DIST_DIR="dist"
DMG_NAME="${APP_NAME}.dmg"

echo ">>> Creating / updating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo ">>> Upgrading pip..."
pip install --upgrade pip

echo ">>> Installing Python dependencies (including PyInstaller)..."
python -m pip install -r requirements.txt "pyinstaller>=6.10,<7"
if [ "${FABSTACK_BUNDLE_MODEL:-0}" == "1" ]; then
    echo ">>> Installing optional local-AI runtime..."
    python -m pip install "llama-cpp-python>=0.3,<0.4"
fi

echo ">>> Building macOS .app bundle with PyInstaller..."
pyinstaller --clean --noconfirm FabStack.spec

if [ ! -d "${DIST_DIR}/${APP_NAME}.app" ]; then
    echo "ERROR: ${DIST_DIR}/${APP_NAME}.app not found after build."
    exit 1
fi

echo ">>> Preparing and verifying a clean signed bundle..."
STAGING_DIR=$(mktemp -d "/private/tmp/${APP_NAME}-release.XXXXXX")
trap 'rm -rf "${STAGING_DIR}"' EXIT
ditto --norsrc --noextattr \
  "${DIST_DIR}/${APP_NAME}.app" \
  "${STAGING_DIR}/${APP_NAME}.app"
xattr -cr "${STAGING_DIR}/${APP_NAME}.app"
codesign --force --deep --sign - "${STAGING_DIR}/${APP_NAME}.app"
codesign --verify --deep --strict "${STAGING_DIR}/${APP_NAME}.app"
ln -s /Applications "${STAGING_DIR}/Applications"

echo ">>> Creating compressed DMG image ${DMG_NAME}..."
hdiutil create \
  -volname "${APP_NAME}" \
  -srcfolder "${STAGING_DIR}" \
  -ov \
  -format UDZO \
  "${DIST_DIR}/${DMG_NAME}"

echo
echo "✅ Build complete."
echo "You can distribute ${DIST_DIR}/${DMG_NAME} to macOS users."
echo "End users just need to:"
echo "  1) Open the DMG"
echo "  2) Drag ${APP_NAME}.app into /Applications"
echo "  3) Double-click ${APP_NAME}.app to run"
