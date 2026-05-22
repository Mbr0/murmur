#!/bin/bash
# Build Murmur as a macOS app

set -e

cd "$(dirname "$0")"
source venv/bin/activate

echo "📦 Installing py2app..."
pip install py2app

echo ""
echo "🔨 Building Murmur.app..."
echo "   This may take a few minutes..."
echo ""

# Clean previous builds
rm -rf build dist

# Build the app
python setup.py py2app

echo ""
echo "✅ Build complete!"
echo ""
echo "📍 Your app is at: $(pwd)/dist/Murmur.app"
echo ""
echo "To install:"
echo "  1. Drag 'dist/Murmur.app' to your Applications folder"
echo "  2. Right-click → Open (first time, to bypass Gatekeeper)"
echo "  3. Grant permissions in System Settings → Privacy & Security:"
echo "     - Microphone"
echo "     - Accessibility" 
echo "     - Input Monitoring"
echo ""
echo "To add to Login Items (start at boot):"
echo "  System Settings → General → Login Items → Add Murmur"
echo ""
