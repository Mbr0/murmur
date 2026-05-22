#!/bin/bash
# Murmur Setup Script for macOS

set -e

echo "🎤 Murmur Setup"
echo "=================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install python@3.11
fi

echo "✅ Python found: $(python3 --version)"

# Check for portaudio (needed for sounddevice)
if ! brew list portaudio &> /dev/null; then
    echo "📦 Installing portaudio..."
    brew install portaudio
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run Murmur:"
echo "  cd $(pwd)"
echo "  source venv/bin/activate"
echo "  python murmur.py"
echo ""
echo "Or use: ./run.sh"
echo ""
echo "⚠️  First run will download the Whisper model (~140MB for 'base')"
echo ""
echo "📋 Grant these permissions in System Preferences → Privacy & Security:"
echo "   - Microphone access"
echo "   - Accessibility (for keyboard shortcuts)"
echo "   - Input Monitoring (for global hotkeys)"
