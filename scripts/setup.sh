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

# cmake builds the bundled whisper.cpp server (decision D2)
if ! command -v cmake &> /dev/null; then
    echo "📦 Installing cmake..."
    brew install cmake
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# MLX (the Voxtral engine) installs on Apple Silicon only; on Intel the
# requirements markers skip it and whisper.cpp is the only engine (decision D7).
if [ "$(uname -m)" = "arm64" ]; then
    echo "✅ Apple Silicon: MLX speech engine installed"
else
    echo "ℹ️  Intel Mac: MLX skipped, whisper.cpp is the engine (decision D7)"
fi

# The whisper.cpp server is a bundled binary, not a wheel (decision D2).
if [ ! -f "${ROOT}/vendor/whispercpp/whisper-server" ]; then
    echo ""
    echo "🔨 Building the whisper.cpp server (a few minutes, once)..."
    bash "${ROOT}/scripts/tools/fetch_whispercpp.sh"
fi

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
echo "⚠️  First run downloads the speech model on demand (Settings → Speech engine)"
echo ""
echo "📋 Grant these permissions in System Settings → Privacy & Security:"
echo "   - Microphone (recording)"
echo "   - Accessibility (paste text at cursor)"
echo "   Global shortcut (⌥ Space) works without extra permissions."
