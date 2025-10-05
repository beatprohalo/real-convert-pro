#!/bin/bash

# Audio Converter Setup Script for macOS
# This script installs all required dependencies

echo "🎵 Audio Converter Setup Script"
echo "================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "⚠️  Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install FFmpeg
echo "🎬 Installing FFmpeg..."
brew install ffmpeg

if [ $? -eq 0 ]; then
    echo "✅ FFmpeg installed successfully"
else
    echo "❌ Failed to install FFmpeg"
    echo "   You can install it manually with: brew install ffmpeg"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the Audio Converter:"
echo "   python3 audio_converter.py"
echo ""
echo "Features included:"
echo "   ✅ Batch audio conversion"
echo "   ✅ Pitch and key shifting"
echo "   ✅ Multiple format support (WAV, MP3, FLAC, OGG, etc.)"
echo "   ✅ Automatic categorization by filename"
echo "   ✅ Folder scanning and file selection"
echo "   ✅ Quality settings and normalization"
echo "   ✅ Progress monitoring"
echo ""