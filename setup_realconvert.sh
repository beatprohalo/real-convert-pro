#!/bin/bash
# Real Convert Setup Script
# This script sets up the Python virtual environment for Real Convert

echo "🎵 Real Convert Setup Script"
echo "================================"

# Check if we're in the right directory
if [ ! -f "audio_converter.py" ]; then
    echo "❌ Please run this script from the Real Convert directory"
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is not installed. Please install it first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "✅ Homebrew found"

# Install Python 3.12 if not already installed
if ! brew list python@3.12 &> /dev/null; then
    echo "📦 Installing Python 3.12..."
    brew install python@3.12
    brew install python-tk@3.12
else
    echo "✅ Python 3.12 already installed"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_realconvert" ]; then
    echo "🐍 Creating virtual environment..."
    /opt/homebrew/bin/python3.12 -m venv venv_realconvert
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo "📦 Installing dependencies..."
source venv_realconvert/bin/activate
pip install --upgrade pip

# Install basic requirements
if [ -f "requirements_basic.txt" ]; then
    pip install -r requirements_basic.txt
else
    echo "Installing core dependencies..."
    pip install librosa soundfile pydub mutagen eyed3 python-magic audioread numpy scipy resampy numba pyloudnorm scikit-learn
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "You can now:"
echo "1. Launch Real Convert by double-clicking the Real Convert.app"
echo "2. Or run from terminal: source venv_realconvert/bin/activate && python audio_converter.py"
echo ""
echo "✅ Real Convert is ready to use!"