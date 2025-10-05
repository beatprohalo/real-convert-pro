#!/usr/bin/env python3
"""
Quick launcher for Audio Converter
Double-click this file to start the application
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from audio_converter import main
    main()
except ImportError as e:
    print(f"Error importing audio_converter: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip3 install librosa soundfile pydub numpy")
    input("Press Enter to exit...")
except Exception as e:
    print(f"Error starting application: {e}")
    input("Press Enter to exit...")