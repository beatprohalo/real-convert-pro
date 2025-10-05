#!/usr/bin/env python3
"""
Create a macOS Application Bundle for the Audio Converter
This script creates a .app bundle that can be double-clicked to launch the audio converter
"""

import os
import shutil
import subprocess
from pathlib import Path

def create_app_bundle():
    """Create a macOS .app bundle for the audio converter"""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    app_name = "Audio Converter ML"
    app_bundle = current_dir / f"{app_name}.app"
    
    print(f"🎵 Creating {app_name}.app bundle...")
    
    # Remove existing bundle if it exists
    if app_bundle.exists():
        shutil.rmtree(app_bundle)
    
    # Create app bundle structure
    contents_dir = app_bundle / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    
    contents_dir.mkdir(parents=True)
    macos_dir.mkdir()
    resources_dir.mkdir()
    
    # Create Info.plist
    info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>audio_converter_launcher</string>
    <key>CFBundleIdentifier</key>
    <string>com.audioml.converter</string>
    <key>CFBundleName</key>
    <string>{app_name}</string>
    <key>CFBundleDisplayName</key>
    <string>{app_name}</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>app_icon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.10</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.music</string>
</dict>
</plist>"""
    
    with open(contents_dir / "Info.plist", "w") as f:
        f.write(info_plist)
    
    # Create launcher script
    launcher_script = f"""#!/bin/bash
# Audio Converter ML Launcher

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
APP_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to the application directory
cd "$APP_DIR"

# Set up Python path
export PYTHONPATH="$APP_DIR:$PYTHONPATH"

# Launch the audio converter GUI
echo "🎵 Launching Audio Converter ML..."
echo "📁 Working directory: $APP_DIR"

# Try different Python commands
if command -v python3 &> /dev/null; then
    python3 professional_tools_gui.py
elif command -v python &> /dev/null; then
    python professional_tools_gui.py
else
    echo "❌ Python not found. Please install Python 3."
    exit 1
fi
"""
    
    launcher_path = macos_dir / "audio_converter_launcher"
    with open(launcher_path, "w") as f:
        f.write(launcher_script)
    
    # Make launcher executable
    os.chmod(launcher_path, 0o755)
    
    print(f"✅ Created app bundle: {app_bundle}")
    print(f"📱 You can now double-click '{app_name}.app' to launch the audio converter!")
    
    return app_bundle

def create_simple_launcher():
    """Create a simple .command file for easy launching"""
    current_dir = Path(__file__).parent
    launcher_file = current_dir / "Launch Audio Converter.command"
    
    launcher_content = f"""#!/bin/bash
# Simple Audio Converter Launcher

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "🎵 Launching Audio Converter ML..."
echo "📁 Working directory: $SCRIPT_DIR"

# Launch the GUI
if command -v python3 &> /dev/null; then
    python3 professional_tools_gui.py
elif command -v python &> /dev/null; then
    python professional_tools_gui.py
else
    echo "❌ Python not found. Please install Python 3."
    read -p "Press Enter to exit..."
    exit 1
fi
"""
    
    with open(launcher_file, "w") as f:
        f.write(launcher_content)
    
    # Make executable
    os.chmod(launcher_file, 0o755)
    
    print(f"✅ Created simple launcher: {launcher_file}")
    print(f"🚀 Double-click 'Launch Audio Converter.command' to start the app!")

if __name__ == "__main__":
    print("🎵 Audio Converter ML - App Bundle Creator")
    print("=" * 50)
    
    # Create both launchers
    create_app_bundle()
    print()
    create_simple_launcher()
    
    print("\n🎉 Launchers created successfully!")
    print("\nYou now have two ways to launch the app:")
    print("1. Double-click 'Audio Converter ML.app' (macOS app bundle)")
    print("2. Double-click 'Launch Audio Converter.command' (simple launcher)")