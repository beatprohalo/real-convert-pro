#!/usr/bin/env python3
"""
Create Real Convert App Bundle
Professional audio converter with real-time processing
"""

import os
import shutil
import subprocess
from pathlib import Path

def create_real_convert_app():
    """Create Real Convert .app bundle"""
    
    current_dir = Path(__file__).parent
    app_name = "Real Convert"
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
    <string>real_convert_launcher</string>
    <key>CFBundleIdentifier</key>
    <string>com.realconvert.app</string>
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
    <string>real_convert_icon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.10</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.music</string>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>mp3</string>
                <string>wav</string>
                <string>flac</string>
                <string>aac</string>
                <string>m4a</string>
                <string>ogg</string>
            </array>
            <key>CFBundleTypeName</key>
            <string>Audio Files</string>
            <key>CFBundleTypeRole</key>
            <string>Editor</string>
        </dict>
    </array>
</dict>
</plist>"""
    
    with open(contents_dir / "Info.plist", "w") as f:
        f.write(info_plist)
    
    # Create launcher script with better error handling
    launcher_script = f"""#!/bin/bash
# Real Convert Launcher

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
APP_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Create log file for debugging
LOG_FILE="$APP_DIR/launch.log"

{{
    echo "🎵 Real Convert Launch Log - $(date)"
    echo "📁 App Directory: $APP_DIR"
    echo "🔍 Script Directory: $SCRIPT_DIR"
    
    # Change to the application directory
    cd "$APP_DIR"
    echo "✅ Changed to app directory: $(pwd)"
    
    # Set up Python path
    export PYTHONPATH="$APP_DIR:$PYTHONPATH"
    echo "🐍 Python path set"
    
    # Check for main GUI file
    if [ -f "audio_converter.py" ]; then
        echo "✅ Found audio_converter.py"
        GUI_FILE="audio_converter.py"
    elif [ -f "professional_tools_gui.py" ]; then
        echo "✅ Found professional_tools_gui.py"
        GUI_FILE="professional_tools_gui.py"
    else
        echo "❌ No GUI file found!"
        exit 1
    fi
    
    echo "🚀 Launching Real Convert GUI: $GUI_FILE"
    
    # Try different Python commands
    if command -v python3 &> /dev/null; then
        echo "🐍 Using python3"
        python3 "$GUI_FILE"
    elif command -v python &> /dev/null; then
        echo "🐍 Using python"
        python "$GUI_FILE"
    else
        echo "❌ Python not found. Please install Python 3."
        osascript -e 'display alert "Python Required" message "Python 3 is required to run Real Convert. Please install Python from python.org or using Homebrew." buttons {{"OK"}} default button "OK"'
        exit 1
    fi
    
}} >> "$LOG_FILE" 2>&1

# If we get here, launch was successful
echo "✅ Real Convert launched successfully" >> "$LOG_FILE"
"""
    
    launcher_path = macos_dir / "real_convert_launcher"
    with open(launcher_path, "w") as f:
        f.write(launcher_script)
    
    # Make launcher executable
    os.chmod(launcher_path, 0o755)
    
    print(f"✅ Created app bundle: {app_bundle}")
    return app_bundle

def create_simple_launchers():
    """Create simple command line launchers"""
    current_dir = Path(__file__).parent
    
    # Main launcher
    main_launcher = current_dir / "Launch Real Convert.command"
    main_content = f"""#!/bin/bash
# Real Convert Simple Launcher

echo "🎵 Launching Real Convert..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

# Find the right GUI file
if [ -f "audio_converter.py" ]; then
    GUI_FILE="audio_converter.py"
elif [ -f "professional_tools_gui.py" ]; then
    GUI_FILE="professional_tools_gui.py"
else
    echo "❌ No GUI file found!"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "🚀 Starting $GUI_FILE..."

# Launch with Python
if command -v python3 &> /dev/null; then
    python3 "$GUI_FILE"
elif command -v python &> /dev/null; then
    python "$GUI_FILE"
else
    echo "❌ Python not found. Please install Python 3."
    echo "💡 Visit https://python.org or use: brew install python"
    read -p "Press Enter to exit..."
    exit 1
fi
"""
    
    with open(main_launcher, "w") as f:
        f.write(main_content)
    os.chmod(main_launcher, 0o755)
    
    # Desktop launcher
    desktop_launcher = Path.home() / "Desktop" / "Real Convert.command"
    desktop_content = f"""#!/bin/bash
# Real Convert Desktop Launcher

echo "🎵 Launching Real Convert from Desktop..."

# Navigate to app directory
cd "{current_dir}"

# Launch the app bundle if it exists, otherwise launch directly
if [ -d "Real Convert.app" ]; then
    echo "📱 Opening Real Convert app bundle..."
    open "Real Convert.app"
else
    echo "🚀 Launching GUI directly..."
    if [ -f "audio_converter.py" ]; then
        GUI_FILE="audio_converter.py"
    elif [ -f "professional_tools_gui.py" ]; then
        GUI_FILE="professional_tools_gui.py"
    else
        echo "❌ No GUI file found!"
        read -p "Press Enter to exit..."
        exit 1
    fi
    
    if command -v python3 &> /dev/null; then
        python3 "$GUI_FILE"
    elif command -v python &> /dev/null; then
        python "$GUI_FILE"
    else
        echo "❌ Python not found!"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi
"""
    
    try:
        with open(desktop_launcher, "w") as f:
            f.write(desktop_content)
        os.chmod(desktop_launcher, 0o755)
        print(f"✅ Created desktop launcher: {desktop_launcher}")
    except Exception as e:
        print(f"⚠️  Could not create desktop launcher: {e}")
    
    print(f"✅ Created main launcher: {main_launcher}")

def create_real_convert_icon():
    """Create Real Convert branded icon"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        print("🎨 Creating Real Convert icon...")
        
        # Create a 512x512 icon
        size = 512
        icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon)
        
        # Create modern gradient background
        margin = 20
        
        # Create circular background with gradient effect
        for i in range(50):
            alpha = int(255 * (1 - i/50))
            radius_offset = i * 2
            color = (20 + i, 35 + i, 75 + i*2, alpha)
            draw.ellipse([margin + radius_offset, margin + radius_offset, 
                         size - margin - radius_offset, size - margin - radius_offset], 
                        fill=color)
        
        # Add outer ring
        ring_width = 8
        draw.ellipse([margin, margin, size-margin, size-margin], 
                    outline=(100, 180, 255, 255), width=ring_width)
        
        # Add inner tech rings
        center = size // 2
        for i, radius in enumerate([180, 140, 100]):
            alpha = 150 - i * 30
            color = (80 + i*40, 160 + i*20, 255, alpha)
            draw.ellipse([center-radius, center-radius, center+radius, center+radius], 
                       outline=color, width=3)
        
        # Add REAL text
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 85)
        except:
            font = ImageFont.load_default()
        
        real_text = 'REAL'
        bbox = draw.textbbox((0, 0), real_text, font=font)
        real_width = bbox[2] - bbox[0]
        real_x = (size - real_width) // 2
        real_y = center - 70
        
        # Add text shadow
        draw.text((real_x+3, real_y+3), real_text, fill=(0, 0, 0, 120), font=font)
        draw.text((real_x, real_y), real_text, fill=(255, 255, 255, 255), font=font)
        
        # Add CONVERT text
        try:
            convert_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 42)
        except:
            convert_font = ImageFont.load_default()
        
        convert_text = 'CONVERT'
        bbox = draw.textbbox((0, 0), convert_text, font=convert_font)
        convert_width = bbox[2] - bbox[0]
        convert_x = (size - convert_width) // 2
        convert_y = real_y + 90
        
        # Add text shadow
        draw.text((convert_x+2, convert_y+2), convert_text, fill=(0, 0, 0, 120), font=convert_font)
        draw.text((convert_x, convert_y), convert_text, fill=(255, 255, 255, 255), font=convert_font)
        
        # Add audio waveform at bottom
        wave_y = convert_y + 60
        wave_colors = [(0, 200, 255), (50, 220, 255), (100, 240, 255)]
        
        for i in range(0, size-80, 10):
            height = 6 + (i % 20)
            x = i + 40
            if x < size - 40:
                color_idx = (i // 20) % len(wave_colors)
                color = wave_colors[color_idx] + (200,)
                draw.rectangle([x, wave_y, x+6, wave_y + height], fill=color)
        
        # Save files
        icon.save('real_convert_icon.png')
        print('✅ Created real_convert_icon.png')
        
        # Create .icns
        try:
            subprocess.run(['sips', '-s', 'format', 'icns', 'real_convert_icon.png', 
                           '--out', 'real_convert_icon.icns'], 
                          check=True, capture_output=True)
            print('✅ Created real_convert_icon.icns')
            return True
        except:
            print('⚠️  Could not create .icns, PNG available')
            return False
            
    except ImportError:
        print("⚠️  PIL not available, skipping icon creation")
        return False

if __name__ == "__main__":
    print("🎵 Real Convert - App Bundle Creator")
    print("=" * 50)
    
    # Create icon
    icon_created = create_real_convert_icon()
    
    # Create app bundle
    app_bundle = create_real_convert_app()
    
    # Copy icon to app bundle
    if icon_created and app_bundle.exists():
        resources_dir = app_bundle / "Contents" / "Resources"
        current_dir = Path(__file__).parent
        
        icns_file = current_dir / "real_convert_icon.icns"
        if icns_file.exists():
            shutil.copy2(icns_file, resources_dir / "real_convert_icon.icns")
            print("✅ Icon added to app bundle")
        else:
            png_file = current_dir / "real_convert_icon.png"
            if png_file.exists():
                shutil.copy2(png_file, resources_dir / "real_convert_icon.png")
                print("✅ PNG icon added to app bundle")
    
    # Create simple launchers
    create_simple_launchers()
    
    print("\n🎉 Real Convert Setup Complete!")
    print("=" * 50)
    print("\nLaunch options:")
    print("1. 📱 Double-click 'Real Convert.app' (recommended)")
    print("2. 🚀 Double-click 'Launch Real Convert.command'")
    print("3. 🖥️  Desktop shortcut (if created)")
    print("\n✅ Real Convert is ready to go!")