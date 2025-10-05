#!/usr/bin/env python3
"""
Complete App Launcher Setup for Audio Converter ML
This script creates a complete double-click application launcher
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_complete_launcher():
    """Set up everything needed for the double-click launcher"""
    
    print("🎵 Audio Converter ML - Complete Launcher Setup")
    print("=" * 55)
    
    current_dir = Path(__file__).parent
    
    # Step 1: Install Pillow if needed
    print("\n📦 Checking dependencies...")
    try:
        import PIL
        print("✅ Pillow is already installed")
    except ImportError:
        print("⚠️  Installing Pillow for icon creation...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "Pillow"], 
                         check=True, capture_output=True)
            print("✅ Pillow installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Pillow: {e}")
            print("💡 You may need to run: pip install Pillow")
    
    # Step 2: Create icon
    print("\n🎨 Creating application icon...")
    try:
        exec(open(current_dir / "create_icon.py").read())
        print("✅ Icon created successfully")
    except Exception as e:
        print(f"⚠️  Icon creation failed: {e}")
        print("Continuing without custom icon...")
    
    # Step 3: Create app bundle
    print("\n📱 Creating macOS app bundle...")
    try:
        exec(open(current_dir / "create_app_bundle.py").read())
        print("✅ App bundle created successfully")
    except Exception as e:
        print(f"❌ App bundle creation failed: {e}")
        return False
    
    # Step 4: Copy icon to app bundle if it exists
    app_bundle = current_dir / "Audio Converter ML.app"
    if app_bundle.exists():
        resources_dir = app_bundle / "Contents" / "Resources"
        
        # Try to copy .icns file
        icns_file = current_dir / "app_icon.icns"
        if icns_file.exists():
            import shutil
            shutil.copy2(icns_file, resources_dir / "app_icon.icns")
            print("✅ Icon added to app bundle")
        else:
            # Try PNG as fallback
            png_file = current_dir / "app_icon.png"
            if png_file.exists():
                import shutil
                shutil.copy2(png_file, resources_dir / "app_icon.png")
                print("✅ PNG icon added to app bundle")
    
    # Step 5: Create desktop shortcut option
    print("\n🖥️  Creating desktop launcher option...")
    desktop_launcher = Path.home() / "Desktop" / "Audio Converter ML.command"
    
    launcher_content = f"""#!/bin/bash
# Audio Converter ML Desktop Launcher

echo "🎵 Launching Audio Converter ML from Desktop..."

# Navigate to the app directory
cd "{current_dir}"

# Launch the app
if [ -d "Audio Converter ML.app" ]; then
    echo "📱 Opening app bundle..."
    open "Audio Converter ML.app"
else
    echo "🚀 Launching GUI directly..."
    if command -v python3 &> /dev/null; then
        python3 professional_tools_gui.py
    elif command -v python &> /dev/null; then
        python professional_tools_gui.py
    else
        echo "❌ Python not found. Please install Python 3."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi
"""
    
    try:
        with open(desktop_launcher, "w") as f:
            f.write(launcher_content)
        os.chmod(desktop_launcher, 0o755)
        print(f"✅ Desktop launcher created: {desktop_launcher}")
    except Exception as e:
        print(f"⚠️  Could not create desktop launcher: {e}")
    
    # Step 6: Final instructions
    print("\n🎉 Setup Complete!")
    print("=" * 55)
    print("\nYou now have multiple ways to launch Audio Converter ML:")
    print()
    
    if (current_dir / "Audio Converter ML.app").exists():
        print("1. 📱 Double-click 'Audio Converter ML.app' (recommended)")
    
    if (current_dir / "Launch Audio Converter.command").exists():
        print("2. 🚀 Double-click 'Launch Audio Converter.command'")
    
    if desktop_launcher.exists():
        print("3. 🖥️  Double-click desktop shortcut (if created)")
    
    print("\n💡 Tips:")
    print("• The .app bundle is a native macOS application")
    print("• You can drag the .app to your Applications folder")
    print("• You can add it to your Dock for quick access")
    print("• The app will automatically find Python and launch the GUI")
    
    return True

if __name__ == "__main__":
    try:
        success = setup_complete_launcher()
        if success:
            print("\n✅ All launchers created successfully!")
            print("🎵 Ready to rock with Audio Converter ML! 🎵")
        else:
            print("\n❌ Some issues occurred during setup")
            print("💡 Try running the individual scripts manually")
    except KeyboardInterrupt:
        print("\n⚠️  Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("💡 Make sure you have proper permissions and Python is installed")