#!/usr/bin/env python3
"""
Real Convert - Compatibility Launcher
This launcher handles architecture compatibility issues automatically
"""

import sys
import os
import subprocess
from pathlib import Path

def find_working_python():
    """Find a Python installation that can actually run the GUI"""
    
    # First, try the current Python executable that's running this script
    current_python = sys.executable
    print(f"🔍 Testing current Python: {current_python}")
    
    try:
        test_cmd = [
            current_python, '-c', 
            'import soundfile, tkinter; print("CURRENT_OK")'
        ]
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'CURRENT_OK' in result.stdout:
            print(f"✅ Current Python works: {current_python}")
            return current_python
        else:
            print(f"❌ Current Python failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Current Python test error: {e}")
    
    candidates = [
        # Try the Python that usually works from terminal
        '/Library/Frameworks/Python.framework/Versions/3.9/bin/python3',
        '/Library/Frameworks/Python.framework/Versions/3.11/bin/python3',
        '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3',
        '/usr/bin/python3',
        '/opt/homebrew/bin/python3',
        # Skip problematic Intel Homebrew for now
        # '/usr/local/bin/python3',
    ]
    
    for python_path in candidates:
        if not os.path.exists(python_path):
            continue
            
        print(f"🔍 Testing: {python_path}")
        try:
            # Test if this Python can import our dependencies
            test_cmd = [
                python_path, '-c', 
                'import soundfile, tkinter; print("TEST_OK")'
            ]
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'TEST_OK' in result.stdout:
                print(f"✅ Found working Python: {python_path}")
                return python_path
            else:
                print(f"❌ Failed: {result.stderr[:100]}...")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"❌ Error testing {python_path}: {e}")
            continue
    
    return None

def main():
    print("🎵 Real Convert - Smart Launcher")
    print("=" * 40)
    
    # Detect if we're running from app bundle (no terminal input available)
    is_app_bundle = not sys.stdin.isatty()
    
    if is_app_bundle:
        print("📱 Running from app bundle")
    else:
        print("💻 Running from terminal")
    
    # Get the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Find GUI file
    gui_file = None
    if (script_dir / "audio_converter.py").exists():
        gui_file = "audio_converter.py"
    elif (script_dir / "professional_tools_gui.py").exists():
        gui_file = "professional_tools_gui.py"
    
    if not gui_file:
        print("❌ No GUI file found!")
        if not is_app_bundle:
            input("Press Enter to exit...")
        return
    
    print(f"✅ Found GUI file: {gui_file}")
    
    # Find working Python
    print("🔍 Finding compatible Python...")
    python_path = find_working_python()
    
    if not python_path:
        print("❌ No compatible Python installation found!")
        if is_app_bundle:
            # Show macOS alert instead of waiting for input
            try:
                subprocess.run([
                    'osascript', '-e',
                    'display alert "Python Compatibility Issue" message "Real Convert needs a compatible Python installation. Please install Python from python.org or try running from Terminal for more details." buttons {"OK"} default button "OK"'
                ])
            except:
                pass  # Fail silently if osascript doesn't work
        else:
            print("\n💡 Please try:")
            print("   1. Install Python from python.org")
            print("   2. Or use: brew install python")
            print("   3. Then reinstall dependencies: pip3 install -r requirements.txt")
            input("\nPress Enter to exit...")
        return
    
    print(f"✅ Using Python: {python_path}")
    
    # Launch the GUI
    print("🚀 Launching Real Convert...")
    print("-" * 40)
    
    try:
        # Use the working Python to launch the GUI
        subprocess.run([python_path, gui_file], check=True)
        print("✅ Real Convert closed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Real Convert failed with exit code: {e.returncode}")
        if not is_app_bundle:
            input("Press Enter to continue...")
    except KeyboardInterrupt:
        print("⚠️  Real Convert interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if not is_app_bundle:
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()