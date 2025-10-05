#!/usr/bin/env python3
"""
Audio Converter System Test
Run this to verify all dependencies are working correctly
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🧪 Testing Audio Converter Dependencies...")
    print("=" * 50)
    
    tests = [
        ("Python version", lambda: sys.version_info >= (3, 8), f"Python {sys.version}"),
        ("tkinter (GUI)", lambda: __import__('tkinter'), "✅ GUI framework available"),
        ("numpy", lambda: __import__('numpy'), "✅ Numerical computing"),
        ("librosa", lambda: __import__('librosa'), "✅ Audio processing"),
        ("soundfile", lambda: __import__('soundfile'), "✅ Audio I/O"),
        ("pydub", lambda: __import__('pydub'), "✅ Format conversion"),
        ("FFmpeg", lambda: __import__('pydub.utils', fromlist=['which']).which('ffmpeg'), "✅ Audio codec support"),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func, success_msg in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {name}: {success_msg}")
                passed += 1
            else:
                print(f"❌ {name}: Test failed")
        except Exception as e:
            print(f"❌ {name}: {str(e)}")
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Audio Converter is ready to use.")
        print("\nTo start the application:")
        print("  python3 audio_converter.py")
        print("  or double-click: launch_converter.py")
    else:
        print("⚠️  Some tests failed. Please install missing dependencies:")
        print("  pip3 install librosa soundfile pydub numpy")
        if not __import__('pydub.utils', fromlist=['which']).which('ffmpeg'):
            print("  brew install ffmpeg  # for macOS")
    
    return passed == total

def test_audio_converter_import():
    """Test importing the main audio converter module"""
    print("\n🎵 Testing Audio Converter Module...")
    try:
        import audio_converter
        print("✅ Audio converter module loads successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import audio converter: {e}")
        return False

def main():
    """Run all tests"""
    print("🎵 Audio Converter System Test")
    print("Testing all components and dependencies...\n")
    
    deps_ok = test_imports()
    module_ok = test_audio_converter_import()
    
    if deps_ok and module_ok:
        print("\n🚀 System test complete - Everything is working!")
        print("\nYou can now:")
        print("  • Convert audio between formats")
        print("  • Adjust pitch and key")
        print("  • Batch process folders")
        print("  • Auto-categorize by filename")
        print("  • Process individual files or entire directories")
    else:
        print("\n❌ System test failed - Please fix the issues above")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()