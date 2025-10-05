#!/usr/bin/env python3
"""
Real Convert Key Selection Demo
Demonstrates the new key selection feature for audio transposition
"""

import os
import sys
from pathlib import Path

# Add the current directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from audio_converter import AudioConverter
import tkinter as tk
from tkinter import messagebox

def demo_key_selection():
    """Demo the key selection feature"""
    
    print("🎵 Real Convert Key Selection Demo")
    print("=" * 50)
    
    # Initialize converter
    root = tk.Tk()
    root.withdraw()  # Hide the main window for demo
    
    converter = AudioConverter(root)
    
    print("✅ Audio Converter initialized with key selection feature")
    print()
    
    # Show available keys
    print("🎹 Available Musical Keys:")
    for i, key in enumerate(converter.musical_keys):
        if i % 6 == 0 and i > 0:
            print()  # New line every 6 keys
        print(f"{key:>8}", end="  ")
    print("\n")
    
    # Demo key calculations
    print("🧮 Key Transposition Examples:")
    print("-" * 40)
    test_cases = [
        ("Original", "No transposition"),
        ("C", "Transpose to C major"),
        ("D", "Transpose to D major (+2 semitones)"),
        ("F#/Gb", "Transpose to F# major (+6 semitones)"),
        ("Am", "Transpose to A minor (+9 semitones)"),
        ("Em", "Transpose to E minor (+4 semitones)"),
        ("Bb", "Transpose to Bb major (+10 semitones)")
    ]
    
    for key, description in test_cases:
        semitones = converter.calculate_key_transposition(key)
        print(f"{key:>8} -> {description}")
        if semitones != 0:
            print(f"         ({semitones:+d} semitones)")
        print()
    
    print("🎯 How to Use:")
    print("1. Select your audio files/folders")
    print("2. Choose output format (MP3, WAV, FLAC, etc.)")
    print("3. Select target key from dropdown")
    print("4. Click Convert - audio will be transposed automatically!")
    print()
    
    print("💡 Features:")
    print("• 24 musical keys (12 major + 12 minor)")
    print("• Automatic key detection when possible")
    print("• High-quality pitch shifting using librosa")
    print("• Combines with existing pitch shift controls")
    print("• Settings are saved and restored")
    print()
    
    print("🚀 Real Convert now supports key selection alongside format conversion!")
    
    # Show a demo message
    root.deiconify()  # Show window
    root.geometry("400x200")
    root.title("Key Selection Demo")
    
    demo_text = """
🎵 Real Convert Key Selection Feature Added! 🎵

✅ 24 Musical Keys Available
✅ Automatic Transposition  
✅ High-Quality Pitch Shifting
✅ Works with All Audio Formats

The key selection dropdown is now available
right next to the format selection!

Select any key (C, D, Em, F#, etc.) and your
audio will be automatically transposed during
conversion.
    """
    
    label = tk.Label(root, text=demo_text, justify=tk.LEFT, padx=20, pady=20)
    label.pack(expand=True)
    
    def close_demo():
        root.destroy()
        print("\n✨ Demo complete! Launch Real Convert to try the key selection feature.")
    
    tk.Button(root, text="Close Demo", command=close_demo, 
              bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    try:
        demo_key_selection()
    except KeyboardInterrupt:
        print("\n\n🔄 Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("💡 Make sure you're in the Real Convert directory and dependencies are installed")