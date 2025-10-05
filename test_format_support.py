#!/usr/bin/env python3
"""
Test script to verify comprehensive audio format support
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# Import our audio converter
from audio_converter import AudioConverter
import tkinter as tk

def create_test_audio():
    """Create a simple test audio file"""
    duration = 1.0  # 1 second
    sample_rate = 44100
    frequency = 440  # A4 note
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.3 * np.sin(frequency * 2 * np.pi * t)
    
    return audio, sample_rate

def test_format_conversion():
    """Test conversion to various formats"""
    print("🎵 COMPREHENSIVE AUDIO FORMAT CONVERSION TEST")
    print("=" * 60)
    
    # Create test audio
    audio_data, sample_rate = create_test_audio()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create input file
        input_file = os.path.join(temp_dir, "test_input.wav")
        sf.write(input_file, audio_data, sample_rate)
        
        print(f"✅ Created test input file: {os.path.basename(input_file)}")
        
        # Initialize converter (without GUI)
        root = tk.Tk()
        root.withdraw()
        converter = AudioConverter(root)
        
        # Test formats to convert to
        priority_formats = [
            'wav', 'flac', 'mp3', 'aac', 'm4a', 'ogg', 'opus',
            'aiff', 'au', 'caf', 'wma', 'ac3'
        ]
        
        print(f"\\n🔧 System Check:")
        deps = converter.check_system_dependencies()
        print(f"  FFmpeg: {'✅ Available' if deps['ffmpeg'] else '❌ Missing'}")
        print(f"  Total formats in system: {len(converter.formats)}")
        
        print(f"\\n🎯 Testing Priority Formats:")
        print("-" * 40)
        
        successful_conversions = []
        failed_conversions = []
        
        for format_name in priority_formats:
            try:
                # Check format support
                supported, reason = converter.validate_format_compatibility(format_name)
                
                if supported:
                    # Set the format in converter
                    converter.selected_format.set(format_name)
                    
                    # Create output file path
                    output_file = os.path.join(temp_dir, f"test_output.{format_name}")
                    
                    # Attempt conversion
                    success = converter.process_audio(input_file, output_file)
                    
                    if success and os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        print(f"✅ {format_name.upper():<8} - Converted successfully ({file_size:,} bytes)")
                        successful_conversions.append(format_name)
                    else:
                        print(f"❌ {format_name.upper():<8} - Conversion failed")
                        failed_conversions.append(format_name)
                else:
                    print(f"⚠️  {format_name.upper():<8} - Not supported: {reason}")
                    failed_conversions.append(format_name)
                    
            except Exception as e:
                print(f"❌ {format_name.upper():<8} - Error: {str(e)[:50]}...")
                failed_conversions.append(format_name)
        
        # Summary
        print("\\n" + "=" * 60)
        print(f"📊 CONVERSION SUMMARY:")
        print(f"  ✅ Successful: {len(successful_conversions)}/{len(priority_formats)}")
        print(f"  ❌ Failed: {len(failed_conversions)}")
        
        if successful_conversions:
            print(f"\\n🎉 Successfully converted to: {', '.join(successful_conversions)}")
        
        if failed_conversions:
            print(f"\\n⚠️  Failed conversions: {', '.join(failed_conversions)}")
        
        print(f"\\n💡 Available Formats Summary:")
        supported_count = 0
        for fmt in converter.formats:
            if converter.validate_format_compatibility(fmt)[0]:
                supported_count += 1
        
        print(f"  📈 Total supported formats: {supported_count}/{len(converter.formats)}")
        print(f"  📊 Support percentage: {(supported_count/len(converter.formats)*100):.1f}%")
        
        # Show format categories
        uncompressed = ['wav', 'aiff', 'au', 'caf']
        lossless = ['flac']
        lossy = ['mp3', 'aac', 'm4a', 'ogg', 'opus', 'wma']
        
        supported_uncompressed = [f for f in uncompressed if f in successful_conversions]
        supported_lossless = [f for f in lossless if f in successful_conversions]
        supported_lossy = [f for f in lossy if f in successful_conversions]
        
        print(f"\\n📁 Format Categories Successfully Tested:")
        if supported_uncompressed:
            print(f"  🔊 Uncompressed: {', '.join(supported_uncompressed)}")
        if supported_lossless:
            print(f"  🗜️  Lossless: {', '.join(supported_lossless)}")
        if supported_lossy:
            print(f"  🎶 Lossy: {', '.join(supported_lossy)}")
        
        return len(successful_conversions), len(failed_conversions)

if __name__ == "__main__":
    try:
        successful, failed = test_format_conversion()
        
        print("\\n" + "=" * 60)
        if successful > failed:
            print("🎉 COMPREHENSIVE FORMAT SUPPORT CONFIRMED!")
            print(f"The audio converter successfully supports {successful} major formats.")
        else:
            print("⚠️  Limited format support detected.")
            print("Consider installing FFmpeg for full format compatibility.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)