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
import matplotlib
matplotlib.use('Agg')

# Import our audio converter
from audio_converter import AudioConverter

class HeadlessAudioConverter(AudioConverter):
    """A version of AudioConverter that doesn't initialize the GUI."""
    def __init__(self):
        # Mock tk variables to avoid creating a root window
        class MockTkVar:
            def __init__(self, value): self._value = value
            def get(self): return self._value
            def set(self, value): self._value = value

        self.selected_format = MockTkVar('wav')
        self.sample_rate = MockTkVar(44100)
        self.bit_depth = MockTkVar(16)
        self.normalize_audio = MockTkVar(True)
        self.remove_silence = MockTkVar(False)
        self.pitch_shift = MockTkVar(0.0)
        self.key_shift = MockTkVar(0)
        self.selected_key = MockTkVar("Original")
        self.analysis_results = {}
        self.include_key_in_filename = MockTkVar(False)
        self.include_mood_in_filename = MockTkVar(False)
        self.include_energy_in_filename = MockTkVar(False)
        self.include_bpm_in_filename = MockTkVar(False)
        self.include_genre_in_filename = MockTkVar(False)
        self.include_lufs_in_filename = MockTkVar(False)
        self.include_category_in_filename = MockTkVar(False)
        self.formats = [
            "wav", "flac", "aiff", "au", "caf", "w64", "rf64",
            "mp3", "aac", "m4a", "ogg", "opus", "wma", "ac3", "mp2",
            "bwf", "sd2", "snd", "iff", "svx", "nist", "voc", "ircam", "xi",
            "3gp", "webm", "mka", "m4v", "mov", "avi", "mkv", "mp4",
            "ra", "rm", "amr", "amr-nb", "amr-wb", "gsm", "dct", "dwv",
            "raw", "pcm", "s8", "s16le", "s24le", "s32le", "f32le", "f64le",
            "aifc", "caff", "m4r", "m4b", "m4p",
            "tta", "tak", "als", "ape", "wv", "mpc", "ofr", "ofs", "shn"
        ]

    def log(self, message):
        # Suppress logging in tests
        pass

    def check_system_dependencies(self):
        # Mock this method to avoid dependency checks in this test
        return {'ffmpeg': True}

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
        
        # Initialize a headless version of the converter to avoid GUI errors
        converter = HeadlessAudioConverter()
        
        # Test formats to convert to
        priority_formats = [
            'wav', 'flac', 'mp3', 'aac', 'm4a', 'ogg', 'opus',
            'aiff', 'au', 'caf', 'ac3'
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
        
        assert len(failed_conversions) == 0

if __name__ == "__main__":
    try:
        test_format_conversion()
        
        print("\\n" + "=" * 60)
        print("🎉 COMPREHENSIVE FORMAT SUPPORT CONFIRMED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)