#!/usr/bin/env python3
"""
Test and Demo Script for Intelligent Audio Analysis Features
This script demonstrates the new audio analysis capabilities.
"""

import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import librosa
import soundfile as sf

# Add the current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from audio_analyzer import IntelligentAudioAnalyzer, AudioAnalysisResult
    print("✓ Successfully imported IntelligentAudioAnalyzer")
except ImportError as e:
    print(f"✗ Failed to import audio analyzer: {e}")
    sys.exit(1)

def create_test_audio_files():
    """Create some test audio files with different characteristics"""
    test_dir = tempfile.mkdtemp(prefix="audio_analysis_test_")
    print(f"Creating test audio files in: {test_dir}")
    
    # Generate different types of test audio
    sr = 44100
    duration = 5  # 5 seconds
    
    test_files = []
    
    # 1. Sine wave at 440Hz (A4) - should detect as A major
    print("Creating sine wave test file (440Hz)...")
    t = np.linspace(0, duration, sr * duration, False)
    sine_wave = 0.3 * np.sin(2 * np.pi * 440 * t)
    sine_file = os.path.join(test_dir, "sine_440hz_A4.wav")
    sf.write(sine_file, sine_wave, sr)
    test_files.append(sine_file)
    
    # 2. White noise - should be hard to analyze for key
    print("Creating white noise test file...")
    noise = 0.1 * np.random.randn(sr * duration)
    noise_file = os.path.join(test_dir, "white_noise.wav")
    sf.write(noise_file, noise, sr)
    test_files.append(noise_file)
    
    # 3. Simple drum pattern - should have detectable BPM
    print("Creating drum pattern test file...")
    drum_pattern = np.zeros(sr * duration)
    bpm = 120
    beat_interval = 60.0 / bpm  # seconds per beat
    samples_per_beat = int(sr * beat_interval)
    
    # Add kick drums on beats 1 and 3, snare on beats 2 and 4
    for beat in range(int(duration / beat_interval)):
        start_sample = beat * samples_per_beat
        if start_sample < len(drum_pattern):
            # Kick drum (low frequency)
            if beat % 4 in [0, 2]:  # beats 1 and 3
                kick_duration = int(0.1 * sr)  # 100ms
                kick_freq = 60  # 60 Hz
                kick_t = np.linspace(0, 0.1, kick_duration, False)
                kick = 0.5 * np.sin(2 * np.pi * kick_freq * kick_t) * np.exp(-kick_t * 20)
                end_sample = min(start_sample + kick_duration, len(drum_pattern))
                drum_pattern[start_sample:end_sample] = kick[:end_sample - start_sample]
            
            # Snare drum (high frequency noise burst)
            if beat % 4 in [1, 3]:  # beats 2 and 4
                snare_duration = int(0.05 * sr)  # 50ms
                snare_noise = 0.3 * np.random.randn(snare_duration) * np.exp(-np.linspace(0, 1, snare_duration) * 10)
                end_sample = min(start_sample + snare_duration, len(drum_pattern))
                drum_pattern[start_sample:end_sample] += snare_noise[:end_sample - start_sample]
    
    drum_file = os.path.join(test_dir, "drum_pattern_120bpm.wav")
    sf.write(drum_file, drum_pattern, sr)
    test_files.append(drum_file)
    
    # 4. C Major chord progression
    print("Creating C major chord progression...")
    chord_duration = duration / 4  # 4 chords
    chord_progression = np.zeros(sr * duration)
    
    # C Major chord (C-E-G)
    c_major = [261.63, 329.63, 392.00]  # C4, E4, G4
    # F Major chord (F-A-C)
    f_major = [349.23, 440.00, 523.25]  # F4, A4, C5
    # G Major chord (G-B-D)
    g_major = [392.00, 493.88, 587.33]  # G4, B4, D5
    # C Major chord again
    chords = [c_major, f_major, g_major, c_major]
    
    for i, chord in enumerate(chords):
        start_sample = int(i * chord_duration * sr)
        end_sample = int((i + 1) * chord_duration * sr)
        chord_t = np.linspace(0, chord_duration, end_sample - start_sample, False)
        
        chord_sound = np.zeros(len(chord_t))
        for freq in chord:
            chord_sound += 0.2 * np.sin(2 * np.pi * freq * chord_t)
        
        # Add some envelope
        envelope = np.exp(-chord_t * 2) + 0.1
        chord_sound *= envelope
        
        chord_progression[start_sample:end_sample] = chord_sound
    
    chord_file = os.path.join(test_dir, "c_major_progression.wav")
    sf.write(chord_file, chord_progression, sr)
    test_files.append(chord_file)
    
    return test_dir, test_files

def run_analysis_tests():
    """Run comprehensive analysis tests"""
    print("=" * 60)
    print("INTELLIGENT AUDIO ANALYSIS TEST")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = IntelligentAudioAnalyzer()
    
    # Create test files
    test_dir, test_files = create_test_audio_files()
    
    print(f"\nAnalyzing {len(test_files)} test files...\n")
    
    results = []
    
    for file_path in test_files:
        filename = os.path.basename(file_path)
        print(f"Analyzing: {filename}")
        print("-" * 40)
        
        # Perform analysis
        result = analyzer.analyze_audio_file(file_path)
        results.append(result)
        
        # Display results
        print(f"Duration: {result.duration:.2f}s")
        print(f"Sample Rate: {result.sample_rate}Hz")
        
        # BPM Analysis
        if result.bpm:
            print(f"BPM: {result.bpm:.1f} ({result.tempo_category})")
        else:
            print("BPM: Unable to detect")
        
        # Key Analysis
        if result.key:
            print(f"Key: {result.key} {result.scale} (confidence: {result.key_confidence:.3f})")
        else:
            print("Key: Unable to detect")
        
        # Loudness Analysis
        if result.lufs_integrated:
            print(f"LUFS: {result.lufs_integrated:.1f}")
        if result.peak_db:
            print(f"Peak: {result.peak_db:.1f}dB")
        
        # Spectral Features
        if result.spectral_centroid:
            print(f"Spectral Centroid: {result.spectral_centroid:.1f}Hz")
        if result.zero_crossing_rate:
            print(f"Zero Crossing Rate: {result.zero_crossing_rate:.4f}")
        
        # Suggested Category
        suggested_category = analyzer.categorize_by_analysis(result)
        print(f"Suggested Category: {suggested_category}")
        
        # Fingerprint
        if result.fingerprint:
            print(f"Fingerprint: {result.fingerprint[:16]}... (length: {len(result.fingerprint)})")
        
        # Errors
        if result.analysis_errors:
            print("Analysis Errors:")
            for error in result.analysis_errors:
                print(f"  - {error}")
        
        print()
    
    # Test duplicate detection
    print("=" * 60)
    print("DUPLICATE DETECTION TEST")
    print("=" * 60)
    
    # Create duplicate of first file
    if test_files:
        original_file = test_files[0]
        duplicate_file = os.path.join(test_dir, "duplicate_" + os.path.basename(original_file))
        
        # Copy the file
        import shutil
        shutil.copy2(original_file, duplicate_file)
        
        # Analyze the duplicate
        print(f"Creating duplicate: {os.path.basename(duplicate_file)}")
        duplicate_result = analyzer.analyze_audio_file(duplicate_file)
        results.append(duplicate_result)
        
        # Find duplicates
        duplicates = analyzer.find_duplicates(results)
        
        print(f"\nDuplicate Detection Results:")
        if duplicates:
            for i, group in enumerate(duplicates, 1):
                print(f"  Group {i}: {group}")
        else:
            print("  No duplicates detected")
    
    # Export analysis report
    print("\n" + "=" * 60)
    print("EXPORTING ANALYSIS REPORT")
    print("=" * 60)
    
    report_file = os.path.join(test_dir, "analysis_report.json")
    success = analyzer.export_analysis_report(results, report_file)
    
    if success:
        print(f"✓ Analysis report exported to: {report_file}")
        
        # Show summary from report
        import json
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        summary = report['analysis_summary']
        print(f"\nSummary Statistics:")
        print(f"  Total Files: {report['total_files']}")
        print(f"  Total Duration: {summary['total_duration']:.1f}s")
        print(f"  Average BPM: {summary['avg_bpm']:.1f}")
        print(f"  Average Loudness: {summary['avg_loudness']:.1f} LUFS")
        print(f"  Tempo Distribution: {summary['tempo_distribution']}")
        print(f"  Key Distribution: {summary['key_distribution']}")
        print(f"  Scale Distribution: {summary['scale_distribution']}")
    else:
        print("✗ Failed to export analysis report")
    
    print(f"\nTest files created in: {test_dir}")
    print("You can manually delete this directory when done testing.")
    
    return results

def test_individual_features():
    """Test individual analysis features"""
    print("\n" + "=" * 60)
    print("INDIVIDUAL FEATURE TESTS")
    print("=" * 60)
    
    analyzer = IntelligentAudioAnalyzer()
    
    # Test tempo categorization
    print("Testing tempo categorization:")
    test_bpms = [65, 85, 110, 130, 170, 250]
    for bpm in test_bpms:
        category = analyzer._categorize_tempo(bpm)
        print(f"  {bpm} BPM -> {category}")
    
    # Test key mapping
    print("\nKey mapping available:")
    for key in analyzer.key_mapping:
        print(f"  {key}")
    
    print("\nTempo categories:")
    for category, (min_bpm, max_bpm) in analyzer.tempo_categories.items():
        print(f"  {category}: {min_bpm}-{max_bpm} BPM")

def main():
    """Main test function"""
    print("Intelligent Audio Analysis Test Suite")
    print("This script tests the new audio analysis features.")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    try:
        import librosa
        print("✓ librosa available")
    except ImportError:
        print("✗ librosa not available")
        return
    
    try:
        import pyloudnorm
        print("✓ pyloudnorm available (LUFS analysis)")
    except ImportError:
        print("⚠ pyloudnorm not available (LUFS analysis disabled)")
    
    try:
        import essentia
        print("✓ essentia available (advanced analysis)")
    except ImportError:
        print("⚠ essentia not available (advanced analysis limited)")
    
    try:
        import chromaprint
        print("✓ chromaprint available (audio fingerprinting)")
    except ImportError:
        print("⚠ chromaprint not available (basic fingerprinting will be used)")
    
    print()
    
    # Run tests
    try:
        results = run_analysis_tests()
        test_individual_features()
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Analyzed {len(results)} audio files")
        print("Check the temporary directory for generated test files and analysis report.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()