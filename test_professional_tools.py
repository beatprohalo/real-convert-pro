#!/usr/bin/env python3
"""
Test Professional Audio Tools
Tests all professional audio processing features including time-stretching,
audio repair, spectral editing, stereo manipulation, and mastering.
"""

import os
import sys
import numpy as np
from pathlib import Path
from audio_analyzer import ProfessionalAudioProcessor, AudioProcessingResult

def create_test_audio():
    """Create test audio files for demonstration"""
    import librosa
    import soundfile as sf
    
    sample_dir = Path("sample_audio")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a test tone with some noise
    duration = 5.0  # seconds
    sr = 44100
    t = np.linspace(0, duration, int(duration * sr), False)
    
    # Generate a chord (C major)
    frequencies = [261.63, 329.63, 392.00]  # C, E, G
    signal = np.zeros_like(t)
    
    for freq in frequencies:
        signal += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Add some harmonics and noise for realistic testing
    signal += 0.1 * np.sin(2 * np.pi * 523.25 * t)  # octave
    signal += 0.05 * np.random.normal(0, 1, len(t))  # noise
    
    # Add some clicks for declick testing
    click_positions = [int(0.5 * sr), int(2.1 * sr), int(3.7 * sr)]
    for pos in click_positions:
        if pos < len(signal):
            signal[pos:pos+10] += 0.8
    
    # Add 60Hz hum for dehum testing
    signal += 0.02 * np.sin(2 * np.pi * 60 * t)
    signal += 0.01 * np.sin(2 * np.pi * 120 * t)  # harmonic
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Save test file
    test_file = sample_dir / "test_audio.wav"
    sf.write(test_file, signal, sr)
    
    print(f"Created test audio file: {test_file}")
    return str(test_file)

def test_time_stretching(processor, test_file):
    """Test time-stretching functionality"""
    print("\n" + "="*50)
    print("TESTING TIME-STRETCHING")
    print("="*50)
    
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    test_cases = [
        {"factor": 0.5, "description": "2x faster (half length)"},
        {"factor": 1.5, "description": "1.5x slower"},
        {"factor": 2.0, "description": "2x slower (double length)"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['description']}")
        print("-" * 30)
        
        output_path = output_dir / f"stretched_{case['factor']:.1f}x.wav"
        
        result = processor.time_stretch(
            test_file,
            str(output_path),
            stretch_factor=case['factor'],
            preserve_pitch=True,
            quality="high"
        )
        
        if result.success:
            print(f"✓ SUCCESS: {result.message}")
            print(f"  Output: {result.output_path}")
            print(f"  Processing time: {result.processing_time:.2f}s")
        else:
            print(f"✗ FAILED: {result.message}")

def test_audio_repair(processor, test_file):
    """Test audio repair functionality"""
    print("\n" + "="*50)
    print("TESTING AUDIO REPAIR")
    print("="*50)
    
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    repair_tests = [
        {
            "name": "Declick only",
            "operations": ["declick"],
            "description": "Remove clicks and pops"
        },
        {
            "name": "Denoise only", 
            "operations": ["denoise"],
            "description": "Remove background noise"
        },
        {
            "name": "Dehum only",
            "operations": ["dehum"], 
            "description": "Remove 50/60Hz hum"
        },
        {
            "name": "Full repair",
            "operations": ["declick", "denoise", "dehum", "normalize"],
            "description": "Complete audio restoration"
        }
    ]
    
    for i, test in enumerate(repair_tests, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Description: {test['description']}")
        print("-" * 30)
        
        output_path = output_dir / f"repaired_{test['name'].lower().replace(' ', '_')}.wav"
        
        result = processor.repair_audio(
            test_file,
            str(output_path),
            operations=test['operations']
        )
        
        if result.success:
            print(f"✓ SUCCESS: {result.message}")
            print(f"  Output: {result.output_path}")
            if result.quality_metrics:
                snr_improvement = result.quality_metrics.get('snr_improvement_db', 0)
                print(f"  SNR improvement: {snr_improvement:.1f}dB")
        else:
            print(f"✗ FAILED: {result.message}")

def test_spectral_editing(processor, test_file):
    """Test spectral editing functionality"""
    print("\n" + "="*50)
    print("TESTING SPECTRAL EDITING")
    print("="*50)
    
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    spectral_tests = [
        {
            "name": "Remove bass",
            "freq_ranges": [(20, 200)],
            "operation": "remove",
            "description": "Remove low frequencies"
        },
        {
            "name": "Remove mids",
            "freq_ranges": [(500, 2000)],
            "operation": "remove", 
            "description": "Remove midrange frequencies"
        },
        {
            "name": "Isolate highs",
            "freq_ranges": [(4000, 20000)],
            "operation": "isolate",
            "description": "Keep only high frequencies"
        },
        {
            "name": "Notch filter",
            "freq_ranges": [(390, 400), (520, 530)],
            "operation": "remove",
            "description": "Remove specific frequency bands"
        }
    ]
    
    for i, test in enumerate(spectral_tests, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Description: {test['description']}")
        print(f"Frequency ranges: {test['freq_ranges']}")
        print("-" * 30)
        
        output_path = output_dir / f"spectral_{test['name'].lower().replace(' ', '_')}.wav"
        
        result = processor.spectral_edit(
            test_file,
            str(output_path),
            freq_ranges=test['freq_ranges'],
            operation=test['operation']
        )
        
        if result.success:
            print(f"✓ SUCCESS: {result.message}")
            print(f"  Output: {result.output_path}")
        else:
            print(f"✗ FAILED: {result.message}")

def test_stereo_manipulation(processor, test_file):
    """Test stereo field manipulation"""
    print("\n" + "="*50)
    print("TESTING STEREO FIELD MANIPULATION")
    print("="*50)
    
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    stereo_tests = [
        {
            "name": "Mono to stereo (delay)",
            "operation": "mono_to_stereo",
            "parameters": {"technique": "delay", "delay_ms": 15},
            "description": "Create stereo width using delay"
        },
        {
            "name": "Mono to stereo (chorus)",
            "operation": "mono_to_stereo", 
            "parameters": {"technique": "chorus", "lfo_freq": 0.3, "lfo_depth": 0.03},
            "description": "Create stereo width using chorus"
        },
        {
            "name": "Narrow stereo",
            "operation": "width_control",
            "parameters": {"width": 0.5},
            "description": "Reduce stereo width"
        },
        {
            "name": "Wide stereo",
            "operation": "width_control",
            "parameters": {"width": 1.8},
            "description": "Increase stereo width"
        },
        {
            "name": "Pan left",
            "operation": "pan",
            "parameters": {"position": -0.7},
            "description": "Pan audio to the left"
        }
    ]
    
    for i, test in enumerate(stereo_tests, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Description: {test['description']}")
        print("-" * 30)
        
        output_path = output_dir / f"stereo_{test['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.wav"
        
        result = processor.stereo_field_manipulation(
            test_file,
            str(output_path),
            operation=test['operation'],
            parameters=test['parameters']
        )
        
        if result.success:
            print(f"✓ SUCCESS: {result.message}")
            print(f"  Output: {result.output_path}")
        else:
            print(f"✗ FAILED: {result.message}")

def test_mastering_chain(processor, test_file):
    """Test complete mastering chain"""
    print("\n" + "="*50)
    print("TESTING MASTERING CHAIN")
    print("="*50)
    
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    mastering_configs = [
        {
            "name": "Gentle mastering",
            "config": {
                "eq": {
                    "bands": [
                        {"frequency": 80, "gain": 1.0, "q": 0.7},
                        {"frequency": 2000, "gain": 0.5, "q": 1.0}
                    ]
                },
                "compressor": {
                    "threshold": -15.0,
                    "ratio": 2.0,
                    "attack": 20.0,
                    "release": 200.0
                },
                "limiter": {
                    "ceiling": -0.3,
                    "lookahead": 10.0
                },
                "target_lufs": -16.0
            }
        },
        {
            "name": "Aggressive mastering",
            "config": {
                "eq": {
                    "bands": [
                        {"frequency": 100, "gain": 2.5, "q": 0.8},
                        {"frequency": 3000, "gain": 2.0, "q": 1.2},
                        {"frequency": 8000, "gain": 1.5, "q": 0.9}
                    ]
                },
                "compressor": {
                    "threshold": -10.0,
                    "ratio": 4.0,
                    "attack": 5.0,
                    "release": 50.0
                },
                "stereo_enhance": {
                    "width": 1.3
                },
                "limiter": {
                    "ceiling": -0.1,
                    "lookahead": 5.0
                },
                "target_lufs": -12.0
            }
        }
    ]
    
    for i, test in enumerate(mastering_configs, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-" * 30)
        
        output_path = output_dir / f"mastered_{test['name'].lower().replace(' ', '_')}.wav"
        
        result = processor.mastering_chain(
            test_file,
            str(output_path),
            chain_config=test['config']
        )
        
        if result.success:
            print(f"✓ SUCCESS: {result.message}")
            print(f"  Output: {result.output_path}")
            if result.quality_metrics:
                metrics = result.quality_metrics
                print(f"  Peak level: {metrics.get('peak_level', 0):.3f}")
                print(f"  Target LUFS: {metrics.get('target_lufs', 0):.1f}")
        else:
            print(f"✗ FAILED: {result.message}")

def main():
    """Run all professional audio tools tests"""
    print("PROFESSIONAL AUDIO TOOLS TEST SUITE")
    print("="*60)
    
    # Initialize processor
    processor = ProfessionalAudioProcessor()
    
    # Create or find test audio
    test_file = None
    sample_dir = Path("sample_audio")
    
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*.wav"))
        if sample_files:
            test_file = str(sample_files[0])
            print(f"Using existing sample file: {Path(test_file).name}")
    
    if not test_file:
        print("No sample audio found. Creating test audio...")
        test_file = create_test_audio()
    
    # Run all tests
    try:
        test_time_stretching(processor, test_file)
        test_audio_repair(processor, test_file)
        test_spectral_editing(processor, test_file)
        test_stereo_manipulation(processor, test_file)
        test_mastering_chain(processor, test_file)
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        print("Check the 'test_outputs' directory for processed audio files.")
        print("\nProfessional audio tools are ready for use!")
        
    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()