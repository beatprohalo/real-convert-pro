#!/usr/bin/env python3
"""
Test Advanced Stem Separation
Demonstrates the enhanced AI-powered stem separation capabilities.
"""

import os
import sys
from pathlib import Path
from audio_analyzer import ProfessionalAudioProcessor

def test_basic_separation():
    """Test basic stem separation"""
    print("\n" + "="*60)
    print("TESTING BASIC STEM SEPARATION")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    input_file = "sample_audio/electronic_beat.wav"
    output_dir = "test_outputs/stem_separation_basic"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    print(f"Input: {input_file}")
    print("Method: Basic harmonic-percussive separation")
    print("Stems: vocals, drums, bass, other")
    
    result = processor.stem_separation(
        input_path=input_file,
        output_dir=output_dir,
        stems=['vocals', 'drums', 'bass', 'other'],
        method='basic',
        quality='balanced'
    )
    
    print_separation_results(result)

def test_advanced_separation():
    """Test advanced stem separation"""
    print("\n" + "="*60)
    print("TESTING ADVANCED STEM SEPARATION")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    input_file = "sample_audio/piano_melody.wav"
    output_dir = "test_outputs/stem_separation_advanced"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    print(f"Input: {input_file}")
    print("Method: Advanced multi-technique separation")
    print("Quality: High")
    print("Stems: vocals, piano, other")
    
    result = processor.stem_separation(
        input_path=input_file,
        output_dir=output_dir,
        stems=['vocals', 'piano', 'other'],
        method='advanced',
        quality='high'
    )
    
    print_separation_results(result)

def test_spectral_separation():
    """Test spectral-based stem separation"""
    print("\n" + "="*60)
    print("TESTING SPECTRAL STEM SEPARATION")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    input_file = "sample_audio/rock_guitar.wav"
    output_dir = "test_outputs/stem_separation_spectral"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    print(f"Input: {input_file}")
    print("Method: High-resolution spectral separation")
    print("Quality: Ultra")
    print("Stems: vocals, drums, bass, other")
    
    result = processor.stem_separation(
        input_path=input_file,
        output_dir=output_dir,
        stems=['vocals', 'drums', 'bass', 'other'],
        method='spectral',
        quality='ultra'
    )
    
    print_separation_results(result)

def test_nmf_separation():
    """Test NMF-based stem separation"""
    print("\n" + "="*60)
    print("TESTING NMF STEM SEPARATION")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    input_file = "sample_audio/ambient_pad.wav"
    output_dir = "test_outputs/stem_separation_nmf"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    print(f"Input: {input_file}")
    print("Method: Non-negative Matrix Factorization")
    print("Quality: High")
    print("Stems: vocals, bass, other")
    
    result = processor.stem_separation(
        input_path=input_file,
        output_dir=output_dir,
        stems=['vocals', 'bass', 'other'],
        method='nmf',
        quality='high'
    )
    
    print_separation_results(result)

def test_ai_separation():
    """Test AI-based stem separation (falls back to advanced)"""
    print("\n" + "="*60)
    print("TESTING AI STEM SEPARATION")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    input_file = "sample_audio/electronic_beat.wav"
    output_dir = "test_outputs/stem_separation_ai"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    print(f"Input: {input_file}")
    print("Method: AI-based separation (simplified implementation)")
    print("Quality: Ultra")
    print("Stems: vocals, drums, bass, piano, other")
    
    result = processor.stem_separation(
        input_path=input_file,
        output_dir=output_dir,
        stems=['vocals', 'drums', 'bass', 'piano', 'other'],
        method='ai',
        quality='ultra'
    )
    
    print_separation_results(result)

def test_quality_comparison():
    """Test different quality settings"""
    print("\n" + "="*60)
    print("TESTING QUALITY COMPARISON")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    input_file = "sample_audio/piano_melody.wav"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    qualities = ['fast', 'balanced', 'high', 'ultra']
    stems = ['vocals', 'other']
    
    print(f"Input: {input_file}")
    print("Comparing quality settings for advanced method")
    print(f"Stems: {', '.join(stems)}")
    
    for quality in qualities:
        print(f"\n🔧 Testing {quality} quality...")
        output_dir = f"test_outputs/stem_separation_quality_{quality}"
        
        result = processor.stem_separation(
            input_path=input_file,
            output_dir=output_dir,
            stems=stems,
            method='advanced',
            quality=quality
        )
        
        if result.success:
            print(f"✅ {quality.capitalize()} quality: {result.processing_time:.2f}s")
            if result.quality_metrics:
                metrics = result.quality_metrics
                print(f"   Energy conservation: {metrics.get('energy_conservation', 0):.3f}")
                print(f"   Success rate: {metrics.get('success_rate', 0):.1%}")
        else:
            print(f"❌ {quality.capitalize()} quality failed: {result.message}")

def print_separation_results(result):
    """Print formatted separation results"""
    if result.success:
        print(f"✅ {result.message}")
        print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
        print(f"📁 Output directory: {result.output_path}")
        
        if result.additional_outputs:
            print("🎵 Separated stems:")
            for stem, path in result.additional_outputs.items():
                file_size = Path(path).stat().st_size / 1024  # KB
                print(f"   - {stem}: {Path(path).name} ({file_size:.1f} KB)")
        
        if result.quality_metrics:
            metrics = result.quality_metrics
            print("📊 Quality metrics:")
            print(f"   - Stems created: {metrics.get('stems_created', 0)}")
            print(f"   - Success rate: {metrics.get('success_rate', 0):.1%}")
            print(f"   - Energy conservation: {metrics.get('energy_conservation', 0):.3f}")
            
            # Show individual stem RMS levels
            print("   - Stem RMS levels:")
            for key, value in metrics.items():
                if key.endswith('_rms'):
                    stem_name = key.replace('_rms', '')
                    print(f"     • {stem_name}: {value:.4f}")
    else:
        print(f"❌ {result.message}")

def benchmark_methods():
    """Benchmark different separation methods"""
    print("\n" + "="*60)
    print("BENCHMARKING SEPARATION METHODS")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    input_file = "sample_audio/electronic_beat.wav"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    methods = ['basic', 'advanced', 'spectral', 'nmf']
    stems = ['vocals', 'drums', 'bass']
    quality = 'balanced'
    
    print(f"Input: {input_file}")
    print(f"Quality: {quality}")
    print(f"Stems: {', '.join(stems)}")
    
    results = {}
    
    for method in methods:
        print(f"\n🔧 Testing {method} method...")
        output_dir = f"test_outputs/benchmark_{method}"
        
        result = processor.stem_separation(
            input_path=input_file,
            output_dir=output_dir,
            stems=stems,
            method=method,
            quality=quality
        )
        
        if result.success:
            results[method] = {
                'time': result.processing_time,
                'energy_conservation': result.quality_metrics.get('energy_conservation', 0),
                'success_rate': result.quality_metrics.get('success_rate', 0)
            }
            print(f"✅ {method.capitalize()}: {result.processing_time:.2f}s")
        else:
            print(f"❌ {method.capitalize()} failed: {result.message}")
    
    # Summary
    if results:
        print("\n📊 Benchmark Summary:")
        print("Method       | Time (s) | Energy Cons. | Success Rate")
        print("-" * 50)
        for method, metrics in results.items():
            print(f"{method:<12} | {metrics['time']:>7.2f} | {metrics['energy_conservation']:>11.3f} | {metrics['success_rate']:>10.1%}")

def main():
    """Run all advanced stem separation tests"""
    print("🎵 ADVANCED STEM SEPARATION TEST SUITE")
    print("Testing AI-powered stem separation capabilities...")
    
    # Create output directories
    os.makedirs("test_outputs", exist_ok=True)
    
    # Test different methods
    test_basic_separation()
    test_advanced_separation()
    test_spectral_separation()
    test_nmf_separation()
    test_ai_separation()
    
    # Quality and performance tests
    test_quality_comparison()
    benchmark_methods()
    
    print("\n" + "="*60)
    print("ADVANCED STEM SEPARATION TESTS COMPLETED")
    print("="*60)
    print("Check the test_outputs/ directory for separated stems")
    print("\nKey improvements in this version:")
    print("• Multiple separation algorithms (Basic, Advanced, Spectral, NMF, AI)")
    print("• Quality settings (Fast, Balanced, High, Ultra)")
    print("• Enhanced post-processing for each stem type")
    print("• Comprehensive quality metrics")
    print("• Support for additional stems (piano, accompaniment)")
    print("• Better frequency analysis and component assignment")

if __name__ == "__main__":
    main()