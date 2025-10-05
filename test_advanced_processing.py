#!/usr/bin/env python3
"""
Test script for advanced processing features
Demonstrates GPU acceleration, multi-core optimization, progress estimation, and memory optimization
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Import our modules
try:
    from advanced_processing import (
        AdvancedAudioProcessor, ProcessingConfig, 
        GPUAccelerator, MemoryOptimizer, ProgressEstimator,
        demo_advanced_processing
    )
    from audio_analyzer import IntelligentAudioAnalyzer
    HAS_MODULES = True
except ImportError as e:
    print(f"Import error: {e}")
    HAS_MODULES = False

def create_test_audio_files():
    """Create test audio files if they don't exist"""
    import librosa
    import soundfile as sf
    
    sample_dir = Path("sample_audio")
    sample_dir.mkdir(exist_ok=True)
    
    test_files = []
    
    # Create test files with different characteristics
    test_configs = [
        {"name": "test_ambient.wav", "freq": 220, "duration": 5, "type": "ambient"},
        {"name": "test_electronic.wav", "freq": 440, "duration": 3, "type": "electronic"},
        {"name": "test_piano.wav", "freq": 523, "duration": 4, "type": "piano"},
        {"name": "test_rock.wav", "freq": 330, "duration": 6, "type": "rock"},
        {"name": "test_large.wav", "freq": 880, "duration": 10, "type": "large"}
    ]
    
    sr = 22050
    
    for config in test_configs:
        file_path = sample_dir / config["name"]
        
        if not file_path.exists():
            print(f"Creating test file: {config['name']}")
            
            # Generate test audio based on type
            t = np.linspace(0, config["duration"], int(config["duration"] * sr))
            
            if config["type"] == "ambient":
                # Slow-changing ambient pad
                y = 0.3 * np.sin(2 * np.pi * config["freq"] * t) * np.exp(-t/3)
                y += 0.1 * np.random.randn(len(t))  # Add noise
                
            elif config["type"] == "electronic":
                # Electronic-style with harmonics
                y = 0.5 * np.sin(2 * np.pi * config["freq"] * t)
                y += 0.2 * np.sin(2 * np.pi * config["freq"] * 2 * t)
                y += 0.1 * np.sin(2 * np.pi * config["freq"] * 3 * t)
                
            elif config["type"] == "piano":
                # Piano-like with decay
                y = 0.6 * np.sin(2 * np.pi * config["freq"] * t) * np.exp(-t/2)
                y += 0.3 * np.sin(2 * np.pi * config["freq"] * 2 * t) * np.exp(-t/1.5)
                
            elif config["type"] == "rock":
                # Rock-style with distortion simulation
                y = 0.7 * np.sin(2 * np.pi * config["freq"] * t)
                y = np.tanh(y * 3)  # Distortion
                
            elif config["type"] == "large":
                # Larger file for testing memory optimization
                y = 0.4 * np.sin(2 * np.pi * config["freq"] * t)
                # Add complexity
                for harmonic in range(2, 6):
                    y += (0.1 / harmonic) * np.sin(2 * np.pi * config["freq"] * harmonic * t)
            
            # Normalize
            y = y / np.max(np.abs(y)) * 0.8
            
            # Save file
            sf.write(str(file_path), y, sr)
            
        test_files.append(str(file_path))
    
    return test_files

def progress_callback(progress):
    """Progress callback for demonstrations"""
    print(f"\rProgress: {progress.progress_percentage:.1f}% "
          f"({progress.files_processed}/{progress.total_files}) | "
          f"Current: {progress.current_file} | "
          f"Rate: {progress.processing_rate:.1f} files/sec | "
          f"Memory: {progress.memory_usage:.0f}MB | "
          f"ETA: {progress.estimated_remaining}", end="")

def test_basic_functionality():
    """Test basic advanced processing functionality"""
    print("\n" + "="*60)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    # Test GPU accelerator
    gpu = GPUAccelerator()
    print(f"GPU Backend: {gpu.backend}")
    print(f"GPU Available: {gpu.is_available()}")
    
    if gpu.is_available():
        device_info = gpu.device_info
        print(f"Device: {device_info.get('device_name', 'Unknown')}")
        memory_info = gpu.get_memory_info()
        print(f"GPU Memory: {memory_info['total']:.0f}MB total, {memory_info['used']:.0f}MB used")
    
    # Test memory optimizer
    memory_opt = MemoryOptimizer(memory_limit_mb=2048)
    memory_usage = memory_opt.get_memory_usage()
    print(f"\nSystem Memory: {memory_usage['total_mb']:.0f}MB total, {memory_usage['available_mb']:.0f}MB available")
    print(f"Current usage: {memory_usage['rss_mb']:.0f}MB ({memory_usage['percent']:.1f}%)")
    
    # Test progress estimator
    progress_est = ProgressEstimator()
    progress_est.start_processing()
    
    print("\nProgress Estimation Test:")
    for i in range(3):
        time.sleep(0.2)
        progress_est.record_file_completion(f"test_file_{i}.wav", 0.2)
        rate = progress_est.get_processing_rate()
        eta = progress_est.get_eta(i + 1, 10)
        print(f"  File {i+1}/10 processed - Rate: {rate:.1f} files/sec - ETA: {eta}")

def test_gpu_acceleration():
    """Test GPU acceleration features"""
    print("\n" + "="*60)
    print("TESTING GPU ACCELERATION")
    print("="*60)
    
    # Create test audio
    sr = 22050
    duration = 5
    t = np.linspace(0, duration, int(duration * sr))
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
    
    gpu = GPUAccelerator()
    
    if not gpu.is_available():
        print("GPU not available - skipping GPU tests")
        return
    
    print("Testing GPU-accelerated STFT...")
    
    # CPU benchmark
    start_time = time.time()
    import librosa
    cpu_stft = librosa.stft(test_audio)
    cpu_time = time.time() - start_time
    
    # GPU benchmark
    start_time = time.time()
    gpu_stft = gpu.accelerate_stft(test_audio)
    gpu_time = time.time() - start_time
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 1
    
    print(f"CPU STFT: {cpu_time:.4f}s")
    print(f"GPU STFT: {gpu_time:.4f}s")
    print(f"Speedup: {speedup:.1f}x")
    
    # Test batch processing
    print("\nTesting batch GPU processing...")
    audio_batch = [test_audio] * 5  # 5 copies
    
    start_time = time.time()
    batch_results = gpu.batch_process_spectrograms(audio_batch)
    batch_time = time.time() - start_time
    
    print(f"Batch processed {len(audio_batch)} files in {batch_time:.4f}s")
    print(f"Average per file: {batch_time/len(audio_batch):.4f}s")

def test_memory_optimization():
    """Test memory optimization features"""
    print("\n" + "="*60)
    print("TESTING MEMORY OPTIMIZATION")
    print("="*60)
    
    memory_opt = MemoryOptimizer(memory_limit_mb=1024)  # 1GB limit
    
    # Test memory monitoring
    initial_memory = memory_opt.get_memory_usage()
    print(f"Initial memory usage: {initial_memory['rss_mb']:.1f}MB")
    
    # Create large array to test memory management
    print("Creating large array...")
    large_array = np.random.randn(10_000_000)  # ~76MB array
    
    current_memory = memory_opt.get_memory_usage()
    print(f"Memory after allocation: {current_memory['rss_mb']:.1f}MB")
    print(f"Memory increase: {current_memory['rss_mb'] - initial_memory['rss_mb']:.1f}MB")
    
    # Test memory cleanup
    print("Testing memory cleanup...")
    del large_array
    memory_opt.cleanup_memory()
    
    final_memory = memory_opt.get_memory_usage()
    print(f"Memory after cleanup: {final_memory['rss_mb']:.1f}MB")
    
    # Test chunk processing
    print("\nTesting chunk processing...")
    large_signal = np.random.randn(1_000_000)
    
    def simple_rms(x):
        return np.sqrt(np.mean(x**2))
    
    # Process in chunks
    chunk_result = memory_opt.chunk_audio_processing(
        large_signal, 
        chunk_size=100_000, 
        process_func=simple_rms
    )
    
    # Compare with direct processing
    direct_result = simple_rms(large_signal)
    
    chunk_rms_mean = np.mean(chunk_result)
    print(f"Chunk processing result (mean of chunks): {chunk_rms_mean:.6f}")
    print(f"Direct processing result: {direct_result:.6f}")
    print(f"Difference: {abs(chunk_rms_mean - direct_result):.6f}")

def test_audio_analysis_advanced():
    """Test advanced audio analysis capabilities"""
    print("\n" + "="*60)
    print("TESTING ADVANCED AUDIO ANALYSIS")
    print("="*60)
    
    # Create test files
    test_files = create_test_audio_files()
    print(f"Created {len(test_files)} test files")
    
    # Initialize advanced analyzer
    analyzer = IntelligentAudioAnalyzer(enable_advanced_processing=True)
    
    print("\n1. Standard vs Advanced Analysis Comparison")
    print("-" * 40)
    
    test_file = test_files[0]
    
    # Standard analysis
    start_time = time.time()
    standard_result = analyzer.analyze_audio_file(test_file)
    standard_time = time.time() - start_time
    
    # Advanced analysis
    start_time = time.time()
    advanced_result = analyzer.analyze_audio_file_optimized(test_file)
    advanced_time = time.time() - start_time
    
    print(f"Standard analysis: {standard_time:.4f}s")
    print(f"Advanced analysis: {advanced_time:.4f}s")
    print(f"BPM (standard): {standard_result.bpm}")
    print(f"BPM (advanced): {advanced_result.bpm}")
    
    print("\n2. Batch Analysis with Advanced Processing")
    print("-" * 40)
    
    # Batch analysis
    start_time = time.time()
    batch_results = analyzer.batch_analyze_library_advanced(
        directory="sample_audio",
        max_files=len(test_files),
        progress_callback=None,
        use_gpu=True
    )
    batch_time = time.time() - start_time
    
    print(f"Batch processed {len(batch_results)} files in {batch_time:.2f}s")
    print(f"Average per file: {batch_time/len(batch_results):.4f}s")
    
    # Display results
    for result in batch_results:
        if result.bpm:
            print(f"  {result.filename}: {result.duration:.1f}s, "
                  f"BPM: {result.bpm:.0f}, "
                  f"Energy: {result.energy_level:.3f}" if result.energy_level else "N/A")
    
    print("\n3. GPU-Accelerated Spectral Analysis")
    print("-" * 40)
    
    spectral_results = analyzer.gpu_accelerated_batch_spectral_analysis(test_files[:3])
    
    for result in spectral_results:
        if 'error' not in result:
            print(f"  {result['filename']}: "
                  f"Centroid: {result['spectral_centroid']:.0f}Hz, "
                  f"Energy: {result['energy']:.3f}")

def test_full_system_integration():
    """Test the complete advanced processing system"""
    print("\n" + "="*60)
    print("TESTING FULL SYSTEM INTEGRATION")
    print("="*60)
    
    # Create configuration
    config = ProcessingConfig(
        use_gpu=True,
        cpu_workers=2,
        memory_limit_mb=2048,
        enable_progress_estimation=True,
        progress_callback=progress_callback
    )
    
    # Initialize processor
    processor = AdvancedAudioProcessor(config)
    
    # Create test files
    test_files = create_test_audio_files()
    
    # Test batch processing with all optimizations
    print("\nRunning complete batch analysis with all optimizations...")
    
    def comprehensive_analysis(file_path: str) -> Dict[str, Any]:
        """Comprehensive analysis function"""
        try:
            import librosa
            y, sr = librosa.load(file_path, duration=10)  # Max 10 seconds
            
            analysis = {
                "filename": os.path.basename(file_path),
                "duration": len(y) / sr,
                "sample_rate": sr,
                "rms_energy": float(np.sqrt(np.mean(y**2))),
                "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
                "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
            }
            
            # Tempo analysis
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                analysis["bpm"] = float(tempo)
            except:
                analysis["bpm"] = None
            
            # MFCC features
            try:
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
                analysis["mfcc_mean"] = [float(x) for x in np.mean(mfccs, axis=1)]
            except:
                analysis["mfcc_mean"] = []
            
            return analysis
            
        except Exception as e:
            return {"filename": os.path.basename(file_path), "error": str(e)}
    
    # Run batch processing
    start_time = time.time()
    results = processor.multicore_processor.batch_process_with_memory_management(
        test_files,
        comprehensive_analysis,
        processor.memory_optimizer,
        progress_callback=lambda completed, total, current_file, error=None: 
            print(f"\rProcessed {completed}/{total}: {os.path.basename(current_file)}", end="")
    )
    total_time = time.time() - start_time
    
    print(f"\n\nBatch processing complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Files processed: {len(results)}")
    print(f"Average per file: {total_time/len(results):.4f}s")
    
    # Display results
    print("\nResults Summary:")
    for file_path, result in results:
        if isinstance(result, dict) and 'error' not in result:
            print(f"  {result['filename']}: "
                  f"{result['duration']:.1f}s, "
                  f"BPM: {result.get('bpm', 'N/A')}, "
                  f"Energy: {result['rms_energy']:.3f}, "
                  f"Centroid: {result['spectral_centroid']:.0f}Hz")
    
    # Get final statistics
    stats = processor.get_processing_stats()
    print(f"\nFinal System Statistics:")
    print(f"  Memory Usage: {stats['system']['memory_usage_mb']:.0f}MB")
    print(f"  CPU Usage: {stats['system']['cpu_usage']:.0f}%")
    if stats['gpu']['available']:
        print(f"  GPU Usage: {stats['gpu']['usage_percent']:.0f}%")
    print(f"  Errors: {stats['errors']}")

def main():
    """Main test function"""
    if not HAS_MODULES:
        print("Required modules not available. Please ensure audio_analyzer.py and advanced_processing.py are available.")
        return
    
    print("ADVANCED AUDIO PROCESSING TEST SUITE")
    print("=" * 60)
    
    try:
        # Run all tests
        test_basic_functionality()
        test_gpu_acceleration()
        test_memory_optimization()
        test_audio_analysis_advanced()
        test_full_system_integration()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Run the demo
        print("\nRunning comprehensive demo...")
        demo_advanced_processing()
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()