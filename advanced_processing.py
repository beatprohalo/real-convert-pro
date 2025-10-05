#!/usr/bin/env python3
"""
Advanced Processing Module
GPU acceleration, multi-core optimization, progress estimation, and memory optimization
for high-performance audio processing of large libraries.
"""

import os
import sys
import time
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Callable, Iterator
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import psutil
import gc
from dataclasses import dataclass
from collections import deque
import warnings

# Try to import GPU acceleration libraries
try:
    import cupy as cp
    HAS_CUPY = True
    print("✓ CuPy detected - GPU acceleration available")
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        print(f"✓ PyTorch detected - GPU acceleration available on {torch.cuda.get_device_name()}")
    else:
        print("✓ PyTorch detected - CPU only")
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
    print("✓ JAX detected - XLA acceleration available")
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

# Audio processing imports
import librosa
import soundfile as sf

@dataclass
class ProcessingProgress:
    """Container for processing progress information"""
    current_file: str = ""
    files_processed: int = 0
    total_files: int = 0
    current_operation: str = ""
    start_time: Optional[datetime] = None
    estimated_finish: Optional[datetime] = None
    processing_rate: float = 0.0  # files per second
    memory_usage: float = 0.0  # MB
    cpu_usage: float = 0.0  # percentage
    gpu_usage: float = 0.0  # percentage if available
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def progress_percentage(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.files_processed / self.total_files) * 100
    
    @property
    def elapsed_time(self) -> timedelta:
        if self.start_time:
            return datetime.now() - self.start_time
        return timedelta()
    
    @property
    def estimated_remaining(self) -> timedelta:
        if self.estimated_finish:
            return self.estimated_finish - datetime.now()
        return timedelta()

@dataclass
class ProcessingConfig:
    """Configuration for advanced processing"""
    use_gpu: bool = True
    gpu_batch_size: int = 32
    cpu_workers: int = None  # None = auto-detect
    memory_limit_mb: int = 8192  # 8GB default
    chunk_size: int = 1024  # For streaming processing
    enable_progress_estimation: bool = True
    progress_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    use_memory_mapping: bool = True
    optimize_for_large_files: bool = True
    
    def __post_init__(self):
        if self.cpu_workers is None:
            self.cpu_workers = min(mp.cpu_count(), 8)  # Reasonable default

class GPUAccelerator:
    """GPU acceleration utilities for audio processing"""
    
    def __init__(self):
        self.backend = self._detect_gpu_backend()
        self.device_info = self._get_device_info()
        
    def _detect_gpu_backend(self) -> str:
        """Detect available GPU backend"""
        if HAS_CUPY and cp.cuda.is_available():
            return "cupy"
        elif HAS_TORCH and torch.cuda.is_available():
            return "torch"
        elif HAS_JAX:
            try:
                # Check if JAX can use GPU
                jax.device_get(jax.device_put(1))
                return "jax"
            except:
                pass
        return "none"
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information"""
        info = {"backend": self.backend, "available": False}
        
        if self.backend == "cupy":
            try:
                info.update({
                    "available": True,
                    "device_count": cp.cuda.runtime.getDeviceCount(),
                    "device_name": cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                    "memory_total": cp.cuda.runtime.memGetInfo()[1] / 1024**2,  # MB
                })
            except:
                pass
        elif self.backend == "torch":
            try:
                info.update({
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**2,  # MB
                })
            except:
                pass
        elif self.backend == "jax":
            try:
                devices = jax.devices()
                info.update({
                    "available": True,
                    "device_count": len(devices),
                    "device_name": str(devices[0]),
                })
            except:
                pass
        
        return info
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available"""
        return self.device_info["available"]
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get GPU memory usage information"""
        if not self.is_available():
            return {"used": 0, "total": 0, "free": 0}
        
        if self.backend == "cupy":
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                return {
                    "used": (total - free) / 1024**2,
                    "total": total / 1024**2,
                    "free": free / 1024**2
                }
            except:
                pass
        elif self.backend == "torch":
            try:
                return {
                    "used": torch.cuda.memory_allocated() / 1024**2,
                    "total": torch.cuda.get_device_properties(0).total_memory / 1024**2,
                    "free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
                }
            except:
                pass
        
        return {"used": 0, "total": 0, "free": 0}
    
    def accelerate_stft(self, y: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """GPU-accelerated STFT computation"""
        if not self.is_available():
            return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        try:
            if self.backend == "cupy":
                y_gpu = cp.asarray(y)
                # Use CuPy's FFT
                stft_gpu = cp.fft.stft(y_gpu, nperseg=n_fft, noverlap=n_fft-hop_length)
                return cp.asnumpy(stft_gpu)
            
            elif self.backend == "torch":
                y_tensor = torch.tensor(y, device='cuda')
                stft_tensor = torch.stft(
                    y_tensor, 
                    n_fft=n_fft, 
                    hop_length=hop_length,
                    return_complex=True
                )
                return stft_tensor.cpu().numpy()
            
        except Exception as e:
            print(f"GPU STFT failed, falling back to CPU: {e}")
            
        return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    def accelerate_mfcc(self, y: np.ndarray, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
        """GPU-accelerated MFCC computation"""
        if not self.is_available():
            return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        try:
            if self.backend == "cupy":
                # Use GPU-accelerated STFT then CPU MFCC (librosa doesn't have GPU MFCC)
                stft = self.accelerate_stft(y)
                magnitude = cp.abs(cp.asarray(stft))
                return librosa.feature.mfcc(S=cp.asnumpy(magnitude), sr=sr, n_mfcc=n_mfcc)
            
            elif self.backend == "torch":
                # Similar approach for PyTorch
                stft = self.accelerate_stft(y)
                magnitude = np.abs(stft)
                return librosa.feature.mfcc(S=magnitude, sr=sr, n_mfcc=n_mfcc)
                
        except Exception as e:
            print(f"GPU MFCC failed, falling back to CPU: {e}")
            
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    def batch_process_spectrograms(self, audio_batch: List[np.ndarray], 
                                  n_fft: int = 2048, hop_length: int = 512) -> List[np.ndarray]:
        """Batch process multiple audio files for spectrograms on GPU"""
        if not self.is_available() or not audio_batch:
            return [librosa.stft(y, n_fft=n_fft, hop_length=hop_length) for y in audio_batch]
        
        try:
            if self.backend == "cupy":
                results = []
                for y in audio_batch:
                    y_gpu = cp.asarray(y)
                    stft_gpu = cp.fft.stft(y_gpu, nperseg=n_fft, noverlap=n_fft-hop_length)
                    results.append(cp.asnumpy(stft_gpu))
                return results
            
            elif self.backend == "torch":
                results = []
                for y in audio_batch:
                    y_tensor = torch.tensor(y, device='cuda')
                    stft_tensor = torch.stft(
                        y_tensor, 
                        n_fft=n_fft, 
                        hop_length=hop_length,
                        return_complex=True
                    )
                    results.append(stft_tensor.cpu().numpy())
                return results
                
        except Exception as e:
            print(f"GPU batch processing failed, falling back to CPU: {e}")
        
        return [librosa.stft(y, n_fft=n_fft, hop_length=hop_length) for y in audio_batch]

class MemoryOptimizer:
    """Memory optimization utilities for large-scale processing"""
    
    def __init__(self, memory_limit_mb: int = 8192):
        self.memory_limit_mb = memory_limit_mb
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self._memory_usage_history = deque(maxlen=100)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024**2,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024**2,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024**2,
            "total_mb": psutil.virtual_memory().total / 1024**2
        }
    
    def is_memory_available(self, required_mb: float) -> bool:
        """Check if enough memory is available for operation"""
        current_usage = self.get_memory_usage()
        return current_usage["available_mb"] >= required_mb
    
    def estimate_audio_memory(self, file_path: str, channels: int = 2) -> float:
        """Estimate memory usage for loading an audio file (in MB)"""
        try:
            file_size = os.path.getsize(file_path)
            # Rough estimation: uncompressed audio is typically 4-10x larger than file size
            estimated_mb = (file_size * 8 * channels) / 1024**2
            return estimated_mb
        except:
            return 100  # Default estimation
    
    def create_memory_mapped_audio(self, file_path: str, mode: str = 'r') -> Optional[np.memmap]:
        """Create memory-mapped array for large audio files"""
        try:
            # Load audio info without loading data
            info = sf.info(file_path)
            
            # Create temporary file for memory mapping
            temp_path = file_path + ".memmap"
            
            if not os.path.exists(temp_path):
                # Load and save as memory-mappable format
                y, sr = librosa.load(file_path, sr=None)
                memmap_array = np.memmap(temp_path, dtype='float32', mode='w+', shape=y.shape)
                memmap_array[:] = y.astype('float32')
                del memmap_array  # Flush to disk
            
            # Return memory-mapped array
            y, sr = librosa.load(file_path, sr=None)
            return np.memmap(temp_path, dtype='float32', mode=mode, shape=y.shape)
            
        except Exception as e:
            print(f"Memory mapping failed for {file_path}: {e}")
            return None
    
    def chunk_audio_processing(self, audio: np.ndarray, chunk_size: int, 
                              process_func: Callable, overlap: int = 0) -> np.ndarray:
        """Process audio in chunks to manage memory usage"""
        if len(audio) <= chunk_size:
            return process_func(audio)
        
        results = []
        
        for i in range(0, len(audio), chunk_size - overlap):
            end_idx = min(i + chunk_size, len(audio))
            chunk = audio[i:end_idx]
            
            # Process chunk
            result_chunk = process_func(chunk)
            
            # Handle overlap removal for array results
            if not np.isscalar(result_chunk):
                if overlap > 0 and i > 0:
                    result_chunk = result_chunk[overlap//2:]
                if overlap > 0 and end_idx < len(audio):
                    result_chunk = result_chunk[:-overlap//2]
            
            results.append(result_chunk)
            
            # Force garbage collection after each chunk
            gc.collect()
        
        if not results:
            return np.array([])

        if np.isscalar(results[0]):
            return np.mean(results)
        else:
            return np.concatenate(results)
    
    def smart_batch_size(self, file_sizes: List[int], target_memory_mb: float = 1024) -> int:
        """Calculate optimal batch size based on file sizes and memory"""
        if not file_sizes:
            return 1
        
        avg_file_size = np.mean(file_sizes)
        estimated_memory_per_file = (avg_file_size * 8) / 1024**2  # Rough estimate
        
        optimal_batch_size = max(1, int(target_memory_mb / estimated_memory_per_file))
        return min(optimal_batch_size, 32)  # Cap at reasonable maximum
    
    def cleanup_memory(self):
        """Force memory cleanup"""
        gc.collect()
        
        # Clear GPU memory if available
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if HAS_CUPY:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass

class ProgressEstimator:
    """Smart progress estimation with time remaining calculations"""
    
    def __init__(self):
        self.start_time = None
        self.processing_times = deque(maxlen=50)  # Store last 50 processing times
        self.file_sizes = deque(maxlen=50)
        self.completion_times = deque(maxlen=100)
        
    def start_processing(self):
        """Mark the start of processing"""
        self.start_time = datetime.now()
        self.processing_times.clear()
        self.file_sizes.clear()
        self.completion_times.clear()
    
    def record_file_completion(self, file_path: str, processing_time: float):
        """Record completion of a file"""
        try:
            file_size = os.path.getsize(file_path)
            self.processing_times.append(processing_time)
            self.file_sizes.append(file_size)
            self.completion_times.append(datetime.now())
        except:
            # If we can't get file size, use processing time only
            self.processing_times.append(processing_time)
    
    def estimate_remaining_time(self, files_processed: int, total_files: int, 
                              remaining_files: List[str] = None) -> timedelta:
        """Estimate remaining processing time"""
        if files_processed == 0 or not self.processing_times:
            return timedelta(seconds=0)
        
        remaining = total_files - files_processed
        if remaining <= 0:
            return timedelta(seconds=0)
        
        # Method 1: Simple average of recent processing times
        avg_time_per_file = np.mean(self.processing_times)
        simple_estimate = avg_time_per_file * remaining
        
        # Method 2: Size-based estimation if we have file sizes
        if self.file_sizes and remaining_files:
            try:
                # Calculate processing rate per byte
                total_processed_size = sum(self.file_sizes)
                total_processing_time = sum(self.processing_times)
                
                if total_processed_size > 0:
                    rate_per_byte = total_processing_time / total_processed_size
                    
                    # Estimate remaining size
                    remaining_size = sum(os.path.getsize(f) for f in remaining_files[:10])  # Sample first 10
                    avg_remaining_size = remaining_size / min(10, len(remaining_files))
                    total_remaining_size = avg_remaining_size * remaining
                    
                    size_based_estimate = rate_per_byte * total_remaining_size
                    
                    # Weighted average of both methods
                    final_estimate = (simple_estimate * 0.3 + size_based_estimate * 0.7)
                else:
                    final_estimate = simple_estimate
            except:
                final_estimate = simple_estimate
        else:
            final_estimate = simple_estimate
        
        # Method 3: Trend analysis for improved accuracy
        if len(self.completion_times) >= 5:
            try:
                recent_times = list(self.completion_times)[-5:]
                recent_intervals = [(recent_times[i] - recent_times[i-1]).total_seconds() 
                                 for i in range(1, len(recent_times))]
                
                if recent_intervals:
                    current_rate = len(recent_intervals) / sum(recent_intervals)  # files per second
                    trend_estimate = remaining / current_rate if current_rate > 0 else final_estimate
                    
                    # Blend with other estimates
                    final_estimate = (final_estimate * 0.7 + trend_estimate * 0.3)
            except:
                pass
        
        return timedelta(seconds=max(0, final_estimate))
    
    def get_processing_rate(self) -> float:
        """Get current processing rate (files per second)"""
        if len(self.completion_times) < 2:
            return 0.0
        
        recent_times = list(self.completion_times)[-10:]  # Last 10 completions
        if len(recent_times) < 2:
            return 0.0
        
        time_span = (recent_times[-1] - recent_times[0]).total_seconds()
        if time_span > 0:
            return (len(recent_times) - 1) / time_span
        
        return 0.0
    
    def get_eta(self, files_processed: int, total_files: int) -> Optional[datetime]:
        """Get estimated time of completion"""
        remaining_time = self.estimate_remaining_time(files_processed, total_files)
        if remaining_time.total_seconds() > 0:
            return datetime.now() + remaining_time
        return None

class MultiCoreProcessor:
    """Multi-core processing optimization for audio operations"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.cpu_count = mp.cpu_count()
        self.optimal_workers = min(config.cpu_workers, self.cpu_count)
        
    def process_files_parallel(self, file_paths: List[str], 
                              process_func: Callable, 
                              progress_callback: Optional[Callable] = None) -> List[Any]:
        """Process files in parallel using multiple CPU cores"""
        
        if len(file_paths) == 1:
            # Single file - no need for parallel processing
            return [process_func(file_paths[0])]
        
        results = []
        completed = 0
        
        # Use ProcessPoolExecutor for CPU-intensive tasks
        with ProcessPoolExecutor(max_workers=self.optimal_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_func, file_path): file_path 
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append((file_path, result))
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(file_paths), file_path)
                        
                except Exception as e:
                    results.append((file_path, f"Error: {str(e)}"))
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(file_paths), file_path, error=str(e))
        
        return results
    
    def batch_process_with_memory_management(self, file_paths: List[str],
                                           process_func: Callable,
                                           memory_optimizer: MemoryOptimizer,
                                           progress_callback: Optional[Callable] = None) -> List[Any]:
        """Process files in batches with memory management"""
        
        # Calculate optimal batch size
        file_sizes = []
        for file_path in file_paths[:20]:  # Sample first 20 files
            try:
                file_sizes.append(os.path.getsize(file_path))
            except:
                file_sizes.append(50 * 1024 * 1024)  # 50MB default
        
        batch_size = memory_optimizer.smart_batch_size(file_sizes, target_memory_mb=self.config.memory_limit_mb // 4)
        
        all_results = []
        total_files = len(file_paths)
        
        # Process in batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            
            # Check memory before processing batch
            if not memory_optimizer.is_memory_available(self.config.memory_limit_mb // 2):
                memory_optimizer.cleanup_memory()
                time.sleep(1)  # Give system time to clean up
            
            # Process batch
            batch_results = self.process_files_parallel(batch, process_func, progress_callback)
            all_results.extend(batch_results)
            
            # Clean up after batch
            memory_optimizer.cleanup_memory()
            
            # Update progress for batch completion
            if progress_callback:
                progress_callback(min(i + batch_size, total_files), total_files, "Batch completed")
        
        return all_results

class AdvancedAudioProcessor:
    """Main advanced processing class combining all optimizations"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.gpu_accelerator = GPUAccelerator()
        self.memory_optimizer = MemoryOptimizer(self.config.memory_limit_mb)
        self.progress_estimator = ProgressEstimator()
        self.multicore_processor = MultiCoreProcessor(self.config)
        self.current_progress = ProcessingProgress()
        
        # Print system info
        self._print_system_info()
    
    def _print_system_info(self):
        """Print system capabilities"""
        print("\n" + "="*60)
        print("ADVANCED PROCESSING SYSTEM INITIALIZATION")
        print("="*60)
        
        # CPU info
        print(f"CPU Cores: {mp.cpu_count()} (using {self.config.cpu_workers} workers)")
        
        # Memory info
        memory = self.memory_optimizer.get_memory_usage()
        print(f"Memory: {memory['total_mb']:.1f}MB total, {memory['available_mb']:.1f}MB available")
        print(f"Memory limit: {self.config.memory_limit_mb}MB")
        
        # GPU info
        if self.gpu_accelerator.is_available():
            info = self.gpu_accelerator.device_info
            print(f"GPU: {info['device_name']} ({info['backend']})")
            gpu_memory = self.gpu_accelerator.get_memory_info()
            print(f"GPU Memory: {gpu_memory['total']:.1f}MB total")
        else:
            print("GPU: Not available - using CPU only")
        
        print("="*60)
    
    def batch_analyze_library(self, directory: str, 
                             file_extensions: List[str] = None,
                             analysis_func: Callable = None,
                             max_files: int = None) -> List[Any]:
        """
        Advanced batch analysis of entire music library
        
        Args:
            directory: Root directory to analyze
            file_extensions: List of file extensions to process
            analysis_func: Function to analyze each file
            max_files: Maximum number of files to process (for testing)
            
        Returns:
            List of analysis results
        """
        if file_extensions is None:
            file_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff']
        
        # Find all audio files
        print(f"Scanning directory: {directory}")
        audio_files = []
        
        for ext in file_extensions:
            audio_files.extend(Path(directory).rglob(f'*{ext}'))
            audio_files.extend(Path(directory).rglob(f'*{ext.upper()}'))
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        total_files = len(audio_files)
        print(f"Found {total_files} audio files")
        
        if total_files == 0:
            return []
        
        # Initialize progress tracking
        self.current_progress = ProcessingProgress(
            total_files=total_files,
            start_time=datetime.now()
        )
        self.progress_estimator.start_processing()
        
        # Default analysis function
        if analysis_func is None:
            from audio_analyzer import IntelligentAudioAnalyzer
            analyzer = IntelligentAudioAnalyzer()
            analysis_func = analyzer.analyze_audio_file
        
        # Create progress callback
        def progress_callback(completed: int, total: int, current_file: str, error: str = None):
            self._update_progress(completed, total, current_file, error)
        
        # Process files with advanced optimizations
        results = self.multicore_processor.batch_process_with_memory_management(
            [str(f) for f in audio_files],
            analysis_func,
            self.memory_optimizer,
            progress_callback
        )
        
        print(f"\nProcessing complete! Analyzed {len(results)} files")
        return results
    
    def _update_progress(self, completed: int, total: int, current_file: str, error: str = None):
        """Update progress information"""
        self.current_progress.files_processed = completed
        self.current_progress.current_file = os.path.basename(current_file)
        
        # Record completion for estimation
        if completed > self.current_progress.files_processed:
            processing_time = time.time() - getattr(self, '_last_file_start', time.time())
            self.progress_estimator.record_file_completion(current_file, processing_time)
        
        # Update estimates
        self.current_progress.processing_rate = self.progress_estimator.get_processing_rate()
        self.current_progress.estimated_finish = self.progress_estimator.get_eta(completed, total)
        
        # Update system metrics
        memory_info = self.memory_optimizer.get_memory_usage()
        self.current_progress.memory_usage = memory_info['rss_mb']
        self.current_progress.cpu_usage = psutil.cpu_percent()
        
        if self.gpu_accelerator.is_available():
            gpu_memory = self.gpu_accelerator.get_memory_info()
            self.current_progress.gpu_usage = (gpu_memory['used'] / gpu_memory['total']) * 100
        
        # Handle errors
        if error:
            self.current_progress.errors.append(f"{current_file}: {error}")
        
        # Call user callback if provided
        if self.config.progress_callback:
            self.config.progress_callback(self.current_progress)
        
        # Print progress
        self._print_progress()
        
        self._last_file_start = time.time()
    
    def _print_progress(self):
        """Print current progress"""
        p = self.current_progress
        
        print(f"\rProgress: {p.progress_percentage:.1f}% "
              f"({p.files_processed}/{p.total_files}) | "
              f"Rate: {p.processing_rate:.1f} files/sec | "
              f"Memory: {p.memory_usage:.0f}MB | "
              f"CPU: {p.cpu_usage:.0f}% | "
              f"ETA: {p.estimated_remaining}", end="")
        
        if p.files_processed == p.total_files:
            print()  # New line when complete
    
    def optimize_audio_batch_gpu(self, audio_files: List[str], 
                                operation: str = "spectral_analysis") -> List[np.ndarray]:
        """
        GPU-optimized batch processing for spectral operations
        
        Args:
            audio_files: List of audio file paths
            operation: Type of operation ('spectral_analysis', 'mfcc', 'stft')
            
        Returns:
            List of processed results
        """
        if not self.gpu_accelerator.is_available():
            print("GPU not available, falling back to CPU")
            return self._cpu_batch_process(audio_files, operation)
        
        results = []
        batch_size = self.config.gpu_batch_size
        
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i + batch_size]
            
            # Load audio batch
            audio_batch = []
            for file_path in batch:
                try:
                    y, sr = librosa.load(file_path, sr=22050, duration=30)  # Standardize
                    audio_batch.append(y)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    audio_batch.append(np.zeros(22050))  # Placeholder
            
            # GPU batch processing
            if operation == "stft":
                batch_results = self.gpu_accelerator.batch_process_spectrograms(audio_batch)
            elif operation == "mfcc":
                batch_results = [
                    self.gpu_accelerator.accelerate_mfcc(y, sr=22050) 
                    for y in audio_batch
                ]
            else:  # spectral_analysis
                batch_results = [
                    np.abs(self.gpu_accelerator.accelerate_stft(y))
                    for y in audio_batch
                ]
            
            results.extend(batch_results)
            
            # Memory cleanup
            self.memory_optimizer.cleanup_memory()
        
        return results
    
    def _cpu_batch_process(self, audio_files: List[str], operation: str) -> List[np.ndarray]:
        """CPU fallback for batch processing"""
        results = []
        
        for file_path in audio_files:
            try:
                y, sr = librosa.load(file_path, sr=22050, duration=30)
                
                if operation == "stft":
                    result = librosa.stft(y)
                elif operation == "mfcc":
                    result = librosa.feature.mfcc(y=y, sr=sr)
                else:  # spectral_analysis
                    result = np.abs(librosa.stft(y))
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results.append(np.zeros((129, 1)))  # Placeholder
        
        return results
    
    def process_large_audio_file(self, file_path: str, 
                                process_func: Callable,
                                chunk_duration: float = 30.0) -> Any:
        """
        Process very large audio files in chunks to manage memory
        
        Args:
            file_path: Path to large audio file
            process_func: Function to apply to each chunk
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            Combined result from all chunks
        """
        try:
            # Get file info
            info = sf.info(file_path)
            duration = info.duration
            sr = info.samplerate
            
            if duration <= chunk_duration:
                # File is small enough to process normally
                y, sr = librosa.load(file_path, sr=None)
                return process_func(y, sr)
            
            print(f"Processing large file ({duration:.1f}s) in chunks...")
            
            chunk_samples = int(chunk_duration * sr)
            chunks_processed = 0
            total_chunks = int(np.ceil(duration / chunk_duration))
            
            results = []
            
            # Process file in chunks
            with sf.SoundFile(file_path) as f:
                while True:
                    chunk = f.read(chunk_samples)
                    
                    if len(chunk) == 0:
                        break
                    
                    # Convert to mono if stereo
                    if len(chunk.shape) > 1:
                        chunk = np.mean(chunk, axis=1)
                    
                    # Process chunk
                    chunk_result = process_func(chunk, sr)
                    results.append(chunk_result)
                    
                    chunks_processed += 1
                    print(f"\rProcessed chunk {chunks_processed}/{total_chunks}", end="")
                    
                    # Clean up memory
                    if chunks_processed % 10 == 0:
                        self.memory_optimizer.cleanup_memory()
            
            print()  # New line
            
            # Combine results (implementation depends on the specific function)
            if results and hasattr(results[0], 'shape'):
                # Numpy arrays - concatenate
                if len(results[0].shape) == 1:
                    return np.concatenate(results)
                else:
                    return np.concatenate(results, axis=1)
            else:
                # Other types - return list
                return results
                
        except Exception as e:
            print(f"Error processing large file {file_path}: {e}")
            return None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        memory_info = self.memory_optimizer.get_memory_usage()
        gpu_info = self.gpu_accelerator.get_memory_info() if self.gpu_accelerator.is_available() else {}
        
        return {
            "progress": {
                "percentage": self.current_progress.progress_percentage,
                "files_processed": self.current_progress.files_processed,
                "total_files": self.current_progress.total_files,
                "processing_rate": self.current_progress.processing_rate,
                "estimated_finish": self.current_progress.estimated_finish.isoformat() if self.current_progress.estimated_finish else None,
                "elapsed_time": str(self.current_progress.elapsed_time)
            },
            "system": {
                "cpu_usage": psutil.cpu_percent(),
                "cpu_cores": mp.cpu_count(),
                "cpu_workers": self.config.cpu_workers,
                "memory_usage_mb": memory_info['rss_mb'],
                "memory_available_mb": memory_info['available_mb'],
                "memory_total_mb": memory_info['total_mb'],
                "memory_limit_mb": self.config.memory_limit_mb
            },
            "gpu": {
                "available": self.gpu_accelerator.is_available(),
                "backend": self.gpu_accelerator.backend,
                "memory_used_mb": gpu_info.get('used', 0),
                "memory_total_mb": gpu_info.get('total', 0),
                "usage_percent": (gpu_info.get('used', 0) / gpu_info.get('total', 1)) * 100 if gpu_info.get('total') else 0
            },
            "errors": len(self.current_progress.errors),
            "error_list": self.current_progress.errors[-5:]  # Last 5 errors
        }

def demo_advanced_processing():
    """Demonstrate advanced processing capabilities"""
    print("\n" + "="*60)
    print("ADVANCED PROCESSING DEMO")
    print("="*60)
    
    # Create configuration
    config = ProcessingConfig(
        use_gpu=True,
        cpu_workers=4,
        memory_limit_mb=4096,
        enable_progress_estimation=True
    )
    
    # Initialize processor
    processor = AdvancedAudioProcessor(config)
    
    # Demo 1: GPU acceleration
    print("\n1. GPU ACCELERATION DEMO")
    print("-" * 30)
    
    sample_dir = Path("sample_audio")
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*.wav"))[:3]  # Test with 3 files
        
        if sample_files:
            print("Testing GPU-accelerated batch processing...")
            
            start_time = time.time()
            results = processor.optimize_audio_batch_gpu(
                [str(f) for f in sample_files], 
                operation="spectral_analysis"
            )
            gpu_time = time.time() - start_time
            
            print(f"✓ Processed {len(sample_files)} files in {gpu_time:.2f}s")
            print(f"  Average: {gpu_time/len(sample_files):.3f}s per file")
            
            # Compare with CPU
            start_time = time.time()
            cpu_results = processor._cpu_batch_process(
                [str(f) for f in sample_files], 
                "spectral_analysis"
            )
            cpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1
            print(f"  CPU time: {cpu_time:.2f}s")
            print(f"  Speedup: {speedup:.1f}x")
    
    # Demo 2: Memory optimization
    print("\n2. MEMORY OPTIMIZATION DEMO")
    print("-" * 30)
    
    memory_info = processor.memory_optimizer.get_memory_usage()
    print(f"Current memory usage: {memory_info['rss_mb']:.1f}MB")
    print(f"Available memory: {memory_info['available_mb']:.1f}MB")
    
    # Test memory-mapped processing
    if sample_files:
        test_file = str(sample_files[0])
        print(f"Testing memory mapping for: {Path(test_file).name}")
        
        # Estimate memory usage
        estimated_mb = processor.memory_optimizer.estimate_audio_memory(test_file)
        print(f"Estimated memory for loading: {estimated_mb:.1f}MB")
        
        # Check if memory is available
        has_memory = processor.memory_optimizer.is_memory_available(estimated_mb)
        print(f"Sufficient memory available: {has_memory}")
    
    # Demo 3: Multi-core processing
    print("\n3. MULTI-CORE PROCESSING DEMO")
    print("-" * 30)
    
    if sample_files:
        def simple_analysis(file_path):
            """Simple analysis function for demo"""
            try:
                y, sr = librosa.load(file_path, duration=10)  # 10 seconds max
                return {
                    "file": os.path.basename(file_path),
                    "duration": len(y) / sr,
                    "rms": float(np.sqrt(np.mean(y**2))),
                    "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
                }
            except Exception as e:
                return {"file": os.path.basename(file_path), "error": str(e)}
        
        print(f"Processing {len(sample_files)} files with {config.cpu_workers} workers...")
        
        start_time = time.time()
        results = processor.multicore_processor.process_files_parallel(
            [str(f) for f in sample_files],
            simple_analysis
        )
        parallel_time = time.time() - start_time
        
        print(f"✓ Parallel processing completed in {parallel_time:.2f}s")
        
        for file_path, result in results:
            if isinstance(result, dict) and "error" not in result:
                print(f"  {result['file']}: {result['duration']:.1f}s, "
                      f"RMS: {result['rms']:.3f}, "
                      f"Centroid: {result['spectral_centroid']:.0f}Hz")
    
    # Demo 4: Progress estimation
    print("\n4. PROGRESS ESTIMATION DEMO")
    print("-" * 30)
    
    # Simulate processing with progress tracking
    processor.progress_estimator.start_processing()
    
    print("Simulating file processing with progress estimation...")
    for i in range(5):
        time.sleep(0.5)  # Simulate processing time
        processor.progress_estimator.record_file_completion(f"file_{i}.wav", 0.5)
        
        rate = processor.progress_estimator.get_processing_rate()
        eta = processor.progress_estimator.get_eta(i + 1, 10)
        
        print(f"  File {i+1}/10 - Rate: {rate:.1f} files/sec - ETA: {eta}")
    
    # Demo 5: System statistics
    print("\n5. SYSTEM STATISTICS")
    print("-" * 30)
    
    stats = processor.get_processing_stats()
    
    print("System Information:")
    sys_info = stats["system"]
    print(f"  CPU: {sys_info['cpu_cores']} cores, {sys_info['cpu_usage']:.1f}% usage")
    print(f"  Memory: {sys_info['memory_usage_mb']:.0f}MB used, "
          f"{sys_info['memory_available_mb']:.0f}MB available")
    
    gpu_info = stats["gpu"]
    if gpu_info["available"]:
        print(f"  GPU: {gpu_info['backend']}, {gpu_info['usage_percent']:.1f}% used")
    else:
        print("  GPU: Not available")
    
    print("\n" + "="*60)
    print("ADVANCED PROCESSING DEMO COMPLETE")
    print("="*60)
    print("\nAdvanced features demonstrated:")
    print("• GPU acceleration for spectral operations")
    print("• Memory optimization and monitoring")
    print("• Multi-core parallel processing")
    print("• Smart progress estimation")
    print("• Comprehensive system monitoring")
    print("• Batch processing with memory management")

if __name__ == "__main__":
    demo_advanced_processing()