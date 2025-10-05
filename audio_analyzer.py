#!/usr/bin/env python3
"""
Intelligent Audio Analysis Module
Provides advanced audio analysis features including BPM detection, key detection,
audio fingerprinting, and loudness analysis.
"""

import os
import sys
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
from collections import defaultdict

# Advanced processing imports
try:
    from advanced_processing import (
        AdvancedAudioProcessor, ProcessingConfig, GPUAccelerator, 
        MemoryOptimizer, ProgressEstimator, MultiCoreProcessor
    )
    HAS_ADVANCED_PROCESSING = True
except ImportError:
    HAS_ADVANCED_PROCESSING = False
    print("Note: Advanced processing features not available. Run with advanced_processing.py for GPU acceleration and optimization.")

# Try to import optional dependencies
try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False
    print("Warning: pyloudnorm not found. LUFS analysis will be disabled.")

try:
    import essentia
    import essentia.standard as es
    HAS_ESSENTIA = True
except ImportError:
    HAS_ESSENTIA = False
    print("Warning: essentia not found. Advanced analysis features will be limited.")

try:
    import chromaprint
    HAS_CHROMAPRINT = True
except ImportError:
    HAS_CHROMAPRINT = False
    print("Warning: chromaprint not found. Audio fingerprinting will be disabled.")

try:
    from music21 import pitch, key as music21_key, stream, note
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False
    print("Warning: music21 not found. Advanced key detection will be limited.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not found. ML classification features will be disabled.")

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: tensorflow not found. Deep learning features will be disabled.")

try:
    import scipy
    from scipy import signal, interpolate
    from scipy.signal import filtfilt, butter, find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Advanced signal processing will be limited.")

try:
    import pyrubberband as pyrb
    HAS_PYRUBBERBAND = True
except ImportError:
    HAS_PYRUBBERBAND = False
    print("Warning: pyrubberband not found. Time-stretching features will be limited.")

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("Warning: noisereduce not found. Noise reduction features will be limited.")

@dataclass
class AudioAnalysisResult:
    """Container for audio analysis results"""
    filename: str
    duration: float
    sample_rate: int
    
    # BPM Analysis
    bpm: Optional[float] = None
    tempo_category: Optional[str] = None
    
    # Key Analysis
    key: Optional[str] = None
    key_confidence: Optional[float] = None
    scale: Optional[str] = None  # major/minor
    
    # Loudness Analysis
    lufs_integrated: Optional[float] = None
    lufs_short_term: Optional[float] = None
    lufs_momentary: Optional[float] = None
    peak_db: Optional[float] = None
    
    # Audio Fingerprint
    fingerprint: Optional[str] = None
    fingerprint_duration: Optional[float] = None
    
    # Additional Analysis
    spectral_centroid: Optional[float] = None
    spectral_rolloff: Optional[float] = None
    zero_crossing_rate: Optional[float] = None
    mfcc_features: Optional[List[float]] = None
    
    # Genre and mood analysis
    genre_prediction: Optional[str] = None
    genre_confidence: Optional[float] = None
    mood_prediction: Optional[str] = None
    mood_confidence: Optional[float] = None
    energy_level: Optional[float] = None
    danceability: Optional[float] = None
    valence: Optional[float] = None  # Musical positivity
    
    # Instrument recognition
    detected_instruments: Optional[List[str]] = None
    instrument_confidence: Optional[Dict[str, float]] = None
    
    # Advanced ML features
    audio_embeddings: Optional[np.ndarray] = None
    content_category: Optional[str] = None  # speech, music, noise, etc.
    
    # Rhythm analysis
    rhythmic_complexity: Optional[float] = None
    beat_strength: Optional[float] = None
    onset_density: Optional[float] = None
    
    # Content-based search features
    chroma_features: Optional[np.ndarray] = None  # For harmonic matching
    rhythm_features: Optional[np.ndarray] = None  # For rhythm matching
    spectral_features: Optional[np.ndarray] = None  # For spectral search
    similarity_hash: Optional[str] = None  # For audio similarity search
    
    # Error information
    analysis_errors: List[str] = None
    
    def __post_init__(self):
        if self.analysis_errors is None:
            self.analysis_errors = []

@dataclass
class AudioProcessingResult:
    """Container for audio processing results"""
    success: bool
    message: str
    output_path: Optional[str] = None
    processing_time: Optional[float] = None
    parameters_used: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, float]] = None
    additional_outputs: Optional[Dict[str, str]] = None

class ProfessionalAudioProcessor:
    """Professional audio processing tools for studio-quality operations"""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.flac', '.aiff', '.mp3', '.aac', '.m4a', '.ogg', '.opus', '.wma', '.au', '.caf', '.w64', '.aifc', '.bwf']
        self.temp_dir = Path("temp_processing")
        self.temp_dir.mkdir(exist_ok=True)
        
    def time_stretch(self, input_path: str, output_path: str, stretch_factor: float, 
                    preserve_pitch: bool = True, quality: str = "high") -> AudioProcessingResult:
        """
        Time-stretch audio without pitch change using advanced algorithms
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file
            stretch_factor: Time stretch factor (1.0 = no change, 2.0 = double length, 0.5 = half length)
            preserve_pitch: Whether to preserve pitch during stretching
            quality: Processing quality ('low', 'medium', 'high', 'ultra')
        """
        start_time = datetime.now()
        
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None)
            
            if HAS_PYRUBBERBAND and quality in ['high', 'ultra']:
                # Use Rubber Band for high-quality time stretching
                options = []
                if quality == 'ultra':
                    options.extend(['--formant-corrected', '--smoothing'])
                if preserve_pitch:
                    options.append('--pitch-hq')
                
                # Process with pyrubberband
                y_stretched = pyrb.time_stretch(y, sr, stretch_factor, rbargs=options)
                
            else:
                # Fallback to librosa phase vocoder
                if preserve_pitch:
                    # Use phase vocoder for pitch preservation
                    stft = librosa.stft(y, hop_length=512, n_fft=2048)
                    stft_stretched = librosa.phase_vocoder(stft, rate=1/stretch_factor)
                    y_stretched = librosa.istft(stft_stretched, hop_length=512)
                else:
                    # Simple resampling (will change pitch)
                    new_length = int(len(y) * stretch_factor)
                    y_stretched = scipy.signal.resample(y, new_length)
            
            # Save processed audio
            sf.write(output_path, y_stretched, sr)
            
            # Calculate quality metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AudioProcessingResult(
                success=True,
                message=f"Time-stretched by factor {stretch_factor:.2f}",
                output_path=output_path,
                processing_time=processing_time,
                parameters_used={
                    'stretch_factor': stretch_factor,
                    'preserve_pitch': preserve_pitch,
                    'quality': quality
                }
            )
            
        except Exception as e:
            return AudioProcessingResult(
                success=False,
                message=f"Time-stretch failed: {str(e)}"
            )
    
    def repair_audio(self, input_path: str, output_path: str, 
                    operations: List[str] = None) -> AudioProcessingResult:
        """
        Comprehensive audio repair including declicking, denoising, and dehumming
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file
            operations: List of repair operations ['declick', 'denoise', 'dehum', 'normalize']
        """
        if operations is None:
            operations = ['declick', 'denoise', 'dehum']
        
        start_time = datetime.now()
        
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None)
            y_original = y.copy()
            
            repair_log = []
            
            # Declick - remove clicks and pops
            if 'declick' in operations:
                y = self._declick_audio(y, sr)
                repair_log.append("Applied declicking")
            
            # Denoise - remove background noise
            if 'denoise' in operations:
                y = self._denoise_audio(y, sr)
                repair_log.append("Applied noise reduction")
            
            # Dehum - remove 50/60Hz hum and harmonics
            if 'dehum' in operations:
                y = self._dehum_audio(y, sr)
                repair_log.append("Applied dehum filtering")
            
            # Normalize if requested
            if 'normalize' in operations:
                y = self._normalize_audio(y, target_lufs=-23.0)
                repair_log.append("Applied normalization")
            
            # Save repaired audio
            sf.write(output_path, y, sr)
            
            # Calculate improvement metrics
            snr_before = self._calculate_snr(y_original)
            snr_after = self._calculate_snr(y)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AudioProcessingResult(
                success=True,
                message=f"Audio repair completed: {', '.join(repair_log)}",
                output_path=output_path,
                processing_time=processing_time,
                parameters_used={'operations': operations},
                quality_metrics={
                    'snr_improvement_db': snr_after - snr_before,
                    'snr_before': snr_before,
                    'snr_after': snr_after
                }
            )
            
        except Exception as e:
            return AudioProcessingResult(
                success=False,
                message=f"Audio repair failed: {str(e)}"
            )
    
    def spectral_edit(self, input_path: str, output_path: str, 
                     freq_ranges: List[Tuple[float, float]], 
                     operation: str = "remove") -> AudioProcessingResult:
        """
        Spectral editing - remove or isolate specific frequency ranges
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file
            freq_ranges: List of (low_freq, high_freq) tuples in Hz
            operation: 'remove', 'isolate', or 'attenuate'
        """
        start_time = datetime.now()
        
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None)
            
            # Convert to frequency domain
            stft = librosa.stft(y, n_fft=4096, hop_length=1024)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Create frequency bins
            freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
            
            # Apply spectral editing
            for low_freq, high_freq in freq_ranges:
                # Find frequency bin indices
                low_bin = np.argmin(np.abs(freqs - low_freq))
                high_bin = np.argmin(np.abs(freqs - high_freq))
                
                if operation == "remove":
                    # Zero out the specified frequency range
                    magnitude[low_bin:high_bin+1, :] = 0
                elif operation == "isolate":
                    # Keep only the specified frequency range
                    mask = np.ones_like(magnitude)
                    mask[low_bin:high_bin+1, :] = 0
                    magnitude *= (1 - mask)
                elif operation == "attenuate":
                    # Reduce the specified frequency range by -20dB
                    magnitude[low_bin:high_bin+1, :] *= 0.1
            
            # Reconstruct audio
            stft_modified = magnitude * np.exp(1j * phase)
            y_processed = librosa.istft(stft_modified, hop_length=1024)
            
            # Save processed audio
            sf.write(output_path, y_processed, sr)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AudioProcessingResult(
                success=True,
                message=f"Spectral editing completed: {operation} {len(freq_ranges)} frequency ranges",
                output_path=output_path,
                processing_time=processing_time,
                parameters_used={
                    'freq_ranges': freq_ranges,
                    'operation': operation
                }
            )
            
        except Exception as e:
            return AudioProcessingResult(
                success=False,
                message=f"Spectral editing failed: {str(e)}"
            )
    
    def stereo_field_manipulation(self, input_path: str, output_path: str,
                                operation: str, parameters: Dict[str, Any]) -> AudioProcessingResult:
        """
        Stereo field manipulation - mono to stereo, width control, panning
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file
            operation: 'mono_to_stereo', 'width_control', 'pan', 'mid_side'
            parameters: Operation-specific parameters
        """
        start_time = datetime.now()
        
        try:
            # Load audio (preserve stereo if present)
            y, sr = librosa.load(input_path, sr=None, mono=False)
            
            if len(y.shape) == 1:
                # Mono audio - convert to stereo for processing
                y = np.array([y, y])
            
            if operation == "mono_to_stereo":
                # Create stereo from mono using various techniques
                technique = parameters.get('technique', 'delay')
                
                if technique == 'delay':
                    delay_ms = parameters.get('delay_ms', 10)
                    delay_samples = int(delay_ms * sr / 1000)
                    y_right = np.pad(y[0], (delay_samples, 0), mode='constant')[:-delay_samples]
                    y = np.array([y[0], y_right])
                
                elif technique == 'chorus':
                    # Add slight modulation to create width
                    lfo_freq = parameters.get('lfo_freq', 0.5)
                    lfo_depth = parameters.get('lfo_depth', 0.02)
                    t = np.arange(len(y[0])) / sr
                    lfo = np.sin(2 * np.pi * lfo_freq * t) * lfo_depth
                    
                    # Apply modulated delay to right channel
                    y_right = self._apply_modulated_delay(y[0], lfo, sr)
                    y = np.array([y[0], y_right])
                
                elif technique == 'reverb':
                    # Add different reverb to each channel
                    y_right = self._apply_simple_reverb(y[0], sr, 0.3, 0.05)
                    y = np.array([y[0], y_right])
            
            elif operation == "width_control":
                # Adjust stereo width
                width = parameters.get('width', 1.0)  # 0 = mono, 1 = normal, >1 = wide
                
                # Convert to mid-side
                mid = (y[0] + y[1]) / 2
                side = (y[0] - y[1]) / 2
                
                # Adjust side signal
                side *= width
                
                # Convert back to left-right
                y[0] = mid + side
                y[1] = mid - side
            
            elif operation == "pan":
                # Pan the audio
                pan_position = parameters.get('position', 0.0)  # -1 = left, 0 = center, 1 = right
                
                # Calculate gain factors
                left_gain = np.sqrt(0.5 * (1 - pan_position))
                right_gain = np.sqrt(0.5 * (1 + pan_position))
                
                # Apply panning
                mono_signal = (y[0] + y[1]) / 2
                y[0] = mono_signal * left_gain
                y[1] = mono_signal * right_gain
            
            elif operation == "mid_side":
                # Mid-side processing
                mode = parameters.get('mode', 'encode')
                
                if mode == 'encode':
                    # Convert L/R to M/S
                    mid = (y[0] + y[1]) / 2
                    side = (y[0] - y[1]) / 2
                    y = np.array([mid, side])
                else:  # decode
                    # Convert M/S to L/R
                    left = y[0] + y[1]
                    right = y[0] - y[1]
                    y = np.array([left, right])
            
            # Save processed audio
            sf.write(output_path, y.T, sr)  # Transpose for soundfile format
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AudioProcessingResult(
                success=True,
                message=f"Stereo field manipulation completed: {operation}",
                output_path=output_path,
                processing_time=processing_time,
                parameters_used=parameters
            )
            
        except Exception as e:
            return AudioProcessingResult(
                success=False,
                message=f"Stereo field manipulation failed: {str(e)}"
            )
    
    def mastering_chain(self, input_path: str, output_path: str,
                       chain_config: Dict[str, Any]) -> AudioProcessingResult:
        """
        Complete mastering chain with EQ, compression, and limiting
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file
            chain_config: Configuration for the mastering chain
        """
        start_time = datetime.now()
        
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None, mono=False)
            if len(y.shape) == 1:
                y = np.array([y, y])  # Convert to stereo
            
            processing_log = []
            
            # EQ Stage
            if 'eq' in chain_config:
                eq_config = chain_config['eq']
                y = self._apply_eq(y, sr, eq_config)
                processing_log.append("Applied EQ")
            
            # Compression Stage
            if 'compressor' in chain_config:
                comp_config = chain_config['compressor']
                y = self._apply_compressor(y, sr, comp_config)
                processing_log.append("Applied compression")
            
            # Stereo Enhancement
            if 'stereo_enhance' in chain_config:
                enhance_config = chain_config['stereo_enhance']
                y = self._apply_stereo_enhancement(y, enhance_config)
                processing_log.append("Applied stereo enhancement")
            
            # Limiting Stage (always last)
            if 'limiter' in chain_config:
                limit_config = chain_config['limiter']
                y = self._apply_limiter(y, sr, limit_config)
                processing_log.append("Applied limiting")
            
            # Final normalization to target LUFS
            target_lufs = chain_config.get('target_lufs', -14.0)
            if HAS_PYLOUDNORM:
                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(y.T)
                y = pyln.normalize.loudness(y.T, loudness, target_lufs).T
                processing_log.append(f"Normalized to {target_lufs} LUFS")
            
            # Save mastered audio
            sf.write(output_path, y.T, sr)
            
            # Calculate quality metrics
            peak_level = np.max(np.abs(y))
            rms_level = np.sqrt(np.mean(y**2))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AudioProcessingResult(
                success=True,
                message=f"Mastering chain completed: {', '.join(processing_log)}",
                output_path=output_path,
                processing_time=processing_time,
                parameters_used=chain_config,
                quality_metrics={
                    'peak_level': float(peak_level),
                    'rms_level': float(rms_level),
                    'target_lufs': target_lufs
                }
            )
            
        except Exception as e:
            return AudioProcessingResult(
                success=False,
                message=f"Mastering chain failed: {str(e)}"
            )
    
    # Helper methods for audio processing
    def _declick_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Remove clicks and pops from audio"""
        # Simple declick using median filtering for impulse noise
        from scipy.signal import medfilt
        
        # Apply median filter to remove clicks
        y_filtered = medfilt(y, kernel_size=3)
        
        # Detect and replace only the clicks
        diff = np.abs(y - y_filtered)
        threshold = np.std(diff) * 3
        click_mask = diff > threshold
        
        y_output = y.copy()
        y_output[click_mask] = y_filtered[click_mask]
        
        return y_output
    
    def _denoise_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Remove background noise"""
        if HAS_NOISEREDUCE:
            # Use noisereduce library for advanced noise reduction
            return nr.reduce_noise(y=y, sr=sr)
        else:
            # Simple spectral gating approach
            stft = librosa.stft(y, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            
            # Estimate noise floor from quiet sections
            power = magnitude ** 2
            noise_floor = np.percentile(power, 10, axis=1, keepdims=True)
            
            # Apply soft gating
            gate_ratio = 0.1
            mask = power / (noise_floor + 1e-10)
            soft_mask = np.tanh(mask * gate_ratio)
            
            # Apply mask and reconstruct
            stft_denoised = stft * soft_mask
            return librosa.istft(stft_denoised, hop_length=512)
    
    def _dehum_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Remove 50/60Hz hum and harmonics"""
        if not HAS_SCIPY:
            return y
        
        # Remove 50Hz, 60Hz and their harmonics
        hum_freqs = [50, 60, 100, 120, 150, 180]
        
        for freq in hum_freqs:
            if freq < sr / 2:  # Below Nyquist frequency
                # Create notch filter
                Q = 30  # Quality factor
                w0 = freq / (sr / 2)
                b, a = signal.iirnotch(w0, Q)
                y = signal.filtfilt(b, a, y)
        
        return y
    
    def _normalize_audio(self, y: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
        """Normalize audio to target LUFS"""
        if HAS_PYLOUDNORM:
            meter = pyln.Meter(44100)  # Standard sample rate for LUFS
            y_resampled = librosa.resample(y, orig_sr=len(y), target_sr=44100)
            loudness = meter.integrated_loudness(y_resampled.reshape(1, -1))
            return pyln.normalize.loudness(y.reshape(1, -1), loudness, target_lufs).flatten()
        else:
            # Simple peak normalization
            peak = np.max(np.abs(y))
            if peak > 0:
                return y / peak * 0.9  # Leave some headroom
            return y
    
    def _calculate_snr(self, y: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Simple SNR estimation
        signal_power = np.mean(y ** 2)
        # Estimate noise from quiet sections
        sorted_samples = np.sort(np.abs(y))
        noise_power = np.mean(sorted_samples[:len(sorted_samples)//10] ** 2)
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        return float('inf')
    
    def _apply_modulated_delay(self, y: np.ndarray, lfo: np.ndarray, sr: int) -> np.ndarray:
        """Apply modulated delay for chorus effect"""
        max_delay = int(0.02 * sr)  # 20ms max delay
        delay_line = np.zeros(len(y) + max_delay)
        delay_line[:len(y)] = y
        
        output = np.zeros_like(y)
        for i in range(len(y)):
            delay_samples = int(lfo[i] * max_delay)
            if i + delay_samples < len(delay_line):
                output[i] = delay_line[i + delay_samples]
        
        return output
    
    def _apply_simple_reverb(self, y: np.ndarray, sr: int, 
                           reverb_time: float, wet_level: float) -> np.ndarray:
        """Apply simple reverb effect"""
        # Create impulse response for reverb
        ir_length = int(reverb_time * sr)
        ir = np.random.randn(ir_length) * np.exp(-np.arange(ir_length) / (sr * 0.1))
        
        # Convolve with impulse response
        reverb = scipy.signal.convolve(y, ir, mode='same')
        
        # Mix dry and wet signals
        return y * (1 - wet_level) + reverb * wet_level
    
    def _apply_eq(self, y: np.ndarray, sr: int, eq_config: Dict[str, Any]) -> np.ndarray:
        """Apply EQ to audio"""
        if not HAS_SCIPY:
            return y
        
        # Apply multiple EQ bands
        for band in eq_config.get('bands', []):
            freq = band['frequency']
            gain_db = band['gain']
            q = band.get('q', 1.0)
            
            # Design peaking filter
            w0 = freq / (sr / 2)
            A = 10 ** (gain_db / 40)
            alpha = np.sin(w0) / (2 * q)
            
            # Peaking EQ coefficients
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
            
            # Normalize
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1, a1, a2]) / a0
            
            # Apply filter to each channel
            for ch in range(len(y)):
                y[ch] = signal.filtfilt(b, a, y[ch])
        
        return y
    
    def _apply_compressor(self, y: np.ndarray, sr: int, comp_config: Dict[str, Any]) -> np.ndarray:
        """Apply compression to audio"""
        threshold_db = comp_config.get('threshold', -12.0)
        ratio = comp_config.get('ratio', 4.0)
        attack_ms = comp_config.get('attack', 10.0)
        release_ms = comp_config.get('release', 100.0)
        
        # Convert to linear
        threshold = 10 ** (threshold_db / 20)
        
        # Calculate envelope follower parameters
        attack_coeff = np.exp(-1 / (attack_ms * sr / 1000))
        release_coeff = np.exp(-1 / (release_ms * sr / 1000))
        
        # Process each channel
        for ch in range(len(y)):
            envelope = 0
            output = np.zeros_like(y[ch])
            
            for i in range(len(y[ch])):
                # Envelope follower
                input_level = abs(y[ch][i])
                if input_level > envelope:
                    envelope = input_level * (1 - attack_coeff) + envelope * attack_coeff
                else:
                    envelope = input_level * (1 - release_coeff) + envelope * release_coeff
                
                # Compression
                if envelope > threshold:
                    # Calculate gain reduction
                    excess = envelope - threshold
                    compressed_excess = excess / ratio
                    gain_reduction = (threshold + compressed_excess) / envelope
                else:
                    gain_reduction = 1.0
                
                output[i] = y[ch][i] * gain_reduction
            
            y[ch] = output
        
        return y
    
    def _apply_stereo_enhancement(self, y: np.ndarray, enhance_config: Dict[str, Any]) -> np.ndarray:
        """Apply stereo enhancement"""
        width = enhance_config.get('width', 1.2)
        
        # Convert to mid-side
        mid = (y[0] + y[1]) / 2
        side = (y[0] - y[1]) / 2
        
        # Enhance side signal
        side *= width
        
        # Convert back to left-right
        y[0] = mid + side
        y[1] = mid - side
        
        return y
    
    def _apply_limiter(self, y: np.ndarray, sr: int, limit_config: Dict[str, Any]) -> np.ndarray:
        """Apply limiting to audio"""
        ceiling_db = limit_config.get('ceiling', -0.1)
        lookahead_ms = limit_config.get('lookahead', 5.0)
        
        ceiling = 10 ** (ceiling_db / 20)
        lookahead_samples = int(lookahead_ms * sr / 1000)
        
        # Simple brick-wall limiting with lookahead
        for ch in range(len(y)):
            # Delay the signal by lookahead amount
            delayed_signal = np.concatenate([np.zeros(lookahead_samples), y[ch]])
            
            # Calculate gain reduction
            gain = np.ones(len(delayed_signal))
            for i in range(len(y[ch])):
                # Look ahead for peaks
                window = delayed_signal[i:i+lookahead_samples+1]
                peak = np.max(np.abs(window))
                if peak > ceiling:
                    gain[i:i+lookahead_samples+1] = np.minimum(
                        gain[i:i+lookahead_samples+1], ceiling / peak)
            
            # Apply gain reduction
            y[ch] = delayed_signal[:len(y[ch])] * gain[:len(y[ch])]
        
        return y

    # New Professional Features
    
    def stem_separation(self, input_path: str, output_dir: str, 
                       stems: List[str] = None, method: str = "advanced",
                       quality: str = "high") -> AudioProcessingResult:
        """
        Advanced audio stem separation with multiple algorithms
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory for output stem files
            stems: List of stems to extract ['vocals', 'drums', 'bass', 'other', 'piano', 'accompaniment']
            method: Separation method ('basic', 'advanced', 'ai', 'spectral', 'nmf')
            quality: Processing quality ('fast', 'balanced', 'high', 'ultra')
        """
        if stems is None:
            stems = ['vocals', 'drums', 'bass', 'other']
        
        start_time = datetime.now()
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Load audio with appropriate quality settings
            sr_target = {'fast': 22050, 'balanced': 44100, 'high': 44100, 'ultra': 48000}[quality]
            y, sr = librosa.load(input_path, sr=sr_target, mono=False)
            
            if len(y.shape) == 1:
                y = np.array([y, y])  # Convert to stereo
            
            separated_stems = {}
            output_paths = {}
            separation_log = []
            
            # Choose separation method
            if method == "ai" and HAS_TENSORFLOW:
                separated_stems, method_log = self._ai_stem_separation(y, sr, stems, quality)
                separation_log.extend(method_log)
            elif method == "spectral":
                separated_stems, method_log = self._spectral_stem_separation(y, sr, stems, quality)
                separation_log.extend(method_log)
            elif method == "nmf":
                separated_stems, method_log = self._nmf_stem_separation(y, sr, stems, quality)
                separation_log.extend(method_log)
            elif method == "advanced":
                separated_stems, method_log = self._advanced_stem_separation(y, sr, stems, quality)
                separation_log.extend(method_log)
            else:
                # Fallback to basic method
                separated_stems, method_log = self._basic_stem_separation(y, sr, stems)
                separation_log.extend(method_log)
            
            # Post-processing and cleanup
            for stem_name, stem_audio in separated_stems.items():
                if stem_name in stems:
                    # Apply post-processing
                    stem_audio = self._post_process_stem(stem_audio, stem_name, quality)
                    
                    # Save stem
                    output_path = output_dir / f"{Path(input_path).stem}_{stem_name}.wav"
                    
                    # Ensure stereo output
                    if len(stem_audio.shape) == 1:
                        stem_audio = np.array([stem_audio, stem_audio])
                    
                    sf.write(output_path, stem_audio.T, sr)
                    output_paths[stem_name] = str(output_path)
                    separation_log.append(f"Saved {stem_name} stem")
            
            # Calculate quality metrics
            quality_metrics = self._calculate_separation_quality(y, separated_stems, stems)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AudioProcessingResult(
                success=True,
                message=f"Separated {len(output_paths)} stems using {method} method: {', '.join(output_paths.keys())}",
                output_path=str(output_dir),
                processing_time=processing_time,
                parameters_used={'stems': stems, 'method': method, 'quality': quality},
                quality_metrics=quality_metrics,
                additional_outputs=output_paths
            )
            
        except Exception as e:
            return AudioProcessingResult(
                success=False,
                message=f"Stem separation failed: {str(e)}"
            )
    
    # Advanced Stem Separation Helper Methods
    
    def _ai_stem_separation(self, y: np.ndarray, sr: int, stems: List[str], quality: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """AI-based stem separation using deep learning models"""
        log = []
        separated_stems = {}
        
        try:
            # Simplified AI separation - in production, use Spleeter, OpenUnmix, or similar
            log.append("Using AI-based separation (simplified implementation)")
            
            # For now, use advanced spectral methods as AI placeholder
            # In a full implementation, this would load pre-trained models
            separated_stems, spec_log = self._advanced_stem_separation(y, sr, stems, quality)
            log.extend(spec_log)
            log.append("Note: Full AI separation requires Spleeter or similar models")
            
        except Exception as e:
            log.append(f"AI separation failed, falling back to advanced method: {e}")
            separated_stems, spec_log = self._advanced_stem_separation(y, sr, stems, quality)
            log.extend(spec_log)
        
        return separated_stems, log
    
    def _spectral_stem_separation(self, y: np.ndarray, sr: int, stems: List[str], quality: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Advanced spectral-based stem separation"""
        log = []
        separated_stems = {}
        
        # Use higher resolution for better quality
        n_fft = {'fast': 2048, 'balanced': 4096, 'high': 8192, 'ultra': 16384}[quality]
        hop_length = n_fft // 4
        
        log.append(f"Using spectral separation with {n_fft}-point FFT")
        
        # Multi-resolution analysis
        stft_low = librosa.stft(y[0], n_fft=n_fft//2, hop_length=hop_length//2)
        stft_mid = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)
        stft_high = librosa.stft(y[0], n_fft=n_fft*2, hop_length=hop_length*2) if quality in ['high', 'ultra'] else stft_mid
        
        # Enhanced frequency-based separation with intelligent masking
        for stem in stems:
            if stem == 'vocals':
                # Advanced vocal isolation
                vocal_audio = self._isolate_vocals_advanced(y, sr, n_fft, hop_length)
                separated_stems[stem] = vocal_audio
                log.append("Applied advanced vocal isolation")
                
            elif stem == 'drums':
                # Percussive component with transient enhancement
                drums_audio = self._isolate_drums_advanced(y, sr, n_fft, hop_length)
                separated_stems[stem] = drums_audio
                log.append("Applied percussive isolation with transient enhancement")
                
            elif stem == 'bass':
                # Low-frequency harmonic content
                bass_audio = self._isolate_bass_advanced(y, sr, n_fft, hop_length)
                separated_stems[stem] = bass_audio
                log.append("Applied low-frequency harmonic isolation")
                
            elif stem == 'piano':
                # Harmonic content in piano frequency range
                piano_audio = self._isolate_piano_advanced(y, sr, n_fft, hop_length)
                separated_stems[stem] = piano_audio
                log.append("Applied piano-specific harmonic isolation")
                
            else:  # 'other' or any remaining stem
                # Residual after removing other components
                other_audio = self._isolate_other_advanced(y, sr, separated_stems)
                separated_stems[stem] = other_audio
                log.append("Applied residual isolation for remaining components")
        
        return separated_stems, log
    
    def _nmf_stem_separation(self, y: np.ndarray, sr: int, stems: List[str], quality: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Non-negative Matrix Factorization based stem separation"""
        log = []
        separated_stems = {}
        
        try:
            from sklearn.decomposition import NMF
            
            # Parameters based on quality
            n_components = {'fast': 8, 'balanced': 16, 'high': 32, 'ultra': 64}[quality]
            max_iter = {'fast': 100, 'balanced': 200, 'high': 500, 'ultra': 1000}[quality]
            
            log.append(f"Using NMF separation with {n_components} components")
            
            # Convert to magnitude spectrogram
            stft = librosa.stft(y[0], n_fft=4096, hop_length=1024)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply NMF
            nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=42)
            W = nmf.fit_transform(magnitude)  # Basis functions
            H = nmf.components_  # Activations
            
            # Assign components to stems based on frequency characteristics
            component_assignments = self._assign_nmf_components(W, stems, sr)
            
            for stem in stems:
                if stem in component_assignments:
                    # Reconstruct from assigned components
                    component_indices = component_assignments[stem]
                    reconstructed_magnitude = W[:, component_indices] @ H[component_indices, :]
                    
                    # Apply original phase
                    reconstructed_stft = reconstructed_magnitude * np.exp(1j * phase)
                    stem_audio = librosa.istft(reconstructed_stft, hop_length=1024)
                    
                    separated_stems[stem] = np.array([stem_audio, stem_audio])
                    log.append(f"Reconstructed {stem} from {len(component_indices)} NMF components")
                else:
                    # Fallback to basic method
                    basic_stems, _ = self._basic_stem_separation(y, sr, [stem])
                    separated_stems.update(basic_stems)
                    log.append(f"Used fallback method for {stem}")
            
        except ImportError:
            log.append("NMF not available, falling back to spectral method")
            separated_stems, spec_log = self._spectral_stem_separation(y, sr, stems, quality)
            log.extend(spec_log)
        except Exception as e:
            log.append(f"NMF separation failed: {e}, falling back to spectral method")
            separated_stems, spec_log = self._spectral_stem_separation(y, sr, stems, quality)
            log.extend(spec_log)
        
        return separated_stems, log
    
    def _advanced_stem_separation(self, y: np.ndarray, sr: int, stems: List[str], quality: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Advanced stem separation combining multiple techniques"""
        log = []
        separated_stems = {}
        
        log.append("Using advanced multi-technique separation")
        
        # Harmonic-percussive separation as foundation
        y_harmonic, y_percussive = librosa.effects.hpss(y[0], margin=8.0)
        
        # Spectral centroid for frequency characteristics
        spectral_centroids = librosa.feature.spectral_centroid(y=y[0], sr=sr)[0]
        avg_centroid = np.mean(spectral_centroids)
        
        for stem in stems:
            if stem == 'vocals':
                # Combine harmonic content with vocal-specific processing
                vocal_audio = self._enhance_vocal_separation(y_harmonic, y, sr)
                separated_stems[stem] = np.array([vocal_audio, vocal_audio])
                log.append("Applied enhanced vocal separation")
                
            elif stem == 'drums':
                # Enhanced percussive with drum-specific processing
                drum_audio = self._enhance_drum_separation(y_percussive, y, sr)
                separated_stems[stem] = np.array([drum_audio, drum_audio])
                log.append("Applied enhanced drum separation")
                
            elif stem == 'bass':
                # Low-frequency focused extraction
                bass_audio = self._enhance_bass_separation(y, sr)
                separated_stems[stem] = np.array([bass_audio, bass_audio])
                log.append("Applied enhanced bass separation")
                
            elif stem == 'piano':
                # Piano-specific harmonic extraction
                piano_audio = self._enhance_piano_separation(y_harmonic, y, sr)
                separated_stems[stem] = np.array([piano_audio, piano_audio])
                log.append("Applied piano-specific separation")
                
            else:  # 'other', 'accompaniment', etc.
                # Residual after removing primary stems
                other_audio = y[0].copy()
                # Subtract already separated components
                for existing_stem, existing_audio in separated_stems.items():
                    if len(existing_audio.shape) > 1:
                        other_audio -= existing_audio[0] * 0.3  # Partial subtraction to avoid artifacts
                separated_stems[stem] = np.array([other_audio, other_audio])
                log.append(f"Applied residual extraction for {stem}")
        
        return separated_stems, log
    
    def _basic_stem_separation(self, y: np.ndarray, sr: int, stems: List[str]) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Basic stem separation using simple spectral filtering"""
        log = ["Using basic spectral separation"]
        separated_stems = {}
        
        # Simple harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y[0])
        
        for stem in stems:
            if stem == 'vocals':
                separated_stems[stem] = np.array([y_harmonic, y_harmonic])
            elif stem == 'drums':
                separated_stems[stem] = np.array([y_percussive, y_percussive])
            elif stem == 'bass':
                # Simple low-pass filter for bass
                bass_audio = self._apply_low_pass_filter(y[0], sr, cutoff=250)
                separated_stems[stem] = np.array([bass_audio, bass_audio])
            else:
                # Everything else
                other_audio = y[0] - y_harmonic * 0.5 - y_percussive * 0.5
                separated_stems[stem] = np.array([other_audio, other_audio])
        
        return separated_stems, log
    
    def _post_process_stem(self, stem_audio: np.ndarray, stem_name: str, quality: str) -> np.ndarray:
        """Post-process separated stems for better quality"""
        
        # Ensure mono input for processing
        if len(stem_audio.shape) > 1:
            audio = stem_audio[0]
        else:
            audio = stem_audio
        
        # Apply stem-specific post-processing
        if stem_name == 'vocals':
            # Vocal enhancement
            audio = self._enhance_vocals(audio, quality)
        elif stem_name == 'drums':
            # Drum enhancement
            audio = self._enhance_drums(audio, quality)
        elif stem_name == 'bass':
            # Bass enhancement
            audio = self._enhance_bass(audio, quality)
        
        # General cleanup
        if quality in ['high', 'ultra']:
            # Noise gate for cleaner stems
            audio = self._apply_noise_gate(audio, threshold=0.001)
            
            # Gentle limiting to prevent clipping
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio * (0.95 / peak)
        
        return audio
    
    def _calculate_separation_quality(self, original: np.ndarray, separated_stems: Dict[str, np.ndarray], 
                                    requested_stems: List[str]) -> Dict[str, Any]:
        """Calculate quality metrics for stem separation"""
        metrics = {}
        
        # Basic metrics
        metrics['stems_created'] = len(separated_stems)
        metrics['requested_stems'] = len(requested_stems)
        metrics['success_rate'] = len(separated_stems) / len(requested_stems) if requested_stems else 0
        
        # Signal quality metrics
        original_rms = np.sqrt(np.mean(original[0]**2))
        total_separated_rms = 0
        
        for stem_name, stem_audio in separated_stems.items():
            if len(stem_audio.shape) > 1:
                stem_rms = np.sqrt(np.mean(stem_audio[0]**2))
            else:
                stem_rms = np.sqrt(np.mean(stem_audio**2))
            total_separated_rms += stem_rms
            metrics[f'{stem_name}_rms'] = float(stem_rms)
        
        # Energy conservation ratio
        metrics['energy_conservation'] = float(total_separated_rms / original_rms) if original_rms > 0 else 0
        
        return metrics
    
    # Stem-specific enhancement methods
    
    def _isolate_vocals_advanced(self, y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
        """Advanced vocal isolation using multiple techniques"""
        # Harmonic-percussive separation
        y_harmonic, _ = librosa.effects.hpss(y[0], margin=8.0)
        
        # Vocal frequency emphasis (roughly 85Hz - 8kHz for vocals)
        stft = librosa.stft(y_harmonic, n_fft=n_fft, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Create vocal-focused mask
        vocal_mask = np.zeros_like(freqs)
        vocal_range = (freqs >= 85) & (freqs <= 8000)
        vocal_mask[vocal_range] = 1.0
        
        # Apply frequency emphasis
        filtered_stft = stft * vocal_mask[:, np.newaxis]
        vocal_audio = librosa.istft(filtered_stft, hop_length=hop_length)
        
        return vocal_audio
    
    def _isolate_drums_advanced(self, y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
        """Advanced drum isolation with transient enhancement"""
        # Percussive component
        _, y_percussive = librosa.effects.hpss(y[0], margin=8.0)
        
        # Enhance transients (drum hits)
        stft = librosa.stft(y_percussive, n_fft=n_fft, hop_length=hop_length)
        
        # Transient detection and enhancement
        onset_frames = librosa.onset.onset_detect(y=y_percussive, sr=sr, hop_length=hop_length)
        
        # Create transient enhancement mask
        enhanced_stft = stft.copy()
        for onset_frame in onset_frames:
            # Enhance a window around each onset
            start_frame = max(0, onset_frame - 2)
            end_frame = min(stft.shape[1], onset_frame + 3)
            enhanced_stft[:, start_frame:end_frame] *= 1.5
        
        drum_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
        return drum_audio
    
    def _isolate_bass_advanced(self, y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
        """Advanced bass isolation focusing on low frequencies"""
        # Focus on low frequencies (20-250 Hz typically for bass)
        bass_audio = self._apply_low_pass_filter(y[0], sr, cutoff=250)
        
        # Additional harmonic enhancement for bass fundamentals
        stft = librosa.stft(bass_audio, n_fft=n_fft, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Enhance bass harmonics (fundamental + first few harmonics)
        bass_mask = np.ones_like(freqs)
        bass_range = freqs <= 500  # Include some harmonics
        bass_mask[~bass_range] *= 0.1  # Reduce non-bass frequencies
        
        filtered_stft = stft * bass_mask[:, np.newaxis]
        enhanced_bass = librosa.istft(filtered_stft, hop_length=hop_length)
        
        return enhanced_bass
    
    def _isolate_piano_advanced(self, y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
        """Piano-specific harmonic isolation"""
        # Piano covers a wide frequency range but has specific harmonic characteristics
        stft = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)
        
        # Piano frequency range (roughly 27.5 Hz - 4.2 kHz for fundamentals)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        piano_mask = np.zeros_like(freqs)
        piano_range = (freqs >= 25) & (freqs <= 8000)  # Include harmonics
        piano_mask[piano_range] = 1.0
        
        # Emphasize piano-typical frequencies
        filtered_stft = stft * piano_mask[:, np.newaxis]
        piano_audio = librosa.istft(filtered_stft, hop_length=hop_length)
        
        return piano_audio
    
    def _isolate_other_advanced(self, y: np.ndarray, sr: int, separated_stems: Dict[str, np.ndarray]) -> np.ndarray:
        """Isolate remaining components after primary stems"""
        other_audio = y[0].copy()
        
        # Subtract already separated components (with reduced weight to avoid artifacts)
        for stem_name, stem_audio in separated_stems.items():
            if len(stem_audio.shape) > 1:
                other_audio -= stem_audio[0] * 0.4  # Partial subtraction
        
        return other_audio
    
    def _enhance_vocal_separation(self, y_harmonic: np.ndarray, y_full: np.ndarray, sr: int) -> np.ndarray:
        """Enhanced vocal separation using advanced techniques"""
        # Center channel extraction for stereo (vocals often in center)
        if len(y_full.shape) > 1 and y_full.shape[0] == 2:
            # Mid-side processing
            mid = (y_full[0] + y_full[1]) / 2
            side = (y_full[0] - y_full[1]) / 2
            
            # Vocals typically in mid channel
            vocal_audio = mid
        else:
            vocal_audio = y_harmonic
        
        # Apply vocal-specific filtering
        vocal_audio = self._apply_band_pass_filter(vocal_audio, sr, low_freq=80, high_freq=8000)
        
        return vocal_audio
    
    def _enhance_drum_separation(self, y_percussive: np.ndarray, y_full: np.ndarray, sr: int) -> np.ndarray:
        """Enhanced drum separation with transient emphasis"""
        # Stereo width enhancement for drums (often panned)
        if len(y_full.shape) > 1 and y_full.shape[0] == 2:
            # Use side information for drums
            side = (y_full[0] - y_full[1]) / 2
            drum_audio = y_percussive + side * 0.3
        else:
            drum_audio = y_percussive
        
        # Enhance attack transients
        drum_audio = self._enhance_transients(drum_audio, sr)
        
        return drum_audio
    
    def _enhance_bass_separation(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Enhanced bass separation"""
        # Extract low frequencies
        bass_audio = self._apply_low_pass_filter(y[0], sr, cutoff=200)
        
        # Add some low-mid for bass harmonics
        bass_harmonics = self._apply_band_pass_filter(y[0], sr, low_freq=200, high_freq=600)
        bass_audio += bass_harmonics * 0.3
        
        return bass_audio
    
    def _enhance_piano_separation(self, y_harmonic: np.ndarray, y_full: np.ndarray, sr: int) -> np.ndarray:
        """Enhanced piano separation"""
        # Piano-specific frequency shaping
        piano_audio = self._apply_band_pass_filter(y_harmonic, sr, low_freq=25, high_freq=8000)
        
        # Emphasize piano attack characteristics
        piano_audio = self._enhance_transients(piano_audio, sr, attack_emphasis=True)
        
        return piano_audio
    
    def _assign_nmf_components(self, W: np.ndarray, stems: List[str], sr: int) -> Dict[str, List[int]]:
        """Assign NMF components to stems based on frequency characteristics"""
        assignments = {}
        n_components = W.shape[1]
        
        # Analyze each component's frequency characteristics
        component_characteristics = []
        for i in range(n_components):
            component = W[:, i]
            
            # Calculate frequency center of mass
            freqs = librosa.fft_frequencies(sr=sr, n_fft=(len(component)-1)*2)
            freq_center = np.sum(component * freqs) / np.sum(component)
            
            # Calculate spectral spread
            freq_spread = np.sqrt(np.sum(component * (freqs - freq_center)**2) / np.sum(component))
            
            component_characteristics.append({
                'index': i,
                'freq_center': freq_center,
                'freq_spread': freq_spread,
                'energy': np.sum(component)
            })
        
        # Sort components by frequency center
        component_characteristics.sort(key=lambda x: x['freq_center'])
        
        # Assign components to stems
        for stem in stems:
            if stem == 'bass':
                # Low frequency components
                assignments[stem] = [c['index'] for c in component_characteristics[:n_components//4]
                                   if c['freq_center'] < 300]
            elif stem == 'drums':
                # Wide spread, high energy components
                assignments[stem] = [c['index'] for c in component_characteristics
                                   if c['freq_spread'] > np.percentile([x['freq_spread'] for x in component_characteristics], 75)]
            elif stem == 'vocals':
                # Mid-frequency components with moderate spread
                assignments[stem] = [c['index'] for c in component_characteristics
                                   if 200 < c['freq_center'] < 4000 and c['freq_spread'] < np.percentile([x['freq_spread'] for x in component_characteristics], 50)]
            else:  # 'other'
                # Remaining components
                used_indices = set()
                for other_stem, indices in assignments.items():
                    used_indices.update(indices)
                assignments[stem] = [c['index'] for c in component_characteristics
                                   if c['index'] not in used_indices]
        
        return assignments
    
    def _enhance_vocals(self, audio: np.ndarray, quality: str) -> np.ndarray:
        """Vocal-specific enhancement"""
        if quality in ['high', 'ultra']:
            # Gentle high-frequency emphasis for vocal clarity
            audio = self._apply_high_shelf_filter(audio, freq=3000, gain_db=2)
        return audio
    
    def _enhance_drums(self, audio: np.ndarray, quality: str) -> np.ndarray:
        """Drum-specific enhancement"""
        if quality in ['high', 'ultra']:
            # Enhance attack transients
            audio = self._enhance_transients(audio, sr=44100)
        return audio
    
    def _enhance_bass(self, audio: np.ndarray, quality: str) -> np.ndarray:
        """Bass-specific enhancement"""
        if quality in ['high', 'ultra']:
            # Low-frequency emphasis
            audio = self._apply_low_shelf_filter(audio, freq=100, gain_db=1)
        return audio
    
    def _apply_low_pass_filter(self, audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
        """Apply low-pass filter"""
        if HAS_SCIPY:
            nyquist = sr / 2
            normalized_cutoff = cutoff / nyquist
            b, a = scipy.signal.butter(4, normalized_cutoff, btype='low')
            return scipy.signal.filtfilt(b, a, audio)
        else:
            # Simple approximation using librosa
            stft = librosa.stft(audio)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-2)
            mask = freqs <= cutoff
            stft[~mask] *= 0.1
            return librosa.istft(stft)
    
    def _apply_band_pass_filter(self, audio: np.ndarray, sr: int, low_freq: float, high_freq: float) -> np.ndarray:
        """Apply band-pass filter"""
        if HAS_SCIPY:
            nyquist = sr / 2
            low_normalized = low_freq / nyquist
            high_normalized = high_freq / nyquist
            b, a = scipy.signal.butter(4, [low_normalized, high_normalized], btype='band')
            return scipy.signal.filtfilt(b, a, audio)
        else:
            # Simple approximation
            stft = librosa.stft(audio)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-2)
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            stft[~mask] *= 0.1
            return librosa.istft(stft)
    
    def _apply_high_shelf_filter(self, audio: np.ndarray, freq: float, gain_db: float) -> np.ndarray:
        """Apply high-shelf filter (simplified)"""
        # Simplified high-frequency emphasis
        stft = librosa.stft(audio)
        freqs = librosa.fft_frequencies(sr=44100, n_fft=stft.shape[0]*2-2)
        gain_linear = 10**(gain_db/20)
        mask = freqs >= freq
        stft[mask] *= gain_linear
        return librosa.istft(stft)
    
    def _apply_low_shelf_filter(self, audio: np.ndarray, freq: float, gain_db: float) -> np.ndarray:
        """Apply low-shelf filter (simplified)"""
        # Simplified low-frequency emphasis
        stft = librosa.stft(audio)
        freqs = librosa.fft_frequencies(sr=44100, n_fft=stft.shape[0]*2-2)
        gain_linear = 10**(gain_db/20)
        mask = freqs <= freq
        stft[mask] *= gain_linear
        return librosa.istft(stft)
    
    def _enhance_transients(self, audio: np.ndarray, sr: int, attack_emphasis: bool = False) -> np.ndarray:
        """Enhance transient content"""
        # Simple transient enhancement using high-pass filtered signal
        if HAS_SCIPY:
            # High-pass filter to isolate transients
            nyquist = sr / 2
            normalized_cutoff = 1000 / nyquist  # 1kHz high-pass
            b, a = scipy.signal.butter(2, normalized_cutoff, btype='high')
            transients = scipy.signal.filtfilt(b, a, audio)
            
            # Add back transients with emphasis
            enhancement_factor = 1.2 if attack_emphasis else 1.1
            return audio + transients * (enhancement_factor - 1)
        else:
            return audio  # No enhancement without scipy
    
    def _apply_noise_gate(self, audio: np.ndarray, threshold: float) -> np.ndarray:
        """Apply simple noise gate"""
        # Simple noise gate implementation
        gate_mask = np.abs(audio) > threshold
        gated_audio = audio * gate_mask
        
        # Smooth the gating to avoid clicks
        if HAS_SCIPY:
            from scipy import ndimage
            # Smooth the mask
            smoothed_mask = ndimage.gaussian_filter1d(gate_mask.astype(float), sigma=100)
            gated_audio = audio * smoothed_mask
        
        return gated_audio

    def format_conversion_with_metadata(self, input_path: str, output_path: str,
                                      target_format: str, preserve_metadata: bool = True,
                                      quality_settings: Dict[str, Any] = None) -> AudioProcessingResult:
        """
        Convert audio format while preserving metadata
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file
            target_format: Target format ('wav', 'flac', 'mp3', 'aiff')
            preserve_metadata: Whether to preserve metadata tags
            quality_settings: Format-specific quality settings
        """
        start_time = datetime.now()
        
        if quality_settings is None:
            quality_settings = {
                'mp3_bitrate': 320,
                'flac_compression': 5,
                'wav_bit_depth': 24
            }
        
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None, mono=False)
            
            # Extract metadata if preserving
            metadata = {}
            if preserve_metadata:
                try:
                    import mutagen
                    audio_file = mutagen.File(input_path)
                    if audio_file:
                        metadata = dict(audio_file.tags) if audio_file.tags else {}
                except ImportError:
                    print("Warning: mutagen not found. Metadata preservation disabled.")
                except Exception as e:
                    print(f"Warning: Could not extract metadata: {e}")
            
            # Set output format parameters
            output_path = Path(output_path)
            if not output_path.suffix:
                output_path = output_path.with_suffix(f'.{target_format}')
            
            # Convert and save with appropriate settings
            if target_format.lower() == 'mp3':
                # For MP3, we'd need pydub or similar
                print("Note: MP3 encoding requires additional libraries (pydub, ffmpeg)")
                # Fallback to WAV
                target_format = 'wav'
                output_path = output_path.with_suffix('.wav')
            
            if target_format.lower() in ['wav', 'flac', 'aiff']:
                # Use soundfile for lossless formats
                subtype = None
                if target_format.lower() == 'flac':
                    subtype = 'PCM_24'
                elif 'wav_bit_depth' in quality_settings:
                    bit_depth = quality_settings['wav_bit_depth']
                    subtype = f'PCM_{bit_depth}'
                
                if len(y.shape) == 1:
                    sf.write(output_path, y, sr, subtype=subtype)
                else:
                    sf.write(output_path, y.T, sr, subtype=subtype)
            
            # Apply metadata to output file
            if preserve_metadata and metadata:
                try:
                    import mutagen
                    output_file = mutagen.File(str(output_path))
                    if output_file is not None:
                        for key, value in metadata.items():
                            try:
                                output_file[key] = value
                            except:
                                pass
                        output_file.save()
                except:
                    pass
            
            # Calculate quality metrics
            original_size = Path(input_path).stat().st_size
            converted_size = output_path.stat().st_size
            compression_ratio = converted_size / original_size
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AudioProcessingResult(
                success=True,
                message=f"Converted to {target_format.upper()} format",
                output_path=str(output_path),
                processing_time=processing_time,
                parameters_used={
                    'target_format': target_format,
                    'preserve_metadata': preserve_metadata,
                    'quality_settings': quality_settings
                },
                quality_metrics={
                    'compression_ratio': compression_ratio,
                    'metadata_preserved': bool(metadata) if preserve_metadata else False,
                    'original_size_bytes': original_size,
                    'converted_size_bytes': converted_size
                }
            )
            
        except Exception as e:
            return AudioProcessingResult(
                success=False,
                message=f"Format conversion failed: {str(e)}"
            )
    
    def broadcast_standards_compliance(self, input_path: str, output_path: str,
                                     standard: str = "EBU_R128",
                                     auto_correct: bool = True) -> AudioProcessingResult:
        """
        Check and ensure broadcast standards compliance (EBU R128, etc.)
        
        Args:
            input_path: Path to input audio file
            output_path: Path for compliant output file
            standard: Broadcast standard ('EBU_R128', 'ATSC_A85', 'ITU_BS1770')
            auto_correct: Whether to automatically correct non-compliant audio
        """
        start_time = datetime.now()
        
        # Standard specifications
        standards = {
            'EBU_R128': {
                'target_lufs': -23.0,
                'max_true_peak': -1.0,
                'lra_max': 20.0,  # Loudness Range
                'gate_threshold': -70.0
            },
            'ATSC_A85': {
                'target_lufs': -24.0,
                'max_true_peak': -2.0,
                'lra_max': None,
                'gate_threshold': -70.0
            },
            'ITU_BS1770': {
                'target_lufs': -23.0,
                'max_true_peak': -1.0,
                'lra_max': None,
                'gate_threshold': -70.0
            }
        }
        
        if standard not in standards:
            return AudioProcessingResult(
                success=False,
                message=f"Unknown standard: {standard}. Available: {list(standards.keys())}"
            )
        
        std_params = standards[standard]
        
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None, mono=False)
            if len(y.shape) == 1:
                y = np.array([y, y])  # Convert to stereo
            
            compliance_report = {}
            corrections_applied = []
            
            # Measure current loudness
            if HAS_PYLOUDNORM:
                meter = pyln.Meter(sr)
                
                # Integrated loudness
                loudness = meter.integrated_loudness(y.T)
                compliance_report['integrated_loudness'] = loudness
                compliance_report['target_loudness'] = std_params['target_lufs']
                compliance_report['loudness_compliant'] = abs(loudness - std_params['target_lufs']) <= 0.1
                
                # True peak
                true_peak = 20 * np.log10(np.max(np.abs(y)))
                compliance_report['true_peak'] = true_peak
                compliance_report['max_true_peak'] = std_params['max_true_peak']
                compliance_report['peak_compliant'] = true_peak <= std_params['max_true_peak']
                
                # Loudness Range (if supported by standard)
                if std_params['lra_max'] is not None:
                    try:
                        # Simple LRA estimation (would need more sophisticated implementation)
                        short_term_loudness = []
                        window_size = int(3.0 * sr)  # 3 second windows
                        for i in range(0, len(y[0]) - window_size, window_size // 2):
                            window = y[:, i:i+window_size]
                            if window.shape[1] > 0:
                                st_loudness = meter.integrated_loudness(window.T)
                                if st_loudness > -70:  # Above gate threshold
                                    short_term_loudness.append(st_loudness)
                        
                        if short_term_loudness:
                            lra = np.percentile(short_term_loudness, 95) - np.percentile(short_term_loudness, 10)
                            compliance_report['loudness_range'] = lra
                            compliance_report['lra_compliant'] = lra <= std_params['lra_max']
                        else:
                            compliance_report['loudness_range'] = None
                            compliance_report['lra_compliant'] = True
                    except:
                        compliance_report['loudness_range'] = None
                        compliance_report['lra_compliant'] = True
                
                # Auto-correct if requested and not compliant
                y_corrected = y.copy()
                
                if auto_correct:
                    # Correct loudness
                    if not compliance_report['loudness_compliant']:
                        y_corrected = pyln.normalize.loudness(
                            y_corrected.T, loudness, std_params['target_lufs']
                        ).T
                        corrections_applied.append(f"Loudness normalized to {std_params['target_lufs']} LUFS")
                    
                    # Correct true peak
                    current_peak = 20 * np.log10(np.max(np.abs(y_corrected)))
                    if current_peak > std_params['max_true_peak']:
                        peak_reduction = std_params['max_true_peak'] - current_peak
                        gain_factor = 10 ** (peak_reduction / 20)
                        y_corrected *= gain_factor
                        corrections_applied.append(f"Peak limited to {std_params['max_true_peak']} dB")
                
                # Save corrected audio
                sf.write(output_path, y_corrected.T, sr)
                
            else:
                # Fallback without pyloudnorm
                peak_db = 20 * np.log10(np.max(np.abs(y)))
                compliance_report = {
                    'peak_level': peak_db,
                    'max_true_peak': std_params['max_true_peak'],
                    'peak_compliant': peak_db <= std_params['max_true_peak'],
                    'note': 'Limited analysis without pyloudnorm'
                }
                
                # Simple peak limiting if needed
                y_corrected = y.copy()
                if auto_correct and peak_db > std_params['max_true_peak']:
                    peak_reduction = std_params['max_true_peak'] - peak_db
                    gain_factor = 10 ** (peak_reduction / 20)
                    y_corrected *= gain_factor
                    corrections_applied.append(f"Peak limited to {std_params['max_true_peak']} dB")
                
                sf.write(output_path, y_corrected.T, sr)
            
            # Overall compliance
            compliant_checks = [v for k, v in compliance_report.items() if k.endswith('_compliant')]
            overall_compliant = all(compliant_checks) if compliant_checks else False
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AudioProcessingResult(
                success=True,
                message=f"Broadcast compliance check completed for {standard}",
                output_path=output_path,
                processing_time=processing_time,
                parameters_used={
                    'standard': standard,
                    'auto_correct': auto_correct
                },
                quality_metrics={
                    'compliance_report': compliance_report,
                    'overall_compliant': overall_compliant,
                    'corrections_applied': corrections_applied
                }
            )
            
        except Exception as e:
            return AudioProcessingResult(
                success=False,
                message=f"Broadcast compliance check failed: {str(e)}"
            )
    
    def quality_assurance_check(self, input_path: str, 
                              thresholds: Dict[str, float] = None) -> AudioProcessingResult:
        """
        Comprehensive quality assurance - detect clipping, distortion, noise, etc.
        
        Args:
            input_path: Path to input audio file
            thresholds: Custom thresholds for quality metrics
        """
        if thresholds is None:
            thresholds = {
                'clip_threshold': 0.99,      # Digital clipping threshold
                'thd_threshold': 1.0,        # Total Harmonic Distortion threshold (%)
                'snr_minimum': 40.0,         # Minimum SNR in dB
                'dc_offset_max': 0.01,       # Maximum DC offset
                'silence_threshold': -60.0,   # Silence detection threshold (dB)
                'peak_margin': -3.0          # Minimum headroom below 0dB
            }
        
        start_time = datetime.now()
        
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None, mono=False)
            if len(y.shape) == 1:
                y = np.array([y])
            
            qa_report = {
                'file_info': {
                    'duration': len(y[0]) / sr,
                    'sample_rate': sr,
                    'channels': len(y),
                    'bit_depth': 'unknown'  # Would need different loading method to detect
                }
            }
            
            issues_found = []
            warnings = []
            
            for ch_idx, channel in enumerate(y):
                ch_name = f"Channel_{ch_idx + 1}" if len(y) > 1 else "Mono"
                ch_report = {}
                
                # 1. Clipping Detection
                clipped_samples = np.sum(np.abs(channel) >= thresholds['clip_threshold'])
                clip_percentage = (clipped_samples / len(channel)) * 100
                ch_report['clipping'] = {
                    'clipped_samples': int(clipped_samples),
                    'clip_percentage': clip_percentage,
                    'is_clipped': clip_percentage > 0.01  # More than 0.01% clipped
                }
                
                if ch_report['clipping']['is_clipped']:
                    issues_found.append(f"{ch_name}: {clip_percentage:.2f}% clipped samples detected")
                
                # 2. Peak Level Analysis
                peak_level = 20 * np.log10(np.max(np.abs(channel)))
                ch_report['peak_analysis'] = {
                    'peak_level_db': peak_level,
                    'headroom_db': 0 - peak_level,
                    'sufficient_headroom': peak_level <= thresholds['peak_margin']
                }
                
                if not ch_report['peak_analysis']['sufficient_headroom']:
                    warnings.append(f"{ch_name}: Peak level {peak_level:.1f}dB - insufficient headroom")
                
                # 3. DC Offset Detection
                dc_offset = np.mean(channel)
                ch_report['dc_offset'] = {
                    'offset_value': float(dc_offset),
                    'offset_percentage': abs(dc_offset) * 100,
                    'has_dc_offset': abs(dc_offset) > thresholds['dc_offset_max']
                }
                
                if ch_report['dc_offset']['has_dc_offset']:
                    issues_found.append(f"{ch_name}: DC offset detected ({dc_offset:.4f})")
                
                # 4. Silence Detection
                silence_threshold_linear = 10 ** (thresholds['silence_threshold'] / 20)
                silent_samples = np.sum(np.abs(channel) < silence_threshold_linear)
                silence_percentage = (silent_samples / len(channel)) * 100
                ch_report['silence_analysis'] = {
                    'silent_samples': int(silent_samples),
                    'silence_percentage': silence_percentage,
                    'mostly_silent': silence_percentage > 50.0
                }
                
                if ch_report['silence_analysis']['mostly_silent']:
                    warnings.append(f"{ch_name}: {silence_percentage:.1f}% silence detected")
                
                # 5. Dynamic Range Analysis
                rms_level = np.sqrt(np.mean(channel**2))
                rms_db = 20 * np.log10(rms_level) if rms_level > 0 else -np.inf
                crest_factor = peak_level - rms_db if rms_db > -np.inf else np.inf
                
                ch_report['dynamic_range'] = {
                    'rms_level_db': rms_db,
                    'crest_factor_db': crest_factor,
                    'compressed': crest_factor < 6.0  # Less than 6dB crest factor indicates heavy compression
                }
                
                if ch_report['dynamic_range']['compressed']:
                    warnings.append(f"{ch_name}: Low dynamic range - possibly over-compressed")
                
                # 6. Frequency Analysis
                freqs = np.fft.fftfreq(len(channel), 1/sr)
                fft = np.fft.fft(channel)
                magnitude_spectrum = np.abs(fft)
                
                # Check for frequency imbalances
                low_freq_energy = np.sum(magnitude_spectrum[(freqs >= 20) & (freqs <= 200)])
                mid_freq_energy = np.sum(magnitude_spectrum[(freqs >= 200) & (freqs <= 2000)])
                high_freq_energy = np.sum(magnitude_spectrum[(freqs >= 2000) & (freqs <= 20000)])
                
                total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
                if total_energy > 0:
                    freq_balance = {
                        'low_freq_ratio': low_freq_energy / total_energy,
                        'mid_freq_ratio': mid_freq_energy / total_energy,
                        'high_freq_ratio': high_freq_energy / total_energy
                    }
                    
                    ch_report['frequency_balance'] = freq_balance
                    
                    # Check for extreme imbalances
                    if freq_balance['low_freq_ratio'] > 0.7:
                        warnings.append(f"{ch_name}: Excessive low frequency content")
                    elif freq_balance['high_freq_ratio'] < 0.05:
                        warnings.append(f"{ch_name}: Very limited high frequency content")
                
                # 7. Simple Distortion Estimation
                # Using zero-crossing rate as a rough distortion indicator
                zcr = librosa.feature.zero_crossing_rate(channel)[0]
                avg_zcr = np.mean(zcr)
                ch_report['distortion_estimate'] = {
                    'zero_crossing_rate': float(avg_zcr),
                    'possibly_distorted': avg_zcr > 0.15  # High ZCR can indicate distortion
                }
                
                if ch_report['distortion_estimate']['possibly_distorted']:
                    warnings.append(f"{ch_name}: Possible distortion detected (high zero-crossing rate)")
                
                qa_report[ch_name] = ch_report
            
            # Overall assessment
            qa_report['overall_assessment'] = {
                'issues_found': issues_found,
                'warnings': warnings,
                'quality_rating': self._calculate_quality_rating(qa_report),
                'recommendation': self._generate_qa_recommendation(issues_found, warnings)
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AudioProcessingResult(
                success=True,
                message=f"Quality assurance check completed - {len(issues_found)} issues, {len(warnings)} warnings",
                output_path=None,  # No output file for QA check
                processing_time=processing_time,
                parameters_used=thresholds,
                quality_metrics=qa_report
            )
            
        except Exception as e:
            return AudioProcessingResult(
                success=False,
                message=f"Quality assurance check failed: {str(e)}"
            )
    
    def _calculate_quality_rating(self, qa_report: Dict[str, Any]) -> str:
        """Calculate overall quality rating based on QA metrics"""
        issues = qa_report['overall_assessment']['issues_found']
        warnings = qa_report['overall_assessment']['warnings']
        
        if len(issues) == 0 and len(warnings) == 0:
            return "Excellent"
        elif len(issues) == 0 and len(warnings) <= 2:
            return "Good"
        elif len(issues) <= 1 and len(warnings) <= 3:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_qa_recommendation(self, issues: List[str], warnings: List[str]) -> str:
        """Generate recommendation based on QA results"""
        if not issues and not warnings:
            return "Audio quality is excellent. No issues detected."
        
        recommendations = []
        
        if any("clipped" in issue.lower() for issue in issues):
            recommendations.append("Re-master with lower input levels to avoid clipping")
        
        if any("dc offset" in issue.lower() for issue in issues):
            recommendations.append("Apply DC offset removal filter")
        
        if any("headroom" in warning.lower() for warning in warnings):
            recommendations.append("Leave more headroom for mastering (aim for -6dB to -3dB peaks)")
        
        if any("compressed" in warning.lower() for warning in warnings):
            recommendations.append("Consider reducing compression to preserve dynamic range")
        
        if any("frequency" in warning.lower() for warning in warnings):
            recommendations.append("Review EQ settings for better frequency balance")
        
        if any("distortion" in warning.lower() for warning in warnings):
            recommendations.append("Check for overdriven input stages or excessive processing")
        
        if not recommendations:
            recommendations.append("Review and address the detected issues")
        
        return "; ".join(recommendations)

class IntelligentAudioAnalyzer:
    """Advanced audio analysis with multiple algorithms"""
    
    def __init__(self, enable_advanced_processing: bool = True):
        self.tempo_categories = {
            "very_slow": (0, 70),
            "slow": (70, 90),
            "moderate": (90, 120),
            "fast": (120, 160),
            "very_fast": (160, 200),
            "extreme": (200, 300)
        }
        
        # Initialize advanced processing if available
        self.advanced_processor = None
        if enable_advanced_processing and HAS_ADVANCED_PROCESSING:
            try:
                config = ProcessingConfig(
                    use_gpu=True,
                    cpu_workers=None,  # Auto-detect
                    memory_limit_mb=4096,
                    enable_progress_estimation=True
                )
                self.advanced_processor = AdvancedAudioProcessor(config)
                print("✓ Advanced processing initialized")
            except Exception as e:
                print(f"Advanced processing initialization failed: {e}")
        
        # Key mapping for different notation systems
        self.key_mapping = {
            'C': 'C', 'C#': 'C#', 'Db': 'Db', 'D': 'D', 'D#': 'D#', 'Eb': 'Eb',
            'E': 'E', 'F': 'F', 'F#': 'F#', 'Gb': 'Gb', 'G': 'G', 'G#': 'G#', 
            'Ab': 'Ab', 'A': 'A', 'A#': 'A#', 'Bb': 'Bb', 'B': 'B'
        }
        
        # Cache for duplicate detection
        self.fingerprint_cache = {}
        
        # Try to load pre-trained ML models
        self.load_trained_models()
        
    def analyze_audio_file(self, file_path: str) -> AudioAnalysisResult:
        """Perform comprehensive audio analysis on a file"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Initialize result
            result = AudioAnalysisResult(
                filename=os.path.basename(file_path),
                duration=duration,
                sample_rate=sr
            )
            
            # Perform all analysis types
            self._analyze_tempo(y, sr, result)
            self._analyze_key(y, sr, result)
            self._analyze_loudness(y, sr, result)
            self._analyze_fingerprint(file_path, result)
            self._analyze_spectral_features(y, sr, result)
            self._analyze_genre_and_mood(y, sr, result)
            self._analyze_rhythm_characteristics(y, sr, result)
            
            # Content-based search features
            self._analyze_content_search_features(y, sr, result)
            
            # Advanced ML analysis
            try:
                # Create audio embeddings for similarity analysis
                embeddings = self.create_audio_embeddings(y, sr)
                if embeddings is not None:
                    result.audio_embeddings = embeddings
                
                # Use trained ML model if available
                self.classify_with_ml_model(y, sr, result)
                
            except Exception as e:
                result.analysis_errors.append(f"Advanced ML analysis failed: {str(e)}")
            
            return result
            
        except Exception as e:
            # Return minimal result with error
            result = AudioAnalysisResult(
                filename=os.path.basename(file_path),
                duration=0.0,
                sample_rate=0,
                analysis_errors=[f"Failed to analyze file: {str(e)}"]
            )
            return result
    
    def _analyze_tempo(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Analyze BPM and tempo category with improved rhythm detection"""
        try:
            # Method 1: Standard librosa beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Method 2: Onset-based tempo estimation for better rhythm detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > 1:
                # Calculate tempo from onset intervals
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                intervals = np.diff(onset_times)
                if len(intervals) > 0:
                    # Convert intervals to BPM
                    avg_interval = np.median(intervals)
                    if avg_interval > 0:
                        onset_tempo = 60.0 / avg_interval
                        # Use onset tempo if it's in reasonable range and different from beat tempo
                        if 40 <= onset_tempo <= 300 and abs(tempo - onset_tempo) < 40:
                            tempo = (tempo + onset_tempo) / 2
                        elif 40 <= onset_tempo <= 300:
                            tempo = onset_tempo
            
            # Method 3: Autocorrelation-based tempo for rhythmic patterns
            if len(y) > sr * 2:  # At least 2 seconds
                try:
                    # Compute autocorrelation of onset strength
                    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
                    autocorr = np.correlate(onset_strength, onset_strength, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    # Find peaks in autocorrelation
                    hop_length = 512
                    time_per_frame = hop_length / sr
                    min_bpm, max_bpm = 60, 200
                    min_period = 60 / max_bpm / time_per_frame
                    max_period = 60 / min_bpm / time_per_frame
                    
                    if max_period < len(autocorr):
                        period_range = autocorr[int(min_period):int(max_period)]
                        if len(period_range) > 0:
                            peak_idx = np.argmax(period_range) + int(min_period)
                            autocorr_tempo = 60 / (peak_idx * time_per_frame)
                            
                            # Use autocorr tempo if it's reasonable
                            if 60 <= autocorr_tempo <= 200 and abs(tempo - autocorr_tempo) < 30:
                                tempo = (tempo + autocorr_tempo) / 2
                except:
                    pass  # Fall back to standard tempo
            
            result.bpm = float(tempo)
            
            # Alternative method using essentia if available
            if HAS_ESSENTIA and len(y) > sr:  # At least 1 second of audio
                try:
                    # Convert to essentia format
                    audio_essentia = y.astype(np.float32)
                    
                    # BPM detection with essentia
                    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
                    bpm_essentia, _, _, _, _ = rhythm_extractor(audio_essentia)
                    
                    # Use essentia result if it's reasonable and different from librosa
                    if 60 <= bpm_essentia <= 300:
                        # Average the two methods if both are reasonable
                        if abs(result.bpm - bpm_essentia) < 20:
                            result.bpm = (result.bpm + float(bpm_essentia)) / 2
                        else:
                            # Use the one that seems more reasonable
                            if 80 <= bpm_essentia <= 180:  # Common musical range
                                result.bpm = float(bpm_essentia)
                except:
                    pass  # Fall back to librosa result
            
            # Categorize tempo
            result.tempo_category = self._categorize_tempo(result.bpm)
            
        except Exception as e:
            result.analysis_errors.append(f"Tempo analysis failed: {str(e)}")
    
    def _analyze_key(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Analyze musical key and scale"""
        try:
            # Method 1: Chromagram-based key detection
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Find the most prominent pitch class
            key_idx = np.argmax(chroma_mean)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Determine major/minor using harmonic analysis
            # Calculate major and minor profiles
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            # Normalize profiles
            major_profile = major_profile / np.sum(major_profile)
            minor_profile = minor_profile / np.sum(minor_profile)
            
            # Test all possible keys
            major_scores = []
            minor_scores = []
            
            for i in range(12):
                # Rotate chroma to test key
                rotated_chroma = np.roll(chroma_mean, -i)
                rotated_chroma = rotated_chroma / np.sum(rotated_chroma)
                
                # Calculate correlation with profiles
                major_score = np.corrcoef(rotated_chroma, major_profile)[0, 1]
                minor_score = np.corrcoef(rotated_chroma, minor_profile)[0, 1]
                
                major_scores.append(major_score)
                minor_scores.append(minor_score)
            
            # Find best matches
            best_major_idx = np.argmax(major_scores)
            best_minor_idx = np.argmax(minor_scores)
            best_major_score = major_scores[best_major_idx]
            best_minor_score = minor_scores[best_minor_idx]
            
            # Choose major or minor
            if best_major_score > best_minor_score:
                result.key = key_names[best_major_idx]
                result.scale = "major"
                result.key_confidence = float(best_major_score)
            else:
                result.key = key_names[best_minor_idx]
                result.scale = "minor"
                result.key_confidence = float(best_minor_score)
            
            # Alternative method using essentia if available
            if HAS_ESSENTIA:
                try:
                    audio_essentia = y.astype(np.float32)
                    key_extractor = es.KeyExtractor()
                    key_est, scale_est, strength = key_extractor(audio_essentia)
                    
                    # Use essentia result if confidence is higher
                    if strength > result.key_confidence:
                        result.key = key_est
                        result.scale = scale_est
                        result.key_confidence = float(strength)
                except:
                    pass
                    
        except Exception as e:
            result.analysis_errors.append(f"Key analysis failed: {str(e)}")
    
    def _analyze_loudness(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Analyze loudness using LUFS standard"""
        try:
            # Peak analysis
            result.peak_db = float(20 * np.log10(np.max(np.abs(y))))
            
            # LUFS analysis if pyloudnorm is available
            if HAS_PYLOUDNORM:
                try:
                    # Create loudness meter
                    meter = pyln.Meter(sr)
                    
                    # Convert mono to stereo if needed for LUFS
                    if len(y.shape) == 1:
                        y_stereo = np.stack([y, y], axis=0).T
                    else:
                        y_stereo = y
                    
                    # Measure loudness
                    result.lufs_integrated = float(meter.integrated_loudness(y_stereo))
                    
                    # Short-term and momentary measurements (if audio is long enough)
                    if len(y) >= sr * 3:  # At least 3 seconds
                        try:
                            # Window-based analysis for short-term LUFS
                            window_size = int(3 * sr)  # 3-second windows
                            short_term_values = []
                            
                            for i in range(0, len(y) - window_size, window_size // 2):
                                window = y[i:i + window_size]
                                if len(window.shape) == 1:
                                    window_stereo = np.stack([window, window], axis=0).T
                                else:
                                    window_stereo = window
                                    
                                st_lufs = meter.integrated_loudness(window_stereo)
                                if not np.isnan(st_lufs) and not np.isinf(st_lufs):
                                    short_term_values.append(st_lufs)
                            
                            if short_term_values:
                                result.lufs_short_term = float(np.mean(short_term_values))
                        except:
                            pass
                    
                except Exception as e:
                    result.analysis_errors.append(f"LUFS analysis failed: {str(e)}")
            
            # Alternative RMS-based loudness if LUFS not available
            if result.lufs_integrated is None:
                rms = np.sqrt(np.mean(y**2))
                if rms > 0:
                    result.lufs_integrated = float(20 * np.log10(rms))
                    
        except Exception as e:
            result.analysis_errors.append(f"Loudness analysis failed: {str(e)}")
    
    def _analyze_fingerprint(self, file_path: str, result: AudioAnalysisResult):
        """Generate audio fingerprint for duplicate detection"""
        try:
            if HAS_CHROMAPRINT:
                # Use chromaprint for audio fingerprinting
                try:
                    import subprocess
                    
                    # Use fpcalc command line tool if available
                    cmd = ['fpcalc', '-json', file_path]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if proc.returncode == 0:
                        fingerprint_data = json.loads(proc.stdout)
                        result.fingerprint = fingerprint_data.get('fingerprint', '')
                        result.fingerprint_duration = fingerprint_data.get('duration', 0)
                except:
                    pass
            
            # Fallback: create a simple spectral fingerprint
            if result.fingerprint is None:
                try:
                    # Load a short segment for fingerprinting
                    y, sr = librosa.load(file_path, duration=30, sr=22050)  # 30 seconds max
                    
                    # Compute spectral features for fingerprinting
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
                    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                    
                    # Combine features
                    features = np.concatenate([
                        np.mean(chroma, axis=1),
                        np.mean(mfcc, axis=1),
                        np.mean(spectral_contrast, axis=1)
                    ])
                    
                    # Create hash from features
                    feature_bytes = features.tobytes()
                    result.fingerprint = hashlib.md5(feature_bytes).hexdigest()
                    result.fingerprint_duration = float(len(y) / sr)
                    
                except Exception as e:
                    result.analysis_errors.append(f"Fingerprint generation failed: {str(e)}")
                    
        except Exception as e:
            result.analysis_errors.append(f"Fingerprint analysis failed: {str(e)}")
    
    def _analyze_spectral_features(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Analyze spectral characteristics"""
        try:
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            result.spectral_centroid = float(np.mean(spectral_centroids))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            result.spectral_rolloff = float(np.mean(spectral_rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            result.zero_crossing_rate = float(np.mean(zcr))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            result.mfcc_features = [float(x) for x in np.mean(mfccs, axis=1)]
            
        except Exception as e:
            result.analysis_errors.append(f"Spectral analysis failed: {str(e)}")
    
    def _analyze_genre_and_mood(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Advanced ML-based genre, mood, and instrument analysis"""
        try:
            # Basic audio features
            self._extract_basic_features(y, sr, result)
            
            # Advanced ML classifications
            self._classify_content_type(y, sr, result)
            self._detect_instruments(y, sr, result)
            self._classify_genre_ml(y, sr, result)
            self._analyze_mood_ml(y, sr, result)
            
        except Exception as e:
            result.analysis_errors.append(f"ML analysis failed: {str(e)}")
    
    def _extract_basic_features(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Extract basic audio features for ML analysis"""
        try:
            # Energy level (RMS energy)
            rms = librosa.feature.rms(y=y)[0]
            result.energy_level = float(np.mean(rms))
            
            # Danceability (combination of tempo consistency and rhythm strength)
            if result.bpm and result.bpm > 0:
                # Calculate tempo stability
                try:
                    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
                    if len(onset_strength) > 10:
                        onset_diff = np.diff(onset_strength)
                        tempo_stability = 1.0 - (np.std(onset_diff) / (np.mean(np.abs(onset_diff)) + 1e-8))
                        tempo_stability = max(0, min(1, tempo_stability))
                    else:
                        tempo_stability = 0.5
                except:
                    tempo_stability = 0.5
                
                dance_tempo_score = 1.0 if 100 <= result.bpm <= 140 else max(0, 1.0 - abs(result.bpm - 120) / 100)
                result.danceability = float((tempo_stability * 0.4 + result.energy_level * 0.3 + dance_tempo_score * 0.3))
            
            # Valence (musical positivity)
            valence_score = 0.5
            if result.scale == "major":
                valence_score += 0.3
            elif result.scale == "minor":
                valence_score -= 0.2
            
            if result.spectral_centroid:
                brightness_score = min(1.0, result.spectral_centroid / 3000.0)
                valence_score += (brightness_score - 0.5) * 0.2
            
            result.valence = float(max(0, min(1, valence_score)))
            
        except Exception as e:
            result.analysis_errors.append(f"Basic feature extraction failed: {str(e)}")
    
    def _classify_content_type(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Classify audio content as music, speech, noise, etc."""
        try:
            # Extract features for content classification
            features = []
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            features.extend([
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])
            
            # Harmonic vs percussive content
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_ratio = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-8)
            percussive_ratio = np.mean(np.abs(y_percussive)) / (np.mean(np.abs(y)) + 1e-8)
            
            features.extend([harmonic_ratio, percussive_ratio])
            
            # Zero crossing rate variability (speech has more variability)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # Simple rule-based classification
            if harmonic_ratio > 0.7:
                if zcr.std() > 0.05:
                    result.content_category = "speech"
                else:
                    result.content_category = "music"
            elif percussive_ratio > 0.6:
                result.content_category = "drums"
            elif np.mean(spectral_centroid) < 1000:
                result.content_category = "ambient"
            else:
                result.content_category = "music"
                
        except Exception as e:
            result.analysis_errors.append(f"Content classification failed: {str(e)}")
    
    def _detect_instruments(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Detect instruments using spectral and harmonic analysis"""
        try:
            detected_instruments = []
            confidence_scores = {}
            
            # Extract features for instrument detection
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Percussion detection
            percussive_ratio = np.mean(np.abs(y_percussive)) / (np.mean(np.abs(y)) + 1e-8)
            if percussive_ratio > 0.3:
                detected_instruments.append("drums")
                confidence_scores["drums"] = min(1.0, percussive_ratio)
            
            # Harmonic instrument detection
            if np.mean(np.abs(y_harmonic)) > 0.1:
                # Analyze harmonic content
                chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
                spectral_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
                
                avg_centroid = np.mean(spectral_centroid)
                
                # Piano detection (broad harmonic content, moderate brightness)
                if 800 <= avg_centroid <= 2500 and np.std(chroma) > 0.15:
                    detected_instruments.append("piano")
                    confidence_scores["piano"] = 0.7
                
                # Guitar detection (harmonic content with specific spectral shape)
                if 1000 <= avg_centroid <= 3000:
                    # Check for guitar-like harmonic structure
                    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
                    if np.std(tonnetz) > 0.05:
                        detected_instruments.append("guitar")
                        confidence_scores["guitar"] = 0.6
                
                # Violin/strings detection (high frequency content)
                if avg_centroid > 2000:
                    detected_instruments.append("strings")
                    confidence_scores["strings"] = min(1.0, (avg_centroid - 2000) / 2000)
                
                # Bass detection (low frequency harmonic content)
                if avg_centroid < 800:
                    detected_instruments.append("bass")
                    confidence_scores["bass"] = min(1.0, (800 - avg_centroid) / 800)
                
                # Vocal detection (formant-like structure)
                mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
                if np.std(mfcc[1:4]) > 10:  # Formant variation
                    detected_instruments.append("vocals")
                    confidence_scores["vocals"] = 0.6
            
            # Synthesizer detection (electronic characteristics)
            if result.content_category != "speech":
                # Look for electronic/synthetic characteristics
                spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
                if np.mean(spectral_flatness) > 0.5:
                    detected_instruments.append("synthesizer")
                    confidence_scores["synthesizer"] = np.mean(spectral_flatness)
            
            result.detected_instruments = detected_instruments if detected_instruments else None
            result.instrument_confidence = confidence_scores if confidence_scores else None
            
        except Exception as e:
            result.analysis_errors.append(f"Instrument detection failed: {str(e)}")
    
    def _classify_genre_ml(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """ML-based genre classification using audio features"""
        try:
            # Extract comprehensive features for genre classification
            features = self._extract_genre_features(y, sr, result)
            
            # Rule-based genre classification (can be replaced with trained ML model)
            genre_scores = {}
            
            # Electronic/EDM detection
            if (result.bpm and result.bpm >= 120 and 
                result.energy_level and result.energy_level > 0.4 and
                "synthesizer" in (result.detected_instruments or [])):
                
                if result.bpm >= 140:
                    genre_scores["techno"] = 0.8
                    genre_scores["trance"] = 0.7
                elif result.bpm >= 128:
                    genre_scores["house"] = 0.8
                    genre_scores["dance"] = 0.9
                else:
                    genre_scores["electronic"] = 0.7
            
            # Rock/Metal detection
            if ("guitar" in (result.detected_instruments or []) and
                "drums" in (result.detected_instruments or []) and
                result.energy_level and result.energy_level > 0.5):
                
                if result.energy_level > 0.8:
                    genre_scores["metal"] = 0.8
                    genre_scores["hard rock"] = 0.7
                else:
                    genre_scores["rock"] = 0.8
                    genre_scores["alternative"] = 0.6
            
            # Hip-hop detection
            if (result.bpm and 80 <= result.bpm <= 110 and
                "drums" in (result.detected_instruments or []) and
                result.rhythmic_complexity and result.rhythmic_complexity > 0.3):
                genre_scores["hip-hop"] = 0.8
                if "vocals" in (result.detected_instruments or []):
                    genre_scores["rap"] = 0.7
            
            # Jazz detection
            if ("piano" in (result.detected_instruments or []) and
                result.rhythmic_complexity and result.rhythmic_complexity > 0.4 and
                result.key_confidence and result.key_confidence < 0.7):  # Complex harmony
                genre_scores["jazz"] = 0.7
                
            # Classical detection
            if ("strings" in (result.detected_instruments or []) and
                result.scale == "major" and
                result.energy_level and result.energy_level < 0.6 and
                not any(instr in (result.detected_instruments or []) for instr in ["drums", "guitar"])):
                genre_scores["classical"] = 0.8
                genre_scores["orchestral"] = 0.7
            
            # Pop detection
            if ("vocals" in (result.detected_instruments or []) and
                result.bpm and 100 <= result.bpm <= 130 and
                result.valence and result.valence > 0.5):
                genre_scores["pop"] = 0.7
            
            # Ambient detection
            if (result.energy_level and result.energy_level < 0.3 and
                result.bpm and result.bpm < 90 and
                "synthesizer" in (result.detected_instruments or [])):
                genre_scores["ambient"] = 0.8
                genre_scores["chillout"] = 0.7
            
            # Country detection
            if ("guitar" in (result.detected_instruments or []) and
                "vocals" in (result.detected_instruments or []) and
                result.scale == "major" and
                result.bpm and 80 <= result.bpm <= 140):
                genre_scores["country"] = 0.6
            
            # Select best genre match
            if genre_scores:
                best_genre = max(genre_scores.items(), key=lambda x: x[1])
                result.genre_prediction = best_genre[0]
                result.genre_confidence = best_genre[1]
            
        except Exception as e:
            result.analysis_errors.append(f"Genre classification failed: {str(e)}")
    
    def _extract_genre_features(self, y: np.ndarray, sr: int, result: AudioAnalysisResult) -> list:
        """Extract comprehensive features for genre classification"""
        features = []
        
        try:
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            
            features.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth, spectral_flatness])
            
            # MFCC features (first 13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
            
            # Rhythm features
            features.extend([
                result.bpm or 0,
                result.energy_level or 0,
                result.rhythmic_complexity or 0,
                result.beat_strength or 0
            ])
            
            # Harmonic features
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_ratio = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-8)
            percussive_ratio = np.mean(np.abs(y_percussive)) / (np.mean(np.abs(y)) + 1e-8)
            
            features.extend([harmonic_ratio, percussive_ratio])
            
        except Exception as e:
            # Return zeros if feature extraction fails
            features = [0.0] * 20
            
        return features
    
    def _analyze_mood_ml(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Advanced mood detection using multiple audio features"""
        try:
            # Multi-dimensional mood analysis
            mood_scores = {}
            
            # Energy-based moods
            if result.energy_level:
                if result.energy_level > 0.7:
                    mood_scores["energetic"] = result.energy_level
                    mood_scores["excited"] = result.energy_level * 0.8
                elif result.energy_level < 0.3:
                    mood_scores["calm"] = 1.0 - result.energy_level
                    mood_scores["relaxed"] = (1.0 - result.energy_level) * 0.9
            
            # Valence-based moods
            if result.valence:
                if result.valence > 0.6:
                    mood_scores["happy"] = result.valence
                    mood_scores["uplifting"] = result.valence * 0.8
                elif result.valence < 0.4:
                    mood_scores["sad"] = 1.0 - result.valence
                    mood_scores["melancholic"] = (1.0 - result.valence) * 0.9
            
            # Key-based mood influence
            if result.scale == "minor":
                mood_scores["melancholic"] = mood_scores.get("melancholic", 0) + 0.3
                mood_scores["dramatic"] = 0.6
            elif result.scale == "major":
                mood_scores["happy"] = mood_scores.get("happy", 0) + 0.2
                mood_scores["uplifting"] = mood_scores.get("uplifting", 0) + 0.2
            
            # Tempo-based mood influence
            if result.bpm:
                if result.bpm > 140:
                    mood_scores["energetic"] = mood_scores.get("energetic", 0) + 0.3
                    mood_scores["intense"] = 0.7
                elif result.bpm < 70:
                    mood_scores["peaceful"] = 0.8
                    mood_scores["meditative"] = 0.7
            
            # Instrument-based mood influence
            if result.detected_instruments:
                if "strings" in result.detected_instruments:
                    mood_scores["emotional"] = 0.7
                    mood_scores["dramatic"] = mood_scores.get("dramatic", 0) + 0.2
                if "piano" in result.detected_instruments:
                    mood_scores["contemplative"] = 0.6
                if "synthesizer" in result.detected_instruments:
                    mood_scores["futuristic"] = 0.6
                    mood_scores["atmospheric"] = 0.5
            
            # Complex mood combinations
            if (result.energy_level and result.energy_level > 0.6 and 
                result.valence and result.valence < 0.4):
                mood_scores["aggressive"] = 0.8
                mood_scores["intense"] = mood_scores.get("intense", 0) + 0.2
            
            if (result.energy_level and result.energy_level < 0.4 and
                result.valence and result.valence > 0.5):
                mood_scores["peaceful"] = mood_scores.get("peaceful", 0) + 0.3
                mood_scores["serene"] = 0.7
            
            # Select dominant mood
            if mood_scores:
                best_mood = max(mood_scores.items(), key=lambda x: x[1])
                result.mood_prediction = best_mood[0]
                result.mood_confidence = best_mood[1]
            else:
                result.mood_prediction = "neutral"
                result.mood_confidence = 0.5
            
        except Exception as e:
            result.analysis_errors.append(f"Mood analysis failed: {str(e)}")
    
    def _analyze_rhythm_characteristics(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Analyze rhythm and beat characteristics"""
        try:
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # Onset density (onsets per second)
            if len(y) > 0:
                duration = len(y) / sr
                result.onset_density = float(len(onset_times) / duration)
            
            # Beat strength
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            result.beat_strength = float(np.mean(onset_strength))
            
            # Rhythmic complexity (variability in onset intervals)
            if len(onset_times) > 2:
                intervals = np.diff(onset_times)
                if len(intervals) > 0 and np.mean(intervals) > 0:
                    complexity = np.std(intervals) / np.mean(intervals)
                    result.rhythmic_complexity = float(complexity)
            
        except Exception as e:
            result.analysis_errors.append(f"Rhythm analysis failed: {str(e)}")
    
    def _analyze_content_search_features(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Extract features for content-based search functionality"""
        try:
            # Chroma features for harmonic matching
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            result.chroma_features = np.mean(chroma, axis=1)
            
            # Rhythm features (tempo histogram and beat patterns)
            if len(y) > sr * 2:  # At least 2 seconds
                # Create tempo histogram from windowed analysis
                window_size = sr * 4  # 4-second windows
                hop_size = sr * 2     # 2-second hops
                tempos = []
                
                for start in range(0, len(y) - window_size, hop_size):
                    window = y[start:start + window_size]
                    try:
                        tempo, _ = librosa.beat.beat_track(y=window, sr=sr)
                        tempos.append(tempo)
                    except:
                        continue
                
                if tempos:
                    hist, _ = np.histogram(tempos, bins=10, range=(60, 200))
                    result.rhythm_features = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            
            # Spectral features for frequency content search
            # Compute average spectrum
            D = np.abs(librosa.stft(y))
            avg_spectrum = np.mean(D, axis=1)
            
            # Reduce dimensionality and normalize
            reduced_spectrum = avg_spectrum[::4]  # Take every 4th bin
            result.spectral_features = reduced_spectrum / np.sum(reduced_spectrum) if np.sum(reduced_spectrum) > 0 else reduced_spectrum
            
            # Create similarity hash using chromaprint if available
            if HAS_CHROMAPRINT:
                try:
                    y_int = (y * 32767).astype(np.int16)
                    fingerprint = chromaprint.fingerprint(y_int, sr)
                    result.similarity_hash = fingerprint[1] if fingerprint[0] else ""
                except:
                    pass
            
            # Fallback: create hash from MFCC features
            if not result.similarity_hash:
                try:
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
                    mfcc_hash = hashlib.md5(mfcc.tobytes()).hexdigest()
                    result.similarity_hash = mfcc_hash
                except:
                    pass
                    
        except Exception as e:
            result.analysis_errors.append(f"Content search features extraction failed: {str(e)}")
    
    def _categorize_tempo(self, bpm: float) -> str:
        """Categorize BPM into tempo categories"""
        if bpm is None:
            return "unknown"
            
        for category, (min_bpm, max_bpm) in self.tempo_categories.items():
            if min_bpm <= bpm < max_bpm:
                return category
        
        if bpm >= 300:
            return "extreme"
        else:
            return "unknown"
    
    def find_duplicates(self, analysis_results: List[AudioAnalysisResult], 
                       similarity_threshold: float = 0.95) -> List[List[str]]:
        """Find potential duplicate files based on fingerprints"""
        duplicates = []
        fingerprint_groups = defaultdict(list)
        
        # Group by fingerprint
        for result in analysis_results:
            if result.fingerprint:
                fingerprint_groups[result.fingerprint].append(result.filename)
        
        # Find groups with multiple files
        for fingerprint, filenames in fingerprint_groups.items():
            if len(filenames) > 1:
                duplicates.append(filenames)
        
        return duplicates
    
    def categorize_by_analysis(self, result: AudioAnalysisResult) -> str:
        """Suggest category based on analysis results with enhanced logic"""
        categories = []
        
        # Use genre prediction if available
        if result.genre_prediction:
            categories.append(result.genre_prediction)
        
        # BPM-based categorization
        if result.bpm:
            if result.bpm < 80:
                categories.extend(["ambient", "downtempo", "ballad", "slow"])
            elif result.bpm < 100:
                categories.extend(["hip-hop", "r&b", "chill", "moderate"])
            elif result.bpm < 120:
                categories.extend(["pop", "rock", "indie", "medium"])
            elif result.bpm < 140:
                categories.extend(["dance", "house", "upbeat", "energetic"])
            elif result.bpm < 160:
                categories.extend(["electronic", "techno", "fast", "driving"])
            else:
                categories.extend(["hardcore", "drum-and-bass", "extreme", "intense"])
        
        # Energy and mood-based suggestions
        if result.energy_level:
            if result.energy_level > 0.7:
                categories.extend(["high-energy", "powerful", "intense"])
            elif result.energy_level < 0.3:
                categories.extend(["calm", "relaxed", "quiet"])
        
        if result.mood_prediction:
            categories.append(result.mood_prediction)
        
        # Key and scale-based suggestions
        if result.scale == "minor":
            categories.extend(["emotional", "dark", "melancholic", "moody"])
        elif result.scale == "major":
            categories.extend(["uplifting", "bright", "happy", "positive"])
        
        # Danceability
        if result.danceability and result.danceability > 0.7:
            categories.extend(["danceable", "groove", "rhythmic"])
        
        # Loudness-based suggestions
        if result.lufs_integrated:
            if result.lufs_integrated > -10:
                categories.extend(["loud", "compressed", "commercial"])
            elif result.lufs_integrated < -25:
                categories.extend(["quiet", "dynamic", "natural"])
            elif -14 <= result.lufs_integrated <= -10:
                categories.extend(["broadcast", "mastered"])
        
        # Spectral characteristics
        if result.spectral_centroid:
            if result.spectral_centroid > 3000:
                categories.extend(["bright", "treble", "crisp"])
            elif result.spectral_centroid < 1000:
                categories.extend(["dark", "bass", "warm"])
        
        # Rhythmic characteristics
        if result.rhythmic_complexity and result.rhythmic_complexity > 0.5:
            categories.extend(["complex", "polyrhythmic", "intricate"])
        elif result.rhythmic_complexity and result.rhythmic_complexity < 0.2:
            categories.extend(["simple", "steady", "minimal"])
        
        if result.onset_density:
            if result.onset_density > 5:
                categories.extend(["busy", "dense", "active"])
            elif result.onset_density < 1:
                categories.extend(["sparse", "minimal", "space"])
        
        # Return most likely category with priority system
        if categories:
            from collections import Counter
            category_counts = Counter(categories)
            
            # Priority order for tie-breaking
            priority_categories = [
                "techno", "house", "electronic", "hip-hop", "r&b", "dance",
                "ambient", "downtempo", "pop", "rock", "classical", "acoustic",
                "energetic", "peaceful", "aggressive", "melancholic",
                "danceable", "bright", "dark", "loud", "quiet"
            ]
            
            # Get the most common categories
            most_common = category_counts.most_common()
            max_count = most_common[0][1]
            top_categories = [cat for cat, count in most_common if count == max_count]
            
            # If there's a tie, use priority order
            for priority_cat in priority_categories:
                if priority_cat in top_categories:
                    return priority_cat
            
            # Otherwise, return the first most common
            return most_common[0][0]
        
        return "uncategorized"
    
    def export_analysis_report(self, analysis_results: List[AudioAnalysisResult], 
                              output_file: str):
        """Export analysis results to JSON report"""
        try:
            report_data = {
                "analysis_timestamp": str(datetime.now().isoformat()),
                "total_files": len(analysis_results),
                "analysis_summary": self._generate_summary(analysis_results),
                "files": []
            }
            
            for result in analysis_results:
                file_data = {
                    "filename": result.filename,
                    "duration": result.duration,
                    "sample_rate": result.sample_rate,
                    "bpm": result.bpm,
                    "tempo_category": result.tempo_category,
                    "key": result.key,
                    "scale": result.scale,
                    "key_confidence": result.key_confidence,
                    "lufs_integrated": result.lufs_integrated,
                    "lufs_short_term": result.lufs_short_term,
                    "peak_db": result.peak_db,
                    "spectral_centroid": result.spectral_centroid,
                    "spectral_rolloff": result.spectral_rolloff,
                    "zero_crossing_rate": result.zero_crossing_rate,
                    "genre_prediction": result.genre_prediction,
                    "mood_prediction": result.mood_prediction,
                    "energy_level": result.energy_level,
                    "danceability": result.danceability,
                    "valence": result.valence,
                    "rhythmic_complexity": result.rhythmic_complexity,
                    "beat_strength": result.beat_strength,
                    "onset_density": result.onset_density,
                    "suggested_category": self.categorize_by_analysis(result),
                    "fingerprint": result.fingerprint,
                    "analysis_errors": result.analysis_errors
                }
                report_data["files"].append(file_data)
            
            # Find duplicates
            duplicates = self.find_duplicates(analysis_results)
            report_data["duplicates"] = duplicates
            
            # Write report
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Failed to export analysis report: {e}")
            return False
    
    def _generate_summary(self, analysis_results: List[AudioAnalysisResult]) -> Dict[str, Any]:
        """Generate summary statistics from analysis results"""
        summary = {
            "total_duration": 0,
            "avg_bpm": 0,
            "tempo_distribution": {},
            "key_distribution": {},
            "scale_distribution": {"major": 0, "minor": 0},
            "avg_loudness": 0,
            "loudness_range": {"min": float('inf'), "max": float('-inf')},
        }
        
        valid_bpms = []
        valid_loudness = []
        tempo_counts = defaultdict(int)
        key_counts = defaultdict(int)
        
        for result in analysis_results:
            summary["total_duration"] += result.duration
            
            if result.bpm:
                valid_bpms.append(result.bpm)
            
            if result.tempo_category:
                tempo_counts[result.tempo_category] += 1
            
            if result.key:
                key_counts[result.key] += 1
            
            if result.scale:
                summary["scale_distribution"][result.scale] = summary["scale_distribution"].get(result.scale, 0) + 1
            
            if result.lufs_integrated:
                valid_loudness.append(result.lufs_integrated)
                summary["loudness_range"]["min"] = min(summary["loudness_range"]["min"], result.lufs_integrated)
                summary["loudness_range"]["max"] = max(summary["loudness_range"]["max"], result.lufs_integrated)
        
        # Calculate averages
        if valid_bpms:
            summary["avg_bpm"] = sum(valid_bpms) / len(valid_bpms)
        
        if valid_loudness:
            summary["avg_loudness"] = sum(valid_loudness) / len(valid_loudness)
        
        summary["tempo_distribution"] = dict(tempo_counts)
        summary["key_distribution"] = dict(key_counts)
        
        return summary

    def batch_analyze_library_advanced(self, directory: str, 
                                       file_extensions: List[str] = None,
                                       max_files: int = None,
                                       progress_callback: Callable = None,
                                       use_gpu: bool = True,
                                       max_workers: int = None) -> List[AudioAnalysisResult]:
        """
        Advanced batch analysis with GPU acceleration and multi-core optimization
        
        Args:
            directory: Directory to analyze
            file_extensions: File extensions to process
            max_files: Maximum number of files (for testing)
            progress_callback: Callback for progress updates
            use_gpu: Enable GPU acceleration
            max_workers: Number of worker processes
            
        Returns:
            List of analysis results
        """
        if not HAS_ADVANCED_PROCESSING or not self.advanced_processor:
            print("Advanced processing not available. Using standard batch analysis...")
            return self.batch_analyze_optimized(
                self._find_audio_files(directory, file_extensions, max_files),
                progress_callback
            )
        
        print("\n" + "="*60)
        print("ADVANCED BATCH ANALYSIS")
        print("="*60)
        
        # Use the advanced processor
        results = self.advanced_processor.batch_analyze_library(
            directory=directory,
            file_extensions=file_extensions,
            analysis_func=self.analyze_audio_file_optimized,
            max_files=max_files
        )
        
        # Convert results to AudioAnalysisResult objects
        analysis_results = []
        for file_path, result in results:
            if isinstance(result, AudioAnalysisResult):
                analysis_results.append(result)
            elif isinstance(result, str) and result.startswith("Error:"):
                # Create error result
                error_result = AudioAnalysisResult(
                    filename=os.path.basename(file_path),
                    duration=0.0,
                    sample_rate=0,
                    analysis_errors=[result]
                )
                analysis_results.append(error_result)
        
        print(f"\nAdvanced batch analysis complete: {len(analysis_results)} files processed")
        return analysis_results
    
    def analyze_audio_file_optimized(self, file_path: str) -> AudioAnalysisResult:
        """
        Optimized audio analysis using advanced processing features
        """
        if not HAS_ADVANCED_PROCESSING or not self.advanced_processor:
            return self.analyze_audio_file(file_path)
        
        try:
            # Check if file is very large and needs chunk processing
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            if file_size_mb > 100:  # Files larger than 100MB
                print(f"Processing large file ({file_size_mb:.1f}MB): {os.path.basename(file_path)}")
                return self._analyze_large_file(file_path)
            else:
                # Standard analysis with memory optimization
                return self._analyze_with_memory_optimization(file_path)
                
        except Exception as e:
            # Fallback to standard analysis
            print(f"Advanced analysis failed for {file_path}, using standard method: {e}")
            return self.analyze_audio_file(file_path)
    
    def _analyze_large_file(self, file_path: str) -> AudioAnalysisResult:
        """Analyze very large audio files using chunk processing"""
        
        def chunk_analysis_func(y: np.ndarray, sr: int) -> Dict[str, Any]:
            """Analysis function for audio chunks"""
            chunk_features = {}
            
            # Basic features
            if len(y) > 0:
                chunk_features['rms'] = float(np.sqrt(np.mean(y**2)))
                chunk_features['spectral_centroid'] = float(np.mean(
                    librosa.feature.spectral_centroid(y=y, sr=sr)
                ))
                chunk_features['zero_crossing_rate'] = float(np.mean(
                    librosa.feature.zero_crossing_rate(y)
                ))
            
            # Tempo analysis
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                chunk_features['tempo'] = float(tempo)
            except:
                chunk_features['tempo'] = 0.0
            
            return chunk_features
        
        # Process file in chunks
        chunk_results = self.advanced_processor.process_large_audio_file(
            file_path, 
            chunk_analysis_func,
            chunk_duration=30.0
        )
        
        # Combine chunk results
        if chunk_results and len(chunk_results) > 0:
            # Average the features
            combined_features = {}
            for key in chunk_results[0].keys():
                values = [chunk[key] for chunk in chunk_results if key in chunk and chunk[key] is not None]
                if values:
                    combined_features[key] = np.mean(values)
            
            # Create result
            result = AudioAnalysisResult(
                filename=os.path.basename(file_path),
                duration=len(chunk_results) * 30.0,  # Approximate
                sample_rate=22050  # Standard
            )
            
            # Fill in the averaged features
            result.bpm = combined_features.get('tempo', None)
            result.energy_level = combined_features.get('rms', None)
            result.spectral_centroid = combined_features.get('spectral_centroid', None)
            result.zero_crossing_rate = combined_features.get('zero_crossing_rate', None)
            result.tempo_category = self._categorize_tempo(result.bpm) if result.bpm else None
            
            return result
        
        # Fallback if chunk processing fails
        return self.analyze_audio_file(file_path)
    
    def _analyze_with_memory_optimization(self, file_path: str) -> AudioAnalysisResult:
        """Analyze file with memory optimization"""
        
        # Check memory before loading
        memory_optimizer = self.advanced_processor.memory_optimizer
        estimated_memory = memory_optimizer.estimate_audio_memory(file_path)
        
        if not memory_optimizer.is_memory_available(estimated_memory * 2):  # 2x safety factor
            # Force memory cleanup
            memory_optimizer.cleanup_memory()
            
            # If still not enough memory, use memory mapping
            if not memory_optimizer.is_memory_available(estimated_memory):
                print(f"Using memory mapping for {os.path.basename(file_path)}")
                return self._analyze_with_memory_mapping(file_path)
        
        # Standard analysis
        return self.analyze_audio_file(file_path)
    
    def _analyze_with_memory_mapping(self, file_path: str) -> AudioAnalysisResult:
        """Analyze file using memory mapping"""
        try:
            # Create memory-mapped array
            memory_optimizer = self.advanced_processor.memory_optimizer
            y_mapped = memory_optimizer.create_memory_mapped_audio(file_path)
            
            if y_mapped is not None:
                # Analyze subset of the file to reduce memory usage
                sample_length = min(len(y_mapped), 22050 * 60)  # Max 1 minute
                y_sample = np.array(y_mapped[:sample_length])
                
                # Get file info for sample rate
                info = sf.info(file_path)
                sr = info.samplerate
                
                # Perform analysis on sample
                result = AudioAnalysisResult(
                    filename=os.path.basename(file_path),
                    duration=info.duration,
                    sample_rate=sr
                )
                
                # Basic analysis on sample
                self._analyze_tempo(y_sample, sr, result)
                self._analyze_spectral_features(y_sample, sr, result)
                self._analyze_loudness(y_sample, sr, result)
                
                return result
        
        except Exception as e:
            print(f"Memory mapping failed for {file_path}: {e}")
        
        # Fallback
        return self.analyze_audio_file(file_path)
    
    def _find_audio_files(self, directory: str, file_extensions: List[str] = None, 
                         max_files: int = None) -> List[str]:
        """Find audio files in directory"""
        if file_extensions is None:
            file_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff']
        
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(Path(directory).rglob(f'*{ext}'))
            audio_files.extend(Path(directory).rglob(f'*{ext.upper()}'))
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        return [str(f) for f in audio_files]

    def gpu_accelerated_batch_spectral_analysis(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        GPU-accelerated batch spectral analysis
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            List of spectral analysis results
        """
        if not HAS_ADVANCED_PROCESSING or not self.advanced_processor:
            print("GPU acceleration not available")
            return []
        
        print(f"GPU-accelerated spectral analysis of {len(file_paths)} files...")
        
        # Use GPU batch processing
        spectrograms = self.advanced_processor.optimize_audio_batch_gpu(
            file_paths, 
            operation="spectral_analysis"
        )
        
        results = []
        for i, (file_path, spectrogram) in enumerate(zip(file_paths, spectrograms)):
            try:
                # Extract spectral features from GPU-computed spectrogram
                spectral_features = {
                    'filename': os.path.basename(file_path),
                    'spectral_centroid': float(np.mean(np.sum(spectrogram * np.arange(spectrogram.shape[0])[:, None], axis=0) / np.sum(spectrogram, axis=0))),
                    'spectral_rolloff': float(np.mean(np.percentile(spectrogram, 85, axis=0))),
                    'spectral_bandwidth': float(np.mean(np.sqrt(np.sum(((np.arange(spectrogram.shape[0])[:, None] - np.sum(spectrogram * np.arange(spectrogram.shape[0])[:, None], axis=0) / np.sum(spectrogram, axis=0))**2) * spectrogram, axis=0) / np.sum(spectrogram, axis=0)))),
                    'spectral_contrast': float(np.mean(np.max(spectrogram, axis=0) - np.min(spectrogram, axis=0))),
                    'energy': float(np.mean(np.sum(spectrogram**2, axis=0)))
                }
                results.append(spectral_features)
                
            except Exception as e:
                results.append({
                    'filename': os.path.basename(file_path),
                    'error': str(e)
                })
        
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get advanced processing statistics"""
        if not HAS_ADVANCED_PROCESSING or not self.advanced_processor:
            return {"advanced_processing": False}
        
        return self.advanced_processor.get_processing_stats()
        """Optimized batch analysis for large libraries"""
        results = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            try:
                # Load with duration limit for efficiency
                y, sr = librosa.load(file_path, duration=max_duration, sr=22050)
                
                # Create result object
                result = AudioAnalysisResult(
                    filename=os.path.basename(file_path),
                    duration=librosa.get_duration(filename=file_path),  # Get full duration
                    sample_rate=sr
                )
                
                # Perform optimized analysis
                self._analyze_tempo_fast(y, sr, result)
                self._analyze_key_fast(y, sr, result) 
                self._analyze_loudness_fast(y, sr, result)
                self._analyze_spectral_features_fast(y, sr, result)
                
                results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, total_files, result.filename)
                    
            except Exception as e:
                # Create error result
                result = AudioAnalysisResult(
                    filename=os.path.basename(file_path),
                    duration=0.0,
                    sample_rate=0,
                    analysis_errors=[f"Batch analysis failed: {str(e)}"]
                )
                results.append(result)
        
        return results
    
    def _analyze_tempo_fast(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Fast tempo analysis for batch processing"""
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            result.bpm = float(tempo)
            result.tempo_category = self._categorize_tempo(result.bpm)
        except:
            pass
    
    def _analyze_key_fast(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Fast key analysis for batch processing"""
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            key_idx = np.argmax(chroma_mean)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            result.key = key_names[key_idx]
            result.scale = "major"  # Simplified for speed
            result.key_confidence = float(chroma_mean[key_idx] / np.sum(chroma_mean))
        except:
            pass
    
    def _analyze_loudness_fast(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Fast loudness analysis for batch processing"""
        try:
            # Peak analysis
            result.peak_db = float(20 * np.log10(np.max(np.abs(y))))
            # Simple RMS-based loudness
            rms = np.sqrt(np.mean(y**2))
            if rms > 0:
                result.lufs_integrated = float(20 * np.log10(rms))
        except:
            pass
    
    def _analyze_spectral_features_fast(self, y: np.ndarray, sr: int, result: AudioAnalysisResult):
        """Fast spectral analysis for batch processing"""
        try:
            result.spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            result.zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            result.energy_level = float(np.mean(librosa.feature.rms(y=y)))
        except:
            pass
    
    def find_similar_tracks(self, analysis_results: List[AudioAnalysisResult], 
                           target_result: AudioAnalysisResult,
                           similarity_threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find tracks similar to a target track based on multiple audio features"""
        similar_tracks = []
        
        for result in analysis_results:
            if result.filename == target_result.filename:
                continue  # Skip the target itself
            
            similarity_score = self._calculate_similarity(target_result, result)
            if similarity_score >= similarity_threshold:
                similar_tracks.append((result.filename, similarity_score))
        
        # Sort by similarity (highest first)
        similar_tracks.sort(key=lambda x: x[1], reverse=True)
        return similar_tracks
    
    def _calculate_similarity(self, track1: AudioAnalysisResult, track2: AudioAnalysisResult) -> float:
        """Calculate similarity score between two tracks"""
        scores = []
        weights = []
        
        # BPM similarity
        if track1.bpm and track2.bpm:
            bpm_diff = abs(track1.bpm - track2.bpm)
            bpm_score = max(0, 1 - bpm_diff / 50.0)  # 50 BPM = 0 similarity
            scores.append(bpm_score)
            weights.append(0.25)
        
        # Key similarity
        if track1.key and track2.key:
            key_score = 1.0 if track1.key == track2.key else 0.3  # Same key = high, different = low
            if track1.scale and track2.scale:
                if track1.scale == track2.scale:
                    key_score *= 1.0
                else:
                    key_score *= 0.7  # Different scale reduces similarity
            scores.append(key_score)
            weights.append(0.2)
        
        # Energy similarity
        if track1.energy_level is not None and track2.energy_level is not None:
            energy_diff = abs(track1.energy_level - track2.energy_level)
            energy_score = max(0, 1 - energy_diff)
            scores.append(energy_score)
            weights.append(0.15)
        
        # Spectral similarity
        if track1.spectral_centroid and track2.spectral_centroid:
            spectral_diff = abs(track1.spectral_centroid - track2.spectral_centroid)
            spectral_score = max(0, 1 - spectral_diff / 5000.0)  # 5kHz = 0 similarity
            scores.append(spectral_score)
            weights.append(0.15)
        
        # Genre similarity
        if track1.genre_prediction and track2.genre_prediction:
            genre_score = 1.0 if track1.genre_prediction == track2.genre_prediction else 0.2
            scores.append(genre_score)
            weights.append(0.15)
        
        # Mood similarity
        if track1.mood_prediction and track2.mood_prediction:
            mood_score = 1.0 if track1.mood_prediction == track2.mood_prediction else 0.3
            scores.append(mood_score)
            weights.append(0.1)
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return weighted_score
        
        return 0.0
    
    def train_genre_classifier(self, training_data: List[Tuple[str, str]]) -> bool:
        """Train a genre classifier from labeled audio files"""
        if not HAS_SKLEARN:
            print("Warning: scikit-learn not available. Cannot train classifier.")
            return False
        
        try:
            features = []
            labels = []
            
            print(f"Training genre classifier on {len(training_data)} samples...")
            
            for file_path, genre in training_data:
                try:
                    # Analyze audio file to extract features
                    result = self.analyze_audio_file(file_path)
                    
                    # Extract feature vector
                    y, sr = librosa.load(file_path, sr=22050, duration=30)  # Use 30-second clips
                    feature_vector = self._extract_genre_features(y, sr, result)
                    
                    features.append(feature_vector)
                    labels.append(genre)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
            
            if len(features) < 5:
                print("Not enough valid training samples. Need at least 5.")
                return False
            
            # Train classifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train model
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(features_scaled, labels)
            
            # Save model and scaler
            import joblib
            model_dir = Path.cwd() / "models"
            model_dir.mkdir(exist_ok=True)
            
            joblib.dump(classifier, model_dir / "genre_classifier.pkl")
            joblib.dump(scaler, model_dir / "genre_scaler.pkl")
            
            print(f"Genre classifier trained and saved with {len(features)} samples")
            return True
            
        except Exception as e:
            print(f"Error training classifier: {e}")
            return False
    
    def load_trained_models(self) -> bool:
        """Load pre-trained ML models if available"""
        try:
            model_dir = Path.cwd() / "models"
            
            if (model_dir / "genre_classifier.pkl").exists():
                import joblib
                self.genre_classifier = joblib.load(model_dir / "genre_classifier.pkl")
                self.genre_scaler = joblib.load(model_dir / "genre_scaler.pkl")
                print("Loaded pre-trained genre classifier")
                return True
                
        except Exception as e:
            print(f"Error loading models: {e}")
            
        return False
    
    def classify_with_ml_model(self, y: np.ndarray, sr: int, result: AudioAnalysisResult) -> None:
        """Use trained ML model for classification if available"""
        try:
            if hasattr(self, 'genre_classifier') and hasattr(self, 'genre_scaler'):
                # Extract features and predict
                features = self._extract_genre_features(y, sr, result)
                features_scaled = self.genre_scaler.transform([features])
                
                # Get prediction and confidence
                prediction = self.genre_classifier.predict(features_scaled)[0]
                probabilities = self.genre_classifier.predict_proba(features_scaled)[0]
                confidence = max(probabilities)
                
                # Update result if confidence is high enough
                if confidence > 0.6:
                    result.genre_prediction = prediction
                    result.genre_confidence = confidence
                    
        except Exception as e:
            result.analysis_errors.append(f"ML classification failed: {str(e)}")
    
    def create_audio_embeddings(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Create audio embeddings for similarity analysis"""
        try:
            # Extract comprehensive feature set for embeddings
            features = []
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            
            # Add statistical moments
            for feature in [spectral_centroid, spectral_rolloff, spectral_bandwidth, spectral_flatness]:
                features.extend([
                    np.mean(feature),
                    np.std(feature),
                    np.min(feature),
                    np.max(feature)
                ])
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            for i in range(20):
                features.extend([
                    np.mean(mfccs[i]),
                    np.std(mfccs[i])
                ])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(12):
                features.extend([
                    np.mean(chroma[i]),
                    np.std(chroma[i])
                ])
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            for i in range(6):
                features.extend([
                    np.mean(tonnetz[i]),
                    np.std(tonnetz[i])
                ])
            
            # Tempo and rhythm features
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            
            features.extend([
                tempo,
                np.mean(onset_strength),
                np.std(onset_strength)
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return None
    
    def find_similar_tracks(self, target_file: str, candidate_files: List[str], 
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar tracks using audio embeddings"""
        try:
            # Analyze target file
            target_result = self.analyze_audio_file(target_file)
            y_target, sr_target = librosa.load(target_file, sr=22050, duration=30)
            target_embedding = self.create_audio_embeddings(y_target, sr_target)
            
            if target_embedding is None:
                return []
            
            similarities = []
            
            for candidate_file in candidate_files:
                try:
                    # Analyze candidate file
                    y_candidate, sr_candidate = librosa.load(candidate_file, sr=22050, duration=30)
                    candidate_embedding = self.create_audio_embeddings(y_candidate, sr_candidate)
                    
                    if candidate_embedding is not None:
                        # Calculate cosine similarity
                        from scipy.spatial.distance import cosine
                        similarity = 1 - cosine(target_embedding, candidate_embedding)
                        similarities.append((candidate_file, similarity))
                        
                except Exception as e:
                    print(f"Error processing {candidate_file}: {e}")
                    continue
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error finding similar tracks: {e}")
            return []
    
    def analyze_music_collection(self, directory: str) -> Dict[str, any]:
        """Analyze an entire music collection and provide insights"""
        try:
            audio_files = []
            for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']:
                audio_files.extend(Path(directory).rglob(f'*{ext}'))
            
            print(f"Analyzing {len(audio_files)} audio files...")
            
            results = []
            genre_counts = defaultdict(int)
            mood_counts = defaultdict(int)
            instrument_counts = defaultdict(int)
            
            for i, file_path in enumerate(audio_files[:100]):  # Limit for demo
                try:
                    result = self.analyze_audio_file(str(file_path))
                    results.append(result)
                    
                    # Count genres and moods
                    if result.genre_prediction:
                        genre_counts[result.genre_prediction] += 1
                    if result.mood_prediction:
                        mood_counts[result.mood_prediction] += 1
                    if result.detected_instruments:
                        for instrument in result.detected_instruments:
                            instrument_counts[instrument] += 1
                    
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(audio_files)} files...")
                        
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
                    continue
            
            # Generate collection insights
            insights = {
                "total_files": len(results),
                "total_duration": sum(r.duration for r in results if r.duration),
                "avg_bpm": np.mean([r.bpm for r in results if r.bpm]),
                "genre_distribution": dict(genre_counts),
                "mood_distribution": dict(mood_counts),
                "instrument_distribution": dict(instrument_counts),
                "key_distribution": defaultdict(int),
                "energy_stats": {
                    "avg": np.mean([r.energy_level for r in results if r.energy_level]),
                    "std": np.std([r.energy_level for r in results if r.energy_level])
                }
            }
            
            # Key distribution
            for result in results:
                if result.key and result.scale:
                    key_scale = f"{result.key} {result.scale}"
                    insights["key_distribution"][key_scale] += 1
            
            insights["key_distribution"] = dict(insights["key_distribution"])
            
            return insights
            
        except Exception as e:
            print(f"Error analyzing collection: {e}")
            return {}
    
    def create_smart_playlists(self, analysis_results: List[AudioAnalysisResult]) -> Dict[str, List[str]]:
        """Create smart playlists based on analysis results"""
        playlists = {
            "High Energy": [],
            "Chill & Relaxed": [],
            "Dance Floor": [],
            "Study Music": [],
            "Workout": [],
            "Late Night": [],
            "Happy Vibes": [],
            "Emotional": []
        }
        
        for result in analysis_results:
            filename = result.filename
            
            # High Energy
            if (result.energy_level and result.energy_level > 0.7) or (result.bpm and result.bpm > 140):
                playlists["High Energy"].append(filename)
            
            # Chill & Relaxed
            if (result.energy_level and result.energy_level < 0.3) or (result.bpm and result.bpm < 90):
                playlists["Chill & Relaxed"].append(filename)
            
            # Dance Floor
            if result.danceability and result.danceability > 0.7:
                playlists["Dance Floor"].append(filename)
            
            # Study Music
            if ((result.energy_level and result.energy_level < 0.4) and 
                (result.bpm and 60 <= result.bpm <= 100) and
                (not result.mood_prediction or result.mood_prediction in ["peaceful", "neutral"])):
                playlists["Study Music"].append(filename)
            
            # Workout
            if ((result.energy_level and result.energy_level > 0.6) and 
                (result.bpm and result.bpm > 120)):
                playlists["Workout"].append(filename)
            
            # Late Night
            if ((result.energy_level and result.energy_level < 0.4) and
                (result.scale == "minor" or result.mood_prediction == "melancholic")):
                playlists["Late Night"].append(filename)
            
            # Happy Vibes
            if (result.valence and result.valence > 0.6) or result.scale == "major":
                playlists["Happy Vibes"].append(filename)
            
            # Emotional
            if result.scale == "minor" or result.mood_prediction in ["melancholic", "emotional"]:
                playlists["Emotional"].append(filename)
        
        # Remove empty playlists
        return {name: tracks for name, tracks in playlists.items() if tracks}

def demo_professional_tools():
    """Demonstrate professional audio processing tools"""
    print("\n" + "="*60)
    print("PROFESSIONAL AUDIO TOOLS DEMO")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    # Find sample audio files
    sample_dir = Path("sample_audio")
    if not sample_dir.exists():
        print("Sample audio directory not found. Creating demo files...")
        return
    
    sample_files = list(sample_dir.glob("*.wav"))
    if not sample_files:
        print("No sample audio files found in sample_audio directory.")
        return
    
    # Use the first available sample
    input_file = str(sample_files[0])
    print(f"Using sample file: {Path(input_file).name}")
    
    # Create output directory
    output_dir = Path("processed_audio")
    output_dir.mkdir(exist_ok=True)
    
    print("\n1. TIME-STRETCHING DEMO")
    print("-" * 30)
    
    # Time-stretch demo
    output_path = output_dir / "time_stretched.wav"
    result = processor.time_stretch(
        input_file, 
        str(output_path), 
        stretch_factor=1.5,  # 50% slower
        preserve_pitch=True,
        quality="high"
    )
    
    if result.success:
        print(f"✓ {result.message}")
        print(f"  Output: {result.output_path}")
        print(f"  Processing time: {result.processing_time:.2f}s")
    else:
        print(f"✗ {result.message}")
    
    print("\n2. AUDIO REPAIR DEMO")
    print("-" * 30)
    
    # Audio repair demo
    output_path = output_dir / "repaired.wav"
    result = processor.repair_audio(
        input_file,
        str(output_path),
        operations=['declick', 'denoise', 'dehum', 'normalize']
    )
    
    if result.success:
        print(f"✓ {result.message}")
        print(f"  Output: {result.output_path}")
        if result.quality_metrics:
            print(f"  SNR improvement: {result.quality_metrics.get('snr_improvement_db', 0):.1f}dB")
    else:
        print(f"✗ {result.message}")
    
    print("\n3. SPECTRAL EDITING DEMO")
    print("-" * 30)
    
    # Spectral editing demo - remove frequencies around 1kHz
    output_path = output_dir / "spectral_edited.wav"
    result = processor.spectral_edit(
        input_file,
        str(output_path),
        freq_ranges=[(800, 1200), (2000, 2500)],  # Remove these frequency ranges
        operation="remove"
    )
    
    if result.success:
        print(f"✓ {result.message}")
        print(f"  Removed frequencies: 800-1200Hz, 2000-2500Hz")
    else:
        print(f"✗ {result.message}")
    
    print("\n4. STEREO FIELD MANIPULATION DEMO")
    print("-" * 30)
    
    # Stereo width control
    output_path = output_dir / "wide_stereo.wav"
    result = processor.stereo_field_manipulation(
        input_file,
        str(output_path),
        operation="width_control",
        parameters={'width': 1.5}  # 50% wider
    )
    
    if result.success:
        print(f"✓ {result.message}")
        print(f"  Applied stereo widening")
    else:
        print(f"✗ {result.message}")
    
    print("\n5. MASTERING CHAIN DEMO")
    print("-" * 30)
    
    # Complete mastering chain
    output_path = output_dir / "mastered.wav"
    mastering_config = {
        'eq': {
            'bands': [
                {'frequency': 100, 'gain': 2.0, 'q': 0.7},    # Low boost
                {'frequency': 3000, 'gain': 1.5, 'q': 1.0},  # Presence boost
                {'frequency': 8000, 'gain': 1.0, 'q': 0.8}   # Air boost
            ]
        },
        'compressor': {
            'threshold': -12.0,
            'ratio': 3.0,
            'attack': 10.0,
            'release': 100.0
        },
        'stereo_enhance': {
            'width': 1.1
        },
        'limiter': {
            'ceiling': -0.1,
            'lookahead': 5.0
        },
        'target_lufs': -14.0
    }
    
    result = processor.mastering_chain(
        input_file,
        str(output_path),
        mastering_config
    )
    
    if result.success:
        print(f"✓ {result.message}")
        if result.quality_metrics:
            print(f"  Peak level: {result.quality_metrics.get('peak_level', 0):.3f}")
            print(f"  Target LUFS: {result.quality_metrics.get('target_lufs', 0):.1f}")
    else:
        print(f"✗ {result.message}")
    
    print("\n" + "="*60)
    print("PROFESSIONAL TOOLS DEMO COMPLETE")
    print("="*60)
    print(f"Processed files saved to: {output_dir}")
    print("\nAvailable operations:")
    print("• Time-stretching with pitch preservation")
    print("• Audio repair (declick, denoise, dehum)")
    print("• Spectral editing (frequency-specific processing)")
    print("• Stereo field manipulation (width, panning)")
    print("• Complete mastering chain (EQ, compression, limiting)")


class ContentBasedSearchEngine:
    """
    Advanced content-based audio search engine
    Provides similarity search, harmonic matching, rhythm matching, and spectral search
    """
    
    def __init__(self):
        self.analyzer = IntelligentAudioAnalyzer()
        self.audio_database = {}  # Store analyzed audio features
        self.database_file = "audio_search_database.json"
        self.load_database()
        
    def load_database(self):
        """Load existing audio database from file"""
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'r') as f:
                    # Load basic info (non-numpy arrays)
                    raw_data = json.load(f)
                    for file_path, data in raw_data.items():
                        self.audio_database[file_path] = data
        except Exception as e:
            print(f"Warning: Could not load audio database: {e}")
            
    def save_database(self):
        """Save audio database to file (excluding numpy arrays)"""
        try:
            # Create a serializable version
            serializable_db = {}
            for file_path, data in self.audio_database.items():
                serializable_data = {}
                for key, value in data.items():
                    if not isinstance(value, np.ndarray):
                        serializable_data[key] = value
                serializable_db[file_path] = serializable_data
                
            with open(self.database_file, 'w') as f:
                json.dump(serializable_db, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save audio database: {e}")
    
    def add_to_database(self, file_path: str, force_reanalyze: bool = False) -> bool:
        """
        Add audio file to searchable database
        
        Args:
            file_path: Path to audio file
            force_reanalyze: Force re-analysis even if file exists in database
            
        Returns:
            bool: Success status
        """
        abs_path = os.path.abspath(file_path)
        
        # Check if already analyzed
        if abs_path in self.audio_database and not force_reanalyze:
            return True
            
        try:
            print(f"Analyzing for content-based search: {os.path.basename(file_path)}")
            
            # Perform full analysis
            result = self.analyzer.analyze_audio_file(file_path)
            
            # Load audio for detailed feature extraction
            y, sr = librosa.load(file_path, sr=None)
            
            # Extract content-based search features
            search_features = self._extract_search_features(y, sr, result)
            
            # Store in database
            self.audio_database[abs_path] = {
                'filename': result.filename,
                'duration': result.duration,
                'bpm': result.bpm,
                'key': result.key,
                'scale': result.scale,
                'genre': result.genre_prediction,
                'energy': result.energy_level,
                'last_analyzed': datetime.now().isoformat(),
                **search_features
            }
            
            self.save_database()
            return True
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return False
    
    def _extract_search_features(self, y: np.ndarray, sr: int, analysis_result) -> Dict[str, Any]:
        """Extract features for content-based search"""
        features = {}
        
        try:
            # 1. Audio Similarity Features
            # Create perceptual hash for similarity
            features['similarity_hash'] = self._create_similarity_hash(y, sr)
            
            # Spectral contrast for timbral similarity
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1).tolist()
            
            # MFCCs for timbral characteristics
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
            
            # 2. Harmonic Matching Features
            # Chroma features for key and harmonic content
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
            features['chroma_std'] = np.std(chroma, axis=1).tolist()
            
            # Harmonic content analysis
            harmonic_strength = self._analyze_harmonic_strength(y, sr)
            features['harmonic_strength'] = harmonic_strength
            
            # Key compatibility (circle of fifths distance)
            if analysis_result.key:
                features['key_numeric'] = self._key_to_numeric(analysis_result.key)
            
            # 3. Rhythm Matching Features
            # Tempo and rhythm patterns
            tempo_histogram = self._create_tempo_histogram(y, sr)
            features['tempo_histogram'] = tempo_histogram.tolist()
            
            # Onset pattern analysis
            onset_features = self._analyze_onset_patterns(y, sr)
            features.update(onset_features)
            
            # Beat strength and regularity
            beat_features = self._analyze_beat_characteristics(y, sr)
            features.update(beat_features)
            
            # 4. Spectral Search Features
            # Spectral centroid and spread
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_std'] = float(np.std(spectral_centroid))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # Frequency distribution
            freq_distribution = self._analyze_frequency_distribution(y, sr)
            features['freq_distribution'] = freq_distribution.tolist()
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = float(np.mean(zcr))
            
        except Exception as e:
            print(f"Warning: Feature extraction error: {e}")
            
        return features
    
    def _create_similarity_hash(self, y: np.ndarray, sr: int) -> str:
        """Create a perceptual hash for audio similarity"""
        try:
            # Use chromaprint if available
            if HAS_CHROMAPRINT:
                # Convert to int16 for chromaprint
                y_int = (y * 32767).astype(np.int16)
                fingerprint = chromaprint.fingerprint(y_int, sr)
                return fingerprint[1] if fingerprint[0] else ""
            
            # Fallback: create hash from spectral features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
            mfcc_hash = hashlib.md5(mfcc.tobytes()).hexdigest()
            return mfcc_hash
            
        except Exception:
            return ""
    
    def _analyze_harmonic_strength(self, y: np.ndarray, sr: int) -> float:
        """Analyze the harmonic content strength"""
        try:
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Calculate energy ratio
            harmonic_energy = np.sum(y_harmonic ** 2)
            total_energy = np.sum(y ** 2)
            
            return float(harmonic_energy / total_energy) if total_energy > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _key_to_numeric(self, key: str) -> int:
        """Convert key to numeric value for distance calculations"""
        key_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        return key_map.get(key.replace('m', ''), 0)
    
    def _create_tempo_histogram(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Create histogram of tempo estimates across the track"""
        try:
            # Analyze tempo in overlapping windows
            window_size = sr * 4  # 4-second windows
            hop_size = sr * 2     # 2-second hops
            
            tempos = []
            for start in range(0, len(y) - window_size, hop_size):
                window = y[start:start + window_size]
                tempo, _ = librosa.beat.beat_track(y=window, sr=sr)
                tempos.append(tempo)
            
            # Create histogram
            hist, _ = np.histogram(tempos, bins=20, range=(60, 200))
            return hist / np.sum(hist) if np.sum(hist) > 0 else hist
            
        except Exception:
            return np.zeros(20)
    
    def _analyze_onset_patterns(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze onset patterns for rhythm matching"""
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            if len(onset_times) < 2:
                return {'onset_density': 0.0, 'onset_regularity': 0.0}
            
            # Onset density (onsets per second)
            duration = len(y) / sr
            onset_density = len(onset_times) / duration
            
            # Onset regularity (coefficient of variation of intervals)
            intervals = np.diff(onset_times)
            onset_regularity = 1.0 - (np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0.0
            onset_regularity = max(0.0, min(1.0, onset_regularity))
            
            return {
                'onset_density': float(onset_density),
                'onset_regularity': float(onset_regularity)
            }
            
        except Exception:
            return {'onset_density': 0.0, 'onset_regularity': 0.0}
    
    def _analyze_beat_characteristics(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze beat strength and characteristics"""
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            if len(beats) < 2:
                return {'beat_strength': 0.0, 'beat_regularity': 0.0}
            
            # Beat strength (how strong the beats are)
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            beat_strength = np.mean([onset_strength[beat] for beat in beats if beat < len(onset_strength)])
            
            # Beat regularity
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_intervals = np.diff(beat_times)
            beat_regularity = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals)) if np.mean(beat_intervals) > 0 else 0.0
            beat_regularity = max(0.0, min(1.0, beat_regularity))
            
            return {
                'beat_strength': float(beat_strength),
                'beat_regularity': float(beat_regularity)
            }
            
        except Exception:
            return {'beat_strength': 0.0, 'beat_regularity': 0.0}
    
    def _analyze_frequency_distribution(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Analyze frequency distribution for spectral matching"""
        try:
            # Compute spectrogram
            D = np.abs(librosa.stft(y))
            
            # Average across time to get frequency distribution
            freq_dist = np.mean(D, axis=1)
            
            # Normalize
            freq_dist = freq_dist / np.sum(freq_dist) if np.sum(freq_dist) > 0 else freq_dist
            
            # Reduce dimensionality (take every 4th bin for efficiency)
            return freq_dist[::4]
            
        except Exception:
            return np.zeros(256)
    
    def find_similar_audio(self, query_file: str, similarity_threshold: float = 0.7, 
                          limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find audio files similar to the query file
        
        Args:
            query_file: Path to query audio file
            similarity_threshold: Minimum similarity score (0-1)
            limit: Maximum number of results
            
        Returns:
            List of (file_path, similarity_score) tuples
        """
        # Add query file to database if not already there
        if not self.add_to_database(query_file):
            return []
        
        query_path = os.path.abspath(query_file)
        query_features = self.audio_database.get(query_path)
        
        if not query_features:
            return []
        
        similarities = []
        
        for file_path, features in self.audio_database.items():
            if file_path == query_path:
                continue
                
            similarity = self._calculate_audio_similarity(query_features, features)
            
            if similarity >= similarity_threshold:
                similarities.append((file_path, similarity))
        
        # Sort by similarity (highest first) and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def find_harmonic_matches(self, query_file: str, key_tolerance: int = 2, 
                             limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find audio files in compatible keys (harmonic matching)
        
        Args:
            query_file: Path to query audio file
            key_tolerance: Maximum key distance (semitones) for compatibility
            limit: Maximum number of results
            
        Returns:
            List of (file_path, compatibility_score) tuples
        """
        # Add query file to database if not already there
        if not self.add_to_database(query_file):
            return []
        
        query_path = os.path.abspath(query_file)
        query_features = self.audio_database.get(query_path)
        
        if not query_features or 'chroma_mean' not in query_features:
            return []
        
        matches = []
        
        for file_path, features in self.audio_database.items():
            if file_path == query_path or 'chroma_mean' not in features:
                continue
            
            compatibility = self._calculate_harmonic_compatibility(query_features, features, key_tolerance)
            
            if compatibility > 0:
                matches.append((file_path, compatibility))
        
        # Sort by compatibility (highest first) and limit results
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]
    
    def find_rhythm_matches(self, query_file: str, tempo_tolerance: float = 10.0,
                           limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find audio files with similar rhythm patterns
        
        Args:
            query_file: Path to query audio file
            tempo_tolerance: Maximum BPM difference for matching
            limit: Maximum number of results
            
        Returns:
            List of (file_path, rhythm_similarity) tuples
        """
        # Add query file to database if not already there
        if not self.add_to_database(query_file):
            return []
        
        query_path = os.path.abspath(query_file)
        query_features = self.audio_database.get(query_path)
        
        if not query_features:
            return []
        
        matches = []
        
        for file_path, features in self.audio_database.items():
            if file_path == query_path:
                continue
                
            rhythm_similarity = self._calculate_rhythm_similarity(query_features, features, tempo_tolerance)
            
            if rhythm_similarity > 0:
                matches.append((file_path, rhythm_similarity))
        
        # Sort by similarity (highest first) and limit results
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]
    
    def find_spectral_matches(self, query_file: str, spectral_tolerance: float = 0.5,
                             limit: int = 10) -> List[Tuple[str, float]]:
        """
        Search by frequency content and spectral characteristics
        
        Args:
            query_file: Path to query audio file
            spectral_tolerance: Tolerance for spectral matching (0-1)
            limit: Maximum number of results
            
        Returns:
            List of (file_path, spectral_similarity) tuples
        """
        # Add query file to database if not already there
        if not self.add_to_database(query_file):
            return []
        
        query_path = os.path.abspath(query_file)
        query_features = self.audio_database.get(query_path)
        
        if not query_features:
            return []
        
        matches = []
        
        for file_path, features in self.audio_database.items():
            if file_path == query_path:
                continue
                
            spectral_similarity = self._calculate_spectral_similarity(query_features, features)
            
            if spectral_similarity >= spectral_tolerance:
                matches.append((file_path, spectral_similarity))
        
        # Sort by similarity (highest first) and limit results
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]
    
    def _calculate_audio_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate overall audio similarity using multiple features"""
        similarities = []
        
        # MFCC similarity
        if 'mfcc_mean' in features1 and 'mfcc_mean' in features2:
            mfcc_sim = self._cosine_similarity(features1['mfcc_mean'], features2['mfcc_mean'])
            similarities.append(mfcc_sim * 0.4)  # Weight: 40%
        
        # Spectral contrast similarity
        if 'spectral_contrast_mean' in features1 and 'spectral_contrast_mean' in features2:
            contrast_sim = self._cosine_similarity(features1['spectral_contrast_mean'], features2['spectral_contrast_mean'])
            similarities.append(contrast_sim * 0.3)  # Weight: 30%
        
        # Chroma similarity
        if 'chroma_mean' in features1 and 'chroma_mean' in features2:
            chroma_sim = self._cosine_similarity(features1['chroma_mean'], features2['chroma_mean'])
            similarities.append(chroma_sim * 0.2)  # Weight: 20%
        
        # Tempo similarity
        if 'bpm' in features1 and 'bpm' in features2 and features1['bpm'] and features2['bpm']:
            tempo_diff = abs(features1['bpm'] - features2['bpm'])
            tempo_sim = max(0, 1 - tempo_diff / 60)  # Normalize by 60 BPM
            similarities.append(tempo_sim * 0.1)  # Weight: 10%
        
        return sum(similarities) if similarities else 0.0
    
    def _calculate_harmonic_compatibility(self, features1: Dict, features2: Dict, key_tolerance: int) -> float:
        """Calculate harmonic compatibility between two audio files"""
        compatibility = 0.0
        
        # Chroma similarity (harmonic content)
        if 'chroma_mean' in features1 and 'chroma_mean' in features2:
            chroma_sim = self._cosine_similarity(features1['chroma_mean'], features2['chroma_mean'])
            compatibility += chroma_sim * 0.6
        
        # Key compatibility
        if 'key_numeric' in features1 and 'key_numeric' in features2:
            key_distance = min(
                abs(features1['key_numeric'] - features2['key_numeric']),
                12 - abs(features1['key_numeric'] - features2['key_numeric'])
            )
            if key_distance <= key_tolerance:
                key_compatibility = 1.0 - (key_distance / key_tolerance)
                compatibility += key_compatibility * 0.4
        
        return compatibility
    
    def _calculate_rhythm_similarity(self, features1: Dict, features2: Dict, tempo_tolerance: float) -> float:
        """Calculate rhythm similarity between two audio files"""
        similarities = []
        
        # Tempo similarity
        if 'bpm' in features1 and 'bpm' in features2 and features1['bpm'] and features2['bpm']:
            tempo_diff = abs(features1['bpm'] - features2['bpm'])
            if tempo_diff <= tempo_tolerance:
                tempo_sim = 1.0 - (tempo_diff / tempo_tolerance)
                similarities.append(tempo_sim * 0.4)
        
        # Onset pattern similarity
        if 'onset_density' in features1 and 'onset_density' in features2:
            onset_sim = 1.0 - abs(features1['onset_density'] - features2['onset_density']) / max(features1['onset_density'], features2['onset_density'], 1.0)
            similarities.append(onset_sim * 0.3)
        
        # Beat regularity similarity
        if 'beat_regularity' in features1 and 'beat_regularity' in features2:
            beat_sim = 1.0 - abs(features1['beat_regularity'] - features2['beat_regularity'])
            similarities.append(beat_sim * 0.3)
        
        return sum(similarities) if similarities else 0.0
    
    def _calculate_spectral_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate spectral similarity between two audio files"""
        similarities = []
        
        # Frequency distribution similarity
        if 'freq_distribution' in features1 and 'freq_distribution' in features2:
            freq_sim = self._cosine_similarity(features1['freq_distribution'], features2['freq_distribution'])
            similarities.append(freq_sim * 0.4)
        
        # Spectral centroid similarity
        if 'spectral_centroid_mean' in features1 and 'spectral_centroid_mean' in features2:
            centroid_diff = abs(features1['spectral_centroid_mean'] - features2['spectral_centroid_mean'])
            centroid_sim = max(0, 1 - centroid_diff / 4000)  # Normalize by 4kHz
            similarities.append(centroid_sim * 0.3)
        
        # Zero crossing rate similarity
        if 'zcr_mean' in features1 and 'zcr_mean' in features2:
            zcr_diff = abs(features1['zcr_mean'] - features2['zcr_mean'])
            zcr_sim = max(0, 1 - zcr_diff / 0.2)  # Normalize by 0.2
            similarities.append(zcr_sim * 0.3)
        
        return sum(similarities) if similarities else 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception:
            return 0.0
    
    def batch_analyze_directory(self, directory: str, extensions: List[str] = None) -> int:
        """
        Analyze all audio files in a directory for content-based search
        
        Args:
            directory: Directory path to analyze
            extensions: List of file extensions to include (default: common audio formats)
            
        Returns:
            Number of files successfully analyzed
        """
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.aiff', '.m4a', '.ogg']
        
        analyzed_count = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    if self.add_to_database(file_path):
                        analyzed_count += 1
        
        print(f"Analyzed {analyzed_count} audio files for content-based search")
        return analyzed_count
    
    def search_by_example(self, query_file: str, search_types: List[str] = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        Comprehensive search using multiple algorithms
        
        Args:
            query_file: Path to query audio file
            search_types: Types of search to perform ['similarity', 'harmonic', 'rhythm', 'spectral']
            
        Returns:
            Dictionary with search results for each type
        """
        if search_types is None:
            search_types = ['similarity', 'harmonic', 'rhythm', 'spectral']
        
        results = {}
        
        if 'similarity' in search_types:
            results['similarity'] = self.find_similar_audio(query_file, similarity_threshold=0.6)
        
        if 'harmonic' in search_types:
            results['harmonic'] = self.find_harmonic_matches(query_file, key_tolerance=3)
        
        if 'rhythm' in search_types:
            results['rhythm'] = self.find_rhythm_matches(query_file, tempo_tolerance=15.0)
        
        if 'spectral' in search_types:
            results['spectral'] = self.find_spectral_matches(query_file, spectral_tolerance=0.4)
        
        return results


def demo_content_based_search():
    """Demonstrate content-based search capabilities"""
    print("\n" + "="*70)
    print("CONTENT-BASED AUDIO SEARCH DEMO")
    print("="*70)
    
    # Initialize search engine
    search_engine = ContentBasedSearchEngine()
    
    # Analyze sample audio directory
    sample_dir = "sample_audio"
    if os.path.exists(sample_dir):
        print(f"\n1. BUILDING SEARCH DATABASE")
        print("-" * 40)
        analyzed_count = search_engine.batch_analyze_directory(sample_dir)
        
        if analyzed_count == 0:
            print("No audio files found to analyze")
            return
        
        # Get first audio file as query example
        audio_files = []
        for file in os.listdir(sample_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(sample_dir, file))
        
        if not audio_files:
            print("No audio files found for demo")
            return
        
        query_file = audio_files[0]
        print(f"\n2. SEARCHING WITH QUERY: {os.path.basename(query_file)}")
        print("-" * 50)
        
        # Perform comprehensive search
        results = search_engine.search_by_example(query_file)
        
        # Display results
        for search_type, matches in results.items():
            print(f"\n{search_type.upper()} SEARCH RESULTS:")
            if matches:
                for file_path, score in matches[:3]:  # Show top 3
                    filename = os.path.basename(file_path)
                    print(f"  ✓ {filename} (score: {score:.3f})")
            else:
                print("  No matches found")
        
    print(f"\n3. SEARCH DATABASE SAVED")
    print("-" * 30)
    print(f"Database file: {search_engine.database_file}")
    print(f"Total entries: {len(search_engine.audio_database)}")
    
    print("\n" + "="*70)
    print("CONTENT-BASED SEARCH DEMO COMPLETE")
    print("="*70)
    print("Available search types:")
    print("• Audio similarity search - find samples that sound alike")
    print("• Harmonic matching - find samples in compatible keys")
    print("• Rhythm matching - find samples with similar groove")
    print("• Spectral search - search by frequency content")

def main():
    """Main function with enhanced menu system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Audio Analysis with Professional Tools")
    parser.add_argument("file", nargs="?", help="Audio file to analyze")
    parser.add_argument("--demo-tools", action="store_true", help="Demonstrate professional audio tools")
    parser.add_argument("--demo-search", action="store_true", help="Demonstrate content-based search")
    parser.add_argument("--demo-metadata", action="store_true", help="Demonstrate metadata management")
    parser.add_argument("--save-metadata", action="store_true", help="Save analysis results to metadata database")
    parser.add_argument("--format", choices=["json", "detailed"], default="detailed", help="Output format")
    
    args = parser.parse_args()
    
    if args.demo_tools:
        demo_professional_tools()
        return
    
    if args.demo_search:
        demo_content_based_search()
        return
    
    if args.demo_metadata:
        try:
            from metadata_workflow_demo import main as demo_metadata_main
            demo_metadata_main()
        except ImportError:
            print("Metadata management demo not available. Please ensure metadata_manager.py is available.")
        return
    
    if not args.file:
        print("Intelligent Audio Analysis System")
        print("=" * 50)
        print("Options:")
        print("1. Analyze audio file: python audio_analyzer.py <file>")
        print("2. Demo professional tools: python audio_analyzer.py --demo-tools")
        print("3. Demo content-based search: python audio_analyzer.py --demo-search")
        print("4. Demo metadata management: python audio_analyzer.py --demo-metadata")
        print("5. JSON output: python audio_analyzer.py <file> --format json")
        print("6. Save to metadata DB: python audio_analyzer.py <file> --save-metadata")
        return
    
    # Verify file exists
    if not Path(args.file).exists():
        print(f"Error: File '{args.file}' not found.")
        return
    
    # Perform analysis
    analyzer = IntelligentAudioAnalyzer()
    print(f"Analyzing: {args.file}")
    print("-" * 50)
    
    result = analyzer.analyze_audio_file(args.file)
    
    # Save to metadata database if requested
    if args.save_metadata:
        try:
            from metadata_manager import MetadataManager
            metadata_manager = MetadataManager()
            
            # Convert analysis result to metadata and save
            metadata = metadata_manager.extract_file_metadata(args.file)
            audio_id = metadata_manager.save_metadata(metadata)
            
            if audio_id > 0:
                print(f"✓ Saved to metadata database (ID: {audio_id})")
            else:
                print("✗ Failed to save to metadata database")
        except ImportError:
            print("Warning: Metadata management not available")
        except Exception as e:
            print(f"Warning: Could not save to metadata database: {e}")
    
    if args.format == "json":
        # Convert to JSON (excluding non-serializable fields)
        json_result = {
            "filename": result.filename,
            "duration": result.duration,
            "sample_rate": result.sample_rate,
            "bpm": result.bpm,
            "tempo_category": result.tempo_category,
            "key": result.key,
            "scale": result.scale,
            "lufs_integrated": result.lufs_integrated,
            "peak_db": result.peak_db,
            "genre_prediction": result.genre_prediction,
            "mood_prediction": result.mood_prediction,
            "energy_level": result.energy_level,
            "analysis_errors": result.analysis_errors
        }
        print(json.dumps(json_result, indent=2))
    else:
        # Detailed output
        print(f"File: {result.filename}")
        print(f"Duration: {result.duration:.2f} seconds")
        print(f"Sample Rate: {result.sample_rate}Hz")
        print(f"BPM: {result.bpm:.1f} ({result.tempo_category})" if result.bpm else "BPM: Unable to detect")
        print(f"Key: {result.key} {result.scale}" if result.key else "Key: Unable to detect")
        print(f"LUFS: {result.lufs_integrated:.1f}" if result.lufs_integrated else "LUFS: Unable to measure")
        print(f"Peak: {result.peak_db:.1f}dB" if result.peak_db else "Peak: Unable to measure")
        print(f"Suggested Category: {analyzer.categorize_by_analysis(result)}")
        
        if result.analysis_errors:
            print("\nErrors:")
            for error in result.analysis_errors:
                print(f"  - {error}")

if __name__ == "__main__":
    main()