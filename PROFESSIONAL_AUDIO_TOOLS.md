# Professional Audio Tools Documentation

## Overview

The Professional Audio Tools module provides studio-quality audio processing capabilities for professional audio production. These tools are designed to meet the standards of professional audio engineers and producers.

## Features

### 1. Time-Stretching (Pitch-Preserving)

**Purpose**: Change the duration of audio without affecting pitch, essential for tempo matching and audio synchronization.

**Capabilities**:
- High-quality time-stretching using advanced algorithms
- Pitch preservation during time manipulation
- Multiple quality settings (low, medium, high, ultra)
- Support for extreme stretch ratios (0.25x to 4.0x)

**Usage**:
```python
from audio_analyzer import ProfessionalAudioProcessor

processor = ProfessionalAudioProcessor()

# Stretch audio to 1.5x length (slower tempo)
result = processor.time_stretch(
    input_path="input.wav",
    output_path="stretched.wav", 
    stretch_factor=1.5,
    preserve_pitch=True,
    quality="high"
)
```

**Applications**:
- Tempo matching for DJs
- Audio-visual sync correction
- Musical arrangement adjustments
- Speech rate modification

### 2. Audio Repair

**Purpose**: Restore damaged or compromised audio using advanced signal processing techniques.

**Capabilities**:
- **Declicking**: Remove clicks, pops, and impulse noise
- **Denoising**: Remove background noise and hiss
- **Dehumming**: Remove 50/60Hz electrical hum and harmonics
- **Normalization**: Optimize loudness levels

**Usage**:
```python
# Complete audio restoration
result = processor.repair_audio(
    input_path="damaged.wav",
    output_path="repaired.wav",
    operations=['declick', 'denoise', 'dehum', 'normalize']
)
```

**Applications**:
- Restoration of vintage recordings
- Cleaning up field recordings
- Removing electrical interference
- Preparing audio for mastering

### 3. Spectral Editing

**Purpose**: Precision frequency-domain editing for surgical audio corrections.

**Capabilities**:
- Remove specific frequency ranges
- Isolate desired frequency bands
- Attenuate problematic frequencies
- High-resolution spectral analysis (4096-point FFT)

**Usage**:
```python
# Remove unwanted frequency ranges
result = processor.spectral_edit(
    input_path="input.wav",
    output_path="edited.wav",
    freq_ranges=[(800, 1200), (2000, 2500)],  # Hz
    operation="remove"  # or "isolate", "attenuate"
)
```

**Applications**:
- Removing specific instrument bleed
- Eliminating resonant frequencies
- Surgical noise removal
- Creating unique sound effects

### 4. Stereo Field Manipulation

**Purpose**: Control and enhance the stereo image of audio recordings.

**Capabilities**:
- **Mono to Stereo**: Create stereo width from mono sources
  - Delay-based widening
  - Chorus-based expansion
  - Reverb-based spatialization
- **Width Control**: Adjust stereo width (narrow to ultra-wide)
- **Panning**: Position audio in the stereo field
- **Mid-Side Processing**: Advanced stereo manipulation

**Usage**:
```python
# Create stereo width from mono
result = processor.stereo_field_manipulation(
    input_path="mono.wav",
    output_path="stereo.wav",
    operation="mono_to_stereo",
    parameters={
        "technique": "chorus",
        "lfo_freq": 0.3,
        "lfo_depth": 0.03
    }
)

# Adjust stereo width
result = processor.stereo_field_manipulation(
    input_path="stereo.wav",
    output_path="wide.wav", 
    operation="width_control",
    parameters={"width": 1.5}  # 1.0 = normal, >1.0 = wider
)
```

**Applications**:
- Enhancing stereo recordings
- Creating immersive soundscapes
- Fixing mono compatibility issues
- Creative stereo effects

### 5. Mastering Chain

**Purpose**: Complete mastering workflow with professional-grade processing.

**Capabilities**:
- **Multi-band EQ**: Precise frequency shaping
- **Compression**: Dynamic range control
- **Stereo Enhancement**: Width and imaging
- **Limiting**: Peak control and loudness maximization
- **LUFS Normalization**: Broadcast-standard loudness

**Usage**:
```python
# Professional mastering chain
mastering_config = {
    'eq': {
        'bands': [
            {'frequency': 100, 'gain': 2.0, 'q': 0.7},    # Sub boost
            {'frequency': 3000, 'gain': 1.5, 'q': 1.0},  # Presence
            {'frequency': 8000, 'gain': 1.0, 'q': 0.8}   # Air
        ]
    },
    'compressor': {
        'threshold': -12.0,  # dB
        'ratio': 3.0,
        'attack': 10.0,      # ms
        'release': 100.0     # ms
    },
    'stereo_enhance': {
        'width': 1.1
    },
    'limiter': {
        'ceiling': -0.1,     # dB
        'lookahead': 5.0     # ms
    },
    'target_lufs': -14.0
}

result = processor.mastering_chain(
    input_path="mix.wav",
    output_path="mastered.wav",
    chain_config=mastering_config
)
```

**Applications**:
- Final mix preparation
- Streaming platform optimization
- Broadcast compliance
- CD/vinyl mastering

## Quality Metrics

The system provides comprehensive quality metrics for processed audio:

- **SNR Improvement**: Signal-to-noise ratio enhancement
- **Peak Levels**: Maximum amplitude measurements
- **LUFS Measurements**: Perceived loudness standards
- **Processing Time**: Performance benchmarks

## Dependencies

### Required Libraries
```bash
# Core audio processing
pip install librosa soundfile scipy numpy

# Professional processing
pip install pyrubberband noisereduce pyloudnorm

# Optional enhancements
pip install essentia tensorflow
```

### System Dependencies
```bash
# macOS
brew install rubberband ffmpeg

# Ubuntu/Debian
sudo apt install librubberband-dev ffmpeg

# Windows
# Download from respective project websites
```

## Usage Examples

### Quick Start Demo
```bash
# Run the professional tools demo
python audio_analyzer.py --demo-tools

# Test all features comprehensively
python test_professional_tools.py
```

### Integration Example
```python
from audio_analyzer import ProfessionalAudioProcessor

# Initialize processor
processor = ProfessionalAudioProcessor()

# Process audio file
input_file = "source.wav"
output_file = "processed.wav"

# Time-stretch for tempo adjustment
result = processor.time_stretch(input_file, "tempo_adjusted.wav", 0.9)

# Clean up audio
result = processor.repair_audio("tempo_adjusted.wav", "cleaned.wav")

# Master the final result
mastering_config = {
    'target_lufs': -16.0,
    'limiter': {'ceiling': -0.2}
}
result = processor.mastering_chain("cleaned.wav", output_file, mastering_config)

print(f"Processing complete: {result.message}")
```

## Technical Specifications

### Audio Formats
- **Input**: WAV, FLAC, AIFF, MP3 (via FFmpeg)
- **Output**: WAV (recommended), FLAC, AIFF
- **Sample Rates**: 44.1kHz, 48kHz, 88.2kHz, 96kHz, 192kHz
- **Bit Depths**: 16-bit, 24-bit, 32-bit float

### Processing Parameters
- **FFT Size**: 4096 points (spectral editing)
- **Hop Length**: 512-1024 samples
- **Window Function**: Hanning/Hamming
- **Overlap**: 75% (time-stretching)

### Performance
- **Real-time Factor**: 0.1x - 10x (depending on operation)
- **Memory Usage**: Optimized for large files
- **CPU Usage**: Multi-threaded where applicable

## Best Practices

### Workflow Recommendations
1. **Always backup original files**
2. **Process in order**: Repair → Edit → Master
3. **Use highest quality settings for final production**
4. **Monitor levels throughout processing**
5. **A/B test processed vs. original**

### Quality Guidelines
- **Time-stretching**: Limit to ±50% for best quality
- **EQ**: Use broad Q values for musical results
- **Compression**: Gentle ratios (2:1 - 4:1) for transparency
- **Limiting**: Leave headroom (-0.1dB to -0.3dB ceiling)

### File Management
- Use descriptive naming conventions
- Keep processing logs
- Archive intermediate versions
- Document processing parameters

## Troubleshooting

### Common Issues
1. **"Audio must have five channels or less"**
   - Solution: Convert to stereo before processing

2. **"pyrubberband not found"**
   - Solution: Install Rubber Band library system dependency

3. **"Invalid LUFS measurement"**
   - Solution: Check audio levels and duration (>1 second)

### Performance Optimization
- Use smaller audio chunks for large files
- Close unnecessary applications
- Use SSD storage for faster I/O
- Consider processing in batches

## Advanced Features

### Custom Processing Chains
You can create custom processing chains by combining multiple operations:

```python
# Custom vocal processing chain
def process_vocal(input_path, output_path):
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Step 1: Repair
    step1 = temp_dir / "step1.wav"
    processor.repair_audio(input_path, str(step1), ['declick', 'denoise'])
    
    # Step 2: Spectral editing (remove low rumble)
    step2 = temp_dir / "step2.wav" 
    processor.spectral_edit(str(step1), str(step2), [(20, 80)], "remove")
    
    # Step 3: Master
    config = {
        'eq': {'bands': [{'frequency': 2000, 'gain': 2.0, 'q': 1.0}]},
        'compressor': {'threshold': -18.0, 'ratio': 3.0},
        'target_lufs': -20.0
    }
    processor.mastering_chain(str(step2), output_path, config)
```

### Batch Processing
```python
import glob

def batch_process(input_pattern, operation):
    files = glob.glob(input_pattern)
    for file_path in files:
        output_path = file_path.replace('.wav', '_processed.wav')
        result = operation(file_path, output_path)
        print(f"Processed {file_path}: {result.success}")
```

## Conclusion

The Professional Audio Tools provide comprehensive audio processing capabilities suitable for professional production environments. The modular design allows for flexible workflows while maintaining broadcast-quality standards.

For support and advanced usage, refer to the test suite and example implementations in the codebase.