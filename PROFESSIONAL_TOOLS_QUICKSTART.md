# Professional Audio Tools - Quick Start Guide

## Getting Started

### Launch the Tools

**Command Line Interface:**
```bash
# Analyze audio with professional metrics
python3 audio_analyzer.py sample_audio/piano_melody.wav

# Demo all professional tools
python3 audio_analyzer.py --demo-tools

# JSON output for integration
python3 audio_analyzer.py sample_audio/piano_melody.wav --format json
```

**Graphical Interface:**
```bash
# Launch professional tools GUI
python3 professional_tools_gui.py
```

**Test Suite:**
```bash
# Run comprehensive tests
python3 test_professional_tools.py
```

## Common Operations

### 1. Time-Stretching Audio

**Slow down audio by 50% (1.5x length):**
```python
from audio_analyzer import ProfessionalAudioProcessor

processor = ProfessionalAudioProcessor()
result = processor.time_stretch(
    "input.wav", 
    "slow.wav", 
    stretch_factor=1.5,
    preserve_pitch=True,
    quality="high"
)
```

**Speed up audio by 2x (0.5x length):**
```python
result = processor.time_stretch(
    "input.wav", 
    "fast.wav", 
    stretch_factor=0.5,
    preserve_pitch=True
)
```

### 2. Audio Restoration

**Basic noise cleanup:**
```python
result = processor.repair_audio(
    "noisy.wav",
    "clean.wav",
    operations=['declick', 'denoise', 'dehum']
)
```

**Full restoration with normalization:**
```python
result = processor.repair_audio(
    "damaged.wav",
    "restored.wav", 
    operations=['declick', 'denoise', 'dehum', 'normalize']
)
```

### 3. Frequency-Specific Processing

**Remove low-frequency rumble:**
```python
result = processor.spectral_edit(
    "input.wav",
    "no_rumble.wav",
    freq_ranges=[(20, 100)],
    operation="remove"
)
```

**Isolate vocals (rough estimate):**
```python
result = processor.spectral_edit(
    "full_mix.wav",
    "vocals_only.wav", 
    freq_ranges=[(300, 3000)],
    operation="isolate"
)
```

### 4. Stereo Enhancement

**Widen stereo image:**
```python
result = processor.stereo_field_manipulation(
    "narrow.wav",
    "wide.wav",
    operation="width_control",
    parameters={"width": 1.5}
)
```

**Convert mono to stereo:**
```python
result = processor.stereo_field_manipulation(
    "mono.wav",
    "stereo.wav",
    operation="mono_to_stereo",
    parameters={"technique": "delay", "delay_ms": 15}
)
```

### 5. Professional Mastering

**Basic mastering:**
```python
config = {
    'eq': {
        'bands': [
            {'frequency': 100, 'gain': 1.5, 'q': 0.7},
            {'frequency': 3000, 'gain': 1.0, 'q': 1.0}
        ]
    },
    'compressor': {
        'threshold': -15.0,
        'ratio': 2.5,
        'attack': 10.0,
        'release': 100.0
    },
    'limiter': {
        'ceiling': -0.2
    },
    'target_lufs': -16.0
}

result = processor.mastering_chain("mix.wav", "mastered.wav", config)
```

## Using the GUI

### 1. Select Input File
- Click "Browse" to select your audio file
- Supported formats: WAV, FLAC, AIFF, MP3

### 2. Choose Processing Type
- **Time Stretch**: Adjust tempo without changing pitch
- **Audio Repair**: Clean up damaged or noisy audio
- **Spectral Edit**: Remove or isolate specific frequencies
- **Stereo Field**: Enhance stereo width and positioning
- **Mastering**: Apply professional mastering chain

### 3. Adjust Parameters
- Use sliders and controls to set processing parameters
- Real-time parameter display shows current values
- Checkboxes enable/disable specific operations

### 4. Process Audio
- Click the appropriate "Process" button
- Watch the progress bar and log for status updates
- Files are saved to the `gui_outputs` directory

## Quality Settings

### Time-Stretching Quality
- **Low**: Fastest processing, basic quality
- **Medium**: Good balance of speed and quality
- **High**: Professional quality (recommended)
- **Ultra**: Maximum quality, slower processing

### Audio Format Recommendations
- **Input**: Use WAV or FLAC for best quality
- **Working**: Process in 24-bit/48kHz or higher
- **Output**: Match your target format requirements

## Batch Processing

### Process Multiple Files
```python
import glob
from pathlib import Path

processor = ProfessionalAudioProcessor()
input_files = glob.glob("*.wav")

for file_path in input_files:
    output_path = f"processed_{Path(file_path).name}"
    
    # Apply your processing
    result = processor.time_stretch(file_path, output_path, 1.2)
    
    if result.success:
        print(f"✓ Processed: {file_path}")
    else:
        print(f"✗ Failed: {file_path} - {result.message}")
```

### Custom Processing Chain
```python
def custom_vocal_processing(input_path, output_path):
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Step 1: Clean up audio
    step1 = temp_dir / "step1.wav"
    processor.repair_audio(input_path, str(step1), ['declick', 'denoise'])
    
    # Step 2: Remove low frequencies
    step2 = temp_dir / "step2.wav"
    processor.spectral_edit(str(step1), str(step2), [(20, 100)], "remove")
    
    # Step 3: Enhance presence
    config = {
        'eq': {'bands': [{'frequency': 2500, 'gain': 2.0, 'q': 1.0}]},
        'compressor': {'threshold': -18.0, 'ratio': 3.0},
        'target_lufs': -18.0
    }
    processor.mastering_chain(str(step2), output_path, config)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

# Use the custom function
custom_vocal_processing("vocal.wav", "processed_vocal.wav")
```

## Troubleshooting

### Common Issues

**"pyrubberband not found"**
```bash
# Install Rubber Band library
brew install rubberband  # macOS
sudo apt install librubberband-dev  # Ubuntu
```

**"Audio must have five channels or less"**
- Convert to stereo before processing
- Use audio editor to reduce channel count

**"Invalid LUFS measurement"**
- Ensure audio is at least 1 second long
- Check that audio levels are not too low

### Performance Tips

**For Large Files:**
- Process in chunks
- Use lower quality settings for testing
- Close other applications to free memory

**For Best Quality:**
- Use highest sample rate available
- Process in 32-bit float when possible
- Use "ultra" quality for final processing

## Integration Examples

### With FFmpeg
```bash
# Convert to high-quality WAV first
ffmpeg -i input.mp3 -ar 48000 -ac 2 -f wav temp.wav

# Process with professional tools
python3 -c "
from audio_analyzer import ProfessionalAudioProcessor
p = ProfessionalAudioProcessor()
p.mastering_chain('temp.wav', 'output.wav', {'target_lufs': -16.0})
"

# Convert back to desired format
ffmpeg -i output.wav -b:a 320k final.mp3
```

### With Automation Scripts
```python
#!/usr/bin/env python3
import sys
import argparse
from audio_analyzer import ProfessionalAudioProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--operation", choices=['stretch', 'repair', 'master'], required=True)
    
    args = parser.parse_args()
    processor = ProfessionalAudioProcessor()
    
    if args.operation == 'stretch':
        result = processor.time_stretch(args.input, args.output, 1.2)
    elif args.operation == 'repair':
        result = processor.repair_audio(args.input, args.output)
    elif args.operation == 'master':
        config = {'target_lufs': -16.0}
        result = processor.mastering_chain(args.input, args.output, config)
    
    print(f"Result: {result.message}")
    return 0 if result.success else 1

if __name__ == "__main__":
    sys.exit(main())
```

## Next Steps

1. **Experiment**: Try different settings with your audio files
2. **Combine Tools**: Chain multiple operations for complex processing
3. **Automate**: Create scripts for repetitive tasks
4. **Optimize**: Find the best quality/speed balance for your needs
5. **Document**: Keep notes on successful parameter combinations

For detailed documentation, see `PROFESSIONAL_AUDIO_TOOLS.md`.