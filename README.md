# 🎵 Advanced Audio Converter & Professional Tools

A comprehensive audio conversion tool with batch processing, pitch/key adjustment, format conversion, automatic categorization, and professional-grade audio processing capabilities.

## ✨ Features

### Core Functionality
- **Batch Conversion**: Process multiple files and entire folders at once
- **🔥 COMPREHENSIVE FORMAT SUPPORT**: Convert between **62 audio formats** including all major formats
- **Pitch & Key Shifting**: Adjust pitch and key in real-time during conversion
- **Quality Control**: Configurable sample rate (22kHz-96kHz) and bit depth (16/24/32-bit)

### 🎵 Supported Formats (62 Total)
**Popular Formats**: WAV, MP3, FLAC, OGG, AAC, M4A, OPUS, AIFF, WMA, AC3
**Professional**: BWF, CAF, W64, RF64, SD2, IRCAM, NIST, VOC
**Apple Ecosystem**: M4A, M4R, M4B, M4P, CAFF
**Video Containers**: MP4, MOV, AVI, MKV, WEBM, 3GP (audio extraction)
**Lossless Compressed**: TTA, TAK, ALS, APE, WV, SHN
**Legacy/Specialized**: AU, RA, RM, AMR, GSM, and many more

*See [COMPREHENSIVE_FORMAT_SUPPORT.md](COMPREHENSIVE_FORMAT_SUPPORT.md) for the complete list*

### Professional Audio Tools 🎛️
- **Time-Stretching**: Change tempo without affecting pitch using advanced algorithms
- **Audio Repair**: Remove clicks, noise, hum, and restore damaged audio
- **Spectral Editing**: Precision frequency-domain editing and filtering
- **Stereo Field Manipulation**: Width control, mono-to-stereo, panning, mid-side processing
- **Mastering Chain**: Complete mastering workflow with EQ, compression, and limiting

### Smart Organization
- **Auto-Categorization**: Automatically sorts files based on filename keywords
- **Custom Categories**: Fully customizable category system with keyword matching
- **Folder Structure**: Option to preserve original folder hierarchy
- **Multiple Selection**: Choose individual files or scan entire directories

### Audio Processing
- **Normalization**: Automatic level adjustment for consistent volume
- **Silence Removal**: Trim leading and trailing silence
- **High-Quality Processing**: Uses librosa for professional audio processing
- **Real-time Monitoring**: Live progress tracking and detailed logging

### User Interface
- **Tabbed Interface**: Organized sections for conversion, settings, categories, and progress
- **Professional Tools GUI**: Dedicated interface for professional audio processing
- **Visual Feedback**: Progress bars, real-time logs, and status indicators
- **Settings Management**: Save and load custom configurations
- **Intuitive Controls**: Easy-to-use sliders and checkboxes

## 🚀 Quick Start

### 1. Installation

**Option A: Automatic Setup (Recommended)**
```bash
chmod +x setup.sh
./setup.sh
```

**Option B: Manual Installation**
```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Install FFmpeg (macOS)
brew install ffmpeg

# For professional tools (optional)
brew install rubberband

# Run the application
python3 audio_converter.py
```

### 2. Basic Usage

1. **Launch the application**:
   ```bash
   python3 audio_converter.py
   ```

2. **Add audio sources**:
   - Click "Add Folders" to scan entire directories
   - Click "Add Files" to select individual files
   - Mix and match as needed

3. **Choose output settings**:
   - Select output folder
   - Choose target format (WAV, MP3, FLAC, etc.)
   - Adjust pitch/key if needed

4. **Configure processing**:
   - Enable auto-categorization for organized output
   - Set audio quality in the Settings tab
   - Customize categories in the Categories tab

5. **Start conversion**:
   - Click "Start Conversion"
   - Monitor progress in the Progress tab

## 📂 Auto-Categorization

The application automatically sorts files into folders based on filename keywords:

### Default Categories

**Drums**: kick, snare, hihat, cymbal, tom, drum, perc, percussion
**Bass**: bass, sub, 808, low
**Melody**: melody, lead, main, hook, theme
**Vocals**: vocal, voice, singing, rap, spoken
**FX**: fx, effect, sweep, riser, impact, crash
**Loops**: loop, pattern, sequence
**Instruments**: piano, guitar, synth, violin, flute, sax

### Custom Categories

You can modify categories in the "Categories" tab:
1. Edit the text area with your desired categories and keywords
2. Format: `category_name: keyword1, keyword2, keyword3`
3. Click "Save Categories" to apply changes

## ⚙️ Settings

### Audio Quality
- **Sample Rate**: 22kHz, 44.1kHz, 48kHz, 96kHz
- **Bit Depth**: 16-bit, 24-bit, 32-bit
- **Normalization**: Automatic level adjustment
- **Silence Removal**: Trim quiet sections

### Processing Options
- **Auto-categorize**: Sort files by filename keywords
- **Preserve structure**: Maintain original folder hierarchy
- **Pitch shift**: Adjust pitch in semitones (-12 to +12)
- **Key shift**: Additional pitch adjustment for key changes

### Settings Management
- Save current settings to file
- Load previously saved configurations
- Reset to default values

## 🎛️ Professional Audio Tools

The system includes advanced professional audio processing tools for studio-quality work:

### Time-Stretching & Pitch Shifting
```bash
# Demo all professional tools
python3 audio_analyzer.py --demo-tools

# Run professional tools GUI
python3 professional_tools_gui.py

# Test all features
python3 test_professional_tools.py
```

### Available Tools

#### 1. Time-Stretching
- Change audio duration without affecting pitch
- High-quality algorithms (Rubber Band, Phase Vocoder)
- Stretch factors from 0.25x to 4.0x
- Multiple quality settings

#### 2. Audio Repair
- **Declicking**: Remove clicks and pops
- **Denoising**: Background noise reduction
- **Dehumming**: Remove 50/60Hz electrical hum
- **Normalization**: Optimize loudness levels

#### 3. Spectral Editing
- Remove specific frequency ranges
- Isolate desired frequency bands
- Surgical noise removal
- High-resolution FFT processing

#### 4. Stereo Field Manipulation
- Mono to stereo conversion
- Stereo width control
- Precision panning
- Mid-side processing

#### 5. Mastering Chain
- Multi-band EQ
- Professional compression
- Stereo enhancement
- Brick-wall limiting
- LUFS normalization

### Professional Workflow Example
```python
from audio_analyzer import ProfessionalAudioProcessor

processor = ProfessionalAudioProcessor()

# 1. Time-stretch for tempo matching
processor.time_stretch("input.wav", "stretched.wav", 1.2)

# 2. Audio repair
processor.repair_audio("stretched.wav", "cleaned.wav", 
                      ['declick', 'denoise', 'dehum'])

# 3. Mastering
mastering_config = {
    'eq': {'bands': [{'frequency': 3000, 'gain': 1.5, 'q': 1.0}]},
    'compressor': {'threshold': -12.0, 'ratio': 3.0},
    'target_lufs': -14.0
}
processor.mastering_chain("cleaned.wav", "mastered.wav", mastering_config)
```

## 🔧 Advanced Usage

### Batch Processing Multiple Folders
```
1. Add multiple source folders
2. Enable "Preserve folder structure"
3. Set up custom categories
4. Start conversion - files will be organized automatically
```

### Professional Audio Processing
```
1. Go to Settings tab
2. Set sample rate to 48kHz or 96kHz
3. Choose 24-bit or 32-bit depth
4. Enable normalization
5. Configure pitch/key adjustments
```

### Custom Workflow Example
```
Input: /Music/Samples/
├── Drums/
├── Bass/
└── Melody/

Output: /Converted/
├── drums/
├── bass/
├── melody/
└── uncategorized/
```

## 🎯 Use Cases

### Music Production
- Convert sample libraries to consistent format
- Adjust key of samples to match projects
- Organize samples by instrument type
- Batch normalize audio levels

### Audio Post-Production
- Convert between professional formats
- Maintain folder structure for projects
- Apply consistent processing settings
- Quality control with detailed logging

### Sample Library Management
- Scan large collections of audio files
- Auto-categorize by content type
- Convert legacy formats to modern standards
- Batch process with custom settings

## 📋 Supported Formats

### Input Formats
- WAV (all variants)
- MP3 (all bitrates)
- FLAC (lossless)
- OGG Vorbis
- M4A/AAC
- WMA

### Output Formats
- **WAV**: Uncompressed, highest quality
- **FLAC**: Lossless compression
- **MP3**: Universal compatibility
- **OGG**: Open-source alternative
- **M4A**: Apple/iTunes compatible
- **AAC**: Advanced Audio Coding
- **WMA**: Windows Media Audio

## 🛠️ Dependencies

### Required Python Packages
- `librosa`: Advanced audio processing
- `soundfile`: Audio I/O operations
- `pydub`: Format conversion and manipulation
- `numpy`: Numerical computations
- `tkinter`: GUI framework (built-in)

### System Requirements
- **Python**: 3.8 or later
- **FFmpeg**: For compressed format support
- **Memory**: 4GB+ recommended for large files
- **Storage**: Adequate space for output files

## 🐛 Troubleshooting

### Common Issues

**"FFmpeg not found" warning**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

**"Module not found" errors**
```bash
pip3 install -r requirements.txt
```

**Slow processing with large files**
- Reduce sample rate in settings
- Process smaller batches
- Ensure adequate free memory

**Categories not working**
- Check filename contains keywords
- Verify category keywords in Categories tab
- Enable "Auto-categorize" option

### Performance Tips

1. **Large Libraries**: Process in smaller batches
2. **High Quality**: Use 48kHz/24-bit for professional work
3. **Fast Processing**: Use 44.1kHz/16-bit for general use
4. **Memory Usage**: Close other applications during conversion

## 📄 License

This project is open source. Feel free to modify and distribute according to your needs.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional audio formats
- More processing options
- Enhanced categorization algorithms
- Performance optimizations
- UI/UX improvements

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Review the detailed logs in the Progress tab
4. Ensure input files are valid audio formats

---

**Happy Converting! 🎵**