# 🎵 Audio Converter - Quick Start Guide

## How to Use Your New Audio Converter

### 🚀 Starting the Application

**Method 1: Double-click launcher**
```
Double-click: launch_converter.py
```

**Method 2: Terminal**
```bash
cd "/Users/yvhammg/Desktop/YUSEF APPS"
python3 audio_converter.py
```

### 📖 Step-by-Step Tutorial

#### 1. **Add Your Audio Files**
   - **For entire folders**: Click "Add Folders" and select directories containing audio files
   - **For specific files**: Click "Add Files" and choose individual audio files
   - **Mix and match**: You can add both folders and individual files

#### 2. **Choose Output Location**
   - Click "Browse" next to "Output Folder"
   - Select where you want converted files saved
   - The app will create subfolders automatically if categorization is enabled

#### 3. **Select Output Format**
   - Choose from: WAV, MP3, FLAC, OGG, M4A, AAC, WMA
   - **WAV/FLAC**: Best quality, larger files
   - **MP3**: Most compatible, smaller files
   - **FLAC**: Lossless compression

#### 4. **Adjust Audio Settings** (Optional)
   - **Pitch Shift**: Change pitch without affecting tempo (-12 to +12 semitones)
   - **Key Shift**: Additional pitch adjustment for key changes
   - Use sliders or type exact values

#### 5. **Configure Processing Options**
   - ✅ **Auto-categorize**: Automatically sorts files into folders by type
   - ✅ **Preserve folder structure**: Keeps original folder organization

#### 6. **Start Conversion**
   - Click "Start Conversion"
   - Monitor progress in the Progress tab
   - View detailed logs of what's happening

### 🎯 Smart Categorization

Files are automatically sorted based on their names:

**Example Input Files:**
- `kick_drum_heavy.wav` → **drums** folder
- `bass_808_deep.mp3` → **bass** folder  
- `melody_lead_synth.flac` → **melody** folder
- `vocal_chorus_male.wav` → **vocals** folder

**Categories Available:**
- 🥁 **drums**: kick, snare, hihat, cymbal, tom, percussion
- 🎸 **bass**: bass, sub, 808, low
- 🎵 **melody**: melody, lead, main, hook, theme
- 🎤 **vocals**: vocal, voice, singing, rap, spoken
- ✨ **fx**: fx, effect, sweep, riser, impact, crash
- 🔄 **loops**: loop, pattern, sequence
- 🎹 **instruments**: piano, guitar, synth, violin, flute

### ⚙️ Advanced Settings

#### **Audio Quality** (Settings Tab)
- **Sample Rate**: 44.1kHz (standard) or 48kHz (professional)
- **Bit Depth**: 16-bit (standard) or 24-bit (professional)
- **Normalize**: Makes all files the same volume level
- **Remove Silence**: Trims quiet parts at beginning/end

#### **Custom Categories** (Categories Tab)
Edit the text area to add your own categories:
```
my_category: keyword1, keyword2, keyword3
electronic: edm, house, techno, trance
acoustic: guitar, piano, violin, organic
```

### 💡 Pro Tips

#### **Batch Processing Large Libraries**
1. Add multiple folders at once
2. Enable "Preserve folder structure"
3. Choose appropriate quality settings
4. Let it run overnight for huge collections

#### **Key/Pitch Adjustment for Music Production**
1. Set pitch shift to match your project key
2. Use key shift for fine-tuning
3. Process entire sample packs at once
4. Maintain consistent tuning across samples

#### **Quality vs File Size**
- **Highest Quality**: 48kHz, 24-bit, WAV/FLAC
- **Balanced**: 44.1kHz, 16-bit, WAV
- **Compressed**: 44.1kHz, 16-bit, MP3
- **Smallest**: Lower sample rate, MP3

### 🔧 Common Workflows

#### **Sample Library Organization**
```
Input: /Downloads/Random_Samples/
├── various files...

Output: /Organized_Samples/
├── drums/
├── bass/
├── melody/
├── vocals/
└── fx/
```

#### **Format Standardization**
```
Convert everything to: 44.1kHz, 24-bit WAV
Purpose: Professional studio standard
Settings: Normalize ✅, Remove Silence ✅
```

#### **Key Matching for Producers**
```
Shift all samples to C major
Pitch Shift: +2 (example)
Process entire folder
Result: All samples in same key
```

### ⚠️ Important Notes

- **Backup Originals**: Always keep original files safe
- **Test First**: Try with a few files before processing large batches
- **Check Space**: Ensure enough disk space for output files
- **FFmpeg Required**: Needed for MP3, AAC, and other compressed formats

### 🆘 Quick Troubleshooting

**"No audio files found"**
- Check that folders contain audio files
- Verify file extensions are supported

**"FFmpeg not found"**
```bash
brew install ffmpeg
```

**Slow processing**
- Lower sample rate in settings
- Process smaller batches
- Close other applications

**Categories not working**
- Check filename contains keywords
- Verify categories in Categories tab
- Enable "Auto-categorize" checkbox

---

### 🎉 You're Ready to Go!

Your audio converter includes everything you requested:
- ✅ Key and pitch adjustment
- ✅ Multiple format support  
- ✅ Batch conversion
- ✅ Filename-based categorization
- ✅ Folder scanning
- ✅ Individual file selection
- ✅ Custom output locations

**Happy converting! 🎵**