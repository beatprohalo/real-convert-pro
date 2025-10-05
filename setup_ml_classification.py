#!/usr/bin/env python3
"""
Complete Setup Script for ML Audio Classification
Installs dependencies and sets up the environment for machine learning features
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {major}.{minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing ML Audio Classification Dependencies...")
    print("=" * 60)
    
    # Core dependencies
    core_packages = [
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.2.0",
        "pandas>=1.5.0"
    ]
    
    # Optional ML packages
    ml_packages = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    # Advanced audio packages (optional)
    advanced_packages = [
        "pyloudnorm>=0.1.0",
        "music21>=8.1.0"
    ]
    
    def install_package_list(packages, category_name):
        print(f"\\n🔧 Installing {category_name}...")
        for package in packages:
            try:
                print(f"   Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "--quiet"
                ])
                print(f"   ✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️ Failed to install {package}: {e}")
    
    # Install packages
    install_package_list(core_packages, "Core Packages")
    install_package_list(ml_packages, "ML Visualization Packages")
    install_package_list(advanced_packages, "Advanced Audio Packages")
    
    print("\\n📦 Dependency installation completed!")

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        print("✅ FFmpeg is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg not found")
        print("   Install FFmpeg for full audio format support:")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/")
        return False

def create_sample_audio():
    """Create sample audio files for testing"""
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        
        print("\\n🎵 Creating sample audio files for testing...")
        
        # Create samples directory
        samples_dir = Path("sample_audio")
        samples_dir.mkdir(exist_ok=True)
        
        # Sample rate
        sr = 22050
        duration = 5  # 5 seconds
        
        # Create different types of audio samples
        samples = {
            "electronic_beat.wav": create_electronic_sample(sr, duration),
            "piano_melody.wav": create_piano_sample(sr, duration),
            "rock_guitar.wav": create_guitar_sample(sr, duration),
            "ambient_pad.wav": create_ambient_sample(sr, duration)
        }
        
        for filename, audio_data in samples.items():
            filepath = samples_dir / filename
            sf.write(filepath, audio_data, sr)
            print(f"   ✅ Created {filename}")
        
        print(f"\\n📁 Sample audio files created in: {samples_dir}")
        return True
        
    except ImportError:
        print("⚠️ Cannot create sample audio - librosa not installed")
        return False
    except Exception as e:
        print(f"❌ Error creating sample audio: {e}")
        return False

def create_electronic_sample(sr, duration):
    """Create electronic/techno style sample"""
    import numpy as np
    
    t = np.linspace(0, duration, int(sr * duration))
    
    # Bass drum pattern (120 BPM)
    bpm = 120
    beat_interval = 60 / bpm
    kick_pattern = np.zeros_like(t)
    
    for beat in np.arange(0, duration, beat_interval):
        start_idx = int(beat * sr)
        end_idx = min(start_idx + int(0.1 * sr), len(kick_pattern))
        if start_idx < len(kick_pattern):
            # Kick drum synthesis
            kick_env = np.exp(-10 * (t[start_idx:end_idx] - beat))
            kick_osc = np.sin(2 * np.pi * 60 * (t[start_idx:end_idx] - beat))
            kick_pattern[start_idx:end_idx] += kick_osc * kick_env
    
    # Synthesizer lead
    freq = 440  # A4
    synth = 0.3 * np.sin(2 * np.pi * freq * t) * np.sin(2 * np.pi * 0.5 * t)
    
    # Hi-hat pattern
    hihat = np.random.normal(0, 0.1, len(t)) * (np.sin(2 * np.pi * 8 * t) > 0.5)
    
    return kick_pattern + synth + hihat * 0.2

def create_piano_sample(sr, duration):
    """Create piano melody sample"""
    import numpy as np
    
    t = np.linspace(0, duration, int(sr * duration))
    
    # Piano notes (C major scale)
    notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4-C5
    melody = np.zeros_like(t)
    
    note_duration = 0.5
    for i, freq in enumerate(notes):
        start_time = i * note_duration
        if start_time >= duration:
            break
            
        start_idx = int(start_time * sr)
        end_idx = min(start_idx + int(note_duration * sr), len(melody))
        
        if start_idx < len(melody):
            note_t = t[start_idx:end_idx] - start_time
            # Piano-like envelope
            envelope = np.exp(-2 * note_t)
            # Harmonic content for piano timbre
            fundamental = np.sin(2 * np.pi * freq * note_t)
            harmonic2 = 0.5 * np.sin(2 * np.pi * 2 * freq * note_t)
            harmonic3 = 0.25 * np.sin(2 * np.pi * 3 * freq * note_t)
            
            piano_note = (fundamental + harmonic2 + harmonic3) * envelope
            melody[start_idx:end_idx] += piano_note
    
    return melody * 0.5

def create_guitar_sample(sr, duration):
    """Create guitar rock sample"""
    import numpy as np
    
    t = np.linspace(0, duration, int(sr * duration))
    
    # Power chord progression (E5, A5, D5)
    chords = [82.41, 110.00, 146.83]  # E2, A2, D3
    guitar = np.zeros_like(t)
    
    chord_duration = duration / len(chords)
    
    for i, root_freq in enumerate(chords):
        start_time = i * chord_duration
        start_idx = int(start_time * sr)
        end_idx = min(start_idx + int(chord_duration * sr), len(guitar))
        
        if start_idx < len(guitar):
            chord_t = t[start_idx:end_idx] - start_time
            
            # Power chord (root + fifth)
            root = np.sin(2 * np.pi * root_freq * chord_t)
            fifth = np.sin(2 * np.pi * root_freq * 1.5 * chord_t)
            
            # Distortion effect (soft clipping)
            chord_signal = 0.7 * (root + 0.7 * fifth)
            distorted = np.tanh(2 * chord_signal)
            
            guitar[start_idx:end_idx] += distorted
    
    return guitar * 0.6

def create_ambient_sample(sr, duration):
    """Create ambient pad sample"""
    import numpy as np
    
    t = np.linspace(0, duration, int(sr * duration))
    
    # Slow evolving pad sound
    freq1 = 55  # A1
    freq2 = 82.41  # E2
    freq3 = 110  # A2
    
    # Slow LFO for movement
    lfo = 0.1 * np.sin(2 * np.pi * 0.2 * t)
    
    # Multiple oscillators for richness
    osc1 = 0.3 * np.sin(2 * np.pi * freq1 * (1 + lfo) * t)
    osc2 = 0.2 * np.sin(2 * np.pi * freq2 * (1 + lfo * 0.5) * t)
    osc3 = 0.1 * np.sin(2 * np.pi * freq3 * (1 + lfo * 0.3) * t)
    
    # Slow attack envelope
    envelope = 1 - np.exp(-0.5 * t)
    
    ambient = (osc1 + osc2 + osc3) * envelope
    
    return ambient * 0.4

def setup_training_structure():
    """Set up training directory structure"""
    print("\\n📁 Setting up ML training structure...")
    
    training_dir = Path("training_data")
    genres = ["rock", "pop", "electronic", "hip-hop", "classical", "jazz", "ambient", "country"]
    
    for genre in genres:
        genre_dir = training_dir / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        
        # Create README
        readme_path = genre_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"# {genre.title()} Training Samples\\n\\n")
            f.write(f"Add {genre} audio files here for ML training.\\n\\n")
            f.write("**Recommended:**\\n")
            f.write("- 10+ high-quality samples per genre\\n")
            f.write("- Diverse artists and sub-styles\\n")
            f.write("- Files under 5 minutes for faster training\\n")
    
    print(f"✅ Training directories created: {len(genres)} genres")

def run_basic_test():
    """Run basic functionality test"""
    print("\\n🧪 Running basic functionality test...")
    
    try:
        # Test import
        from audio_analyzer import IntelligentAudioAnalyzer
        analyzer = IntelligentAudioAnalyzer()
        print("✅ Audio analyzer imported successfully")
        
        # Test with sample audio if available
        samples_dir = Path("sample_audio")
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*.wav"))
            if sample_files:
                print(f"✅ Found {len(sample_files)} sample audio files")
                
                # Test analysis on first sample
                test_file = sample_files[0]
                print(f"🔬 Testing analysis on: {test_file.name}")
                
                result = analyzer.analyze_audio_file(str(test_file))
                print(f"   Duration: {result.duration:.2f}s")
                print(f"   Content Type: {result.content_category or 'Unknown'}")
                print(f"   Energy Level: {result.energy_level:.2f}" if result.energy_level else "   Energy Level: Not calculated")
                
                if result.analysis_errors:
                    print("   ⚠️ Analysis warnings:")
                    for error in result.analysis_errors:
                        print(f"     - {error}")
                else:
                    print("   ✅ Analysis completed without errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("📋 NEXT STEPS:")
    print()
    print("1. 🎵 Test with your audio files:")
    print("   python test_ml_classification.py")
    print()
    print("2. 🎓 Train custom ML models:")
    print("   python train_ml_classifier.py setup")
    print("   # Add audio files to training_data/ directories")
    print("   python train_ml_classifier.py train")
    print()
    print("3. 📊 Analyze single files:")
    print("   python audio_analyzer.py your_song.mp3")
    print()
    print("4. 🔍 Explore sample audio:")
    print("   ls sample_audio/")
    print("   python audio_analyzer.py sample_audio/electronic_beat.wav")
    print()
    print("📚 DOCUMENTATION:")
    print("   - ML_CLASSIFICATION_GUIDE.md - Detailed ML features guide")
    print("   - README.md - General usage instructions")
    print("   - QUICK_START.md - Quick start guide")
    print()
    print("🎯 FEATURES AVAILABLE:")
    print("   ✅ Instrument Recognition")
    print("   ✅ Genre Classification")
    print("   ✅ Mood Detection")
    print("   ✅ Smart Categorization")
    print("   ✅ Audio Similarity Search")
    print("   ✅ Collection Analysis")
    print("   ✅ Smart Playlists")

def main():
    """Main setup function"""
    print("🚀 ML AUDIO CLASSIFICATION SETUP")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check FFmpeg
    check_ffmpeg()
    
    # Install dependencies
    install_dependencies()
    
    # Create sample audio
    create_sample_audio()
    
    # Setup training structure
    setup_training_structure()
    
    # Run basic test
    if run_basic_test():
        print_next_steps()
        return True
    else:
        print("\\n❌ Setup completed with some issues. Check error messages above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)