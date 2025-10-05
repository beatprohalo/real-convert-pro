# Machine Learning Audio Classification Features

## Overview

The audio analyzer now includes advanced machine learning capabilities for intelligent audio content analysis. These features go beyond basic metadata analysis to understand the actual audio content using AI.

## 🎯 Features

### 1. **Smart Content Classification**
- **Content Type Detection**: Automatically identifies whether audio contains music, speech, noise, or ambient sounds
- **Audio Embeddings**: Creates high-dimensional feature vectors for similarity analysis
- **Rule-based + ML Classification**: Combines expert knowledge with machine learning

### 2. **Instrument Recognition**
- **Multi-instrument Detection**: Identifies multiple instruments playing simultaneously
- **Confidence Scores**: Provides confidence levels for each detected instrument
- **Supported Instruments**:
  - Drums/Percussion
  - Piano/Keyboards
  - Guitar (acoustic/electric)
  - Bass
  - Strings/Violin
  - Vocals
  - Synthesizers

### 3. **Genre Classification**
- **13+ Genre Categories**: Rock, Pop, Electronic, Hip-hop, Classical, Jazz, Ambient, Country, Metal, etc.
- **Feature-based Analysis**: Uses spectral, rhythmic, and harmonic features
- **Trainable Models**: Can be trained on custom datasets
- **Confidence Scoring**: Provides prediction confidence

### 4. **Advanced Mood Detection**
- **Multi-dimensional Mood Analysis**:
  - Energy-based: energetic, calm, intense, peaceful
  - Valence-based: happy, sad, uplifting, melancholic
  - Complex moods: aggressive, contemplative, atmospheric, dramatic
- **Musical Context**: Considers key, tempo, and instrumentation
- **Confidence Metrics**: Provides mood certainty scores

## 🚀 Quick Start

### Basic ML Analysis

```python
from audio_analyzer import IntelligentAudioAnalyzer

# Initialize analyzer (automatically loads trained models if available)
analyzer = IntelligentAudioAnalyzer()

# Analyze audio file
result = analyzer.analyze_audio_file('song.mp3')

# Access ML results
print(f"Genre: {result.genre_prediction} ({result.genre_confidence:.2f})")
print(f"Mood: {result.mood_prediction} ({result.mood_confidence:.2f})")
print(f"Instruments: {result.detected_instruments}")
print(f"Content Type: {result.content_category}")
```

### Audio Features

```python
# Energy and mood characteristics
print(f"Energy Level: {result.energy_level:.2f}")
print(f"Valence (Positivity): {result.valence:.2f}")
print(f"Danceability: {result.danceability:.2f}")

# Technical audio features
print(f"Spectral Centroid: {result.spectral_centroid:.1f} Hz")
print(f"Rhythmic Complexity: {result.rhythmic_complexity:.3f}")
```

## 🧠 Machine Learning Models

### Training Your Own Classifier

1. **Prepare Training Data**:
   ```bash
   python train_ml_classifier.py setup
   ```

2. **Organize Audio Files**:
   ```
   training_data/
   ├── rock/
   │   ├── song1.mp3
   │   └── song2.wav
   ├── pop/
   │   ├── track1.mp3
   │   └── track2.flac
   └── electronic/
       ├── beat1.wav
       └── beat2.mp3
   ```

3. **Train the Model**:
   ```bash
   python train_ml_classifier.py train
   ```

4. **Check Training Progress**:
   ```bash
   python train_ml_classifier.py analyze
   ```

### Model Architecture

- **Feature Extraction**: 100+ audio features including:
  - Spectral features (centroid, rolloff, bandwidth, flatness)
  - MFCC coefficients (1-20)
  - Chroma features (12 pitch classes)
  - Tonnetz features (6 harmonic components)
  - Rhythm and tempo features

- **Classifier**: Random Forest with 100 estimators
- **Preprocessing**: StandardScaler for feature normalization
- **Storage**: Models saved as `.pkl` files for persistence

## 🔍 Advanced Features

### Audio Similarity Search

```python
# Find similar tracks
similar_tracks = analyzer.find_similar_tracks(
    target_file='reference_song.mp3',
    candidate_files=['song1.mp3', 'song2.mp3', 'song3.mp3'],
    top_k=5
)

for track, similarity in similar_tracks:
    print(f"{track}: {similarity:.3f}")
```

### Collection Analysis

```python
# Analyze entire music library
insights = analyzer.analyze_music_collection('/path/to/music/')

print(f"Total files: {insights['total_files']}")
print(f"Genre distribution: {insights['genre_distribution']}")
print(f"Mood distribution: {insights['mood_distribution']}")
print(f"Instrument distribution: {insights['instrument_distribution']}")
```

### Smart Playlists

```python
# Create intelligent playlists
results = [analyzer.analyze_audio_file(f) for f in audio_files]
playlists = analyzer.create_smart_playlists(results)

for playlist_name, tracks in playlists.items():
    print(f"{playlist_name}: {len(tracks)} tracks")
```

## 📊 Testing and Validation

### Run ML Classification Tests

```bash
python test_ml_classification.py
```

This will test:
- ✅ Instrument recognition accuracy
- ✅ Genre classification performance
- ✅ Mood detection reliability
- ✅ Audio similarity matching
- ✅ Collection analysis features

### Expected Output

```
🎵 MACHINE LEARNING AUDIO CLASSIFICATION TEST
============================================================
📁 Testing with: example_song.mp3
------------------------------------------------------------
📊 BASIC ANALYSIS:
   Duration: 185.23 seconds
   Sample Rate: 44100 Hz
   BPM: 128.5
   Key: G major

🤖 MACHINE LEARNING CLASSIFICATION:
   📂 Content Type: music
   🎸 Detected Instruments: guitar, drums, vocals
   📈 Instrument Confidence:
      guitar: 0.85
      drums: 0.92
      vocals: 0.73
   🎭 Genre Prediction: rock (confidence: 0.78)
   😊 Mood Prediction: energetic (confidence: 0.82)

🎶 AUDIO CHARACTERISTICS:
   ⚡ Energy Level: 0.68 (Moderate)
   😄 Valence (Positivity): 0.72 (Positive)
   💃 Danceability: 0.54 (Moderately Danceable)

🏷️ SMART CATEGORIZATION:
   📂 Suggested Category: Rock & Alternative
   🎵 Playlist Recommendations:
      ✓ High Energy
      ✓ Workout
      ✓ Happy Vibes
```

## 🎛️ Configuration

### Feature Selection

Customize which ML features to use:

```python
# In the analyzer initialization
analyzer = IntelligentAudioAnalyzer()

# Configure feature extraction
analyzer.ml_config = {
    'use_mfcc': True,
    'mfcc_coefficients': 20,
    'use_chroma': True,
    'use_tonnetz': True,
    'use_spectral_features': True,
    'embedding_dimensions': 100
}
```

### Model Parameters

Adjust classification sensitivity:

```python
# Minimum confidence thresholds
analyzer.confidence_thresholds = {
    'genre': 0.6,
    'mood': 0.5,
    'instrument': 0.4
}
```

## 🔧 Technical Details

### Dependencies

Core ML libraries:
- `scikit-learn` - Machine learning algorithms
- `tensorflow` - Deep learning framework (optional)
- `joblib` - Model serialization
- `pandas` - Data manipulation
- `scipy` - Scientific computing

Audio processing:
- `librosa` - Audio analysis
- `essentia` - Advanced audio features
- `numpy` - Numerical computing

### Performance

- **Analysis Speed**: ~2-5 seconds per song (depending on length)
- **Memory Usage**: ~100MB for feature extraction
- **Model Size**: ~5MB for trained classifier
- **Accuracy**: 70-85% on diverse datasets (genre-dependent)

### File Support

Supports all audio formats handled by librosa:
- WAV, FLAC (lossless)
- MP3, AAC, OGG (compressed)
- M4A, WMA (with ffmpeg)

## 🚀 Integration Examples

### Batch Processing

```python
import os
from pathlib import Path

analyzer = IntelligentAudioAnalyzer()
results = []

# Process all audio files in directory
audio_dir = Path('/path/to/music')
for audio_file in audio_dir.rglob('*.mp3'):
    try:
        result = analyzer.analyze_audio_file(str(audio_file))
        results.append(result)
        print(f"Processed: {audio_file.name}")
    except Exception as e:
        print(f"Error with {audio_file}: {e}")

# Generate collection insights
insights = analyzer.analyze_music_collection(str(audio_dir))
```

### Music Database Integration

```python
import sqlite3
import json

# Store analysis results in database
def save_to_database(results):
    conn = sqlite3.connect('music_analysis.db')
    
    for result in results:
        conn.execute('''
            INSERT INTO tracks (filename, genre, mood, instruments, 
                              energy, valence, bpm, key_signature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.filename,
            result.genre_prediction,
            result.mood_prediction,
            json.dumps(result.detected_instruments),
            result.energy_level,
            result.valence,
            result.bpm,
            f"{result.key} {result.scale}"
        ))
    
    conn.commit()
    conn.close()
```

## 🎯 Use Cases

1. **Music Library Organization**: Automatically categorize and tag music collections
2. **Playlist Generation**: Create mood-based or activity-based playlists
3. **Music Recommendation**: Find similar tracks based on audio content
4. **Content Analysis**: Analyze podcast or broadcast content
5. **Music Production**: Identify missing instruments or suggest complementary tracks
6. **Research**: Study musical trends and patterns in large datasets

## 🔬 Accuracy and Limitations

### Strengths
- ✅ Excellent at detecting clear instrument signatures
- ✅ Reliable tempo and rhythm analysis
- ✅ Good performance on mainstream genres
- ✅ Robust spectral and harmonic feature extraction

### Limitations
- ⚠️ Genre boundaries can be subjective
- ⚠️ Complex/fusion genres may be misclassified
- ⚠️ Performance varies with audio quality
- ⚠️ Training data bias affects accuracy
- ⚠️ Short audio clips may lack sufficient features

### Improving Accuracy

1. **More Training Data**: Add diverse examples for each genre
2. **Quality Control**: Use high-quality, representative samples
3. **Feature Engineering**: Customize features for specific use cases
4. **Ensemble Methods**: Combine multiple classification approaches
5. **Human Validation**: Review and correct misclassifications

## 📈 Future Enhancements

- **Deep Learning Models**: CNN/RNN for advanced pattern recognition
- **Real-time Analysis**: Stream processing capabilities
- **Multi-label Classification**: Support for genre fusion and subgenres
- **Emotional Granularity**: More detailed mood categorization
- **Cultural Context**: Region-specific classification models
- **Audio Quality Assessment**: Automatic quality scoring

---

*For more information, see the main README.md and test the features using `test_ml_classification.py`*