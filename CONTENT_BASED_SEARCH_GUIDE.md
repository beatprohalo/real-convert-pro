# Content-Based Audio Search Guide

## Overview

The Content-Based Audio Search Engine provides advanced search capabilities for audio files based on their musical and acoustic characteristics rather than just metadata. This enables finding similar-sounding audio, harmonically compatible tracks, rhythmically matching samples, and frequency-based searches.

## Features

### 🎵 Audio Similarity Search
- **Purpose**: Find samples that sound alike
- **Use Cases**: Discovering similar tracks, finding alternatives, grouping by timbre
- **Technology**: Uses MFCC features, spectral contrast, and chroma analysis

### 🎼 Harmonic Matching
- **Purpose**: Find samples in compatible keys
- **Use Cases**: DJ mixing, creating harmonic progressions, key-based organization
- **Technology**: Chroma feature analysis and key detection

### 🥁 Rhythm Matching
- **Purpose**: Find samples with similar groove
- **Use Cases**: Beatmatching, finding drums with similar patterns, tempo organization
- **Technology**: Onset detection, tempo analysis, beat strength measurement

### 🔊 Spectral Search
- **Purpose**: Search by frequency content
- **Use Cases**: Finding sounds with similar frequency profiles, EQ matching
- **Technology**: Spectral analysis, frequency distribution, spectral centroid

## Quick Start

### 1. Basic Analysis
```python
from audio_analyzer import ContentBasedSearchEngine

# Initialize search engine
search_engine = ContentBasedSearchEngine()

# Add a single file to the database
search_engine.add_to_database("path/to/audio/file.wav")
```

### 2. Build Search Database
```python
# Analyze all files in a directory
search_engine.batch_analyze_directory("sample_audio")

# Analyze multiple directories
directories = ["samples", "loops", "tracks"]
for directory in directories:
    search_engine.batch_analyze_directory(directory)
```

### 3. Perform Searches
```python
query_file = "path/to/query/file.wav"

# Find similar audio files
similar = search_engine.find_similar_audio(
    query_file, 
    similarity_threshold=0.7, 
    limit=5
)

# Find harmonically compatible files
harmonic = search_engine.find_harmonic_matches(
    query_file, 
    key_tolerance=2, 
    limit=5
)

# Find rhythmically similar files
rhythm = search_engine.find_rhythm_matches(
    query_file, 
    tempo_tolerance=10.0, 
    limit=5
)

# Find spectrally similar files
spectral = search_engine.find_spectral_matches(
    query_file, 
    spectral_tolerance=0.5, 
    limit=5
)
```

### 4. Comprehensive Search
```python
# Search using all methods
all_results = search_engine.search_by_example(query_file)

for search_type, matches in all_results.items():
    print(f"{search_type.upper()} results:")
    for file_path, score in matches:
        print(f"  {file_path}: {score:.3f}")
```

## Command Line Usage

### Demo Content-Based Search
```bash
python audio_analyzer.py --demo-search
```

### Analyze Single File
```bash
python audio_analyzer.py audio_file.wav
```

### Run Comprehensive Tests
```bash
python test_content_search.py
```

### View Workflow Examples
```bash
python content_search_examples.py
```

## Search Parameters

### Audio Similarity Search
- **similarity_threshold** (0.0-1.0): Minimum similarity score
  - 0.8+: Very similar sounds
  - 0.6-0.8: Moderately similar
  - 0.4-0.6: Somewhat similar
  - <0.4: Different sounds

### Harmonic Matching
- **key_tolerance** (integer): Maximum semitone distance
  - 0: Exact same key only
  - 1-2: Very close keys (perfect harmony)
  - 3-5: Compatible keys (good harmony)
  - 6+: Distant keys (may clash)

### Rhythm Matching
- **tempo_tolerance** (float): Maximum BPM difference
  - 0-5: Very tight tempo matching
  - 5-15: Good tempo matching
  - 15-30: Loose tempo matching
  - 30+: Very loose matching

### Spectral Search
- **spectral_tolerance** (0.0-1.0): Frequency content similarity
  - 0.8+: Very similar frequency content
  - 0.6-0.8: Similar frequency profile
  - 0.4-0.6: Somewhat similar spectrum
  - <0.4: Different frequency content

## Advanced Usage

### Custom Search Criteria
```python
# Combine multiple search types
results = {}

# High-quality similarity
results['high_sim'] = search_engine.find_similar_audio(
    query_file, similarity_threshold=0.8, limit=3
)

# Perfect key matches only
results['perfect_key'] = search_engine.find_harmonic_matches(
    query_file, key_tolerance=0, limit=3
)

# Exact tempo matches
results['exact_tempo'] = search_engine.find_rhythm_matches(
    query_file, tempo_tolerance=1.0, limit=3
)
```

### Database Management
```python
# Force re-analysis of existing files
search_engine.add_to_database("file.wav", force_reanalyze=True)

# Save database manually
search_engine.save_database()

# Check database size
print(f"Database contains {len(search_engine.audio_database)} files")
```

### Feature Extraction Details
```python
# Access detailed features from analysis
analyzer = IntelligentAudioAnalyzer()
result = analyzer.analyze_audio_file("file.wav")

# Content-based features
if hasattr(result, 'chroma_features'):
    print(f"Chroma features: {result.chroma_features.shape}")
if hasattr(result, 'rhythm_features'):
    print(f"Rhythm features: {result.rhythm_features.shape}")
if hasattr(result, 'spectral_features'):
    print(f"Spectral features: {result.spectral_features.shape}")
```

## Workflow Examples

### DJ Workflow
```python
# Build database from music library
search_engine.batch_analyze_directory("music_library")

# Currently playing track
current_track = "current_song.mp3"

# Find tracks in compatible keys (for smooth transitions)
compatible_keys = search_engine.find_harmonic_matches(
    current_track, key_tolerance=2, limit=5
)

# Find tracks with similar tempo (for beatmatching)
similar_tempo = search_engine.find_rhythm_matches(
    current_track, tempo_tolerance=5.0, limit=5
)
```

### Producer Workflow
```python
# Analyze sample library
search_engine.batch_analyze_directory("samples")

# Reference sound
reference = "inspiration.wav"

# Find samples with similar frequency content
spectral_matches = search_engine.find_spectral_matches(
    reference, spectral_tolerance=0.4, limit=10
)

# Find overall similar samples
similar_samples = search_engine.find_similar_audio(
    reference, similarity_threshold=0.3, limit=10
)
```

### Music Organization
```python
# Analyze entire music library
search_engine.batch_analyze_directory("music")

# Create smart playlists based on a seed track
seed_track = "favorite_song.mp3"

# Chill playlist (similar vibes)
chill_playlist = search_engine.find_similar_audio(
    seed_track, similarity_threshold=0.5, limit=20
)

# Harmonic journey (key-compatible tracks)
harmonic_playlist = search_engine.find_harmonic_matches(
    seed_track, key_tolerance=3, limit=20
)
```

## Performance Tips

### Database Size
- Larger databases provide more search options but slower analysis
- Consider separate databases for different music styles/genres
- Regularly clean up duplicate entries

### Search Efficiency
- Start with stricter thresholds and relax if needed
- Use smaller `limit` values for faster results
- Cache frequently used search results

### Memory Usage
- The database stores feature vectors in memory
- For very large libraries, consider batch processing
- Save database regularly to avoid re-analysis

## Technical Details

### Feature Extraction
1. **Audio Similarity**: MFCC coefficients, spectral contrast, chroma features
2. **Harmonic Matching**: Chromagram analysis, key detection
3. **Rhythm Matching**: Onset detection, tempo histogram, beat strength
4. **Spectral Search**: Frequency distribution, spectral centroid, zero-crossing rate

### Similarity Calculations
- **Cosine Similarity**: For comparing feature vectors
- **Euclidean Distance**: For tempo and key differences
- **Weighted Scoring**: Combines multiple features with different weights

### Database Format
- JSON file with extracted features
- Excludes large numpy arrays for portability
- Automatically loads/saves on initialization/updates

## Troubleshooting

### Common Issues

1. **No results found**: Try lowering thresholds
2. **Too many results**: Increase thresholds or use stricter parameters
3. **Analysis errors**: Check audio file format and quality
4. **Slow performance**: Use smaller databases or batch processing

### Dependencies
- **Required**: librosa, numpy, soundfile
- **Optional**: chromaprint (better fingerprinting), essentia (advanced analysis)

### File Formats
- **Supported**: WAV, FLAC, MP3, AIFF, M4A
- **Recommended**: Uncompressed formats (WAV, FLAC) for best analysis quality

## Examples and Testing

Run the included examples to see the search engine in action:

```bash
# Basic functionality test
python test_content_search.py

# Practical workflow examples
python content_search_examples.py

# Built-in demo
python audio_analyzer.py --demo-search
```

These examples demonstrate real-world applications and help you understand the capabilities and limitations of each search type.