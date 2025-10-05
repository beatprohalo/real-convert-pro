#!/usr/bin/env python3
"""
Machine Learning Audio Classification Demo
Tests the new ML-based classification features including:
- Instrument recognition
- Genre classification
- Mood detection
- Smart categorization
"""

import os
import sys
import json
from pathlib import Path
from audio_analyzer import IntelligentAudioAnalyzer

def test_ml_classification():
    """Test ML-based classification features"""
    print("=" * 60)
    print("🎵 MACHINE LEARNING AUDIO CLASSIFICATION TEST")
    print("=" * 60)
    
    analyzer = IntelligentAudioAnalyzer()
    
    # Test with any audio file in the current directory
    audio_files = []
    for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']:
        audio_files.extend(Path('.').glob(f'*{ext}'))
        audio_files.extend(Path('.').glob(f'**/*{ext}'))
    
    if not audio_files:
        print("❌ No audio files found for testing.")
        print("Please add some audio files to test the ML classification features.")
        return
    
    # Test file
    test_file = str(audio_files[0])
    print(f"📁 Testing with: {test_file}")
    print("-" * 60)
    
    # Perform comprehensive analysis
    result = analyzer.analyze_audio_file(test_file)
    
    # Display basic info
    print("📊 BASIC ANALYSIS:")
    print(f"   Duration: {result.duration:.2f} seconds")
    print(f"   Sample Rate: {result.sample_rate} Hz")
    print(f"   BPM: {result.bpm:.1f}" if result.bpm else "   BPM: Not detected")
    print(f"   Key: {result.key} {result.scale}" if result.key else "   Key: Not detected")
    print()
    
    # Display ML classification results
    print("🤖 MACHINE LEARNING CLASSIFICATION:")
    
    # Content Type
    if result.content_category:
        print(f"   📂 Content Type: {result.content_category}")
    
    # Instrument Recognition
    if result.detected_instruments:
        print(f"   🎸 Detected Instruments: {', '.join(result.detected_instruments)}")
        if result.instrument_confidence:
            print("   📈 Instrument Confidence:")
            for instrument, confidence in result.instrument_confidence.items():
                print(f"      {instrument}: {confidence:.2f}")
    else:
        print("   🎸 Detected Instruments: None identified")
    
    # Genre Classification
    if result.genre_prediction:
        confidence_str = f" (confidence: {result.genre_confidence:.2f})" if result.genre_confidence else ""
        print(f"   🎭 Genre Prediction: {result.genre_prediction}{confidence_str}")
    else:
        print("   🎭 Genre Prediction: Not determined")
    
    # Mood Detection
    if result.mood_prediction:
        confidence_str = f" (confidence: {result.mood_confidence:.2f})" if result.mood_confidence else ""
        print(f"   😊 Mood Prediction: {result.mood_prediction}{confidence_str}")
    else:
        print("   😊 Mood Prediction: Not determined")
    print()
    
    # Audio Characteristics
    print("🎶 AUDIO CHARACTERISTICS:")
    if result.energy_level is not None:
        energy_desc = get_energy_description(result.energy_level)
        print(f"   ⚡ Energy Level: {result.energy_level:.2f} ({energy_desc})")
    
    if result.valence is not None:
        valence_desc = get_valence_description(result.valence)
        print(f"   😄 Valence (Positivity): {result.valence:.2f} ({valence_desc})")
    
    if result.danceability is not None:
        dance_desc = get_danceability_description(result.danceability)
        print(f"   💃 Danceability: {result.danceability:.2f} ({dance_desc})")
    print()
    
    # Smart Categorization
    print("🏷️  SMART CATEGORIZATION:")
    category = analyzer.categorize_by_analysis(result)
    print(f"   📂 Suggested Category: {category}")
    
    # Playlist recommendations
    print(f"   🎵 Playlist Recommendations:")
    playlists = analyzer.create_smart_playlists([result])
    for playlist_name, tracks in playlists.items():
        if tracks:  # Only show playlists this track fits in
            print(f"      ✓ {playlist_name}")
    print()
    
    # Technical Features
    print("🔬 TECHNICAL FEATURES:")
    if result.spectral_centroid:
        print(f"   📊 Spectral Centroid: {result.spectral_centroid:.1f} Hz")
    if result.spectral_rolloff:
        print(f"   📈 Spectral Rolloff: {result.spectral_rolloff:.1f} Hz")
    if result.zero_crossing_rate:
        print(f"   〰️  Zero Crossing Rate: {result.zero_crossing_rate:.4f}")
    if result.rhythmic_complexity:
        print(f"   🥁 Rhythmic Complexity: {result.rhythmic_complexity:.3f}")
    print()
    
    # Error reporting
    if result.analysis_errors:
        print("⚠️  ANALYSIS WARNINGS:")
        for error in result.analysis_errors:
            print(f"   • {error}")
        print()
    
    print("=" * 60)
    print("✅ ML CLASSIFICATION TEST COMPLETED")
    print("=" * 60)

def get_energy_description(energy_level):
    """Get human-readable energy level description"""
    if energy_level < 0.2:
        return "Very Low"
    elif energy_level < 0.4:
        return "Low"
    elif energy_level < 0.6:
        return "Moderate"
    elif energy_level < 0.8:
        return "High"
    else:
        return "Very High"

def get_valence_description(valence):
    """Get human-readable valence description"""
    if valence < 0.2:
        return "Very Negative"
    elif valence < 0.4:
        return "Negative"
    elif valence < 0.6:
        return "Neutral"
    elif valence < 0.8:
        return "Positive"
    else:
        return "Very Positive"

def get_danceability_description(danceability):
    """Get human-readable danceability description"""
    if danceability < 0.2:
        return "Not Danceable"
    elif danceability < 0.4:
        return "Slightly Danceable"
    elif danceability < 0.6:
        return "Moderately Danceable"
    elif danceability < 0.8:
        return "Very Danceable"
    else:
        return "Extremely Danceable"

def test_collection_analysis():
    """Test collection-wide analysis"""
    print("\n" + "=" * 60)
    print("📚 COLLECTION ANALYSIS TEST")
    print("=" * 60)
    
    analyzer = IntelligentAudioAnalyzer()
    
    # Analyze current directory
    insights = analyzer.analyze_music_collection('.')
    
    if insights:
        print("📊 COLLECTION INSIGHTS:")
        print(f"   📁 Total Files: {insights.get('total_files', 0)}")
        print(f"   ⏱️  Total Duration: {insights.get('total_duration', 0):.1f} seconds")
        print(f"   🎵 Average BPM: {insights.get('avg_bpm', 0):.1f}")
        print()
        
        # Genre distribution
        if insights.get('genre_distribution'):
            print("🎭 GENRE DISTRIBUTION:")
            for genre, count in sorted(insights['genre_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {genre}: {count} tracks")
            print()
        
        # Mood distribution
        if insights.get('mood_distribution'):
            print("😊 MOOD DISTRIBUTION:")
            for mood, count in sorted(insights['mood_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {mood}: {count} tracks")
            print()
        
        # Instrument distribution
        if insights.get('instrument_distribution'):
            print("🎸 INSTRUMENT DISTRIBUTION:")
            for instrument, count in sorted(insights['instrument_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {instrument}: {count} tracks")
            print()
    else:
        print("❌ No collection insights available.")

def test_similarity_search():
    """Test audio similarity search"""
    print("\n" + "=" * 60)
    print("🔍 SIMILARITY SEARCH TEST")
    print("=" * 60)
    
    analyzer = IntelligentAudioAnalyzer()
    
    # Find audio files
    audio_files = []
    for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']:
        audio_files.extend(Path('.').glob(f'*{ext}'))
        audio_files.extend(Path('.').glob(f'**/*{ext}'))
    
    if len(audio_files) < 2:
        print("❌ Need at least 2 audio files for similarity testing.")
        return
    
    # Use first file as target
    target_file = str(audio_files[0])
    candidate_files = [str(f) for f in audio_files[1:]]
    
    print(f"🎯 Target File: {target_file}")
    print(f"🔍 Searching among {len(candidate_files)} candidates...")
    
    # Find similar tracks
    similar_tracks = analyzer.find_similar_tracks(target_file, candidate_files, top_k=3)
    
    if similar_tracks:
        print("\n📈 MOST SIMILAR TRACKS:")
        for i, (track, similarity) in enumerate(similar_tracks, 1):
            print(f"   {i}. {Path(track).name} (similarity: {similarity:.3f})")
    else:
        print("❌ No similar tracks found.")

if __name__ == "__main__":
    try:
        # Test ML classification
        test_ml_classification()
        
        # Test collection analysis
        test_collection_analysis()
        
        # Test similarity search
        test_similarity_search()
        
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()