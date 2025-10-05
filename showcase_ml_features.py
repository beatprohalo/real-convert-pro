#!/usr/bin/env python3
"""
ML Classification Feature Showcase
Demonstrates all the machine learning audio classification capabilities
"""

from audio_analyzer import IntelligentAudioAnalyzer
from pathlib import Path
import json

def showcase_ml_features():
    """Comprehensive showcase of ML classification features"""
    print("🎵" + "="*58 + "🎵")
    print("🤖 MACHINE LEARNING AUDIO CLASSIFICATION SHOWCASE 🤖")
    print("🎵" + "="*58 + "🎵")
    
    analyzer = IntelligentAudioAnalyzer()
    
    # Test all sample files
    sample_files = list(Path("sample_audio").glob("*.wav"))
    
    if not sample_files:
        print("❌ No sample files found. Run: python3 setup_ml_classification.py")
        return
    
    print(f"\\n📁 Testing {len(sample_files)} sample audio files...")
    print("="*60)
    
    all_results = []
    
    for i, audio_file in enumerate(sample_files, 1):
        print(f"\\n🎵 [{i}/{len(sample_files)}] ANALYZING: {audio_file.name}")
        print("-" * 50)
        
        # Perform comprehensive analysis
        result = analyzer.analyze_audio_file(str(audio_file))
        all_results.append(result)
        
        # Display results in an organized manner
        print_analysis_results(result, analyzer)
    
    # Collection insights
    print("\\n" + "="*60)
    print("📊 COLLECTION INSIGHTS")
    print("="*60)
    
    # Generate collection analysis
    playlists = analyzer.create_smart_playlists(all_results)
    display_collection_insights(all_results, playlists)
    
    # Similarity analysis
    print("\\n" + "="*60)
    print("🔍 AUDIO SIMILARITY ANALYSIS")
    print("="*60)
    
    if len(sample_files) >= 2:
        target_file = str(sample_files[0])
        candidate_files = [str(f) for f in sample_files[1:]]
        
        similar_tracks = analyzer.find_similar_tracks(target_file, candidate_files, top_k=3)
        
        print(f"🎯 Reference: {Path(target_file).name}")
        print("📈 Most Similar Tracks:")
        for track, similarity in similar_tracks:
            print(f"   • {Path(track).name}: {similarity:.3f} similarity")
    
    # ML Training Status
    print("\\n" + "="*60)
    print("🎓 ML TRAINING STATUS")
    print("="*60)
    
    models_dir = Path("models")
    if models_dir.exists() and (models_dir / "genre_classifier.pkl").exists():
        print("✅ Custom ML models trained and loaded")
        print(f"   📁 Model directory: {models_dir}")
        print("   🎭 Genre classifier: Available")
        print("   📊 Feature scaler: Available")
    else:
        print("⚠️ No custom ML models found")
        print("   💡 Train custom models with: python3 train_ml_classifier.py train")
    
    print("\\n" + "🎵" + "="*58 + "🎵")
    print("✅ ML CLASSIFICATION SHOWCASE COMPLETED!")
    print("🎵" + "="*58 + "🎵")

def print_analysis_results(result, analyzer):
    """Print formatted analysis results"""
    
    # Basic info
    print(f"⏱️  Duration: {result.duration:.2f}s")
    print(f"📊 Sample Rate: {result.sample_rate} Hz")
    
    # Core analysis
    if result.bpm:
        print(f"🥁 BPM: {result.bpm:.1f} ({result.tempo_category})")
    if result.key and result.scale:
        print(f"🎼 Key: {result.key} {result.scale}")
    
    print()
    
    # ML Classification Results
    print("🤖 ML CLASSIFICATION:")
    
    # Content classification
    if result.content_category:
        print(f"   📂 Content Type: {result.content_category}")
    
    # Instrument recognition
    if result.detected_instruments:
        print(f"   🎸 Instruments: {', '.join(result.detected_instruments)}")
        if result.instrument_confidence:
            print("   📈 Confidence Scores:")
            for instrument, confidence in result.instrument_confidence.items():
                confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
                print(f"      {instrument}: {confidence:.2f} [{confidence_bar}]")
    else:
        print("   🎸 Instruments: None detected")
    
    # Genre classification
    if result.genre_prediction:
        confidence_str = f" ({result.genre_confidence:.2f})" if result.genre_confidence else ""
        print(f"   🎭 Genre: {result.genre_prediction}{confidence_str}")
    else:
        print("   🎭 Genre: Not classified")
    
    # Mood detection
    if result.mood_prediction:
        confidence_str = f" ({result.mood_confidence:.2f})" if result.mood_confidence else ""
        print(f"   😊 Mood: {result.mood_prediction}{confidence_str}")
    else:
        print("   😊 Mood: Not determined")
    
    print()
    
    # Audio characteristics
    print("🎶 AUDIO CHARACTERISTICS:")
    if result.energy_level is not None:
        energy_bar = "█" * int(result.energy_level * 20) + "░" * (20 - int(result.energy_level * 20))
        print(f"   ⚡ Energy: {result.energy_level:.2f} [{energy_bar}]")
    
    if result.valence is not None:
        valence_bar = "█" * int(result.valence * 20) + "░" * (20 - int(result.valence * 20))
        print(f"   😄 Valence: {result.valence:.2f} [{valence_bar}]")
    
    if result.danceability is not None:
        dance_bar = "█" * int(result.danceability * 20) + "░" * (20 - int(result.danceability * 20))
        print(f"   💃 Danceability: {result.danceability:.2f} [{dance_bar}]")
    
    # Smart categorization
    category = analyzer.categorize_by_analysis(result)
    print(f"\\n🏷️  Smart Category: {category}")

def display_collection_insights(results, playlists):
    """Display collection-wide insights"""
    
    total_duration = sum(r.duration for r in results if r.duration)
    avg_bpm = sum(r.bpm for r in results if r.bpm) / len([r for r in results if r.bpm]) if any(r.bpm for r in results) else 0
    
    print(f"📁 Total Files: {len(results)}")
    print(f"⏱️  Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"🥁 Average BPM: {avg_bpm:.1f}")
    
    # Genre distribution
    genres = [r.genre_prediction for r in results if r.genre_prediction]
    if genres:
        print(f"\\n🎭 Genre Distribution:")
        genre_counts = {}
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {genre}: {count} tracks")
    
    # Mood distribution
    moods = [r.mood_prediction for r in results if r.mood_prediction]
    if moods:
        print(f"\\n😊 Mood Distribution:")
        mood_counts = {}
        for mood in moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        for mood, count in sorted(mood_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {mood}: {count} tracks")
    
    # Instrument frequency
    all_instruments = []
    for result in results:
        if result.detected_instruments:
            all_instruments.extend(result.detected_instruments)
    
    if all_instruments:
        print(f"\\n🎸 Instrument Frequency:")
        instrument_counts = {}
        for instrument in all_instruments:
            instrument_counts[instrument] = instrument_counts.get(instrument, 0) + 1
        for instrument, count in sorted(instrument_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            print(f"   {instrument}: {count} tracks ({percentage:.1f}%)")
    
    # Smart playlists
    if playlists:
        print(f"\\n🎵 Generated Smart Playlists:")
        for playlist_name, tracks in playlists.items():
            print(f"   {playlist_name}: {len(tracks)} tracks")

if __name__ == "__main__":
    try:
        showcase_ml_features()
    except KeyboardInterrupt:
        print("\\n❌ Showcase interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error during showcase: {e}")
        import traceback
        traceback.print_exc()