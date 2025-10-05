#!/usr/bin/env python3
"""
Content-Based Audio Search Examples
Practical examples of using the content-based search engine
"""

import os
from audio_analyzer import ContentBasedSearchEngine

def example_dj_workflow():
    """Example: DJ looking for compatible tracks"""
    print("🎧 DJ WORKFLOW EXAMPLE")
    print("=" * 40)
    
    search_engine = ContentBasedSearchEngine()
    
    # Build database from sample tracks
    if os.path.exists("sample_audio"):
        print("Building music database...")
        search_engine.batch_analyze_directory("sample_audio")
        
        # Example: Find tracks that mix well together
        sample_files = [f for f in os.listdir("sample_audio") 
                       if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        
        if sample_files:
            current_track = os.path.join("sample_audio", sample_files[0])
            print(f"Currently playing: {sample_files[0]}")
            
            # Find harmonically compatible tracks (for smooth key transitions)
            print("\n🎵 Tracks in compatible keys:")
            harmonic_matches = search_engine.find_harmonic_matches(
                current_track, key_tolerance=2, limit=3
            )
            for track, score in harmonic_matches:
                print(f"  • {os.path.basename(track)} (compatibility: {score:.2f})")
            
            # Find tracks with similar tempo (for beatmatching)
            print("\n🥁 Tracks with similar tempo:")
            rhythm_matches = search_engine.find_rhythm_matches(
                current_track, tempo_tolerance=5.0, limit=3
            )
            for track, score in rhythm_matches:
                print(f"  • {os.path.basename(track)} (rhythm similarity: {score:.2f})")

def example_producer_workflow():
    """Example: Producer looking for sample inspiration"""
    print("\n🎹 PRODUCER WORKFLOW EXAMPLE")
    print("=" * 40)
    
    search_engine = ContentBasedSearchEngine()
    
    # Build database from training samples
    if os.path.exists("training_data"):
        print("Analyzing sample library...")
        search_engine.batch_analyze_directory("training_data")
        
        # Example: Find samples with similar sonic characteristics
        if os.path.exists("sample_audio"):
            sample_files = [f for f in os.listdir("sample_audio") 
                           if f.lower().endswith(('.wav', '.mp3', '.flac'))]
            
            if sample_files:
                reference_track = os.path.join("sample_audio", sample_files[0])
                print(f"Reference sound: {sample_files[0]}")
                
                # Find samples with similar frequency content
                print("\n🔊 Samples with similar frequency content:")
                spectral_matches = search_engine.find_spectral_matches(
                    reference_track, spectral_tolerance=0.4, limit=3
                )
                for track, score in spectral_matches:
                    print(f"  • {os.path.basename(track)} (spectral similarity: {score:.2f})")
                
                # Find samples that sound similar overall
                print("\n🎶 Similar sounding samples:")
                similar_tracks = search_engine.find_similar_audio(
                    reference_track, similarity_threshold=0.4, limit=3
                )
                for track, score in similar_tracks:
                    print(f"  • {os.path.basename(track)} (overall similarity: {score:.2f})")

def example_music_organization():
    """Example: Organizing a music library by content"""
    print("\n📚 MUSIC LIBRARY ORGANIZATION EXAMPLE")
    print("=" * 50)
    
    search_engine = ContentBasedSearchEngine()
    
    # Analyze all available audio
    directories = ["sample_audio", "training_data"]
    total_files = 0
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"Cataloging {directory}...")
            count = search_engine.batch_analyze_directory(directory)
            total_files += count
    
    print(f"Total files analyzed: {total_files}")
    
    if total_files > 0:
        # Group files by similarity
        print("\n🎯 Creating smart playlists based on audio content:")
        
        # Get all files in database
        all_files = list(search_engine.audio_database.keys())
        
        if len(all_files) >= 2:
            # Example: Create a "chill" playlist based on first ambient track
            ambient_files = [f for f in all_files if 'ambient' in os.path.basename(f).lower()]
            
            if ambient_files:
                reference_file = ambient_files[0]
                print(f"\n🌙 'Chill Vibes' playlist (based on {os.path.basename(reference_file)}):")
                
                similar_tracks = search_engine.find_similar_audio(
                    reference_file, similarity_threshold=0.3, limit=5
                )
                
                for i, (track, score) in enumerate(similar_tracks, 1):
                    print(f"  {i}. {os.path.basename(track)} (match: {score:.2f})")
            
            # Example: Create playlist of harmonically compatible tracks
            if len(all_files) >= 1:
                reference_file = all_files[0]
                print(f"\n🎼 'Harmonic Journey' playlist (key-compatible with {os.path.basename(reference_file)}):")
                
                harmonic_tracks = search_engine.find_harmonic_matches(
                    reference_file, key_tolerance=3, limit=5
                )
                
                for i, (track, score) in enumerate(harmonic_tracks, 1):
                    print(f"  {i}. {os.path.basename(track)} (harmony: {score:.2f})")

def example_sound_replacement():
    """Example: Finding replacement sounds for missing samples"""
    print("\n🔄 SOUND REPLACEMENT EXAMPLE")
    print("=" * 40)
    
    search_engine = ContentBasedSearchEngine()
    
    # Build comprehensive database
    if os.path.exists("sample_audio"):
        search_engine.batch_analyze_directory("sample_audio")
        
        sample_files = [f for f in os.listdir("sample_audio") 
                       if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        
        if len(sample_files) >= 2:
            # Simulate a missing sample scenario
            missing_sample = os.path.join("sample_audio", sample_files[0])
            print(f"🚨 Missing sample: {sample_files[0]}")
            print("Finding suitable replacements...")
            
            # Find the best alternatives using multiple criteria
            results = search_engine.search_by_example(missing_sample)
            
            print("\n📋 Replacement candidates:")
            
            # Combine and score all results
            all_candidates = {}
            
            for search_type, matches in results.items():
                for track, score in matches:
                    if track != missing_sample:  # Exclude the original
                        if track not in all_candidates:
                            all_candidates[track] = {}
                        all_candidates[track][search_type] = score
            
            # Score candidates by how many search types found them
            for track, scores in all_candidates.items():
                total_score = sum(scores.values()) / len(scores)
                match_types = len(scores)
                
                print(f"  • {os.path.basename(track)}")
                print(f"    Overall score: {total_score:.3f} (found in {match_types} search types)")
                
                for search_type, score in scores.items():
                    print(f"    - {search_type}: {score:.3f}")
                print()

def example_genre_clustering():
    """Example: Automatically group tracks by genre/style"""
    print("\n🎨 GENRE/STYLE CLUSTERING EXAMPLE")
    print("=" * 40)
    
    search_engine = ContentBasedSearchEngine()
    
    # Analyze training data which has genre folders
    if os.path.exists("training_data"):
        search_engine.batch_analyze_directory("training_data")
        
        print("🔍 Analyzing musical similarity patterns...")
        
        # Get a representative from each genre folder
        genre_representatives = {}
        
        for root, dirs, files in os.walk("training_data"):
            if files:  # Has audio files
                genre_name = os.path.basename(root)
                audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac'))]
                
                if audio_files:
                    representative = os.path.join(root, audio_files[0])
                    genre_representatives[genre_name] = representative
        
        # For each genre, find the most similar tracks
        for genre, representative in genre_representatives.items():
            if os.path.exists(representative):
                print(f"\n🎵 Tracks similar to {genre} style:")
                
                similar_tracks = search_engine.find_similar_audio(
                    representative, similarity_threshold=0.3, limit=4
                )
                
                for track, score in similar_tracks:
                    if track != representative:  # Exclude the representative itself
                        track_genre = None
                        for g, rep in genre_representatives.items():
                            if os.path.dirname(track) == os.path.dirname(rep):
                                track_genre = g
                                break
                        
                        if track_genre:
                            match_indicator = "✓" if track_genre == genre else "→"
                            print(f"  {match_indicator} {os.path.basename(track)} ({track_genre}) - similarity: {score:.3f}")

def main():
    """Run all workflow examples"""
    print("CONTENT-BASED AUDIO SEARCH WORKFLOW EXAMPLES")
    print("=" * 60)
    
    # Example workflows for different use cases
    example_dj_workflow()
    example_producer_workflow()
    example_music_organization()
    example_sound_replacement()
    example_genre_clustering()
    
    print("\n" + "=" * 60)
    print("✨ WORKFLOW EXAMPLES COMPLETE")
    print("=" * 60)
    print("These examples demonstrate practical applications of:")
    print("• Audio similarity search - find samples that sound alike")
    print("• Harmonic matching - find samples in compatible keys")
    print("• Rhythm matching - find samples with similar groove")
    print("• Spectral search - search by frequency content")
    print("\nUse cases:")
    print("• DJ set preparation and live mixing")
    print("• Music production and sample discovery")
    print("• Music library organization and smart playlists")
    print("• Sound replacement and alternative finding")
    print("• Genre analysis and style clustering")

if __name__ == "__main__":
    main()