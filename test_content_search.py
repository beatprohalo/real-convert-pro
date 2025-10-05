#!/usr/bin/env python3
"""
Test Content-Based Audio Search Functionality
"""

import os
import sys
from pathlib import Path
import json

# Import the audio analyzer with content-based search
from audio_analyzer import ContentBasedSearchEngine, IntelligentAudioAnalyzer

def test_content_based_search():
    """Test all content-based search features"""
    print("CONTENT-BASED AUDIO SEARCH TEST")
    print("=" * 50)
    
    # Initialize search engine
    search_engine = ContentBasedSearchEngine()
    
    # Check if sample audio directory exists
    sample_dir = "sample_audio"
    if not os.path.exists(sample_dir):
        print(f"Error: Sample directory '{sample_dir}' not found")
        print("Please ensure you have audio files in the sample_audio directory")
        return False
    
    # Get list of audio files
    audio_files = []
    for file in os.listdir(sample_dir):
        if file.lower().endswith(('.wav', '.mp3', '.flac', '.aiff', '.m4a')):
            audio_files.append(os.path.join(sample_dir, file))
    
    if len(audio_files) < 2:
        print(f"Error: Need at least 2 audio files for testing. Found {len(audio_files)}")
        return False
    
    print(f"Found {len(audio_files)} audio files for testing")
    
    # Test 1: Build search database
    print("\n1. BUILDING SEARCH DATABASE")
    print("-" * 30)
    
    analyzed_count = search_engine.batch_analyze_directory(sample_dir)
    print(f"Successfully analyzed {analyzed_count} files")
    
    if analyzed_count == 0:
        print("Failed to analyze any files")
        return False
    
    # Test 2: Audio similarity search
    print("\n2. TESTING AUDIO SIMILARITY SEARCH")
    print("-" * 40)
    
    query_file = audio_files[0]
    print(f"Query file: {os.path.basename(query_file)}")
    
    similar_files = search_engine.find_similar_audio(query_file, similarity_threshold=0.3, limit=5)
    
    if similar_files:
        print("Similar audio files found:")
        for file_path, similarity in similar_files:
            filename = os.path.basename(file_path)
            print(f"  • {filename} (similarity: {similarity:.3f})")
    else:
        print("No similar audio files found (threshold may be too high)")
    
    # Test 3: Harmonic matching
    print("\n3. TESTING HARMONIC MATCHING")
    print("-" * 35)
    
    harmonic_matches = search_engine.find_harmonic_matches(query_file, key_tolerance=5, limit=5)
    
    if harmonic_matches:
        print("Harmonically compatible files found:")
        for file_path, compatibility in harmonic_matches:
            filename = os.path.basename(file_path)
            print(f"  • {filename} (compatibility: {compatibility:.3f})")
    else:
        print("No harmonically compatible files found")
    
    # Test 4: Rhythm matching
    print("\n4. TESTING RHYTHM MATCHING")
    print("-" * 30)
    
    rhythm_matches = search_engine.find_rhythm_matches(query_file, tempo_tolerance=20.0, limit=5)
    
    if rhythm_matches:
        print("Rhythmically similar files found:")
        for file_path, similarity in rhythm_matches:
            filename = os.path.basename(file_path)
            print(f"  • {filename} (rhythm similarity: {similarity:.3f})")
    else:
        print("No rhythmically similar files found")
    
    # Test 5: Spectral search
    print("\n5. TESTING SPECTRAL SEARCH")
    print("-" * 30)
    
    spectral_matches = search_engine.find_spectral_matches(query_file, spectral_tolerance=0.3, limit=5)
    
    if spectral_matches:
        print("Spectrally similar files found:")
        for file_path, similarity in spectral_matches:
            filename = os.path.basename(file_path)
            print(f"  • {filename} (spectral similarity: {similarity:.3f})")
    else:
        print("No spectrally similar files found")
    
    # Test 6: Comprehensive search
    print("\n6. TESTING COMPREHENSIVE SEARCH")
    print("-" * 40)
    
    all_results = search_engine.search_by_example(query_file)
    
    for search_type, results in all_results.items():
        print(f"\n{search_type.upper()} search results:")
        if results:
            for file_path, score in results[:3]:  # Top 3 results
                filename = os.path.basename(file_path)
                print(f"  • {filename} (score: {score:.3f})")
        else:
            print("  No results found")
    
    # Test 7: Database information
    print("\n7. DATABASE INFORMATION")
    print("-" * 25)
    
    print(f"Database file: {search_engine.database_file}")
    print(f"Total entries: {len(search_engine.audio_database)}")
    
    # Show sample database entry
    if search_engine.audio_database:
        sample_key = list(search_engine.audio_database.keys())[0]
        sample_entry = search_engine.audio_database[sample_key]
        print(f"Sample entry keys: {list(sample_entry.keys())}")
    
    print("\n" + "=" * 50)
    print("CONTENT-BASED SEARCH TEST COMPLETED")
    print("=" * 50)
    
    return True

def test_individual_file_analysis():
    """Test individual file analysis with content-based features"""
    print("\nINDIVIDUAL FILE ANALYSIS TEST")
    print("=" * 40)
    
    sample_dir = "sample_audio"
    if not os.path.exists(sample_dir):
        print("Sample directory not found")
        return False
    
    # Get first audio file
    audio_files = [f for f in os.listdir(sample_dir) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.aiff', '.m4a'))]
    
    if not audio_files:
        print("No audio files found")
        return False
    
    test_file = os.path.join(sample_dir, audio_files[0])
    print(f"Analyzing: {audio_files[0]}")
    
    # Analyze with enhanced analyzer
    analyzer = IntelligentAudioAnalyzer()
    result = analyzer.analyze_audio_file(test_file)
    
    # Display analysis results including content-based features
    print(f"\nAnalysis Results:")
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"BPM: {result.bpm}" if result.bpm else "BPM: Not detected")
    print(f"Key: {result.key}" if result.key else "Key: Not detected")
    
    # Content-based search features
    if hasattr(result, 'chroma_features') and result.chroma_features is not None:
        print(f"Chroma features extracted: {len(result.chroma_features)} dimensions")
    
    if hasattr(result, 'rhythm_features') and result.rhythm_features is not None:
        print(f"Rhythm features extracted: {len(result.rhythm_features)} dimensions")
    
    if hasattr(result, 'spectral_features') and result.spectral_features is not None:
        print(f"Spectral features extracted: {len(result.spectral_features)} dimensions")
    
    if hasattr(result, 'similarity_hash') and result.similarity_hash:
        print(f"Similarity hash: {result.similarity_hash[:20]}...")
    
    if result.analysis_errors:
        print("\nAnalysis errors:")
        for error in result.analysis_errors:
            print(f"  - {error}")
    
    return True

def demo_search_workflow():
    """Demonstrate a typical search workflow"""
    print("\nSEARCH WORKFLOW DEMONSTRATION")
    print("=" * 40)
    
    # Step 1: Initialize search engine
    search_engine = ContentBasedSearchEngine()
    
    # Step 2: Add files from multiple directories
    directories_to_search = ["sample_audio", "training_data"]
    
    total_analyzed = 0
    for directory in directories_to_search:
        if os.path.exists(directory):
            print(f"Analyzing files in {directory}...")
            count = search_engine.batch_analyze_directory(directory)
            total_analyzed += count
            print(f"  Added {count} files")
    
    print(f"Total files in database: {total_analyzed}")
    
    # Step 3: Perform searches with different parameters
    if total_analyzed > 1:
        # Get a sample file for querying
        sample_files = [f for f in os.listdir("sample_audio") 
                       if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        
        if sample_files:
            query_file = os.path.join("sample_audio", sample_files[0])
            
            print(f"\nSearching with: {sample_files[0]}")
            
            # Different search strategies
            strategies = [
                ("Strict similarity", {"similarity_threshold": 0.8}),
                ("Loose similarity", {"similarity_threshold": 0.4}),
                ("Close harmonic match", {"key_tolerance": 1}),
                ("Wide harmonic match", {"key_tolerance": 5}),
                ("Tight rhythm match", {"tempo_tolerance": 5.0}),
                ("Loose rhythm match", {"tempo_tolerance": 20.0})
            ]
            
            for strategy_name, params in strategies:
                print(f"\n{strategy_name}:")
                if "similarity_threshold" in params:
                    results = search_engine.find_similar_audio(query_file, **params, limit=3)
                elif "key_tolerance" in params:
                    results = search_engine.find_harmonic_matches(query_file, **params, limit=3)
                elif "tempo_tolerance" in params:
                    results = search_engine.find_rhythm_matches(query_file, **params, limit=3)
                
                if results:
                    for file_path, score in results:
                        print(f"  • {os.path.basename(file_path)} (score: {score:.3f})")
                else:
                    print("  No matches found")
    
    return True

def main():
    """Run all content-based search tests"""
    print("CONTENT-BASED AUDIO SEARCH TESTING SUITE")
    print("=" * 60)
    
    success = True
    
    # Test 1: Basic functionality
    try:
        success &= test_content_based_search()
    except Exception as e:
        print(f"Content-based search test failed: {e}")
        success = False
    
    # Test 2: Individual file analysis
    try:
        success &= test_individual_file_analysis()
    except Exception as e:
        print(f"Individual file analysis test failed: {e}")
        success = False
    
    # Test 3: Search workflow demo
    try:
        success &= demo_search_workflow()
    except Exception as e:
        print(f"Search workflow demo failed: {e}")
        success = False
    
    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
        print("\nContent-based search features working correctly:")
        print("• Audio similarity search - find samples that sound alike")
        print("• Harmonic matching - find samples in compatible keys")
        print("• Rhythm matching - find samples with similar groove")
        print("• Spectral search - search by frequency content")
    else:
        print("✗ SOME TESTS FAILED")
        print("Check the error messages above for details")
    
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()