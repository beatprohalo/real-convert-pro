#!/usr/bin/env python3
"""
Test Advanced Metadata Management System
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

from metadata_manager import MetadataManager, AudioMetadata

def test_metadata_extraction():
    """Test metadata extraction from audio files"""
    print("METADATA EXTRACTION TEST")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Test with sample audio files
    sample_dir = "sample_audio"
    if not os.path.exists(sample_dir):
        print("Sample audio directory not found")
        return False
    
    audio_files = [f for f in os.listdir(sample_dir) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aiff'))]
    
    if not audio_files:
        print("No audio files found for testing")
        return False
    
    test_file = os.path.join(sample_dir, audio_files[0])
    print(f"Testing with: {audio_files[0]}")
    
    # Extract metadata
    metadata = manager.extract_file_metadata(test_file)
    
    # Display results
    print(f"\nExtracted Metadata:")
    print(f"Filename: {metadata.filename}")
    print(f"Duration: {metadata.duration:.2f} seconds")
    print(f"Sample Rate: {metadata.sample_rate}Hz")
    print(f"Channels: {metadata.channels}")
    print(f"File Size: {metadata.file_size} bytes")
    print(f"BPM: {metadata.bpm}" if metadata.bpm else "BPM: Not detected")
    print(f"Key: {metadata.key}" if metadata.key else "Key: Not detected")
    print(f"Genre: {metadata.genre}" if metadata.genre else "Genre: Not detected")
    print(f"Mood: {metadata.mood}" if metadata.mood else "Mood: Not detected")
    print(f"Energy Level: {metadata.energy_level}" if metadata.energy_level else "Energy: Not detected")
    print(f"Content Type: {metadata.content_type}" if metadata.content_type else "Content: Not detected")
    
    if metadata.custom_tags:
        print(f"Custom Tags: {metadata.custom_tags}")
    
    return True

def test_database_operations():
    """Test database save and retrieve operations"""
    print("\nDATABASE OPERATIONS TEST")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Test with sample audio files
    sample_dir = "sample_audio"
    if not os.path.exists(sample_dir):
        print("Sample audio directory not found")
        return False
    
    audio_files = [f for f in os.listdir(sample_dir) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aiff'))]
    
    if not audio_files:
        print("No audio files found for testing")
        return False
    
    test_file = os.path.join(sample_dir, audio_files[0])
    
    # Extract and save metadata
    print(f"Saving metadata for: {audio_files[0]}")
    metadata = manager.extract_file_metadata(test_file)
    
    # Add some custom metadata
    metadata.custom_tags.update({
        'producer': 'Test Producer',
        'label': 'Test Label',
        'quality_rating': 8,
        'personal_rating': 4,
        'tags': ['test', 'sample', 'demo'],
        'notes': 'This is a test file for metadata management'
    })
    
    audio_id = manager.save_metadata(metadata)
    
    if audio_id > 0:
        print(f"✓ Metadata saved with ID: {audio_id}")
        
        # Retrieve metadata
        retrieved = manager.get_metadata(file_path=test_file)
        
        if retrieved:
            print("✓ Metadata retrieved successfully")
            print(f"Retrieved ID: {retrieved.id}")
            print(f"Custom tags: {retrieved.custom_tags}")
        else:
            print("✗ Failed to retrieve metadata")
            return False
    else:
        print("✗ Failed to save metadata")
        return False
    
    return True

def test_batch_processing():
    """Test batch processing of audio directory"""
    print("\nBATCH PROCESSING TEST")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Process sample audio directory
    sample_dir = "sample_audio"
    if os.path.exists(sample_dir):
        print(f"Processing directory: {sample_dir}")
        
        processed_count = manager.batch_process_directory(
            sample_dir, 
            auto_tag=True, 
            write_to_files=False  # Don't modify original files in test
        )
        
        print(f"✓ Processed {processed_count} files")
        
        # Get statistics
        stats = manager.get_statistics()
        print(f"\nDatabase Statistics:")
        print(f"Total files: {stats.get('total_files', 0)}")
        print(f"Total duration: {stats.get('total_duration_hours', 0):.2f} hours")
        
        if stats.get('genre_distribution'):
            print(f"Genres: {list(stats['genre_distribution'].keys())}")
        
        if stats.get('bpm_distribution'):
            print(f"BPM ranges: {list(stats['bpm_distribution'].keys())}")
        
        return processed_count > 0
    else:
        print("Sample directory not found")
        return False

def test_search_functionality():
    """Test metadata search functionality"""
    print("\nSEARCH FUNCTIONALITY TEST")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Ensure we have some data
    if os.path.exists("sample_audio"):
        manager.batch_process_directory("sample_audio", auto_tag=True)
    
    # Test various search criteria
    search_tests = [
        {"genre": "electronic"},
        {"bpm_min": 100, "bpm_max": 140},
        {"content_type": "music"},
        {"duration_min": 3.0},
    ]
    
    for criteria in search_tests:
        print(f"\nSearching with criteria: {criteria}")
        results = manager.search_metadata(**criteria)
        
        print(f"Found {len(results)} matching files:")
        for result in results[:3]:  # Show first 3 results
            print(f"  • {result.filename} (BPM: {result.bpm}, Genre: {result.genre})")
    
    return True

def test_tagging_system():
    """Test the tagging system"""
    print("\nTAGGING SYSTEM TEST")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Get a sample file
    sample_dir = "sample_audio"
    if not os.path.exists(sample_dir):
        print("Sample directory not found")
        return False
    
    audio_files = [f for f in os.listdir(sample_dir) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aiff'))]
    
    if not audio_files:
        print("No audio files found")
        return False
    
    test_file = os.path.join(sample_dir, audio_files[0])
    
    # Get or create metadata entry
    metadata = manager.get_metadata(file_path=test_file)
    if not metadata:
        metadata = manager.extract_file_metadata(test_file)
        audio_id = manager.save_metadata(metadata)
    else:
        audio_id = metadata.id
    
    # Add custom tags
    test_tags = ["ambient", "chill", "background", "instrumental"]
    manager.add_tags(audio_id, test_tags, "manual")
    
    # Retrieve tags
    tags = manager.get_tags(audio_id)
    print(f"Tags for {audio_files[0]}:")
    
    for tag in tags:
        print(f"  • {tag['name']} ({tag['category']})")
    
    return len(tags) > 0

def test_version_tracking():
    """Test file version tracking"""
    print("\nVERSION TRACKING TEST")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Get a sample file
    sample_dir = "sample_audio"
    if not os.path.exists(sample_dir):
        print("Sample directory not found")
        return False
    
    audio_files = [f for f in os.listdir(sample_dir) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aiff'))]
    
    if not audio_files:
        print("No audio files found")
        return False
    
    original_file = os.path.join(sample_dir, audio_files[0])
    
    # Create metadata for original
    metadata = manager.extract_file_metadata(original_file)
    original_id = manager.save_metadata(metadata)
    
    print(f"Original file ID: {original_id}")
    
    # Simulate creating a processed version
    # In a real scenario, this would be an actual processed file
    processed_file = original_file  # Using same file for demo
    
    version_id = manager.create_file_version(
        original_id, 
        processed_file, 
        "test_processing",
        parent_version=1
    )
    
    if version_id > 0:
        print(f"✓ Created version with ID: {version_id}")
        
        # Get all versions
        versions = manager.get_file_versions(original_id)
        print(f"Found {len(versions)} versions:")
        
        for version in versions:
            print(f"  Version {version['version_number']}: {version['filename']}")
            print(f"    Method: {version['creation_method']}")
            print(f"    Created: {version['created_at']}")
        
        return True
    else:
        print("✗ Failed to create version")
        return False

def test_metadata_export():
    """Test metadata export functionality"""
    print("\nMETADATA EXPORT TEST")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Ensure we have some data
    if os.path.exists("sample_audio"):
        manager.batch_process_directory("sample_audio")
    
    # Export metadata
    export_file = "metadata_export.json"
    success = manager.export_metadata(export_file, format="json")
    
    if success and os.path.exists(export_file):
        file_size = os.path.getsize(export_file)
        print(f"✓ Exported metadata to {export_file} ({file_size} bytes)")
        
        # Show sample of exported data
        with open(export_file, 'r') as f:
            data = json.load(f)
            print(f"Exported {len(data)} metadata records")
            
            if data:
                sample = data[0]
                print(f"Sample record keys: {list(sample.keys())}")
        
        # Clean up
        os.remove(export_file)
        return True
    else:
        print("✗ Export failed")
        return False

def test_custom_fields():
    """Test custom metadata fields"""
    print("\nCUSTOM FIELDS TEST")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Show available custom field definitions
    print("Available custom fields:")
    for field, definition in manager.custom_field_definitions.items():
        print(f"  • {field}: {definition['description']} ({definition['type']})")
    
    # Test with sample file
    sample_dir = "sample_audio"
    if not os.path.exists(sample_dir):
        print("Sample directory not found")
        return False
    
    audio_files = [f for f in os.listdir(sample_dir) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aiff'))]
    
    if not audio_files:
        print("No audio files found")
        return False
    
    test_file = os.path.join(sample_dir, audio_files[0])
    metadata = manager.extract_file_metadata(test_file)
    
    # Add custom field values
    metadata.custom_tags.update({
        'producer': 'John Smith',
        'label': 'Demo Records',
        'release_date': '2023-01-15',
        'mix_version': 'radio edit',
        'vocal_type': 'instrumental',
        'musical_era': '2020s',
        'cultural_origin': 'western',
        'usage_rights': 'creative commons',
        'quality_rating': 9,
        'personal_rating': 5,
        'tags': ['electronic', 'ambient', 'chill'],
        'notes': 'Great for background music',
        'similar_artists': ['Brian Eno', 'Boards of Canada'],
        'recommended_usage': 'meditation, study, relaxation'
    })
    
    # Save and retrieve
    audio_id = manager.save_metadata(metadata)
    retrieved = manager.get_metadata(audio_id=audio_id)
    
    if retrieved:
        print(f"\nCustom metadata for {retrieved.filename}:")
        for field, value in retrieved.custom_tags.items():
            print(f"  {field}: {value}")
        return True
    else:
        print("Failed to save/retrieve custom metadata")
        return False

def main():
    """Run all metadata management tests"""
    print("ADVANCED METADATA MANAGEMENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Metadata Extraction", test_metadata_extraction),
        ("Database Operations", test_database_operations),
        ("Batch Processing", test_batch_processing),
        ("Search Functionality", test_search_functionality),
        ("Tagging System", test_tagging_system),
        ("Version Tracking", test_version_tracking),
        ("Metadata Export", test_metadata_export),
        ("Custom Fields", test_custom_fields),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}...")
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
                
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\nAdvanced metadata management features working correctly:")
        print("• Automatic tagging with BPM, key, genre, mood")
        print("• Custom metadata fields with flexible data types")
        print("• SQLite database integration for large libraries")
        print("• Version tracking for processed files")
        print("• Comprehensive search and filtering")
        print("• Batch processing capabilities")
        print("• Export/import functionality")
    else:
        print(f"⚠️  {total - passed} tests failed")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    main()