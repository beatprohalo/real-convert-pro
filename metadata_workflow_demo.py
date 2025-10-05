#!/usr/bin/env python3
"""
Advanced Metadata Management Workflow Examples
Practical demonstrations of metadata management features
"""

import os
import json
from pathlib import Path
from datetime import datetime

from metadata_manager import MetadataManager

def demo_automatic_tagging():
    """Demonstrate automatic tagging workflow"""
    print("🏷️  AUTOMATIC TAGGING DEMO")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Process sample audio with automatic tagging
    if os.path.exists("sample_audio"):
        print("Processing audio files with automatic tagging...")
        
        processed_count = manager.batch_process_directory("sample_audio", auto_tag=True)
        print(f"✓ Processed {processed_count} files with automatic tags")
        
        # Show auto-generated tags for each file
        results = manager.search_metadata()  # Get all files
        
        for metadata in results[:3]:  # Show first 3 files
            print(f"\n📁 {metadata.filename}:")
            print(f"   BPM: {metadata.bpm}")
            print(f"   Key: {metadata.key}")
            print(f"   Genre: {metadata.genre}")
            print(f"   Mood: {metadata.mood}")
            print(f"   Energy: {metadata.energy_level}")
            
            # Show auto-generated tags
            tags = manager.get_tags(metadata.id)
            auto_tags = [tag['name'] for tag in tags if tag['category'] == 'auto-generated']
            if auto_tags:
                print(f"   Auto-tags: {', '.join(auto_tags)}")

def demo_custom_metadata_workflow():
    """Demonstrate custom metadata fields workflow"""
    print("\n📝 CUSTOM METADATA WORKFLOW DEMO")
    print("=" * 45)
    
    manager = MetadataManager()
    
    # Get a sample file
    sample_files = []
    if os.path.exists("sample_audio"):
        sample_files = [f for f in os.listdir("sample_audio") 
                       if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
    
    if not sample_files:
        print("No sample files found")
        return
    
    test_file = os.path.join("sample_audio", sample_files[0])
    
    # Extract basic metadata
    metadata = manager.extract_file_metadata(test_file)
    
    print(f"Adding custom metadata to: {metadata.filename}")
    
    # Add comprehensive custom metadata
    metadata.custom_tags.update({
        # Production info
        'producer': 'Alex Thompson',
        'label': 'Indie Electronics',
        'release_date': '2023-06-15',
        'mix_version': 'extended mix',
        
        # Musical characteristics
        'vocal_type': 'instrumental',
        'musical_era': '2020s',
        'cultural_origin': 'electronic/western',
        
        # Rights and usage
        'usage_rights': 'royalty-free',
        'recommended_usage': 'background music, meditation, study',
        
        # Personal ratings
        'quality_rating': 8,
        'personal_rating': 4,
        
        # Tags and classification
        'tags': ['ambient', 'electronic', 'chill', 'atmospheric'],
        'similar_artists': ['Brian Eno', 'Stars of the Lid', 'Tim Hecker'],
        
        # Notes
        'notes': 'Perfect for quiet moments and introspective activities. Builds slowly with layered textures.'
    })
    
    # Save to database
    audio_id = manager.save_metadata(metadata)
    
    print(f"✓ Saved custom metadata (ID: {audio_id})")
    
    # Display the custom metadata
    print("\n📋 Custom Metadata Fields:")
    for field, value in metadata.custom_tags.items():
        if isinstance(value, list):
            value = ', '.join(value)
        print(f"   {field}: {value}")

def demo_library_organization():
    """Demonstrate library organization using metadata"""
    print("\n📚 LIBRARY ORGANIZATION DEMO")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Build comprehensive metadata database
    directories = ["sample_audio", "training_data"]
    total_processed = 0
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"Cataloging {directory}...")
            count = manager.batch_process_directory(directory, auto_tag=True)
            total_processed += count
            print(f"  Added {count} files")
    
    print(f"✓ Total library size: {total_processed} files")
    
    # Create smart collections based on metadata
    print("\n🎯 Smart Collections:")
    
    # Collection 1: High-energy tracks
    high_energy = manager.search_metadata(bpm_min=120)
    print(f"   🔥 High Energy ({len(high_energy)} tracks)")
    for track in high_energy[:3]:
        print(f"      • {track.filename} ({track.bpm} BPM)")
    
    # Collection 2: Ambient/Chill tracks
    ambient_tracks = manager.search_metadata(genre="ambient")
    if not ambient_tracks:
        # Fallback search
        ambient_tracks = manager.search_metadata(bpm_max=90)
    print(f"   🌙 Ambient/Chill ({len(ambient_tracks)} tracks)")
    for track in ambient_tracks[:3]:
        print(f"      • {track.filename} ({track.genre or 'unspecified genre'})")
    
    # Collection 3: By key for harmonic mixing
    c_major_tracks = manager.search_metadata(key="C")
    print(f"   🎼 C Major/Minor ({len(c_major_tracks)} tracks)")
    for track in c_major_tracks[:3]:
        print(f"      • {track.filename} (Key: {track.key})")
    
    # Get library statistics
    stats = manager.get_statistics()
    print(f"\n📊 Library Statistics:")
    print(f"   Total files: {stats.get('total_files', 0)}")
    print(f"   Total duration: {stats.get('total_duration_hours', 0):.1f} hours")
    
    if stats.get('genre_distribution'):
        print("   Genre distribution:")
        for genre, count in stats['genre_distribution'].items():
            print(f"      {genre}: {count} tracks")
    
    if stats.get('bpm_distribution'):
        print("   BPM ranges:")
        for bpm_range, count in list(stats['bpm_distribution'].items())[:5]:
            print(f"      {bpm_range} BPM: {count} tracks")

def demo_version_tracking_workflow():
    """Demonstrate version tracking for processed files"""
    print("\n🔄 VERSION TRACKING DEMO")
    print("=" * 35)
    
    manager = MetadataManager()
    
    # Get original file
    sample_files = []
    if os.path.exists("sample_audio"):
        sample_files = [f for f in os.listdir("sample_audio") 
                       if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    
    if not sample_files:
        print("No sample files found")
        return
    
    original_file = os.path.join("sample_audio", sample_files[0])
    
    # Create metadata for original
    print(f"📁 Original file: {sample_files[0]}")
    original_metadata = manager.extract_file_metadata(original_file)
    original_id = manager.save_metadata(original_metadata)
    
    print(f"   ID: {original_id}")
    print(f"   Duration: {original_metadata.duration:.2f}s")
    print(f"   BPM: {original_metadata.bpm}")
    
    # Simulate creating processed versions
    # In real workflow, these would be actual processed files
    processing_operations = [
        ("normalized", "Normalized to -23 LUFS"),
        ("eq_adjusted", "Applied EQ curve"),
        ("mastered", "Full mastering chain"),
        ("radio_edit", "Radio edit version")
    ]
    
    print(f"\n🔧 Creating processed versions:")
    
    for operation, description in processing_operations:
        # In real scenario, this would be the path to the actual processed file
        processed_file_path = f"processed_{operation}_{sample_files[0]}"
        
        # Record the processing operation
        manager.add_processing_record(
            original_id,
            operation,
            {"description": description, "timestamp": datetime.now().isoformat()},
            original_file,
            processed_file_path,
            True,
            description
        )
        
        print(f"   ✓ {operation}: {description}")
    
    # Show version history (simulated)
    print(f"\n📋 Processing History for {sample_files[0]}:")
    print("   Version 1.0: Original file")
    for i, (operation, description) in enumerate(processing_operations, 2):
        print(f"   Version {i}.0: {description}")

def demo_advanced_search():
    """Demonstrate advanced search capabilities"""
    print("\n🔍 ADVANCED SEARCH DEMO")
    print("=" * 35)
    
    manager = MetadataManager()
    
    # Ensure we have data
    if os.path.exists("sample_audio"):
        manager.batch_process_directory("sample_audio", auto_tag=True)
    
    # Complex search scenarios
    search_scenarios = [
        {
            "name": "DJ Set Preparation",
            "description": "Find tracks for a 120-130 BPM set",
            "criteria": {"bpm_min": 120, "bpm_max": 130},
        },
        {
            "name": "Background Music",
            "description": "Find ambient tracks under 4 minutes",
            "criteria": {"bpm_max": 100, "duration_max": 240},
        },
        {
            "name": "Electronic Collection",
            "description": "Find all electronic music",
            "criteria": {"genre": "electronic"},
        },
        {
            "name": "High Energy Workout",
            "description": "Find high-energy tracks",
            "criteria": {"bpm_min": 140},
        }
    ]
    
    for scenario in search_scenarios:
        print(f"\n🎯 {scenario['name']}:")
        print(f"   {scenario['description']}")
        
        results = manager.search_metadata(**scenario['criteria'])
        print(f"   Found {len(results)} matching tracks:")
        
        for result in results[:3]:  # Show top 3
            bpm_str = f"{result.bpm:.0f} BPM" if result.bpm else "Unknown BPM"
            genre_str = result.genre or "Unknown genre"
            print(f"      • {result.filename} - {bpm_str}, {genre_str}")

def demo_export_import_workflow():
    """Demonstrate metadata export and backup"""
    print("\n💾 EXPORT/BACKUP DEMO")
    print("=" * 30)
    
    manager = MetadataManager()
    
    # Ensure we have data
    if os.path.exists("sample_audio"):
        manager.batch_process_directory("sample_audio", auto_tag=True)
    
    # Export metadata
    export_file = "library_metadata_backup.json"
    print(f"📤 Exporting metadata to {export_file}...")
    
    success = manager.export_metadata(export_file)
    
    if success and os.path.exists(export_file):
        file_size = os.path.getsize(export_file)
        print(f"✓ Export successful ({file_size:,} bytes)")
        
        # Show export content summary
        with open(export_file, 'r') as f:
            data = json.load(f)
            
        print(f"📊 Export contains:")
        print(f"   {len(data)} audio files")
        
        # Count files by type
        formats = {}
        total_duration = 0
        
        for record in data:
            fmt = record.get('format', 'unknown')
            formats[fmt] = formats.get(fmt, 0) + 1
            total_duration += record.get('duration', 0)
        
        print(f"   {total_duration/3600:.1f} hours of audio")
        print(f"   Formats: {dict(formats)}")
        
        # Show sample metadata structure
        if data:
            sample = data[0]
            print(f"   Sample metadata fields: {len(sample)} fields")
            
        # Clean up
        os.remove(export_file)
        print(f"   (Cleaned up {export_file})")

def demo_metadata_tagging_strategies():
    """Demonstrate different tagging strategies"""
    print("\n🏷️  TAGGING STRATEGIES DEMO")
    print("=" * 40)
    
    manager = MetadataManager()
    
    # Get sample files
    sample_files = []
    if os.path.exists("sample_audio"):
        sample_files = [f for f in os.listdir("sample_audio") 
                       if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    
    if not sample_files:
        print("No sample files found")
        return
    
    # Process first file with different tagging strategies
    test_file = os.path.join("sample_audio", sample_files[0])
    metadata = manager.extract_file_metadata(test_file)
    audio_id = manager.save_metadata(metadata)
    
    print(f"📁 Tagging strategies for: {metadata.filename}")
    
    # Strategy 1: Genre-based tagging
    genre_tags = []
    if metadata.genre:
        base_genre = metadata.genre.lower()
        genre_tags.extend([base_genre, f"{base_genre}_music"])
        
        # Add related genre tags
        if "electronic" in base_genre:
            genre_tags.extend(["synth", "digital", "produced"])
        elif "acoustic" in base_genre:
            genre_tags.extend(["organic", "natural", "live"])
    
    if genre_tags:
        manager.add_tags(audio_id, genre_tags, "genre-based")
        print(f"   Genre tags: {', '.join(genre_tags)}")
    
    # Strategy 2: Mood and energy tagging
    mood_tags = []
    if metadata.energy_level is not None:
        if metadata.energy_level > 0.7:
            mood_tags.extend(["energetic", "uplifting", "motivational"])
        elif metadata.energy_level < 0.3:
            mood_tags.extend(["calm", "relaxing", "peaceful"])
        else:
            mood_tags.extend(["balanced", "moderate", "neutral"])
    
    if metadata.mood:
        mood_tags.append(metadata.mood.lower())
    
    if mood_tags:
        manager.add_tags(audio_id, mood_tags, "mood-based")
        print(f"   Mood tags: {', '.join(mood_tags)}")
    
    # Strategy 3: Technical/musical tagging
    technical_tags = []
    if metadata.bpm:
        if metadata.bpm < 80:
            technical_tags.extend(["slow_tempo", "ballad_pace"])
        elif metadata.bpm > 140:
            technical_tags.extend(["fast_tempo", "dance_pace"])
        else:
            technical_tags.extend(["medium_tempo", "walking_pace"])
    
    if metadata.key:
        technical_tags.append(f"key_{metadata.key.lower()}")
        if 'm' in metadata.key.lower():
            technical_tags.append("minor_key")
        else:
            technical_tags.append("major_key")
    
    if technical_tags:
        manager.add_tags(audio_id, technical_tags, "technical")
        print(f"   Technical tags: {', '.join(technical_tags)}")
    
    # Strategy 4: Usage context tagging
    usage_tags = []
    if metadata.bpm and metadata.energy_level:
        if metadata.bpm < 90 and metadata.energy_level < 0.5:
            usage_tags.extend(["study_music", "meditation", "background"])
        elif metadata.bpm > 120 and metadata.energy_level > 0.6:
            usage_tags.extend(["workout", "party", "dance"])
        else:
            usage_tags.extend(["general_listening", "casual"])
    
    if usage_tags:
        manager.add_tags(audio_id, usage_tags, "usage-context")
        print(f"   Usage tags: {', '.join(usage_tags)}")
    
    # Show all tags for the file
    all_tags = manager.get_tags(audio_id)
    print(f"\n📋 All tags ({len(all_tags)} total):")
    
    tag_categories = {}
    for tag in all_tags:
        category = tag['category']
        if category not in tag_categories:
            tag_categories[category] = []
        tag_categories[category].append(tag['name'])
    
    for category, tags in tag_categories.items():
        print(f"   {category}: {', '.join(tags)}")

def main():
    """Run all metadata management workflow demos"""
    print("ADVANCED METADATA MANAGEMENT WORKFLOW DEMOS")
    print("=" * 60)
    
    demos = [
        demo_automatic_tagging,
        demo_custom_metadata_workflow,
        demo_library_organization,
        demo_version_tracking_workflow,
        demo_advanced_search,
        demo_export_import_workflow,
        demo_metadata_tagging_strategies,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"Error in demo: {e}")
    
    print("\n" + "=" * 60)
    print("✨ METADATA MANAGEMENT DEMOS COMPLETE")
    print("=" * 60)
    print("Advanced metadata management capabilities demonstrated:")
    print("• 🏷️  Automatic tagging with BPM, key, genre, mood")
    print("• 📝 Custom metadata fields with flexible data types")
    print("• 📚 Database integration for large music libraries")
    print("• 🔄 Version tracking for processed files")
    print("• 🔍 Advanced search and filtering capabilities")
    print("• 📊 Library statistics and analytics")
    print("• 💾 Export/import for backup and portability")
    print("• 🎯 Smart collections and organization strategies")
    
    print("\nUse cases:")
    print("• Professional music production workflows")
    print("• DJ library management and set preparation")
    print("• Content creation and media asset management")
    print("• Music research and academic analysis")
    print("• Personal music collection organization")

if __name__ == "__main__":
    main()