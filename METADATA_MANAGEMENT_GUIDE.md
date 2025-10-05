# Advanced Metadata Management System

A comprehensive audio metadata management system with automatic tagging, custom fields, database integration, and version tracking.

## Overview

The Advanced Metadata Management System provides professional-grade tools for organizing, analyzing, and tracking audio files. It combines intelligent audio analysis with flexible database storage to create a powerful solution for music libraries of any size.

## Features

### 🏷️ Automatic Tagging
- **BPM Detection**: Precise tempo analysis and categorization
- **Key Detection**: Musical key identification (C, D, E, F, G, A, B with major/minor)
- **Genre Classification**: AI-powered genre prediction
- **Mood Analysis**: Emotional characteristics (happy, sad, energetic, calm, etc.)
- **Energy Level**: Quantified energy measurements (0.0-1.0)
- **Content Type**: Music, speech, noise classification
- **Instrument Detection**: Automatic identification of present instruments

### 📝 Custom Metadata Fields
- **Production Info**: Producer, label, release date, mix version
- **Musical Characteristics**: Vocal type, musical era, cultural origin
- **Rights & Usage**: Usage rights, recommended contexts
- **Personal Ratings**: Quality and personal preference ratings (1-10, 1-5)
- **Flexible Tags**: Custom tag lists and categories
- **Rich Text Notes**: Detailed descriptions and observations
- **Artist Relationships**: Similar artists and influences

### 📚 Database Integration
- **SQLite Backend**: Lightweight, serverless database for any library size
- **Efficient Indexing**: Fast searches on BPM, key, genre, and custom fields
- **Relationship Tracking**: Links between files, versions, and processing history
- **Bulk Operations**: Batch processing of entire directories
- **Data Integrity**: Hash-based duplicate detection and verification

### 🔄 Version Tracking
- **Processing History**: Complete audit trail of all modifications
- **File Relationships**: Parent-child version hierarchies
- **Operation Logging**: Parameters and results for each processing step
- **Rollback Support**: Easy navigation between file versions
- **Metadata Inheritance**: Automatic propagation of relevant metadata

## Installation

### Dependencies
```bash
pip install mutagen librosa soundfile numpy sqlite3
```

### Optional Dependencies (for enhanced features)
```bash
pip install essentia chromaprint tensorflow pyrubberband noisereduce
```

## Quick Start

### Basic Usage
```python
from metadata_manager import MetadataManager

# Initialize the system
manager = MetadataManager()

# Analyze a single file
metadata = manager.extract_file_metadata("audio.wav")
audio_id = manager.save_metadata(metadata)

# Batch process a directory
processed_count = manager.batch_process_directory(
    "music_library", 
    auto_tag=True,
    write_to_files=True
)

print(f"Processed {processed_count} files")
```

### Custom Metadata
```python
# Add custom metadata fields
metadata.custom_tags.update({
    'producer': 'John Smith',
    'label': 'Indie Records',
    'quality_rating': 9,
    'tags': ['ambient', 'electronic', 'chill'],
    'notes': 'Perfect for background music'
})

# Save with custom data
audio_id = manager.save_metadata(metadata)
```

### Search and Filter
```python
# Search by BPM range
dance_tracks = manager.search_metadata(bpm_min=120, bpm_max=140)

# Search by genre
electronic = manager.search_metadata(genre="electronic")

# Complex search
ambient_short = manager.search_metadata(
    genre="ambient",
    duration_max=240,  # Under 4 minutes
    energy_max=0.5     # Low energy
)
```

### Version Tracking
```python
# Create a processed version
new_version_id = manager.create_file_version(
    original_id=audio_id,
    new_file_path="processed_audio.wav",
    creation_method="normalized",
    parent_version=1
)

# Add processing record
manager.add_processing_record(
    audio_id,
    operation="normalize",
    parameters={"target_lufs": -23.0},
    input_file="original.wav",
    output_file="normalized.wav",
    success=True,
    notes="Applied broadcast standard normalization"
)
```

## Database Schema

### Core Tables

#### audio_metadata
- **id**: Primary key
- **file_path**: Absolute path to audio file
- **filename**: Base filename
- **file_size, file_hash**: File verification data
- **duration, sample_rate, channels**: Audio properties
- **bpm, key, scale**: Musical analysis
- **genre, mood, energy_level**: Content classification
- **lufs_integrated, peak_db**: Loudness metrics
- **version, original_file**: Version tracking

#### custom_metadata
- **audio_id**: Foreign key to audio_metadata
- **field_name**: Custom field identifier
- **field_value**: Field value (JSON for complex types)
- **field_type**: Data type (string, integer, list, etc.)

#### processing_history
- **audio_id**: Foreign key to audio_metadata
- **operation**: Processing operation name
- **parameters**: Operation parameters (JSON)
- **input_file, output_file**: File paths
- **success**: Operation success status
- **notes**: Additional information

#### tags
- **audio_id**: Foreign key to audio_metadata
- **tag_name**: Tag identifier
- **tag_category**: Tag grouping (auto, manual, genre, etc.)

## Custom Field Types

### Predefined Fields
| Field | Type | Description |
|-------|------|-------------|
| producer | string | Track producer |
| label | string | Record label |
| release_date | date | Release date |
| mix_version | string | Mix type (radio, extended, etc.) |
| vocal_type | string | Vocal characteristics |
| musical_era | string | Musical era/period |
| cultural_origin | string | Cultural/geographical origin |
| usage_rights | string | Usage rights and licensing |
| quality_rating | integer | Quality rating (1-10) |
| personal_rating | integer | Personal rating (1-5) |
| tags | list | Custom tags |
| notes | text | Additional notes |
| similar_artists | list | Similar artists |
| recommended_usage | string | Recommended usage context |

### Adding Custom Fields
```python
# Define new field types in the manager
manager.custom_field_definitions['tempo_feel'] = {
    'type': 'string',
    'description': 'Rhythmic feel (straight, swing, shuffle)'
}

# Use in metadata
metadata.custom_tags['tempo_feel'] = 'swing'
```

## File Format Support

### Read Support
- **MP3**: Full ID3 tag support
- **FLAC**: Vorbis comment support
- **MP4/M4A**: iTunes-style metadata
- **WAV**: Basic RIFF INFO support
- **AIFF**: Limited metadata support

### Write Support
- **MP3**: ID3v2.4 tags
- **FLAC**: Vorbis comments
- **MP4/M4A**: Standard atoms
- **WAV/AIFF**: Limited (database storage recommended)

## Search and Filtering

### Basic Search
```python
# Single criteria
results = manager.search_metadata(genre="jazz")

# Multiple criteria
results = manager.search_metadata(
    bpm_min=90,
    bpm_max=120,
    key="C",
    content_type="music"
)
```

### Advanced Search
```python
# Get all files tagged as "chill"
chill_files = []
all_metadata = manager.search_metadata()

for metadata in all_metadata:
    tags = manager.get_tags(metadata.id)
    tag_names = [tag['name'] for tag in tags]
    if 'chill' in tag_names:
        chill_files.append(metadata)
```

### Statistical Analysis
```python
# Get library statistics
stats = manager.get_statistics()

print(f"Total files: {stats['total_files']}")
print(f"Total duration: {stats['total_duration_hours']:.1f} hours")
print(f"Genres: {list(stats['genre_distribution'].keys())}")
print(f"BPM distribution: {stats['bpm_distribution']}")
```

## Export and Backup

### JSON Export
```python
# Export all metadata
success = manager.export_metadata("backup.json", format="json")

# The export includes all fields except binary data
# Perfect for backups, transfers, and analysis
```

### Database Backup
```python
import shutil

# Simple database file backup
shutil.copy("audio_metadata.db", "backup_audio_metadata.db")
```

## Tagging Strategies

### Automatic Tagging
The system automatically generates tags based on analysis:

- **BPM-based**: slow, moderate, fast, dance_pace
- **Key-based**: major_key, minor_key, key_c, key_f, etc.
- **Genre-based**: electronic, rock, jazz, classical
- **Mood-based**: happy, sad, energetic, calm
- **Energy-based**: high_energy, low_energy, medium_energy
- **Technical**: has_vocals, instrumental, live_recording

### Manual Tagging
```python
# Add contextual tags
manager.add_tags(audio_id, [
    "workout", "motivation", "high_energy"
], category="usage-context")

# Add descriptive tags
manager.add_tags(audio_id, [
    "synthesizer", "analog", "vintage"
], category="sonic-characteristics")
```

### Smart Collections
Create dynamic collections based on metadata:

```python
# DJ Mix Preparation
mix_candidates = manager.search_metadata(
    bpm_min=125,
    bpm_max=130,
    key="C"  # Compatible key
)

# Background Music Collection
background_music = manager.search_metadata(
    energy_max=0.4,
    duration_min=120,  # At least 2 minutes
    content_type="music"
)
```

## Professional Workflows

### Music Production
1. **Sample Organization**: Categorize by BPM, key, and instruments
2. **Version Control**: Track mix iterations and processing history
3. **Quality Control**: Rate and annotate samples for quick selection
4. **Rights Management**: Track usage rights and licensing information

### DJ Library Management
1. **Set Preparation**: Search by BPM and key for harmonic mixing
2. **Energy Management**: Organize by energy levels for set flow
3. **Genre Mixing**: Find crossover tracks between genres
4. **Crowd Analysis**: Tag tracks by venue and audience response

### Content Creation
1. **Project Assets**: Organize by usage context and mood
2. **Client Libraries**: Separate collections with usage rights
3. **Template Building**: Create reusable metadata templates
4. **Delivery Tracking**: Version control for client deliverables

## Performance Optimization

### Database Tuning
```python
# For large libraries, consider these optimizations:

# 1. Regular VACUUM to optimize database size
import sqlite3
conn = sqlite3.connect("audio_metadata.db")
conn.execute("VACUUM")
conn.close()

# 2. Additional indexes for frequently searched fields
# (These are created automatically by the system)

# 3. Batch operations for better performance
files_to_process = [...]  # Large list of files
for batch in chunks(files_to_process, 100):
    for file in batch:
        manager.extract_file_metadata(file)
    # Commit batch
```

### Memory Management
```python
# For very large libraries, process in batches
import os

def process_large_library(root_directory):
    manager = MetadataManager()
    
    for root, dirs, files in os.walk(root_directory):
        audio_files = [f for f in files if f.endswith(('.wav', '.mp3', '.flac'))]
        
        # Process in chunks
        for i in range(0, len(audio_files), 50):
            batch = audio_files[i:i+50]
            for file in batch:
                filepath = os.path.join(root, file)
                metadata = manager.extract_file_metadata(filepath)
                manager.save_metadata(metadata)
```

## Error Handling and Validation

### File Validation
```python
# Check file integrity
metadata = manager.get_metadata(file_path="audio.wav")
current_hash = manager.calculate_file_hash("audio.wav")

if metadata and metadata.file_hash != current_hash:
    print("Warning: File has been modified since analysis")
```

### Analysis Errors
```python
# Check for analysis errors
metadata = manager.extract_file_metadata("problematic.wav")

if metadata.analysis_errors:
    print("Analysis issues:")
    for error in metadata.analysis_errors:
        print(f"  - {error}")
```

## Integration Examples

### Content Management Systems
```python
# Export for web applications
def export_for_web():
    all_metadata = manager.search_metadata()
    
    web_data = []
    for metadata in all_metadata:
        web_data.append({
            'id': metadata.id,
            'filename': metadata.filename,
            'duration': metadata.duration,
            'bpm': metadata.bpm,
            'key': metadata.key,
            'genre': metadata.genre,
            'tags': [tag['name'] for tag in manager.get_tags(metadata.id)]
        })
    
    return web_data
```

### DAW Integration
```python
# Export metadata for Digital Audio Workstations
def export_for_daw(format="csv"):
    if format == "csv":
        import csv
        
        with open("library_export.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File", "BPM", "Key", "Genre", "Duration"])
            
            for metadata in manager.search_metadata():
                writer.writerow([
                    metadata.file_path,
                    metadata.bpm or "",
                    metadata.key or "",
                    metadata.genre or "",
                    metadata.duration
                ])
```

## Testing and Validation

### Run Tests
```bash
python test_metadata_management.py
```

### Demo Workflows
```bash
python metadata_workflow_demo.py
```

## Troubleshooting

### Common Issues

**Database locked errors**
- Solution: Ensure proper connection closing in all operations
- Use context managers for database connections

**Memory usage with large libraries**
- Solution: Process files in batches
- Clear analysis cache regularly

**Slow searches**
- Solution: Database indexes are created automatically
- Consider additional custom indexes for frequent search patterns

**File format issues**
- Solution: Check mutagen library compatibility
- Some formats have limited metadata support

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check analysis results
metadata = manager.extract_file_metadata("debug_file.wav")
print(f"Analysis errors: {metadata.analysis_errors}")
```

## Contributing

The metadata management system is designed to be extensible:

1. **Custom Analysis**: Add new audio analysis features
2. **Format Support**: Extend to additional audio formats
3. **Field Types**: Create new custom field types
4. **Export Formats**: Add new export/import formats
5. **Search Features**: Implement advanced search algorithms

## License

This system is part of the Advanced Audio Analysis Suite and follows the same licensing terms.

---

## Quick Reference

### Common Operations
```python
# Initialize
manager = MetadataManager()

# Single file
metadata = manager.extract_file_metadata("file.wav")
audio_id = manager.save_metadata(metadata)

# Directory
count = manager.batch_process_directory("music/", auto_tag=True)

# Search
results = manager.search_metadata(bpm_min=120, genre="electronic")

# Tags
manager.add_tags(audio_id, ["ambient", "chill"], "manual")
tags = manager.get_tags(audio_id)

# Export
manager.export_metadata("backup.json")

# Statistics
stats = manager.get_statistics()
```

For more examples and advanced usage, see the included demo and test files.