#!/usr/bin/env python3
"""
Advanced Metadata Management System
Provides automatic tagging, custom metadata fields, database integration, and version tracking
"""

import os
import sys
import json
import sqlite3
import hashlib
import shutil
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import mutagen
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TCON, TBPM, TKEY, COMM, TXXX
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.wave import WAVE

# Import our audio analyzer
from audio_analyzer import IntelligentAudioAnalyzer, AudioAnalysisResult

@dataclass
class AudioMetadata:
    """Container for comprehensive audio metadata"""
    # File information
    file_path: str
    filename: str
    file_size: int
    file_hash: str
    created_date: datetime
    modified_date: datetime
    
    # Audio properties
    duration: float
    sample_rate: int
    channels: int
    bitrate: Optional[int] = None
    format: Optional[str] = None
    
    # Musical analysis
    bpm: Optional[float] = None
    key: Optional[str] = None
    scale: Optional[str] = None
    time_signature: Optional[str] = None
    
    # Genre and mood
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy_level: Optional[float] = None
    danceability: Optional[float] = None
    valence: Optional[float] = None
    
    # Technical analysis
    lufs_integrated: Optional[float] = None
    peak_db: Optional[float] = None
    dynamic_range: Optional[float] = None
    
    # Content classification
    content_type: Optional[str] = None  # music, speech, noise, etc.
    instruments: Optional[List[str]] = None
    
    # User-defined metadata
    custom_tags: Dict[str, Any] = None
    
    # Processing history
    processing_history: List[Dict[str, Any]] = None
    original_file: Optional[str] = None
    version: int = 1
    
    # Database metadata
    id: Optional[int] = None
    last_analyzed: Optional[datetime] = None
    analysis_version: str = "1.0"
    
    def __post_init__(self):
        if self.custom_tags is None:
            self.custom_tags = {}
        if self.processing_history is None:
            self.processing_history = []
        if self.instruments is None:
            self.instruments = []

class MetadataManager:
    """Advanced metadata management system"""
    
    def __init__(self, database_path: str = "audio_metadata.db"):
        self.database_path = database_path
        self.analyzer = IntelligentAudioAnalyzer()
        self.init_database()
        
        # Supported audio formats and their metadata handlers
        self.format_handlers = {
            '.mp3': self._handle_mp3_metadata,
            '.flac': self._handle_flac_metadata,
            '.mp4': self._handle_mp4_metadata,
            '.m4a': self._handle_mp4_metadata,
            '.wav': self._handle_wav_metadata,
            '.aiff': self._handle_aiff_metadata
        }
        
        # Custom field definitions
        self.custom_field_definitions = {
            'producer': {'type': 'string', 'description': 'Track producer'},
            'label': {'type': 'string', 'description': 'Record label'},
            'release_date': {'type': 'date', 'description': 'Release date'},
            'mix_version': {'type': 'string', 'description': 'Mix version (radio, extended, etc.)'},
            'vocal_type': {'type': 'string', 'description': 'Vocal characteristics'},
            'musical_era': {'type': 'string', 'description': 'Musical era/period'},
            'cultural_origin': {'type': 'string', 'description': 'Cultural/geographical origin'},
            'usage_rights': {'type': 'string', 'description': 'Usage rights and licensing'},
            'quality_rating': {'type': 'integer', 'description': 'Quality rating (1-10)'},
            'personal_rating': {'type': 'integer', 'description': 'Personal rating (1-5)'},
            'tags': {'type': 'list', 'description': 'Custom tags'},
            'notes': {'type': 'text', 'description': 'Additional notes'},
            'similar_artists': {'type': 'list', 'description': 'Similar artists'},
            'recommended_usage': {'type': 'string', 'description': 'Recommended usage context'}
        }
    
    def init_database(self):
        """Initialize SQLite database for metadata storage"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Main metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    file_size INTEGER,
                    file_hash TEXT,
                    created_date TIMESTAMP,
                    modified_date TIMESTAMP,
                    duration REAL,
                    sample_rate INTEGER,
                    channels INTEGER,
                    bitrate INTEGER,
                    format TEXT,
                    bpm REAL,
                    key TEXT,
                    scale TEXT,
                    time_signature TEXT,
                    genre TEXT,
                    mood TEXT,
                    energy_level REAL,
                    danceability REAL,
                    valence REAL,
                    lufs_integrated REAL,
                    peak_db REAL,
                    dynamic_range REAL,
                    content_type TEXT,
                    instruments TEXT,
                    version INTEGER DEFAULT 1,
                    original_file TEXT,
                    last_analyzed TIMESTAMP,
                    analysis_version TEXT DEFAULT '1.0',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Custom metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS custom_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_id INTEGER,
                    field_name TEXT NOT NULL,
                    field_value TEXT,
                    field_type TEXT DEFAULT 'string',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (audio_id) REFERENCES audio_metadata (id)
                )
            ''')
            
            # Processing history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_id INTEGER,
                    operation TEXT NOT NULL,
                    parameters TEXT,
                    input_file TEXT,
                    output_file TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN,
                    notes TEXT,
                    FOREIGN KEY (audio_id) REFERENCES audio_metadata (id)
                )
            ''')
            
            # Version tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id INTEGER,
                    version_number INTEGER,
                    file_path TEXT NOT NULL,
                    creation_method TEXT,
                    parent_version INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (original_id) REFERENCES audio_metadata (id)
                )
            ''')
            
            # Tags table for flexible tagging
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_id INTEGER,
                    tag_name TEXT NOT NULL,
                    tag_category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (audio_id) REFERENCES audio_metadata (id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON audio_metadata(file_path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON audio_metadata(file_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_genre ON audio_metadata(genre)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bpm ON audio_metadata(bpm)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_key ON audio_metadata(key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags ON tags(tag_name)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for duplicate detection"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    def extract_file_metadata(self, file_path: str) -> AudioMetadata:
        """Extract comprehensive metadata from audio file"""
        file_path = os.path.abspath(file_path)
        
        # Basic file information
        stat = os.stat(file_path)
        file_hash = self.calculate_file_hash(file_path)
        
        # Initialize metadata object
        metadata = AudioMetadata(
            file_path=file_path,
            filename=os.path.basename(file_path),
            file_size=stat.st_size,
            file_hash=file_hash,
            created_date=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            modified_date=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            duration=0.0,
            sample_rate=0,
            channels=0,
            last_analyzed=datetime.now(tz=timezone.utc)
        )
        
        try:
            # Extract audio format information
            audio_file = mutagen.File(file_path)
            if audio_file:
                metadata.duration = getattr(audio_file.info, 'length', 0.0)
                metadata.sample_rate = getattr(audio_file.info, 'sample_rate', 0)
                metadata.channels = getattr(audio_file.info, 'channels', 0)
                metadata.bitrate = getattr(audio_file.info, 'bitrate', None)
                metadata.format = audio_file.mime[0] if audio_file.mime else None
                
                # Extract existing metadata tags
                self._extract_existing_tags(audio_file, metadata)
            
            # Perform audio analysis
            analysis_result = self.analyzer.analyze_audio_file(file_path)
            self._merge_analysis_results(analysis_result, metadata)
            
        except Exception as e:
            print(f"Error extracting metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_existing_tags(self, audio_file, metadata: AudioMetadata):
        """Extract existing metadata tags from audio file"""
        try:
            if isinstance(audio_file, MP3):
                self._extract_id3_tags(audio_file, metadata)
            elif isinstance(audio_file, FLAC):
                self._extract_flac_tags(audio_file, metadata)
            elif isinstance(audio_file, MP4):
                self._extract_mp4_tags(audio_file, metadata)
        except Exception as e:
            print(f"Error extracting existing tags: {e}")
    
    def _extract_id3_tags(self, audio_file, metadata: AudioMetadata):
        """Extract ID3 tags from MP3 files"""
        tags = audio_file.tags
        if not tags:
            return
        
        # Standard tags
        if 'TIT2' in tags:  # Title
            metadata.custom_tags['title'] = str(tags['TIT2'])
        if 'TPE1' in tags:  # Artist
            metadata.custom_tags['artist'] = str(tags['TPE1'])
        if 'TALB' in tags:  # Album
            metadata.custom_tags['album'] = str(tags['TALB'])
        if 'TCON' in tags:  # Genre
            metadata.genre = str(tags['TCON'])
        if 'TBPM' in tags:  # BPM
            try:
                metadata.bpm = float(str(tags['TBPM']))
            except ValueError:
                pass
        if 'TKEY' in tags:  # Key
            metadata.key = str(tags['TKEY'])
    
    def _extract_flac_tags(self, audio_file, metadata: AudioMetadata):
        """Extract Vorbis comments from FLAC files"""
        if not audio_file.tags:
            return
        
        tags = audio_file.tags
        
        # Standard Vorbis comments
        metadata.custom_tags['title'] = tags.get('TITLE', [''])[0]
        metadata.custom_tags['artist'] = tags.get('ARTIST', [''])[0]
        metadata.custom_tags['album'] = tags.get('ALBUM', [''])[0]
        metadata.genre = tags.get('GENRE', [''])[0]
        
        # BPM
        bpm_str = tags.get('BPM', [''])[0]
        if bpm_str:
            try:
                metadata.bpm = float(bpm_str)
            except ValueError:
                pass
        
        # Key
        metadata.key = tags.get('KEY', [''])[0]
    
    def _extract_mp4_tags(self, audio_file, metadata: AudioMetadata):
        """Extract metadata from MP4/M4A files"""
        tags = audio_file.tags
        if not tags:
            return
        
        # MP4 atom names
        metadata.custom_tags['title'] = tags.get('\xa9nam', [''])[0]
        metadata.custom_tags['artist'] = tags.get('\xa9ART', [''])[0]
        metadata.custom_tags['album'] = tags.get('\xa9alb', [''])[0]
        metadata.genre = tags.get('\xa9gen', [''])[0]
    
    def _merge_analysis_results(self, analysis: AudioAnalysisResult, metadata: AudioMetadata):
        """Merge audio analysis results into metadata"""
        metadata.bpm = analysis.bpm or metadata.bpm
        metadata.key = analysis.key or metadata.key
        metadata.scale = analysis.scale or metadata.scale
        metadata.genre = analysis.genre_prediction or metadata.genre
        metadata.mood = analysis.mood_prediction
        metadata.energy_level = analysis.energy_level
        metadata.danceability = analysis.danceability
        metadata.valence = analysis.valence
        metadata.lufs_integrated = analysis.lufs_integrated
        metadata.peak_db = analysis.peak_db
        
        # Estimate dynamic range
        if analysis.lufs_integrated and analysis.peak_db:
            metadata.dynamic_range = analysis.peak_db - analysis.lufs_integrated
        
        # Detect instruments
        if analysis.detected_instruments:
            metadata.instruments = analysis.detected_instruments
        
        # Content classification
        metadata.content_type = analysis.content_category or "music"
    
    def save_metadata(self, metadata: AudioMetadata) -> int:
        """Save metadata to database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Check if file already exists
            cursor.execute('SELECT id FROM audio_metadata WHERE file_path = ?', (metadata.file_path,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                audio_id = existing[0]
                cursor.execute('''
                    UPDATE audio_metadata SET
                        filename = ?, file_size = ?, file_hash = ?, modified_date = ?,
                        duration = ?, sample_rate = ?, channels = ?, bitrate = ?, format = ?,
                        bpm = ?, key = ?, scale = ?, time_signature = ?, genre = ?, mood = ?,
                        energy_level = ?, danceability = ?, valence = ?, lufs_integrated = ?,
                        peak_db = ?, dynamic_range = ?, content_type = ?, instruments = ?,
                        last_analyzed = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    metadata.filename, metadata.file_size, metadata.file_hash, metadata.modified_date,
                    metadata.duration, metadata.sample_rate, metadata.channels, metadata.bitrate, metadata.format,
                    metadata.bpm, metadata.key, metadata.scale, metadata.time_signature, metadata.genre, metadata.mood,
                    metadata.energy_level, metadata.danceability, metadata.valence, metadata.lufs_integrated,
                    metadata.peak_db, metadata.dynamic_range, metadata.content_type, 
                    json.dumps(metadata.instruments) if metadata.instruments else None,
                    metadata.last_analyzed, audio_id
                ))
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO audio_metadata (
                        file_path, filename, file_size, file_hash, created_date, modified_date,
                        duration, sample_rate, channels, bitrate, format, bpm, key, scale,
                        time_signature, genre, mood, energy_level, danceability, valence,
                        lufs_integrated, peak_db, dynamic_range, content_type, instruments,
                        version, original_file, last_analyzed, analysis_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata.file_path, metadata.filename, metadata.file_size, metadata.file_hash,
                    metadata.created_date, metadata.modified_date, metadata.duration, metadata.sample_rate,
                    metadata.channels, metadata.bitrate, metadata.format, metadata.bpm, metadata.key,
                    metadata.scale, metadata.time_signature, metadata.genre, metadata.mood,
                    metadata.energy_level, metadata.danceability, metadata.valence, metadata.lufs_integrated,
                    metadata.peak_db, metadata.dynamic_range, metadata.content_type,
                    json.dumps(metadata.instruments) if metadata.instruments else None,
                    metadata.version, metadata.original_file, metadata.last_analyzed, metadata.analysis_version
                ))
                audio_id = cursor.lastrowid
            
            # Save custom metadata
            if metadata.custom_tags:
                # Delete existing custom metadata
                cursor.execute('DELETE FROM custom_metadata WHERE audio_id = ?', (audio_id,))
                
                # Insert new custom metadata
                for field_name, field_value in metadata.custom_tags.items():
                    field_type = self.custom_field_definitions.get(field_name, {}).get('type', 'string')
                    cursor.execute('''
                        INSERT INTO custom_metadata (audio_id, field_name, field_value, field_type)
                        VALUES (?, ?, ?, ?)
                    ''', (audio_id, field_name, json.dumps(field_value) if isinstance(field_value, (list, dict)) else str(field_value), field_type))
            
            conn.commit()
            metadata.id = audio_id
            return audio_id
            
        except Exception as e:
            conn.rollback()
            print(f"Error saving metadata: {e}")
            return -1
        finally:
            conn.close()
    
    def get_metadata(self, file_path: str = None, audio_id: int = None) -> Optional[AudioMetadata]:
        """Retrieve metadata from database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            if file_path:
                cursor.execute('SELECT * FROM audio_metadata WHERE file_path = ?', (os.path.abspath(file_path),))
            elif audio_id:
                cursor.execute('SELECT * FROM audio_metadata WHERE id = ?', (audio_id,))
            else:
                return None
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Convert row to metadata object
            metadata = self._row_to_metadata(row)
            
            # Load custom metadata
            cursor.execute('SELECT field_name, field_value, field_type FROM custom_metadata WHERE audio_id = ?', (metadata.id,))
            custom_rows = cursor.fetchall()
            
            for field_name, field_value, field_type in custom_rows:
                if field_type in ['list', 'dict']:
                    try:
                        metadata.custom_tags[field_name] = json.loads(field_value)
                    except:
                        metadata.custom_tags[field_name] = field_value
                else:
                    metadata.custom_tags[field_name] = field_value
            
            return metadata
            
        except Exception as e:
            print(f"Error retrieving metadata: {e}")
            return None
        finally:
            conn.close()
    
    def _row_to_metadata(self, row) -> AudioMetadata:
        """Convert database row to AudioMetadata object"""
        return AudioMetadata(
            id=row[0],
            file_path=row[1],
            filename=row[2],
            file_size=row[3],
            file_hash=row[4],
            created_date=datetime.fromisoformat(row[5]) if row[5] else None,
            modified_date=datetime.fromisoformat(row[6]) if row[6] else None,
            duration=row[7] or 0.0,
            sample_rate=row[8] or 0,
            channels=row[9] or 0,
            bitrate=row[10],
            format=row[11],
            bpm=row[12],
            key=row[13],
            scale=row[14],
            time_signature=row[15],
            genre=row[16],
            mood=row[17],
            energy_level=row[18],
            danceability=row[19],
            valence=row[20],
            lufs_integrated=row[21],
            peak_db=row[22],
            dynamic_range=row[23],
            content_type=row[24],
            instruments=json.loads(row[25]) if row[25] else [],
            version=row[26] or 1,
            original_file=row[27],
            last_analyzed=datetime.fromisoformat(row[28]) if row[28] else None,
            analysis_version=row[29] or "1.0"
        )
    
    def write_metadata_to_file(self, metadata: AudioMetadata) -> bool:
        """Write metadata back to audio file"""
        file_ext = os.path.splitext(metadata.file_path)[1].lower()
        
        if file_ext in self.format_handlers:
            return self.format_handlers[file_ext](metadata)
        else:
            print(f"Unsupported format for metadata writing: {file_ext}")
            return False
    
    def _handle_mp3_metadata(self, metadata: AudioMetadata) -> bool:
        """Write metadata to MP3 file"""
        try:
            audio_file = MP3(metadata.file_path, ID3=ID3)
            
            # Add ID3 tag if it doesn't exist
            if audio_file.tags is None:
                audio_file.add_tags()
            
            # Set standard tags
            if metadata.custom_tags.get('title'):
                audio_file.tags.add(TIT2(encoding=3, text=metadata.custom_tags['title']))
            if metadata.custom_tags.get('artist'):
                audio_file.tags.add(TPE1(encoding=3, text=metadata.custom_tags['artist']))
            if metadata.custom_tags.get('album'):
                audio_file.tags.add(TALB(encoding=3, text=metadata.custom_tags['album']))
            if metadata.genre:
                audio_file.tags.add(TCON(encoding=3, text=metadata.genre))
            if metadata.bpm:
                audio_file.tags.add(TBPM(encoding=3, text=str(int(metadata.bpm))))
            if metadata.key:
                audio_file.tags.add(TKEY(encoding=3, text=metadata.key))
            
            # Add custom tags as TXXX frames
            if metadata.mood:
                audio_file.tags.add(TXXX(encoding=3, desc='MOOD', text=metadata.mood))
            if metadata.energy_level is not None:
                audio_file.tags.add(TXXX(encoding=3, desc='ENERGY', text=str(metadata.energy_level)))
            if metadata.content_type:
                audio_file.tags.add(TXXX(encoding=3, desc='CONTENT_TYPE', text=metadata.content_type))
            
            audio_file.save()
            return True
            
        except Exception as e:
            print(f"Error writing MP3 metadata: {e}")
            return False
    
    def _handle_flac_metadata(self, metadata: AudioMetadata) -> bool:
        """Write metadata to FLAC file"""
        try:
            audio_file = FLAC(metadata.file_path)
            
            if audio_file.tags is None:
                audio_file.add_tags()
            
            # Set Vorbis comments
            if metadata.custom_tags.get('title'):
                audio_file.tags['TITLE'] = metadata.custom_tags['title']
            if metadata.custom_tags.get('artist'):
                audio_file.tags['ARTIST'] = metadata.custom_tags['artist']
            if metadata.custom_tags.get('album'):
                audio_file.tags['ALBUM'] = metadata.custom_tags['album']
            if metadata.genre:
                audio_file.tags['GENRE'] = metadata.genre
            if metadata.bpm:
                audio_file.tags['BPM'] = str(int(metadata.bpm))
            if metadata.key:
                audio_file.tags['KEY'] = metadata.key
            if metadata.mood:
                audio_file.tags['MOOD'] = metadata.mood
            if metadata.energy_level is not None:
                audio_file.tags['ENERGY'] = str(metadata.energy_level)
            
            audio_file.save()
            return True
            
        except Exception as e:
            print(f"Error writing FLAC metadata: {e}")
            return False
    
    def _handle_mp4_metadata(self, metadata: AudioMetadata) -> bool:
        """Write metadata to MP4/M4A file"""
        try:
            audio_file = MP4(metadata.file_path)
            
            if audio_file.tags is None:
                audio_file.add_tags()
            
            # Set MP4 atoms
            if metadata.custom_tags.get('title'):
                audio_file.tags['\xa9nam'] = metadata.custom_tags['title']
            if metadata.custom_tags.get('artist'):
                audio_file.tags['\xa9ART'] = metadata.custom_tags['artist']
            if metadata.custom_tags.get('album'):
                audio_file.tags['\xa9alb'] = metadata.custom_tags['album']
            if metadata.genre:
                audio_file.tags['\xa9gen'] = metadata.genre
            
            audio_file.save()
            return True
            
        except Exception as e:
            print(f"Error writing MP4 metadata: {e}")
            return False
    
    def _handle_wav_metadata(self, metadata: AudioMetadata) -> bool:
        """Write metadata to WAV file (limited support)"""
        # WAV has limited metadata support
        print("WAV metadata writing not fully supported")
        return False
    
    def _handle_aiff_metadata(self, metadata: AudioMetadata) -> bool:
        """Write metadata to AIFF file (limited support)"""
        # AIFF has limited metadata support
        print("AIFF metadata writing not fully supported")
        return False
    
    def add_processing_record(self, audio_id: int, operation: str, parameters: Dict[str, Any], 
                            input_file: str, output_file: str, success: bool, notes: str = ""):
        """Add processing history record"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO processing_history (audio_id, operation, parameters, input_file, output_file, success, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (audio_id, operation, json.dumps(parameters), input_file, output_file, success, notes))
            
            conn.commit()
        except Exception as e:
            print(f"Error adding processing record: {e}")
        finally:
            conn.close()
    
    def create_file_version(self, original_id: int, new_file_path: str, creation_method: str, 
                          parent_version: int = None) -> int:
        """Create a new version of a file"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Get next version number
            cursor.execute('SELECT MAX(version_number) FROM file_versions WHERE original_id = ?', (original_id,))
            max_version = cursor.fetchone()[0] or 0
            next_version = max_version + 1
            
            # Insert version record
            cursor.execute('''
                INSERT INTO file_versions (original_id, version_number, file_path, creation_method, parent_version)
                VALUES (?, ?, ?, ?, ?)
            ''', (original_id, next_version, new_file_path, creation_method, parent_version))
            
            version_id = cursor.lastrowid
            
            # Extract and save metadata for the new version
            new_metadata = self.extract_file_metadata(new_file_path)
            new_metadata.version = next_version
            new_metadata.original_file = self.get_metadata(audio_id=original_id).file_path if self.get_metadata(audio_id=original_id) else None
            
            new_audio_id = self.save_metadata(new_metadata)
            
            conn.commit()
            return new_audio_id
            
        except Exception as e:
            print(f"Error creating file version: {e}")
            return -1
        finally:
            conn.close()
    
    def get_file_versions(self, original_id: int) -> List[Dict[str, Any]]:
        """Get all versions of a file"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT fv.*, am.filename, am.file_size, am.duration
                FROM file_versions fv
                LEFT JOIN audio_metadata am ON am.file_path = fv.file_path
                WHERE fv.original_id = ?
                ORDER BY fv.version_number
            ''', (original_id,))
            
            versions = []
            for row in cursor.fetchall():
                versions.append({
                    'version_id': row[0],
                    'version_number': row[2],
                    'file_path': row[3],
                    'creation_method': row[4],
                    'created_at': row[6],
                    'filename': row[7],
                    'file_size': row[8],
                    'duration': row[9]
                })
            
            return versions
            
        except Exception as e:
            print(f"Error getting file versions: {e}")
            return []
        finally:
            conn.close()
    
    def search_metadata(self, **criteria) -> List[AudioMetadata]:
        """Search metadata by various criteria"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM audio_metadata WHERE 1=1"
        params = []
        
        # Build query based on criteria
        if 'genre' in criteria and criteria['genre']:
            query += " AND genre LIKE ?"
            params.append(f"%{criteria['genre']}%")
        
        if 'bpm_min' in criteria and criteria['bpm_min']:
            query += " AND bpm >= ?"
            params.append(criteria['bpm_min'])
        
        if 'bpm_max' in criteria and criteria['bpm_max']:
            query += " AND bpm <= ?"
            params.append(criteria['bpm_max'])
        
        if 'key' in criteria and criteria['key']:
            query += " AND key = ?"
            params.append(criteria['key'])
        
        if 'mood' in criteria and criteria['mood']:
            query += " AND mood LIKE ?"
            params.append(f"%{criteria['mood']}%")
        
        if 'content_type' in criteria and criteria['content_type']:
            query += " AND content_type = ?"
            params.append(criteria['content_type'])
        
        if 'duration_min' in criteria and criteria['duration_min']:
            query += " AND duration >= ?"
            params.append(criteria['duration_min'])
        
        if 'duration_max' in criteria and criteria['duration_max']:
            query += " AND duration <= ?"
            params.append(criteria['duration_max'])
        
        try:
            cursor.execute(query, params)
            results = []
            
            for row in cursor.fetchall():
                metadata = self._row_to_metadata(row)
                
                # Load custom metadata
                cursor.execute('SELECT field_name, field_value, field_type FROM custom_metadata WHERE audio_id = ?', (metadata.id,))
                custom_rows = cursor.fetchall()
                
                for field_name, field_value, field_type in custom_rows:
                    if field_type in ['list', 'dict']:
                        try:
                            metadata.custom_tags[field_name] = json.loads(field_value)
                        except:
                            metadata.custom_tags[field_name] = field_value
                    else:
                        metadata.custom_tags[field_name] = field_value
                
                results.append(metadata)
            
            return results
            
        except Exception as e:
            print(f"Error searching metadata: {e}")
            return []
        finally:
            conn.close()
    
    def add_tags(self, audio_id: int, tags: List[str], category: str = "general"):
        """Add tags to an audio file"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            for tag in tags:
                cursor.execute('''
                    INSERT OR IGNORE INTO tags (audio_id, tag_name, tag_category)
                    VALUES (?, ?, ?)
                ''', (audio_id, tag.strip(), category))
            
            conn.commit()
        except Exception as e:
            print(f"Error adding tags: {e}")
        finally:
            conn.close()
    
    def get_tags(self, audio_id: int) -> List[Dict[str, str]]:
        """Get all tags for an audio file"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT tag_name, tag_category FROM tags WHERE audio_id = ?', (audio_id,))
            return [{'name': row[0], 'category': row[1]} for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting tags: {e}")
            return []
        finally:
            conn.close()
    
    def batch_process_directory(self, directory: str, auto_tag: bool = True, 
                              write_to_files: bool = False) -> int:
        """Process all audio files in a directory"""
        processed_count = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in ['.mp3', '.flac', '.wav', '.m4a', '.aiff']):
                    file_path = os.path.join(root, file)
                    
                    print(f"Processing: {os.path.basename(file_path)}")
                    
                    # Extract and save metadata
                    metadata = self.extract_file_metadata(file_path)
                    audio_id = self.save_metadata(metadata)
                    
                    if audio_id > 0:
                        processed_count += 1
                        
                        # Auto-tag based on analysis
                        if auto_tag:
                            self._auto_tag_file(audio_id, metadata)
                        
                        # Write metadata back to file
                        if write_to_files:
                            self.write_metadata_to_file(metadata)
        
        return processed_count
    
    def _auto_tag_file(self, audio_id: int, metadata: AudioMetadata):
        """Automatically generate tags based on analysis"""
        tags = []
        
        # BPM-based tags
        if metadata.bpm:
            if metadata.bpm < 70:
                tags.extend(["slow", "ballad", "chill"])
            elif metadata.bpm < 90:
                tags.extend(["moderate", "downtempo"])
            elif metadata.bpm < 120:
                tags.extend(["medium", "walking pace"])
            elif metadata.bpm < 140:
                tags.extend(["upbeat", "danceable"])
            else:
                tags.extend(["fast", "high energy", "uptempo"])
        
        # Key-based tags
        if metadata.key:
            if 'm' in metadata.key.lower() or metadata.scale == 'minor':
                tags.append("minor key")
            else:
                tags.append("major key")
        
        # Genre-based tags
        if metadata.genre:
            tags.append(metadata.genre.lower())
        
        # Mood-based tags
        if metadata.mood:
            tags.append(metadata.mood.lower())
        
        # Energy-based tags
        if metadata.energy_level is not None:
            if metadata.energy_level > 0.7:
                tags.append("high energy")
            elif metadata.energy_level < 0.3:
                tags.append("low energy")
            else:
                tags.append("medium energy")
        
        # Content type tags
        if metadata.content_type:
            tags.append(metadata.content_type)
        
        # Instrument tags
        if metadata.instruments:
            tags.extend([f"has {instrument}" for instrument in metadata.instruments])
        
        # Add all tags
        if tags:
            self.add_tags(audio_id, tags, "auto-generated")
    
    def export_metadata(self, output_file: str, format: str = "json") -> bool:
        """Export all metadata to file"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM audio_metadata')
            rows = cursor.fetchall()
            
            metadata_list = []
            for row in rows:
                metadata = self._row_to_metadata(row)
                
                # Load custom metadata
                cursor.execute('SELECT field_name, field_value FROM custom_metadata WHERE audio_id = ?', (metadata.id,))
                custom_rows = cursor.fetchall()
                for field_name, field_value in custom_rows:
                    metadata.custom_tags[field_name] = field_value
                
                # Convert to serializable format
                metadata_dict = asdict(metadata)
                metadata_dict['created_date'] = metadata.created_date.isoformat() if metadata.created_date else None
                metadata_dict['modified_date'] = metadata.modified_date.isoformat() if metadata.modified_date else None
                metadata_dict['last_analyzed'] = metadata.last_analyzed.isoformat() if metadata.last_analyzed else None
                
                metadata_list.append(metadata_dict)
            
            if format.lower() == "json":
                with open(output_file, 'w') as f:
                    json.dump(metadata_list, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error exporting metadata: {e}")
            return False
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        stats = {}
        
        try:
            # Total files
            cursor.execute('SELECT COUNT(*) FROM audio_metadata')
            stats['total_files'] = cursor.fetchone()[0]
            
            # Genre distribution
            cursor.execute('SELECT genre, COUNT(*) FROM audio_metadata WHERE genre IS NOT NULL GROUP BY genre')
            stats['genre_distribution'] = dict(cursor.fetchall())
            
            # BPM distribution
            cursor.execute('SELECT ROUND(bpm/10)*10 as bpm_range, COUNT(*) FROM audio_metadata WHERE bpm IS NOT NULL GROUP BY bpm_range')
            stats['bpm_distribution'] = {f"{int(bpm)}-{int(bpm)+9}": count for bpm, count in cursor.fetchall()}
            
            # Key distribution
            cursor.execute('SELECT key, COUNT(*) FROM audio_metadata WHERE key IS NOT NULL GROUP BY key')
            stats['key_distribution'] = dict(cursor.fetchall())
            
            # Content type distribution
            cursor.execute('SELECT content_type, COUNT(*) FROM audio_metadata WHERE content_type IS NOT NULL GROUP BY content_type')
            stats['content_type_distribution'] = dict(cursor.fetchall())
            
            # Total duration
            cursor.execute('SELECT SUM(duration) FROM audio_metadata')
            total_duration = cursor.fetchone()[0] or 0
            stats['total_duration_hours'] = total_duration / 3600
            
            # File formats
            cursor.execute('SELECT format, COUNT(*) FROM audio_metadata WHERE format IS NOT NULL GROUP BY format')
            stats['format_distribution'] = dict(cursor.fetchall())
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
        finally:
            conn.close()