#!/usr/bin/env python3
"""
ML Classifier Training Script
Train machine learning models for genre and mood classification
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict
from audio_analyzer import IntelligentAudioAnalyzer

def create_training_dataset():
    """Create a training dataset from organized audio files"""
    print("🎓 CREATING TRAINING DATASET")
    print("=" * 50)
    
    # Expected directory structure:
    # training_data/
    #   ├── rock/
    #   ├── pop/
    #   ├── electronic/
    #   ├── hip-hop/
    #   ├── classical/
    #   └── jazz/
    
    training_dir = Path("training_data")
    if not training_dir.exists():
        print("❌ Training data directory not found.")
        print("Please create a 'training_data' directory with subdirectories for each genre:")
        print("  training_data/")
        print("    ├── rock/")
        print("    ├── pop/")
        print("    ├── electronic/")
        print("    ├── hip-hop/")
        print("    ├── classical/")
        print("    └── jazz/")
        return []
    
    training_data = []
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
    
    for genre_dir in training_dir.iterdir():
        if genre_dir.is_dir():
            genre = genre_dir.name
            print(f"📁 Processing {genre} samples...")
            
            genre_files = []
            for ext in audio_extensions:
                genre_files.extend(genre_dir.glob(f'*{ext}'))
            
            print(f"   Found {len(genre_files)} audio files")
            
            for audio_file in genre_files:
                training_data.append((str(audio_file), genre))
    
    print(f"\n✅ Created training dataset with {len(training_data)} samples")
    return training_data

def train_classifiers():
    """Train ML classifiers for audio analysis"""
    print("\n🤖 TRAINING ML CLASSIFIERS")
    print("=" * 50)
    
    # Create training dataset
    training_data = create_training_dataset()
    
    if len(training_data) < 10:
        print("❌ Not enough training data. Need at least 10 labeled samples.")
        print("Please add more audio files to the training_data directories.")
        return False
    
    # Initialize analyzer
    analyzer = IntelligentAudioAnalyzer()
    
    # Train genre classifier
    print("🎭 Training genre classifier...")
    success = analyzer.train_genre_classifier(training_data)
    
    if success:
        print("✅ Genre classifier trained successfully!")
        
        # Test the classifier
        print("\n🧪 Testing classifier...")
        test_classifier(analyzer, training_data[:5])  # Test on first 5 samples
        
        return True
    else:
        print("❌ Failed to train genre classifier")
        return False

def test_classifier(analyzer, test_samples):
    """Test the trained classifier"""
    print("📊 CLASSIFIER TEST RESULTS:")
    print("-" * 30)
    
    correct = 0
    total = len(test_samples)
    
    for file_path, true_genre in test_samples:
        try:
            # Analyze with trained model
            result = analyzer.analyze_audio_file(file_path)
            predicted_genre = result.genre_prediction
            confidence = result.genre_confidence or 0.0
            
            is_correct = predicted_genre == true_genre
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {Path(file_path).name}")
            print(f"   True: {true_genre} | Predicted: {predicted_genre} ({confidence:.2f})")
            
        except Exception as e:
            print(f"❌ Error testing {file_path}: {e}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n📈 Accuracy: {accuracy:.2%} ({correct}/{total})")

def create_sample_training_structure():
    """Create sample training directory structure"""
    print("\n📁 CREATING SAMPLE TRAINING STRUCTURE")
    print("=" * 50)
    
    training_dir = Path("training_data")
    
    # Create genre directories
    genres = ["rock", "pop", "electronic", "hip-hop", "classical", "jazz", "ambient", "country"]
    
    for genre in genres:
        genre_dir = training_dir / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a README file in each directory
        readme_path = genre_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"# {genre.title()} Training Samples\n\n")
            f.write(f"Add {genre} audio files to this directory for training.\n\n")
            f.write("Supported formats: MP3, WAV, FLAC, M4A, OGG\n\n")
            f.write("Tips for good training data:\n")
            f.write("- Use high-quality audio files\n")
            f.write("- Include diverse examples of the genre\n")
            f.write("- Aim for 10+ samples per genre\n")
            f.write("- Keep files under 5 minutes for faster training\n")
    
    print(f"✅ Created training directories for {len(genres)} genres:")
    for genre in genres:
        print(f"   📂 training_data/{genre}/")
    
    print(f"\n💡 Add audio files to these directories and run this script again to train.")

def analyze_training_progress():
    """Analyze current training data status"""
    print("\n📊 TRAINING DATA ANALYSIS")
    print("=" * 50)
    
    training_dir = Path("training_data")
    if not training_dir.exists():
        print("❌ No training data directory found.")
        return
    
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
    genre_counts = {}
    total_files = 0
    
    for genre_dir in training_dir.iterdir():
        if genre_dir.is_dir() and not genre_dir.name.startswith('.'):
            genre = genre_dir.name
            
            genre_files = []
            for ext in audio_extensions:
                genre_files.extend(genre_dir.glob(f'*{ext}'))
            
            count = len(genre_files)
            genre_counts[genre] = count
            total_files += count
    
    print(f"📁 Total audio files: {total_files}")
    print(f"🎭 Genres found: {len(genre_counts)}")
    print()
    
    if genre_counts:
        print("📈 Files per genre:")
        for genre, count in sorted(genre_counts.items()):
            status = "✅" if count >= 10 else "⚠️" if count >= 5 else "❌"
            print(f"   {status} {genre}: {count} files")
        
        print()
        ready_count = sum(1 for count in genre_counts.values() if count >= 5)
        print(f"🎯 Genres ready for training: {ready_count}/{len(genre_counts)}")
        
        if ready_count >= 2 and total_files >= 10:
            print("✅ Ready to train classifier!")
        else:
            print("⚠️ Need more training data. Minimum: 2 genres with 5+ files each.")
    else:
        print("❌ No audio files found in training directories.")

def main():
    """Main training function"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            create_sample_training_structure()
        elif command == "analyze":
            analyze_training_progress()
        elif command == "train":
            train_classifiers()
        else:
            print("❌ Unknown command. Use: setup, analyze, or train")
    else:
        print("🎓 ML CLASSIFIER TRAINING TOOL")
        print("=" * 40)
        print()
        print("Available commands:")
        print("  python train_ml_classifier.py setup    - Create training directories")
        print("  python train_ml_classifier.py analyze  - Check training data status")
        print("  python train_ml_classifier.py train    - Train the classifier")
        print()
        
        # Auto-analyze current status
        analyze_training_progress()

if __name__ == "__main__":
    main()