#!/usr/bin/env python3
"""
Test Professional Audio Features
Demonstrates the new professional audio processing capabilities.
"""

import os
import sys
from pathlib import Path
from audio_analyzer import ProfessionalAudioProcessor

def test_stem_separation():
    """Test audio stem separation functionality"""
    print("\n" + "="*60)
    print("TESTING STEM SEPARATION")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    # Use existing sample audio
    input_file = "sample_audio/electronic_beat.wav"
    output_dir = "test_outputs/stem_separation"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    # Test stem separation
    print(f"Separating stems from: {input_file}")
    print("Extracting: vocals, drums, bass, other")
    
    result = processor.stem_separation(
        input_path=input_file,
        output_dir=output_dir,
        stems=['vocals', 'drums', 'bass', 'other']
    )
    
    if result.success:
        print(f"✅ {result.message}")
        print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
        print(f"📁 Output directory: {result.output_path}")
        
        if result.additional_outputs:
            print("🎵 Separated stems:")
            for stem, path in result.additional_outputs.items():
                print(f"   - {stem}: {path}")
        
        if result.quality_metrics:
            print(f"📊 Quality metrics: {result.quality_metrics}")
    else:
        print(f"❌ {result.message}")

def test_format_conversion():
    """Test format conversion with metadata preservation"""
    print("\n" + "="*60)
    print("TESTING FORMAT CONVERSION WITH METADATA")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    input_file = "sample_audio/piano_melody.wav"
    output_file = "test_outputs/converted_piano_melody.flac"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Converting: {input_file} -> {output_file}")
    print("Target format: FLAC with metadata preservation")
    
    result = processor.format_conversion_with_metadata(
        input_path=input_file,
        output_path=output_file,
        target_format="flac",
        preserve_metadata=True,
        quality_settings={
            'flac_compression': 5,
            'wav_bit_depth': 24
        }
    )
    
    if result.success:
        print(f"✅ {result.message}")
        print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
        print(f"📁 Output file: {result.output_path}")
        
        if result.quality_metrics:
            metrics = result.quality_metrics
            print("📊 Conversion metrics:")
            print(f"   - Compression ratio: {metrics.get('compression_ratio', 'N/A'):.3f}")
            print(f"   - Metadata preserved: {metrics.get('metadata_preserved', 'N/A')}")
            print(f"   - Original size: {metrics.get('original_size_bytes', 0):,} bytes")
            print(f"   - Converted size: {metrics.get('converted_size_bytes', 0):,} bytes")
    else:
        print(f"❌ {result.message}")

def test_broadcast_compliance():
    """Test broadcast standards compliance checking"""
    print("\n" + "="*60)
    print("TESTING BROADCAST STANDARDS COMPLIANCE")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    input_file = "sample_audio/rock_guitar.wav"
    output_file = "test_outputs/broadcast_compliant.wav"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Checking compliance: {input_file}")
    print("Standard: EBU R128 (European Broadcasting Union)")
    
    result = processor.broadcast_standards_compliance(
        input_path=input_file,
        output_path=output_file,
        standard="EBU_R128",
        auto_correct=True
    )
    
    if result.success:
        print(f"✅ {result.message}")
        print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
        print(f"📁 Output file: {result.output_path}")
        
        if result.quality_metrics:
            compliance = result.quality_metrics.get('compliance_report', {})
            corrections = result.quality_metrics.get('corrections_applied', [])
            overall = result.quality_metrics.get('overall_compliant', False)
            
            print("📊 Compliance report:")
            print(f"   - Overall compliant: {'✅' if overall else '❌'}")
            
            if 'integrated_loudness' in compliance:
                print(f"   - Integrated loudness: {compliance['integrated_loudness']:.1f} LUFS")
                print(f"   - Target: {compliance['target_loudness']:.1f} LUFS")
                print(f"   - Loudness compliant: {'✅' if compliance.get('loudness_compliant') else '❌'}")
            
            if 'true_peak' in compliance:
                print(f"   - True peak: {compliance['true_peak']:.1f} dB")
                print(f"   - Max allowed: {compliance['max_true_peak']:.1f} dB")
                print(f"   - Peak compliant: {'✅' if compliance.get('peak_compliant') else '❌'}")
            
            if corrections:
                print("🔧 Corrections applied:")
                for correction in corrections:
                    print(f"   - {correction}")
    else:
        print(f"❌ {result.message}")

def test_quality_assurance():
    """Test quality assurance checking"""
    print("\n" + "="*60)
    print("TESTING QUALITY ASSURANCE")
    print("="*60)
    
    processor = ProfessionalAudioProcessor()
    
    input_file = "sample_audio/ambient_pad.wav"
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    print(f"Quality analysis: {input_file}")
    print("Checking for clipping, distortion, frequency balance, etc.")
    
    result = processor.quality_assurance_check(
        input_path=input_file,
        thresholds={
            'clip_threshold': 0.99,
            'thd_threshold': 1.0,
            'snr_minimum': 40.0,
            'dc_offset_max': 0.01,
            'silence_threshold': -60.0,
            'peak_margin': -3.0
        }
    )
    
    if result.success:
        print(f"✅ {result.message}")
        print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
        
        if result.quality_metrics:
            qa_report = result.quality_metrics
            overall = qa_report.get('overall_assessment', {})
            
            print("📊 Quality Assessment:")
            print(f"   - Quality rating: {overall.get('quality_rating', 'Unknown')}")
            print(f"   - Issues found: {len(overall.get('issues_found', []))}")
            print(f"   - Warnings: {len(overall.get('warnings', []))}")
            
            # File info
            file_info = qa_report.get('file_info', {})
            print("📁 File information:")
            print(f"   - Duration: {file_info.get('duration', 0):.2f} seconds")
            print(f"   - Sample rate: {file_info.get('sample_rate', 0)} Hz")
            print(f"   - Channels: {file_info.get('channels', 0)}")
            
            # Channel analysis
            for key, value in qa_report.items():
                if key.startswith('Channel_') or key == 'Mono':
                    print(f"🎵 {key} analysis:")
                    
                    # Clipping
                    clipping = value.get('clipping', {})
                    if clipping.get('is_clipped'):
                        print(f"   - ⚠️  Clipping: {clipping.get('clip_percentage', 0):.3f}%")
                    else:
                        print(f"   - ✅ No clipping detected")
                    
                    # Peak analysis
                    peak = value.get('peak_analysis', {})
                    print(f"   - Peak level: {peak.get('peak_level_db', 0):.1f} dB")
                    print(f"   - Headroom: {peak.get('headroom_db', 0):.1f} dB")
                    
                    # DC offset
                    dc = value.get('dc_offset', {})
                    if dc.get('has_dc_offset'):
                        print(f"   - ⚠️  DC offset: {dc.get('offset_value', 0):.4f}")
                    
                    # Dynamic range
                    dr = value.get('dynamic_range', {})
                    print(f"   - RMS level: {dr.get('rms_level_db', 0):.1f} dB")
                    print(f"   - Crest factor: {dr.get('crest_factor_db', 0):.1f} dB")
                    if dr.get('compressed'):
                        print(f"   - ⚠️  Possibly over-compressed")
            
            # Issues and warnings
            issues = overall.get('issues_found', [])
            warnings = overall.get('warnings', [])
            
            if issues:
                print("🚨 Issues found:")
                for issue in issues:
                    print(f"   - {issue}")
            
            if warnings:
                print("⚠️  Warnings:")
                for warning in warnings:
                    print(f"   - {warning}")
            
            # Recommendation
            recommendation = overall.get('recommendation', '')
            if recommendation:
                print(f"💡 Recommendation: {recommendation}")
            
    else:
        print(f"❌ {result.message}")

def main():
    """Run all professional feature tests"""
    print("🎵 PROFESSIONAL AUDIO FEATURES TEST SUITE")
    print("Testing advanced audio processing capabilities...")
    
    # Create output directory
    os.makedirs("test_outputs", exist_ok=True)
    
    # Test all features
    test_stem_separation()
    test_format_conversion()
    test_broadcast_compliance()
    test_quality_assurance()
    
    print("\n" + "="*60)
    print("PROFESSIONAL FEATURES TEST COMPLETED")
    print("="*60)
    print("Check the test_outputs/ directory for generated files")

if __name__ == "__main__":
    main()