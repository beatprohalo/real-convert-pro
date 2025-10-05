#!/usr/bin/env python3
"""
Professional Audio Tools GUI Demo
Simple graphical interface to demonstrate professional audio processing capabilities.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
import json
from audio_analyzer import ProfessionalAudioProcessor

class ProfessionalToolsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Audio Tools")
        self.root.geometry("800x600")
        
        # Initialize processor
        self.processor = ProfessionalAudioProcessor()
        self.input_file = None
        self.output_dir = Path("gui_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface"""
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection frame
        file_frame = ttk.Frame(self.root)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(file_frame, text="Input Audio File:").pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.RIGHT)
        
        # Create tabs
        self.create_time_stretch_tab(notebook)
        self.create_repair_tab(notebook)
        self.create_spectral_tab(notebook)
        self.create_stereo_tab(notebook)
        self.create_mastering_tab(notebook)
        # New professional feature tabs
        self.create_stem_separation_tab(notebook)
        self.create_format_conversion_tab(notebook)
        self.create_broadcast_compliance_tab(notebook)
        self.create_quality_assurance_tab(notebook)
        
        # Output log
        log_frame = ttk.LabelFrame(self.root, text="Processing Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
    def create_time_stretch_tab(self, notebook):
        """Create time-stretching tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Time Stretch")
        
        # Stretch factor
        ttk.Label(frame, text="Stretch Factor:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.stretch_var = tk.DoubleVar(value=1.5)
        stretch_scale = ttk.Scale(frame, from_=0.25, to=4.0, variable=self.stretch_var, orient=tk.HORIZONTAL)
        stretch_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Label(frame, textvariable=self.stretch_var).grid(row=0, column=2, padx=5, pady=5)
        
        # Quality
        ttk.Label(frame, text="Quality:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.quality_var = tk.StringVar(value="high")
        quality_combo = ttk.Combobox(frame, textvariable=self.quality_var, 
                                   values=["low", "medium", "high", "ultra"], state="readonly")
        quality_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Preserve pitch
        self.preserve_pitch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Preserve Pitch", variable=self.preserve_pitch_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Process button
        ttk.Button(frame, text="Process Time Stretch", command=self.process_time_stretch).grid(row=3, column=0, columnspan=3, pady=10)
        
        frame.columnconfigure(1, weight=1)
        
    def create_repair_tab(self, notebook):
        """Create audio repair tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Audio Repair")
        
        ttk.Label(frame, text="Select repair operations:").grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        self.declick_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Declick (remove clicks and pops)", variable=self.declick_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        self.denoise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Denoise (remove background noise)", variable=self.denoise_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        self.dehum_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Dehum (remove 50/60Hz hum)", variable=self.dehum_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        self.normalize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Normalize levels", variable=self.normalize_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        ttk.Button(frame, text="Process Audio Repair", command=self.process_repair).grid(row=5, column=0, columnspan=2, pady=10)
        
    def create_spectral_tab(self, notebook):
        """Create spectral editing tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Spectral Edit")
        
        # Frequency range inputs
        ttk.Label(frame, text="Frequency Range (Hz):").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(frame, text="Low Freq:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.low_freq_var = tk.IntVar(value=800)
        ttk.Entry(frame, textvariable=self.low_freq_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="High Freq:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.high_freq_var = tk.IntVar(value=1200)
        ttk.Entry(frame, textvariable=self.high_freq_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # Operation
        ttk.Label(frame, text="Operation:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.spectral_op_var = tk.StringVar(value="remove")
        op_combo = ttk.Combobox(frame, textvariable=self.spectral_op_var,
                              values=["remove", "isolate", "attenuate"], state="readonly")
        op_combo.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Button(frame, text="Process Spectral Edit", command=self.process_spectral).grid(row=4, column=0, columnspan=2, pady=10)
        
    def create_stereo_tab(self, notebook):
        """Create stereo manipulation tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Stereo Field")
        
        # Operation selection
        ttk.Label(frame, text="Operation:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.stereo_op_var = tk.StringVar(value="width_control")
        op_combo = ttk.Combobox(frame, textvariable=self.stereo_op_var,
                              values=["width_control", "mono_to_stereo", "pan"], state="readonly")
        op_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        op_combo.bind('<<ComboboxSelected>>', self.on_stereo_op_change)
        
        # Width control
        self.width_frame = ttk.LabelFrame(frame, text="Width Control")
        self.width_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(self.width_frame, text="Width:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.width_var = tk.DoubleVar(value=1.5)
        width_scale = ttk.Scale(self.width_frame, from_=0.0, to=3.0, variable=self.width_var, orient=tk.HORIZONTAL)
        width_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(self.width_frame, textvariable=self.width_var).grid(row=0, column=2, padx=5, pady=2)
        
        # Pan control
        self.pan_frame = ttk.LabelFrame(frame, text="Pan Control")
        self.pan_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(self.pan_frame, text="Position:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.pan_var = tk.DoubleVar(value=0.0)
        pan_scale = ttk.Scale(self.pan_frame, from_=-1.0, to=1.0, variable=self.pan_var, orient=tk.HORIZONTAL)
        pan_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(self.pan_frame, textvariable=self.pan_var).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Button(frame, text="Process Stereo Field", command=self.process_stereo).grid(row=3, column=0, columnspan=2, pady=10)
        
        frame.columnconfigure(1, weight=1)
        self.width_frame.columnconfigure(1, weight=1)
        self.pan_frame.columnconfigure(1, weight=1)
        
    def create_mastering_tab(self, notebook):
        """Create mastering tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Mastering")
        
        # Target LUFS
        ttk.Label(frame, text="Target LUFS:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.lufs_var = tk.DoubleVar(value=-14.0)
        lufs_scale = ttk.Scale(frame, from_=-30.0, to=-6.0, variable=self.lufs_var, orient=tk.HORIZONTAL)
        lufs_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Label(frame, textvariable=self.lufs_var).grid(row=0, column=2, padx=5, pady=5)
        
        # Compression
        comp_frame = ttk.LabelFrame(frame, text="Compression")
        comp_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(comp_frame, text="Threshold (dB):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.comp_threshold_var = tk.DoubleVar(value=-12.0)
        ttk.Scale(comp_frame, from_=-30.0, to=0.0, variable=self.comp_threshold_var, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(comp_frame, textvariable=self.comp_threshold_var).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(comp_frame, text="Ratio:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.comp_ratio_var = tk.DoubleVar(value=3.0)
        ttk.Scale(comp_frame, from_=1.0, to=10.0, variable=self.comp_ratio_var, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(comp_frame, textvariable=self.comp_ratio_var).grid(row=1, column=2, padx=5, pady=2)
        
        # EQ
        self.eq_enable_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Enable EQ (Low boost, Presence boost)", variable=self.eq_enable_var).grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(frame, text="Process Mastering Chain", command=self.process_mastering).grid(row=3, column=0, columnspan=3, pady=10)
        
        frame.columnconfigure(1, weight=1)
        comp_frame.columnconfigure(1, weight=1)
        
    def on_stereo_op_change(self, event=None):
        """Handle stereo operation change"""
        op = self.stereo_op_var.get()
        if op == "width_control":
            self.width_frame.grid()
            self.pan_frame.grid_remove()
        elif op == "pan":
            self.width_frame.grid_remove()
            self.pan_frame.grid()
        else:  # mono_to_stereo
            self.width_frame.grid_remove()
            self.pan_frame.grid_remove()
    
    # New Professional Feature Tab Methods
    
    def create_stem_separation_tab(self, notebook):
        """Create advanced stem separation tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Stem Separation")
        
        ttk.Label(frame, text="Advanced AI-powered stem separation").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Method selection
        method_frame = ttk.LabelFrame(frame, text="Separation Method")
        method_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(method_frame, text="Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.stem_method_var = tk.StringVar(value="advanced")
        method_combo = ttk.Combobox(method_frame, textvariable=self.stem_method_var,
                                   values=["basic", "advanced", "spectral", "nmf", "ai"], state="readonly")
        method_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        
        ttk.Label(method_frame, text="Quality:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.stem_quality_var = tk.StringVar(value="high")
        quality_combo = ttk.Combobox(method_frame, textvariable=self.stem_quality_var,
                                    values=["fast", "balanced", "high", "ultra"], state="readonly")
        quality_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        
        method_frame.columnconfigure(1, weight=1)
        
        # Stem selection
        stems_frame = ttk.LabelFrame(frame, text="Stems to Extract")
        stems_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        self.stem_vocals_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stems_frame, text="Vocals", variable=self.stem_vocals_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.stem_drums_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stems_frame, text="Drums", variable=self.stem_drums_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.stem_bass_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stems_frame, text="Bass", variable=self.stem_bass_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.stem_piano_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(stems_frame, text="Piano", variable=self.stem_piano_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.stem_other_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stems_frame, text="Other/Accompaniment", variable=self.stem_other_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        ttk.Button(frame, text="Separate Stems", command=self.process_stem_separation).grid(row=3, column=0, columnspan=3, pady=10)
        
        # Method descriptions
        info_frame = ttk.LabelFrame(frame, text="Method Information")
        info_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        info_text = """• Basic: Simple harmonic-percussive separation
• Advanced: Multi-technique combination with enhancement
• Spectral: High-resolution frequency domain separation  
• NMF: Non-negative Matrix Factorization based
• AI: Deep learning based (requires models)"""
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, font=("Courier", 9)).grid(row=0, column=0, padx=5, pady=5)
        
        frame.columnconfigure(1, weight=1)
    
    def create_format_conversion_tab(self, notebook):
        """Create format conversion tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Format Convert")
        
        ttk.Label(frame, text="Convert audio format with metadata preservation").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Target format
        ttk.Label(frame, text="Target Format:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.format_var = tk.StringVar(value="flac")
        format_combo = ttk.Combobox(frame, textvariable=self.format_var,
                                   values=["wav", "flac", "aiff", "mp3", "aac", "m4a", "ogg", "opus", "wma", "ac3", "au", "caf", "w64"], state="readonly")
        format_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Metadata preservation
        self.preserve_metadata_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Preserve metadata", variable=self.preserve_metadata_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Quality settings
        quality_frame = ttk.LabelFrame(frame, text="Quality Settings")
        quality_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(quality_frame, text="Bit Depth:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.bit_depth_var = tk.IntVar(value=24)
        bit_depth_combo = ttk.Combobox(quality_frame, textvariable=self.bit_depth_var,
                                      values=[16, 24, 32], state="readonly", width=10)
        bit_depth_combo.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Button(frame, text="Convert Format", command=self.process_format_conversion).grid(row=4, column=0, columnspan=3, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    def create_broadcast_compliance_tab(self, notebook):
        """Create broadcast compliance tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Broadcast Compliance")
        
        ttk.Label(frame, text="Check and ensure broadcast standards compliance").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Standard selection
        ttk.Label(frame, text="Broadcast Standard:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.broadcast_standard_var = tk.StringVar(value="EBU_R128")
        standard_combo = ttk.Combobox(frame, textvariable=self.broadcast_standard_var,
                                    values=["EBU_R128", "ATSC_A85", "ITU_BS1770"], state="readonly")
        standard_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Auto-correct option
        self.auto_correct_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Automatically correct non-compliant audio", 
                       variable=self.auto_correct_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Standards info
        info_frame = ttk.LabelFrame(frame, text="Standard Information")
        info_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        info_text = """EBU R128: -23 LUFS, -1 dBTP (European Broadcasting)
ATSC A85: -24 LUFS, -2 dBTP (US Digital TV)
ITU BS1770: -23 LUFS, -1 dBTP (International)"""
        ttk.Label(info_frame, text=info_text, font=("Courier", 9)).grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(frame, text="Check Compliance", command=self.process_broadcast_compliance).grid(row=4, column=0, columnspan=3, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    def create_quality_assurance_tab(self, notebook):
        """Create quality assurance tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Quality Assurance")
        
        ttk.Label(frame, text="Comprehensive audio quality analysis").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Threshold settings
        thresholds_frame = ttk.LabelFrame(frame, text="Detection Thresholds")
        thresholds_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        # Clipping threshold
        ttk.Label(thresholds_frame, text="Clipping Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.clip_threshold_var = tk.DoubleVar(value=0.99)
        ttk.Entry(thresholds_frame, textvariable=self.clip_threshold_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # Peak margin
        ttk.Label(thresholds_frame, text="Peak Margin (dB):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.peak_margin_var = tk.DoubleVar(value=-3.0)
        ttk.Entry(thresholds_frame, textvariable=self.peak_margin_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # DC offset threshold
        ttk.Label(thresholds_frame, text="DC Offset Max:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.dc_offset_var = tk.DoubleVar(value=0.01)
        ttk.Entry(thresholds_frame, textvariable=self.dc_offset_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # What will be checked
        checks_frame = ttk.LabelFrame(frame, text="Quality Checks")
        checks_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        checks_text = """• Digital clipping detection
• Peak level and headroom analysis
• DC offset detection
• Dynamic range analysis
• Frequency balance check
• Distortion estimation
• Silence detection"""
        ttk.Label(checks_frame, text=checks_text, justify=tk.LEFT).grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(frame, text="Run Quality Analysis", command=self.process_quality_assurance).grid(row=3, column=0, columnspan=3, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    # Processing methods for new features
    
    def process_stem_separation(self):
        """Process advanced stem separation"""
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input file first")
            return
        
        # Get selected stems
        stems = []
        if self.stem_vocals_var.get():
            stems.append('vocals')
        if self.stem_drums_var.get():
            stems.append('drums')
        if self.stem_bass_var.get():
            stems.append('bass')
        if self.stem_piano_var.get():
            stems.append('piano')
        if self.stem_other_var.get():
            stems.append('other')
        
        if not stems:
            messagebox.showerror("Error", "Please select at least one stem to extract")
            return
        
        # Get method and quality settings
        method = self.stem_method_var.get()
        quality = self.stem_quality_var.get()
        
        output_dir = self.output_dir / "stem_separation" / f"{method}_{quality}"
        
        self.log(f"Starting {method} stem separation ({quality} quality): {', '.join(stems)}")
        self.start_progress()
        
        def process():
            result = self.processor.stem_separation(
                input_path=str(self.input_file),
                output_dir=str(output_dir),
                stems=stems,
                method=method,
                quality=quality
            )
            self.root.after(0, lambda: self.stem_separation_complete(result))
        
        threading.Thread(target=process, daemon=True).start()
    
    def stem_separation_complete(self, result):
        """Handle stem separation completion"""
        if result.success:
            self.log(f"✅ {result.message}")
            if result.additional_outputs:
                self.log("Generated stems:")
                for stem, path in result.additional_outputs.items():
                    self.log(f"  - {stem}: {Path(path).name}")
            
            # Display quality metrics
            if result.quality_metrics:
                metrics = result.quality_metrics
                self.log(f"Separation quality:")
                self.log(f"  - Success rate: {metrics.get('success_rate', 0):.1%}")
                self.log(f"  - Energy conservation: {metrics.get('energy_conservation', 0):.3f}")
                
                # Show RMS levels for each stem
                for stem in result.additional_outputs.keys():
                    rms_key = f'{stem}_rms'
                    if rms_key in metrics:
                        self.log(f"  - {stem} RMS: {metrics[rms_key]:.4f}")
                        
        else:
            self.log(f"❌ {result.message}")
        self.stop_progress()
    
    def process_format_conversion(self):
        """Process format conversion"""
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input file first")
            return
        
        target_format = self.format_var.get()
        input_path = Path(self.input_file)
        output_path = self.output_dir / f"{input_path.stem}_converted.{target_format}"
        
        self.log(f"Converting to {target_format.upper()} format")
        self.start_progress()
        
        def process():
            result = self.processor.format_conversion_with_metadata(
                input_path=str(self.input_file),
                output_path=str(output_path),
                target_format=target_format,
                preserve_metadata=self.preserve_metadata_var.get(),
                quality_settings={'wav_bit_depth': self.bit_depth_var.get()}
            )
            self.root.after(0, lambda: self.format_conversion_complete(result))
        
        threading.Thread(target=process, daemon=True).start()
    
    def format_conversion_complete(self, result):
        """Handle format conversion completion"""
        if result.success:
            self.log(f"✅ {result.message}")
            if result.quality_metrics:
                metrics = result.quality_metrics
                self.log(f"Compression ratio: {metrics.get('compression_ratio', 0):.3f}")
        else:
            self.log(f"❌ {result.message}")
        self.stop_progress()
    
    def process_broadcast_compliance(self):
        """Process broadcast compliance check"""
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input file first")
            return
        
        standard = self.broadcast_standard_var.get()
        input_path = Path(self.input_file)
        output_path = self.output_dir / f"{input_path.stem}_broadcast_compliant.wav"
        
        self.log(f"Checking {standard} compliance")
        self.start_progress()
        
        def process():
            result = self.processor.broadcast_standards_compliance(
                input_path=str(self.input_file),
                output_path=str(output_path),
                standard=standard,
                auto_correct=self.auto_correct_var.get()
            )
            self.root.after(0, lambda: self.broadcast_compliance_complete(result))
        
        threading.Thread(target=process, daemon=True).start()
    
    def broadcast_compliance_complete(self, result):
        """Handle broadcast compliance completion"""
        if result.success:
            self.log(f"✅ {result.message}")
            if result.quality_metrics:
                compliance = result.quality_metrics.get('compliance_report', {})
                overall = result.quality_metrics.get('overall_compliant', False)
                self.log(f"Overall compliant: {'Yes' if overall else 'No'}")
                
                if 'integrated_loudness' in compliance:
                    self.log(f"Loudness: {compliance['integrated_loudness']:.1f} LUFS")
                if 'true_peak' in compliance:
                    self.log(f"True peak: {compliance['true_peak']:.1f} dB")
        else:
            self.log(f"❌ {result.message}")
        self.stop_progress()
    
    def process_quality_assurance(self):
        """Process quality assurance check"""
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input file first")
            return
        
        thresholds = {
            'clip_threshold': self.clip_threshold_var.get(),
            'peak_margin': self.peak_margin_var.get(),
            'dc_offset_max': self.dc_offset_var.get(),
            'thd_threshold': 1.0,
            'snr_minimum': 40.0,
            'silence_threshold': -60.0
        }
        
        self.log("Running comprehensive quality analysis...")
        self.start_progress()
        
        def process():
            result = self.processor.quality_assurance_check(
                input_path=str(self.input_file),
                thresholds=thresholds
            )
            self.root.after(0, lambda: self.quality_assurance_complete(result))
        
        threading.Thread(target=process, daemon=True).start()
    
    def quality_assurance_complete(self, result):
        """Handle quality assurance completion"""
        if result.success:
            self.log(f"✅ {result.message}")
            if result.quality_metrics:
                qa_report = result.quality_metrics
                overall = qa_report.get('overall_assessment', {})
                
                self.log(f"Quality rating: {overall.get('quality_rating', 'Unknown')}")
                
                issues = overall.get('issues_found', [])
                warnings = overall.get('warnings', [])
                
                if issues:
                    self.log("Issues found:")
                    for issue in issues[:3]:  # Show first 3
                        self.log(f"  - {issue}")
                    if len(issues) > 3:
                        self.log(f"  ... and {len(issues) - 3} more")
                
                if warnings:
                    self.log("Warnings:")
                    for warning in warnings[:3]:  # Show first 3
                        self.log(f"  - {warning}")
                    if len(warnings) > 3:
                        self.log(f"  ... and {len(warnings) - 3} more")
                
                recommendation = overall.get('recommendation', '')
                if recommendation:
                    self.log(f"Recommendation: {recommendation}")
        else:
            self.log(f"❌ {result.message}")
        self.stop_progress()
    
    # Helper methods
    
    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def start_progress(self):
        """Start progress bar animation"""
        self.progress.start(10)
    
    def stop_progress(self):
        """Stop progress bar animation"""
        self.progress.stop()
    
    def browse_file(self):
        """Browse for input audio file"""
        file_types = [
            ("Audio files", "*.wav *.mp3 *.flac *.aiff *.m4a"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("FLAC files", "*.flac"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=file_types
        )
        
        if filename:
            self.input_file = filename
            self.file_label.config(text=Path(filename).name, foreground="black")
            self.log(f"Selected: {Path(filename).name}")
    
    def check_input_file(self):
        """Check if input file is selected"""
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input audio file first.")
            return False
        return True

    def run_processing(self, process_func, output_name):
        """Run processing in background thread"""
        if not self.check_input_file():
            return
        
        def process():
            self.progress.start()
            try:
                output_path = self.output_dir / output_name
                result = process_func(str(output_path))
                
                if result.success:
                    self.log(f"✓ SUCCESS: {result.message}")
                    self.log(f"  Output: {result.output_path}")
                    if result.processing_time:
                        self.log(f"  Processing time: {result.processing_time:.2f}s")
                else:
                    self.log(f"✗ FAILED: {result.message}")
                    
            except Exception as e:
                self.log(f"✗ ERROR: {str(e)}")
            finally:
                self.progress.stop()
        
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()
    
    def process_time_stretch(self):
        """Process time stretching"""
        def process_func(output_path):
            return self.processor.time_stretch(
                self.input_file,
                output_path,
                stretch_factor=self.stretch_var.get(),
                preserve_pitch=self.preserve_pitch_var.get(),
                quality=self.quality_var.get()
            )
        
        factor = self.stretch_var.get()
        output_name = f"time_stretched_{factor:.1f}x.wav"
        self.log(f"Starting time stretch: factor={factor:.1f}, quality={self.quality_var.get()}")
        self.run_processing(process_func, output_name)
    
    def process_repair(self):
        """Process audio repair"""
        def process_func(output_path):
            operations = []
            if self.declick_var.get():
                operations.append("declick")
            if self.denoise_var.get():
                operations.append("denoise")
            if self.dehum_var.get():
                operations.append("dehum")
            if self.normalize_var.get():
                operations.append("normalize")
                
            return self.processor.repair_audio(
                self.input_file,
                output_path,
                operations=operations
            )
        
        operations = []
        if self.declick_var.get(): operations.append("declick")
        if self.denoise_var.get(): operations.append("denoise") 
        if self.dehum_var.get(): operations.append("dehum")
        if self.normalize_var.get(): operations.append("normalize")
        
        if not operations:
            messagebox.showwarning("Warning", "Please select at least one repair operation.")
            return
            
        self.log(f"Starting audio repair: {', '.join(operations)}")
        self.run_processing(process_func, "repaired.wav")
    
    def process_spectral(self):
        """Process spectral editing"""
        def process_func(output_path):
            low_freq = self.low_freq_var.get()
            high_freq = self.high_freq_var.get()
            
            if low_freq >= high_freq:
                raise ValueError("Low frequency must be less than high frequency")
                
            return self.processor.spectral_edit(
                self.input_file,
                output_path,
                freq_ranges=[(low_freq, high_freq)],
                operation=self.spectral_op_var.get()
            )
        
        low_freq = self.low_freq_var.get()
        high_freq = self.high_freq_var.get()
        operation = self.spectral_op_var.get()
        
        if low_freq >= high_freq:
            messagebox.showerror("Error", "Low frequency must be less than high frequency.")
            return
            
        self.log(f"Starting spectral edit: {operation} {low_freq}-{high_freq}Hz")
        self.run_processing(process_func, f"spectral_{operation}.wav")
    
    def process_stereo(self):
        """Process stereo field manipulation"""
        def process_func(output_path):
            operation = self.stereo_op_var.get()
            parameters = {}
            
            if operation == "width_control":
                parameters["width"] = self.width_var.get()
            elif operation == "pan":
                parameters["position"] = self.pan_var.get()
            elif operation == "mono_to_stereo":
                parameters = {"technique": "delay", "delay_ms": 15}
                
            return self.processor.stereo_field_manipulation(
                self.input_file,
                output_path,
                operation=operation,
                parameters=parameters
            )
        
        operation = self.stereo_op_var.get()
        self.log(f"Starting stereo processing: {operation}")
        self.run_processing(process_func, f"stereo_{operation}.wav")
    
    def process_mastering(self):
        """Process mastering chain"""
        def process_func(output_path):
            config = {
                'compressor': {
                    'threshold': self.comp_threshold_var.get(),
                    'ratio': self.comp_ratio_var.get(),
                    'attack': 10.0,
                    'release': 100.0
                },
                'limiter': {
                    'ceiling': -0.1,
                    'lookahead': 5.0
                },
                'target_lufs': self.lufs_var.get()
            }
            
            if self.eq_enable_var.get():
                config['eq'] = {
                    'bands': [
                        {'frequency': 100, 'gain': 1.5, 'q': 0.7},
                        {'frequency': 3000, 'gain': 1.0, 'q': 1.0}
                    ]
                }
                
            return self.processor.mastering_chain(
                self.input_file,
                output_path,
                chain_config=config
            )
        
        target_lufs = self.lufs_var.get()
        self.log(f"Starting mastering chain: target LUFS={target_lufs:.1f}")
        self.run_processing(process_func, "mastered.wav")


def main():
    root = tk.Tk()
    app = ProfessionalToolsGUI(root)
    
    # Create menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open Audio File", command=app.browse_file)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
        "About", "Professional Audio Tools GUI\n\nAdvanced audio processing for professional use."))
    
    root.mainloop()

def main():
    """Run the GUI application"""
    root = tk.Tk()
    app = ProfessionalToolsGUI(root)
    
    # Add menu
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open Audio File", command=app.browse_file)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
        "About", "Professional Audio Tools GUI\n\nAdvanced audio processing for professional use."))
    
    root.mainloop()

if __name__ == "__main__":
    main()