#!/usr/bin/env python3
"""
Professional Audio Tools GUI - Simplified Version
Simple graphical interface for professional audio processing.
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
        self.root.geometry("900x700")
        
        # Initialize processor
        self.processor = ProfessionalAudioProcessor()
        self.input_file = None
        self.output_dir = Path("gui_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface"""
        # File selection frame
        file_frame = ttk.Frame(self.root)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(file_frame, text="Input Audio File:").pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.RIGHT)
        
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
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
                                   values=["wav", "flac", "aiff"], state="readonly")
        format_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Metadata preservation
        self.preserve_metadata_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Preserve metadata", variable=self.preserve_metadata_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(frame, text="Convert Format", command=self.process_format_conversion).grid(row=3, column=0, columnspan=3, pady=10)
        
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
        
        ttk.Button(frame, text="Check Compliance", command=self.process_broadcast_compliance).grid(row=3, column=0, columnspan=3, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    def create_quality_assurance_tab(self, notebook):
        """Create quality assurance tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Quality Assurance")
        
        ttk.Label(frame, text="Comprehensive audio quality analysis").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(frame, text="Run Quality Analysis", command=self.process_quality_assurance).grid(row=1, column=0, columnspan=3, pady=10)
        
        frame.columnconfigure(1, weight=1)
    
    # Processing methods
    
    def process_stem_separation(self):
        """Process advanced stem separation"""
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input file first")
            return
        
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
        
        method = self.stem_method_var.get()
        quality = self.stem_quality_var.get()
        output_dir = self.output_dir / "stem_separation" / f"{method}_{quality}"
        
        self.log(f"Starting {method} stem separation ({quality} quality): {', '.join(stems)}")
        self.start_progress()
        
        def process():
            try:
                result = self.processor.stem_separation(
                    input_path=str(self.input_file),
                    output_dir=str(output_dir),
                    stems=stems,
                    method=method,
                    quality=quality
                )
                self.root.after(0, lambda: self.stem_separation_complete(result))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"❌ Error: {str(e)}"))
                self.root.after(0, self.stop_progress)
        
        threading.Thread(target=process, daemon=True).start()
    
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
            try:
                result = self.processor.format_conversion_with_metadata(
                    input_path=str(self.input_file),
                    output_path=str(output_path),
                    target_format=target_format,
                    preserve_metadata=self.preserve_metadata_var.get()
                )
                self.root.after(0, lambda: self.format_conversion_complete(result))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"❌ Error: {str(e)}"))
                self.root.after(0, self.stop_progress)
        
        threading.Thread(target=process, daemon=True).start()
    
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
            try:
                result = self.processor.broadcast_standards_compliance(
                    input_path=str(self.input_file),
                    output_path=str(output_path),
                    standard=standard,
                    auto_correct=self.auto_correct_var.get()
                )
                self.root.after(0, lambda: self.broadcast_compliance_complete(result))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"❌ Error: {str(e)}"))
                self.root.after(0, self.stop_progress)
        
        threading.Thread(target=process, daemon=True).start()
    
    def process_quality_assurance(self):
        """Process quality assurance check"""
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input file first")
            return
        
        self.log("Running comprehensive quality analysis...")
        self.start_progress()
        
        def process():
            try:
                result = self.processor.quality_assurance_check(
                    input_path=str(self.input_file)
                )
                self.root.after(0, lambda: self.quality_assurance_complete(result))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"❌ Error: {str(e)}"))
                self.root.after(0, self.stop_progress)
        
        threading.Thread(target=process, daemon=True).start()
    
    # Completion handlers
    
    def stem_separation_complete(self, result):
        """Handle stem separation completion"""
        if result.success:
            self.log(f"✅ {result.message}")
            if result.additional_outputs:
                self.log("Generated stems:")
                for stem, path in result.additional_outputs.items():
                    self.log(f"  - {stem}: {Path(path).name}")
            
            if result.quality_metrics:
                metrics = result.quality_metrics
                self.log(f"Quality metrics:")
                self.log(f"  - Success rate: {metrics.get('success_rate', 0):.1%}")
                self.log(f"  - Energy conservation: {metrics.get('energy_conservation', 0):.3f}")
        else:
            self.log(f"❌ {result.message}")
        self.stop_progress()
    
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
                    for issue in issues[:3]:
                        self.log(f"  - {issue}")
                    if len(issues) > 3:
                        self.log(f"  ... and {len(issues) - 3} more")
                
                if warnings:
                    self.log("Warnings:")
                    for warning in warnings[:3]:
                        self.log(f"  - {warning}")
                    if len(warnings) > 3:
                        self.log(f"  ... and {len(warnings) - 3} more")
        else:
            self.log(f"❌ {result.message}")
        self.stop_progress()
    
    # Helper methods
    
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
        "About", "Professional Audio Tools GUI\n\nAdvanced audio processing with AI-powered stem separation."))
    
    root.mainloop()

if __name__ == "__main__":
    main()