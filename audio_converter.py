#!/usr/bin/env python3
"""
Advanced Audio Converter
A comprehensive audio conversion tool with batch processing, pitch/key adjustment,
format conversion, and automatic categorization.
"""

import os
import sys
import re
import json
import threading
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.utils import which

# Import our intelligent audio analyzer
from audio_analyzer import IntelligentAudioAnalyzer, AudioAnalysisResult
from metadata_manager import MetadataManager, AudioMetadata

class AudioConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Real Convert Pro - Advanced Audio Converter")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)  # Set minimum size for professional look
        
        # Audio processing settings
        self.input_folders = []
        self.input_files = []
        self.output_folder = ""
        self.selected_format = tk.StringVar(value="wav")
        self.selected_key = tk.StringVar(value="Original")
        self.pitch_shift = tk.DoubleVar(value=0.0)
        self.key_shift = tk.IntVar(value=0)
        self.batch_mode = tk.BooleanVar(value=True)
        self.auto_categorize = tk.BooleanVar(value=True)
        self.preserve_structure = tk.BooleanVar(value=False)
        
        # Category keywords for automatic classification
        self.category_keywords = {
            "drums": ["kick", "snare", "hihat", "cymbal", "tom", "drum", "perc", "percussion"],
            "bass": ["bass", "sub", "808", "low"],
            "melody": ["melody", "lead", "main", "hook", "theme"],
            "vocals": ["vocal", "voice", "singing", "rap", "spoken"],
            "fx": ["fx", "effect", "sweep", "riser", "impact", "crash"],
            "loops": ["loop", "pattern", "sequence"],
            "instruments": ["piano", "guitar", "synth", "violin", "flute", "sax"]
        }
        
        # Supported formats - comprehensive list for all major audio formats
        self.formats = [
            # Uncompressed/Lossless
            "wav", "flac", "aiff", "au", "caf", "w64", "rf64",
            # Lossy compressed
            "mp3", "aac", "m4a", "ogg", "opus", "wma", "ac3", "mp2",
            # Professional/Studio formats
            "bwf", "sd2", "snd", "iff", "svx", "nist", "voc", "ircam", "xi",
            # Video audio formats
            "3gp", "webm", "mka", "m4v", "mov", "avi", "mkv", "mp4",
            # Legacy/Specialized formats
            "ra", "rm", "amr", "amr-nb", "amr-wb", "gsm", "dct", "dwv",
            # Raw formats
            "raw", "pcm", "s8", "s16le", "s24le", "s32le", "f32le", "f64le",
            # Apple formats
            "aifc", "caff", "m4r", "m4b", "m4p",
            # Other formats
            "tta", "tak", "als", "ape", "wv", "mpc", "ofr", "ofs", "shn"
        ]
        
        # Musical keys for transposition
        self.musical_keys = [
            "Original",  # No transposition
            "C", "C#/Db", "D", "D#/Eb", "E", "F", 
            "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B",
            "Cm", "C#m/Dbm", "Dm", "D#m/Ebm", "Em", "Fm",
            "F#m/Gbm", "Gm", "G#m/Abm", "Am", "A#m/Bbm", "Bm"
        ]
        
        # Initialize intelligent audio analyzer
        self.audio_analyzer = IntelligentAudioAnalyzer()
        self.analysis_results = {}  # Store analysis results by filename
        
        # Analysis settings
        self.enable_analysis = tk.BooleanVar(value=True)
        self.analyze_bpm = tk.BooleanVar(value=True)
        self.analyze_key = tk.BooleanVar(value=True)
        self.analyze_loudness = tk.BooleanVar(value=True)
        self.analyze_fingerprint = tk.BooleanVar(value=True)
        self.auto_categorize_by_analysis = tk.BooleanVar(value=False)

        # Metadata management
        self.metadata_manager = MetadataManager()
        self.metadata_file_var = tk.StringVar(value="")
        self.metadata_status_var = tk.StringVar(value="Select an audio file to begin.")
        self.metadata_vars: Dict[str, tk.StringVar] = {}
        self.metadata_comments_text = None
        self.metadata_current_metadata: Optional[AudioMetadata] = None
        self.metadata_analysis_result: Optional[AudioAnalysisResult] = None
        self.metadata_file_display = None
        self.metadata_analyze_button = None
        self.metadata_save_button = None

        # Filename generation toggles - user can select which metadata to include
        self.include_key_in_filename = tk.BooleanVar(value=True)
        self.include_bpm_in_filename = tk.BooleanVar(value=False)
        self.include_genre_in_filename = tk.BooleanVar(value=False)
        self.include_mood_in_filename = tk.BooleanVar(value=True)
        self.include_energy_in_filename = tk.BooleanVar(value=True)
        self.include_lufs_in_filename = tk.BooleanVar(value=False)
        self.include_category_in_filename = tk.BooleanVar(value=False)
        
        self.setup_ui()
        self.conversion_thread = None
        self.is_converting = False
        
        # Analysis control
        self.analysis_thread = None
        self.is_analyzing = False
        self.stop_analysis_flag = False
        
    def setup_ui(self):
        """Setup the user interface"""
        # Configure themes and styles
        self.setup_themes()
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main conversion tab
        self.setup_main_tab(notebook)

        # Metadata & tags management tab
        self.setup_metadata_tab(notebook)

        # Settings tab
        self.setup_settings_tab(notebook)
        
        # Categories tab
        self.setup_categories_tab(notebook)
        
        # Analysis tab (new)
        self.setup_analysis_tab(notebook)
        
        # Progress tab
        self.setup_progress_tab(notebook)
    
    def setup_themes(self):
        """Configure beautiful purple gradient theme like Music Organizer"""
        self.style = ttk.Style()
        self.apply_theme()
    
    def apply_theme(self):
        """Apply a muted dark theme without bright or light tones."""
        self.style.theme_use('clam')

        bg_primary = '#0b0f16'
        bg_secondary = '#111826'
        bg_accent = '#1a2535'
        button_primary = '#273445'
        button_hover = '#324156'
        button_pressed = '#212b3a'
        text_primary = '#c3cad9'
        text_subtle = '#9aa7bc'
        progress_fill = '#3a5246'

        self.style.configure('.', background=bg_primary, foreground=text_primary)
        self.style.configure('TFrame', background=bg_primary, borderwidth=0)

        self.style.configure('TLabel',
                             background=bg_primary,
                             foreground=text_primary,
                             font=('SF Pro Display', 11))

        for button_style in ('TButton', 'Accent.TButton'):
            self.style.configure(button_style,
                                 background=button_primary,
                                 foreground=text_primary,
                                 borderwidth=0,
                                 relief='flat',
                                 font=('SF Pro Display', 10, 'bold'),
                                 padding=(20, 8))
            self.style.map(button_style,
                           background=[('active', button_hover),
                                       ('pressed', button_pressed)])

        self.style.configure('TNotebook', background=bg_primary, borderwidth=0)
        self.style.configure('TNotebook.Tab',
                             background=bg_secondary,
                             foreground=text_primary,
                             padding=[20, 12],
                             borderwidth=0,
                             font=('SF Pro Display', 11))
        self.style.map('TNotebook.Tab',
                       background=[('selected', button_primary),
                                   ('active', bg_accent)],
                       foreground=[('selected', text_primary)])

        self.style.configure('TLabelFrame',
                             background=bg_primary,
                             foreground=text_primary,
                             borderwidth=0,
                             relief='flat')
        self.style.configure('TLabelFrame.Label',
                             background=bg_primary,
                             foreground=text_primary,
                             font=('SF Pro Display', 12, 'bold'))

        self.style.configure('TCheckbutton',
                             background=bg_primary,
                             foreground=text_primary,
                             focuscolor='none',
                             font=('SF Pro Display', 10))

        entry_kwargs = dict(background=bg_accent,
                            foreground=text_primary,
                            fieldbackground=bg_accent,
                            borderwidth=1,
                            relief='flat',
                            insertcolor=text_primary,
                            padding=8)
        self.style.configure('TEntry', **entry_kwargs)
        self.style.configure('TCombobox', **entry_kwargs)
        self.style.map('TCombobox',
                       fieldbackground=[('readonly', bg_accent)],
                       foreground=[('readonly', text_primary)],
                       selectbackground=[('readonly', bg_accent)],
                       selectforeground=[('readonly', text_primary)])

        self.style.configure('TScale',
                             background=bg_primary,
                             troughcolor=bg_accent,
                             borderwidth=0,
                             sliderthickness=15)

        self.style.configure('Horizontal.TScrollbar',
                             background=bg_accent,
                             troughcolor=bg_primary,
                             borderwidth=0,
                             arrowcolor=text_primary)

        self.style.configure('TProgressbar',
                             background=progress_fill,
                             troughcolor=bg_accent,
                             borderwidth=0,
                             lightcolor=progress_fill,
                             darkcolor=progress_fill)

        self.style.configure('Treeview',
                             background=bg_secondary,
                             fieldbackground=bg_secondary,
                             foreground=text_primary,
                             borderwidth=0)
        self.style.map('Treeview', background=[('selected', bg_accent)])
        self.style.configure('Treeview.Heading',
                             background=bg_accent,
                             foreground=text_primary,
                             font=('SF Pro Display', 11, 'bold'))

        self.root.configure(bg=bg_primary)
        self._theme_colors = {
            'bg_primary': bg_primary,
            'bg_secondary': bg_secondary,
            'bg_accent': bg_accent,
            'text_primary': text_primary,
            'text_subtle': text_subtle,
            'button_primary': button_primary,
            'button_hover': button_hover,
            'button_pressed': button_pressed,
            'progress_fill': progress_fill
        }
        
    def setup_main_tab(self, notebook):
        """Setup main conversion interface"""
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Converter")
        
        # Input selection frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Selection", padding=10)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Folder selection
        folder_frame = ttk.Frame(input_frame)
        folder_frame.pack(fill=tk.X, pady=2)
        ttk.Button(folder_frame, text="Add Folders", command=self.select_folders).pack(side=tk.LEFT)
        ttk.Button(folder_frame, text="Add Files", command=self.select_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_frame, text="Clear All", command=self.clear_inputs).pack(side=tk.LEFT)
        
        # Input list
        list_frame = ttk.Frame(input_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.input_listbox = tk.Listbox(
            list_frame,
            height=6,
            bg=self._theme_colors['bg_secondary'],
            fg=self._theme_colors['text_primary'],
            selectbackground=self._theme_colors['button_hover'],
            selectforeground=self._theme_colors['text_primary'],
            highlightthickness=0,
            borderwidth=0,
            font=('SF Pro Display', 10)
        )
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.input_listbox.yview)
        self.input_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.input_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Output selection frame
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding=10)
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Output folder
        folder_select_frame = ttk.Frame(output_frame)
        folder_select_frame.pack(fill=tk.X, pady=2)
        ttk.Label(folder_select_frame, text="Output Folder:").pack(side=tk.LEFT)
        self.output_label = ttk.Label(
            folder_select_frame,
            text="Not selected",
            foreground=self._theme_colors['text_subtle']
        )
        self.output_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(folder_select_frame, text="Browse", command=self.select_output_folder).pack(side=tk.RIGHT)
        
        # Format selection
        format_frame = ttk.Frame(output_frame)
        format_frame.pack(fill=tk.X, pady=5)
        ttk.Label(format_frame, text="Output Format:").pack(side=tk.LEFT)
        format_combo = ttk.Combobox(format_frame, textvariable=self.selected_format, 
                                   values=self.formats, state="readonly", width=15)
        format_combo.pack(side=tk.LEFT, padx=10)
        
        # Key selection (right next to format)
        ttk.Label(format_frame, text="Target Key:").pack(side=tk.LEFT, padx=(20,5))
        key_combo = ttk.Combobox(format_frame, textvariable=self.selected_key,
                                values=self.musical_keys, state="readonly", width=12)
        key_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(format_frame, text="Format Info", command=self.show_format_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(format_frame, text="System Check", command=self.show_system_capabilities).pack(side=tk.LEFT, padx=5)
        
        # Audio processing frame
        processing_frame = ttk.LabelFrame(main_frame, text="Audio Processing", padding=10)
        processing_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Pitch and key controls
        controls_frame = ttk.Frame(processing_frame)
        controls_frame.pack(fill=tk.X)
        
        # Pitch shift
        pitch_frame = ttk.Frame(controls_frame)
        pitch_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(pitch_frame, text="Pitch Shift (semitones):").pack(anchor=tk.W)
        pitch_scale = ttk.Scale(pitch_frame, from_=-12, to=12, variable=self.pitch_shift, 
                               orient=tk.HORIZONTAL, length=200)
        pitch_scale.pack(fill=tk.X)
        self.pitch_value_label = ttk.Label(pitch_frame, text="0.0")
        self.pitch_value_label.pack()
        pitch_scale.configure(command=self.update_pitch_label)
        
        # Key shift
        key_frame = ttk.Frame(controls_frame)
        key_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=20)
        ttk.Label(key_frame, text="Key Shift (steps):").pack(anchor=tk.W)
        key_scale = ttk.Scale(key_frame, from_=-12, to=12, variable=self.key_shift, 
                             orient=tk.HORIZONTAL, length=200)
        key_scale.pack(fill=tk.X)
        self.key_value_label = ttk.Label(key_frame, text="0")
        self.key_value_label.pack()
        key_scale.configure(command=self.update_key_label)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        options_left = ttk.Frame(options_frame)
        options_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Checkbutton(options_left, text="Auto-categorize by filename", 
                       variable=self.auto_categorize).pack(anchor=tk.W)
        ttk.Checkbutton(options_left, text="Preserve folder structure", 
                       variable=self.preserve_structure).pack(anchor=tk.W)
        
        # Convert button
        convert_frame = ttk.Frame(main_frame)
        convert_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.convert_button = ttk.Button(convert_frame, text="Start Conversion", 
                                        command=self.start_conversion, style="Accent.TButton")
        self.convert_button.pack(side=tk.LEFT)
        
        self.stop_button = ttk.Button(convert_frame, text="Stop", 
                                     command=self.stop_conversion, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(convert_frame, variable=self.progress_var,
                                           length=300, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=10)

    def setup_metadata_tab(self, notebook):
        """Create metadata and tagging management tab"""
        metadata_frame = ttk.Frame(notebook)
        notebook.add(metadata_frame, text="Metadata & Tags")

        ttk.Label(
            metadata_frame,
            text="Inspect, analyze, and edit audio metadata. Use auto-analyze to fill BPM and key, then save changes.",
            wraplength=600,
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=5, pady=(5, 0))

        file_frame = ttk.Frame(metadata_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(file_frame, text="Selected File:").pack(side=tk.LEFT)
        self.metadata_file_display = ttk.Label(
            file_frame,
            text="No file selected",
            foreground=self._theme_colors['text_subtle'],
            wraplength=600
        )
        self.metadata_file_display.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        selector_frame = ttk.Frame(metadata_frame)
        selector_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Button(selector_frame, text="Choose File...", command=self.choose_metadata_file).pack(side=tk.LEFT)
        ttk.Button(selector_frame, text="Use Selected Converter File", command=self.use_selected_converter_file).pack(side=tk.LEFT, padx=5)

        fields_frame = ttk.LabelFrame(metadata_frame, text="Core Tags", padding=10)
        fields_frame.pack(fill=tk.X, padx=5, pady=5)

        field_specs = [
            ("Title", "title"),
            ("Artist", "artist"),
            ("Album", "album"),
            ("Genre", "genre"),
            ("BPM", "bpm"),
            ("Key", "key"),
            ("Mood", "mood"),
            ("Energy", "energy")
        ]

        self.metadata_vars = {name: tk.StringVar(value="") for _, name in field_specs}

        for row, (label_text, key) in enumerate(field_specs):
            ttk.Label(fields_frame, text=f"{label_text}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=4)
            entry = tk.Entry(
                fields_frame,
                textvariable=self.metadata_vars[key],
                bg=self._theme_colors['bg_secondary'],
                fg=self._theme_colors['text_primary'],
                insertbackground=self._theme_colors['text_primary'],
                highlightthickness=0,
                highlightcolor=self._theme_colors['bg_secondary'],
                highlightbackground=self._theme_colors['bg_secondary'],
                relief=tk.FLAT
            )
            entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=4)

        fields_frame.columnconfigure(1, weight=1)

        comments_frame = ttk.LabelFrame(metadata_frame, text="Comments & Notes", padding=10)
        comments_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.metadata_comments_text = scrolledtext.ScrolledText(
            comments_frame,
            height=5,
            bg=self._theme_colors['bg_secondary'],
            fg=self._theme_colors['text_primary'],
            insertbackground=self._theme_colors['text_primary'],
            borderwidth=0,
            highlightthickness=0,
            wrap=tk.WORD
        )
        self.metadata_comments_text.pack(fill=tk.BOTH, expand=True)

        actions_frame = ttk.Frame(metadata_frame)
        actions_frame.pack(fill=tk.X, padx=5, pady=5)

        self.metadata_analyze_button = ttk.Button(
            actions_frame,
            text="Auto-Analyze & Fill",
            command=self.auto_analyze_metadata,
            state=tk.DISABLED
        )
        self.metadata_analyze_button.pack(side=tk.LEFT)

        self.metadata_save_button = ttk.Button(
            actions_frame,
            text="Save Metadata",
            command=self.save_metadata_edits,
            state=tk.DISABLED
        )
        self.metadata_save_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(actions_frame, text="Reset", command=self.reset_metadata_fields).pack(side=tk.LEFT)

        self.metadata_status_label = ttk.Label(
            metadata_frame,
            textvariable=self.metadata_status_var,
            foreground=self._theme_colors['text_subtle']
        )
        self.metadata_status_label.pack(anchor=tk.W, padx=5, pady=(0, 8))

    def choose_metadata_file(self):
        """Prompt user to select a file for metadata editing"""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma *.aiff *.aif *.aifc"),
            ("All files", "*.*")
        ]
        file_path = filedialog.askopenfilename(title="Select audio file", filetypes=filetypes)
        if file_path:
            self.load_metadata_for_file(file_path)

    def use_selected_converter_file(self):
        """Load metadata for the file selected in the converter list"""
        selection = self.input_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file from the Converter tab first.")
            return

        index = selection[0]
        folder_count = len(self.input_folders)
        if index < folder_count:
            messagebox.showinfo("Select File", "Please select an individual file entry (🎵) instead of a folder.")
            return

        file_index = index - folder_count
        if 0 <= file_index < len(self.input_files):
            self.load_metadata_for_file(self.input_files[file_index])

    def load_metadata_for_file(self, file_path: str):
        """Load metadata for a given audio file"""
        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("Invalid File", "The selected file could not be found.")
            return

        absolute_path = os.path.abspath(file_path)
        self.metadata_file_var.set(absolute_path)

        if self.metadata_file_display:
            self.metadata_file_display.config(text=absolute_path, foreground=self._theme_colors['text_primary'])

        self.metadata_status_var.set("Loading metadata...")
        if self.metadata_analyze_button:
            self.metadata_analyze_button.config(state=tk.DISABLED)
        if self.metadata_save_button:
            self.metadata_save_button.config(state=tk.DISABLED)

        self.clear_metadata_fields()

        def worker():
            try:
                metadata = self.metadata_manager.extract_file_metadata(absolute_path)
            except Exception as exc:
                self.root.after(0, lambda: self._metadata_load_failed(absolute_path, exc))
                return

            self.root.after(0, lambda: self._metadata_load_success(absolute_path, metadata))

        threading.Thread(target=worker, daemon=True).start()

    def _metadata_load_success(self, file_path: str, metadata: AudioMetadata):
        """Handle successful metadata extraction"""
        self.metadata_current_metadata = metadata
        self.metadata_analysis_result = None
        self.populate_metadata_fields(metadata)
        self.metadata_status_var.set("Metadata loaded. You can edit fields or run Auto-Analyze.")
        if self.metadata_analyze_button:
            self.metadata_analyze_button.config(state=tk.NORMAL)
        if self.metadata_save_button:
            self.metadata_save_button.config(state=tk.NORMAL)
        self.log(f"Loaded metadata for {os.path.basename(file_path)}")

    def _metadata_load_failed(self, file_path: str, error: Exception):
        """Handle metadata load failure"""
        self.metadata_current_metadata = None
        self.metadata_status_var.set(f"Failed to load metadata: {error}")
        messagebox.showerror("Metadata Error", f"Could not load metadata for {os.path.basename(file_path)}:\n{error}")

    def clear_metadata_fields(self):
        """Reset metadata form fields"""
        for var in self.metadata_vars.values():
            var.set("")
        if self.metadata_comments_text:
            self.metadata_comments_text.delete("1.0", tk.END)

    def populate_metadata_fields(self, metadata: Optional[AudioMetadata]):
        """Populate UI fields with metadata information"""
        if not metadata:
            self.clear_metadata_fields()
            return

        def set_custom(field: str) -> str:
            return metadata.custom_tags.get(field, "") if metadata.custom_tags else ""

        self.metadata_vars['title'].set(set_custom('title'))
        self.metadata_vars['artist'].set(set_custom('artist'))
        self.metadata_vars['album'].set(set_custom('album'))
        self.metadata_vars['genre'].set(metadata.genre or "")
        self.metadata_vars['bpm'].set(self._format_numeric(metadata.bpm))
        self.metadata_vars['key'].set(metadata.key or "")
        self.metadata_vars['mood'].set(metadata.mood or "")
        self.metadata_vars['energy'].set(self._format_numeric(metadata.energy_level, precision=3))

        if self.metadata_comments_text:
            self.metadata_comments_text.delete("1.0", tk.END)
            comments = set_custom('comments')
            if comments:
                self.metadata_comments_text.insert(tk.END, comments)

    def reset_metadata_fields(self):
        """Revert metadata fields to the last loaded values"""
        if self.metadata_current_metadata:
            self.populate_metadata_fields(self.metadata_current_metadata)
            self.metadata_status_var.set("Changes reset to last loaded metadata.")
        else:
            self.clear_metadata_fields()

    def _format_numeric(self, value: Optional[float], precision: int = 2) -> str:
        """Format numeric metadata values for display"""
        if value is None:
            return ""
        formatted = f"{value:.{precision}f}"
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
        return formatted

    def auto_analyze_metadata(self):
        """Run automatic analysis for the currently loaded file"""
        file_path = self.metadata_file_var.get()
        if not file_path:
            messagebox.showwarning("No File", "Select a file before running auto-analysis.")
            return

        if self.metadata_analyze_button:
            self.metadata_analyze_button.config(state=tk.DISABLED)
        self.metadata_status_var.set("Analyzing audio for BPM, key, and mood...")

        def worker():
            try:
                analysis_result = self.audio_analyzer.analyze_audio_file(file_path)
            except Exception as exc:
                self.root.after(0, lambda: self._metadata_analysis_failed(exc))
                return
            self.root.after(0, lambda: self._metadata_analysis_complete(file_path, analysis_result))

        threading.Thread(target=worker, daemon=True).start()

    def _metadata_analysis_complete(self, file_path: str, analysis_result: Optional[AudioAnalysisResult]):
        """Handle completion of metadata auto-analysis"""
        if self.metadata_analyze_button:
            self.metadata_analyze_button.config(state=tk.NORMAL)

        if not analysis_result:
            self.metadata_status_var.set("Analysis complete, but no results were returned.")
            return

        if self.metadata_current_metadata:
            self._update_metadata_with_analysis(self.metadata_current_metadata, analysis_result)
            self.metadata_current_metadata.last_analyzed = datetime.now(timezone.utc)
            self.populate_metadata_fields(self.metadata_current_metadata)

        absolute_path = os.path.abspath(file_path)
        self.metadata_analysis_result = analysis_result
        self.analysis_results[absolute_path] = analysis_result
        self.metadata_status_var.set("Analysis complete! Review the populated fields and save to embed them.")
        self.log(f"Auto-analysis finished for {os.path.basename(file_path)}")

        if self.metadata_save_button:
            self.metadata_save_button.config(state=tk.NORMAL)

    def _metadata_analysis_failed(self, error: Exception):
        """Handle analysis failures"""
        if self.metadata_analyze_button:
            self.metadata_analyze_button.config(state=tk.NORMAL)
        self.metadata_status_var.set(f"Analysis failed: {error}")
        messagebox.showerror("Analysis Error", f"Auto-analysis failed:\n{error}")

    def save_metadata_edits(self):
        """Persist metadata edits to the file"""
        if not self.metadata_current_metadata:
            messagebox.showwarning("No Metadata", "Load a file before saving metadata.")
            return

        metadata = self.metadata_current_metadata
        file_path = metadata.file_path

        def set_custom(field: str, value: str):
            if value:
                metadata.custom_tags[field] = value
            elif metadata.custom_tags and field in metadata.custom_tags:
                metadata.custom_tags.pop(field)

        set_custom('title', self.metadata_vars['title'].get().strip())
        set_custom('artist', self.metadata_vars['artist'].get().strip())
        set_custom('album', self.metadata_vars['album'].get().strip())
        set_custom('comments', self.metadata_comments_text.get("1.0", tk.END).strip() if self.metadata_comments_text else "")

        metadata.genre = self.metadata_vars['genre'].get().strip() or None
        metadata.key = self.metadata_vars['key'].get().strip() or None
        metadata.mood = self.metadata_vars['mood'].get().strip() or None

        bpm_text = self.metadata_vars['bpm'].get().strip()
        if bpm_text:
            try:
                metadata.bpm = float(bpm_text)
            except ValueError:
                messagebox.showerror("Invalid BPM", "Please enter a numeric BPM value.")
                return
        else:
            metadata.bpm = None

        energy_text = self.metadata_vars['energy'].get().strip()
        if energy_text:
            try:
                metadata.energy_level = float(energy_text)
            except ValueError:
                messagebox.showerror("Invalid Energy", "Energy should be a numeric value.")
                return
        else:
            metadata.energy_level = None

        # Update timestamps after editing
        metadata.modified_date = datetime.now(timezone.utc)
        metadata.last_analyzed = metadata.last_analyzed or datetime.now(timezone.utc)

        write_success = self.metadata_manager.write_metadata_to_file(metadata)
        if not write_success:
            messagebox.showerror("Save Failed", "Could not write metadata to the selected file.")
            self.metadata_status_var.set("Failed to write metadata.")
            return

        record_id = self.metadata_manager.save_metadata(metadata)
        if record_id != -1 and self.metadata_analysis_result:
            self.metadata_manager.add_processing_record(
                record_id,
                "metadata_edit",
                {"auto_analysis": True},
                file_path,
                file_path,
                True,
                "Metadata edited via Real Convert Pro"
            )

        try:
            stat = os.stat(file_path)
            metadata.file_size = stat.st_size
            metadata.modified_date = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        except FileNotFoundError:
            pass

        self.metadata_status_var.set("Metadata saved successfully.")
        self.log(f"Metadata updated for {os.path.basename(file_path)}")

    def _update_metadata_with_analysis(self, metadata: AudioMetadata, analysis: AudioAnalysisResult):
        """Merge analysis data into metadata object"""
        if not metadata or not analysis:
            return

        if analysis.bpm is not None:
            metadata.bpm = analysis.bpm
        if analysis.key:
            metadata.key = analysis.key
        if analysis.scale:
            metadata.scale = analysis.scale
        if analysis.genre_prediction:
            metadata.genre = analysis.genre_prediction
        if analysis.mood_prediction:
            metadata.mood = analysis.mood_prediction
        if analysis.energy_level is not None:
            metadata.energy_level = analysis.energy_level
        if analysis.danceability is not None:
            metadata.danceability = analysis.danceability
        if analysis.valence is not None:
            metadata.valence = analysis.valence
        if analysis.content_category:
            metadata.content_type = analysis.content_category
        if analysis.lufs_integrated is not None:
            metadata.lufs_integrated = analysis.lufs_integrated
        if analysis.peak_db is not None:
            metadata.peak_db = analysis.peak_db
        if analysis.lufs_integrated is not None and analysis.peak_db is not None:
            metadata.dynamic_range = analysis.peak_db - analysis.lufs_integrated

    def apply_metadata_to_output(self, input_file: str, output_file: str, audio_array: np.ndarray, sample_rate: int):
        """Copy and enrich metadata from input file to output file"""
        input_path = os.path.abspath(input_file)
        output_path = os.path.abspath(output_file)

        try:
            source_metadata = self.metadata_manager.extract_file_metadata(input_path)
        except Exception as exc:
            self.log(f"⚠️ Metadata extraction failed for {os.path.basename(input_file)}: {exc}")
            return

        analysis_result = self.analysis_results.get(input_path)
        if analysis_result:
            self._update_metadata_with_analysis(source_metadata, analysis_result)

        try:
            output_metadata = self.metadata_manager.extract_file_metadata(output_path)
        except Exception as exc:
            self.log(f"⚠️ Unable to prepare metadata for {os.path.basename(output_file)}: {exc}")
            return

        if source_metadata.custom_tags:
            output_metadata.custom_tags.update(source_metadata.custom_tags)

        output_metadata.genre = source_metadata.genre or output_metadata.genre
        output_metadata.bpm = source_metadata.bpm or output_metadata.bpm
        output_metadata.key = source_metadata.key or output_metadata.key
        output_metadata.scale = source_metadata.scale or output_metadata.scale
        output_metadata.mood = source_metadata.mood or output_metadata.mood
        if source_metadata.energy_level is not None:
            output_metadata.energy_level = source_metadata.energy_level
        if source_metadata.danceability is not None:
            output_metadata.danceability = source_metadata.danceability
        if source_metadata.valence is not None:
            output_metadata.valence = source_metadata.valence
        output_metadata.content_type = source_metadata.content_type or output_metadata.content_type

        if audio_array is not None and audio_array.size:
            frames = audio_array.shape[0] if audio_array.ndim == 1 else audio_array.shape[0]
            channels = 1 if audio_array.ndim == 1 else audio_array.shape[1]
            output_metadata.duration = frames / float(sample_rate)
            output_metadata.sample_rate = sample_rate
            output_metadata.channels = channels

        output_metadata.original_file = input_path
        if output_metadata.processing_history is None:
            output_metadata.processing_history = []
        output_metadata.processing_history.append({
            "operation": "format_conversion",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_file": input_path,
            "output_file": output_path,
            "target_format": self.selected_format.get(),
            "sample_rate": sample_rate,
            "bit_depth": self.bit_depth.get()
        })

        write_success = self.metadata_manager.write_metadata_to_file(output_metadata)
        if not write_success:
            self.log(f"⚠️ Unable to write metadata to {os.path.basename(output_file)}")
            return

        record_id = self.metadata_manager.save_metadata(output_metadata)
        if record_id != -1:
            self.metadata_manager.add_processing_record(
                record_id,
                "format_conversion",
                {
                    "target_format": self.selected_format.get(),
                    "sample_rate": sample_rate,
                    "bit_depth": self.bit_depth.get()
                },
                input_path,
                output_path,
                True,
                "Converted via Real Convert Pro"
            )

        self.log(f"Metadata preserved for {os.path.basename(output_file)}")

    def setup_settings_tab(self, notebook):
        """Setup settings interface"""
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        
        # Audio quality settings
        quality_frame = ttk.LabelFrame(settings_frame, text="Audio Quality", padding=10)
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Sample rate
        self.sample_rate = tk.IntVar(value=44100)
        rate_frame = ttk.Frame(quality_frame)
        rate_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rate_frame, text="Sample Rate:").pack(side=tk.LEFT)
        rate_combo = ttk.Combobox(rate_frame, textvariable=self.sample_rate,
                                 values=[22050, 44100, 48000, 96000], state="readonly", width=10)
        rate_combo.pack(side=tk.LEFT, padx=10)
        
        # Bit depth
        self.bit_depth = tk.IntVar(value=16)
        depth_frame = ttk.Frame(quality_frame)
        depth_frame.pack(fill=tk.X, pady=2)
        ttk.Label(depth_frame, text="Bit Depth:").pack(side=tk.LEFT)
        depth_combo = ttk.Combobox(depth_frame, textvariable=self.bit_depth,
                                  values=[16, 24, 32], state="readonly", width=10)
        depth_combo.pack(side=tk.LEFT, padx=10)
        
        # Processing settings
        proc_frame = ttk.LabelFrame(settings_frame, text="Processing", padding=10)
        proc_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.normalize_audio = tk.BooleanVar(value=True)
        ttk.Checkbutton(proc_frame, text="Normalize audio levels", 
                       variable=self.normalize_audio).pack(anchor=tk.W)
        
        self.remove_silence = tk.BooleanVar(value=False)
        ttk.Checkbutton(proc_frame, text="Remove silence from beginning/end", 
                       variable=self.remove_silence).pack(anchor=tk.W)
        
        # Theme Information
        appearance_frame = ttk.LabelFrame(settings_frame, text="🎨 Theme", padding=10)
        appearance_frame.pack(fill=tk.X, padx=5, pady=5)
        
        theme_info_frame = ttk.Frame(appearance_frame)
        theme_info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(theme_info_frame, text="🌙 Subtle Midnight Theme Active").pack(side=tk.LEFT)
        ttk.Label(theme_info_frame, text="Muted dark tones for comfortable editing").pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(proc_frame, text="Remove leading/trailing silence", 
                       variable=self.remove_silence).pack(anchor=tk.W)
        
        # Save/Load settings
        settings_buttons = ttk.Frame(settings_frame)
        settings_buttons.pack(pady=10)
        ttk.Button(settings_buttons, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT)
        ttk.Button(settings_buttons, text="Load Settings", command=self.load_settings).pack(side=tk.LEFT, padx=5)
        
    def setup_categories_tab(self, notebook):
        """Setup category management interface"""
        cat_frame = ttk.Frame(notebook)
        notebook.add(cat_frame, text="Categories")
        
        ttk.Label(cat_frame, text="Automatic Categorization Keywords", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Categories list
        self.categories_text = scrolledtext.ScrolledText(
            cat_frame,
            height=20,
            width=80,
            bg=self._theme_colors['bg_secondary'],
            fg=self._theme_colors['text_primary'],
            font=('SF Pro Display', 10),
            insertbackground=self._theme_colors['text_primary'],
            borderwidth=0,
            highlightthickness=0
        )
        self.categories_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Load current categories
        self.load_categories_display()
        
        # Buttons
        cat_buttons = ttk.Frame(cat_frame)
        cat_buttons.pack(pady=10)
        ttk.Button(cat_buttons, text="Save Categories", command=self.save_categories).pack(side=tk.LEFT)
        ttk.Button(cat_buttons, text="Reset to Default", command=self.reset_categories).pack(side=tk.LEFT, padx=5)
        
    def setup_analysis_tab(self, notebook):
        """Setup intelligent audio analysis interface"""
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Audio Analysis")
        
        # Analysis settings frame
        settings_frame = ttk.LabelFrame(analysis_frame, text="Analysis Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Enable/disable analysis
        ttk.Checkbutton(settings_frame, text="Enable intelligent audio analysis", 
                       variable=self.enable_analysis).pack(anchor=tk.W)
        
        # Analysis options
        options_frame = ttk.Frame(settings_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        left_frame = ttk.Frame(options_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Checkbutton(left_frame, text="Auto-detect BPM", 
                       variable=self.analyze_bpm).pack(anchor=tk.W)
        ttk.Checkbutton(left_frame, text="Detect musical key", 
                       variable=self.analyze_key).pack(anchor=tk.W)
        
        right_frame = ttk.Frame(options_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        ttk.Checkbutton(right_frame, text="LUFS loudness analysis", 
                       variable=self.analyze_loudness).pack(anchor=tk.W)
        ttk.Checkbutton(right_frame, text="Audio fingerprinting", 
                       variable=self.analyze_fingerprint).pack(anchor=tk.W)
        
        # Advanced categorization
        advanced_frame = ttk.Frame(settings_frame)
        advanced_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(advanced_frame, text="Auto-categorize by analysis results (overrides filename categorization)", 
                       variable=self.auto_categorize_by_analysis).pack(anchor=tk.W)
        
        # Analysis controls
        controls_frame = ttk.LabelFrame(analysis_frame, text="Analysis Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill=tk.X)
        
        self.analyze_button = ttk.Button(buttons_frame, text="Analyze Selected Files", 
                  command=self.analyze_selected_files)
        self.analyze_button.pack(side=tk.LEFT)
        
        self.stop_analysis_button = ttk.Button(buttons_frame, text="Stop Analysis", 
                  command=self.stop_analysis, state=tk.DISABLED)
        self.stop_analysis_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(buttons_frame, text="Quick Preview", 
                  command=self.quick_analysis_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Clear Analysis Data", 
                  command=self.clear_analysis_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Export Analysis Report", 
                  command=self.export_analysis_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Find Duplicates", 
                  command=self.find_duplicate_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Create Smart Playlists", 
                  command=self.create_smart_playlists).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Find Similar", 
                  command=self.find_similar_tracks).pack(side=tk.LEFT, padx=5)
        
        # Progress bar for analysis
        progress_frame = ttk.Frame(controls_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(progress_frame, text="Analysis Progress:").pack(anchor=tk.W)
        self.analysis_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate', 
                                               style='Green.Horizontal.TProgressbar')
        self.analysis_progress.pack(fill=tk.X, pady=2)
        
        self.analysis_status_label = ttk.Label(progress_frame, text="Ready to analyze files")
        self.analysis_status_label.pack(anchor=tk.W)
        
        # Analysis results display
        results_frame = ttk.LabelFrame(analysis_frame, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for results
        columns = ("File", "BPM", "Key", "LUFS", "Genre", "Mood", "Energy", "Category", "Duration")
        self.analysis_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=12)
        
        # Configure columns
        self.analysis_tree.heading("File", text="Filename")
        self.analysis_tree.heading("BPM", text="BPM")
        self.analysis_tree.heading("Key", text="Key")
        self.analysis_tree.heading("LUFS", text="LUFS")
        self.analysis_tree.heading("Genre", text="Genre")
        self.analysis_tree.heading("Mood", text="Mood")
        self.analysis_tree.heading("Energy", text="Energy")
        self.analysis_tree.heading("Category", text="Category")
        self.analysis_tree.heading("Duration", text="Duration")
        
        self.analysis_tree.column("File", width=180)
        self.analysis_tree.column("BPM", width=60)
        self.analysis_tree.column("Key", width=60)
        self.analysis_tree.column("LUFS", width=60)
        self.analysis_tree.column("Genre", width=80)
        self.analysis_tree.column("Mood", width=80)
        self.analysis_tree.column("Energy", width=60)
        self.analysis_tree.column("Category", width=100)
        self.analysis_tree.column("Duration", width=60)
        
        # Add scrollbar for treeview
        tree_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.analysis_tree.yview)
        self.analysis_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.analysis_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to show detailed analysis
        self.analysis_tree.bind("<Double-1>", self.show_detailed_analysis)
        
        # Filename customization frame
        filename_frame = ttk.LabelFrame(analysis_frame, text="Filename Generation Settings", padding=10)
        filename_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(filename_frame, text="Select which metadata to include in converted filenames:", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Create two rows of toggles
        toggle_frame = ttk.Frame(filename_frame)
        toggle_frame.pack(fill=tk.X)

        # Top row
        top_row = ttk.Frame(toggle_frame)
        top_row.pack(fill=tk.X, pady=2)

        toggle_kwargs = dict(
            width=10,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            highlightbackground=self._theme_colors['bg_secondary'],
            highlightcolor=self._theme_colors['bg_secondary'],
            bg=self._theme_colors['bg_secondary'],
            fg=self._theme_colors['text_subtle'],
            activebackground=self._theme_colors['button_hover'],
            activeforeground=self._theme_colors['text_primary'],
            font=('SF Pro Display', 10),
            pady=6
        )

        # Create clickable toggle buttons that look highlighted when selected
        self.key_button = tk.Button(top_row, text="Key",
                                   command=lambda: self.toggle_filename_option('key'),
                                   **toggle_kwargs)
        self.key_button.pack(side=tk.LEFT, padx=5)

        self.mood_button = tk.Button(top_row, text="Mood",
                                    command=lambda: self.toggle_filename_option('mood'),
                                    **toggle_kwargs)
        self.mood_button.pack(side=tk.LEFT, padx=5)

        self.energy_button = tk.Button(top_row, text="Energy",
                                      command=lambda: self.toggle_filename_option('energy'),
                                      **toggle_kwargs)
        self.energy_button.pack(side=tk.LEFT, padx=5)

        self.bpm_button = tk.Button(top_row, text="BPM",
                                   command=lambda: self.toggle_filename_option('bpm'),
                                   **toggle_kwargs)
        self.bpm_button.pack(side=tk.LEFT, padx=5)

        # Bottom row
        bottom_row = ttk.Frame(toggle_frame)
        bottom_row.pack(fill=tk.X, pady=2)

        self.genre_button = tk.Button(bottom_row, text="Genre",
                                     command=lambda: self.toggle_filename_option('genre'),
                                     **toggle_kwargs)
        self.genre_button.pack(side=tk.LEFT, padx=5)

        self.lufs_button = tk.Button(bottom_row, text="LUFS",
                                    command=lambda: self.toggle_filename_option('lufs'),
                                    **toggle_kwargs)
        self.lufs_button.pack(side=tk.LEFT, padx=5)

        self.category_button = tk.Button(bottom_row, text="Category",
                                        command=lambda: self.toggle_filename_option('category'),
                                        **toggle_kwargs)
        self.category_button.pack(side=tk.LEFT, padx=5)
        
        # Preview area
        preview_frame = ttk.Frame(filename_frame)
        preview_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(preview_frame, text="Filename Preview:", font=("Arial", 9)).pack(anchor=tk.W)
        self.filename_preview = tk.Label(preview_frame, text="example_track-C_Major-energetic-high.wav",
                                        font=("Arial", 9, "italic"),
                                        fg=self._theme_colors['text_subtle'], anchor=tk.W,
                                        wraplength=600)
        self.filename_preview.pack(anchor=tk.W, fill=tk.X)
        
        # Initialize button states
        self.update_filename_buttons()
        self.update_filename_preview()
        
    def toggle_filename_option(self, option):
        """Toggle filename generation options and update button appearance"""
        option_map = {
            'key': self.include_key_in_filename,
            'mood': self.include_mood_in_filename,
            'energy': self.include_energy_in_filename,
            'bpm': self.include_bpm_in_filename,
            'genre': self.include_genre_in_filename,
            'lufs': self.include_lufs_in_filename,
            'category': self.include_category_in_filename
        }
        
        if option in option_map:
            # Toggle the value
            current_value = option_map[option].get()
            option_map[option].set(not current_value)
            
            # Update button appearance and preview
            self.update_filename_buttons()
            self.update_filename_preview()
    
    def update_filename_buttons(self):
        """Update button appearance based on selection state"""
        buttons = {
            'key': (self.key_button, self.include_key_in_filename),
            'mood': (self.mood_button, self.include_mood_in_filename),
            'energy': (self.energy_button, self.include_energy_in_filename),
            'bpm': (self.bpm_button, self.include_bpm_in_filename),
            'genre': (self.genre_button, self.include_genre_in_filename),
            'lufs': (self.lufs_button, self.include_lufs_in_filename),
            'category': (self.category_button, self.include_category_in_filename)
        }
        
        selected_bg = self._theme_colors['button_primary']
        selected_fg = self._theme_colors['text_primary']
        selected_active = self._theme_colors['button_hover']
        normal_bg = self._theme_colors['bg_secondary']
        normal_fg = self._theme_colors['text_subtle']
        normal_active = self._theme_colors['bg_accent']

        for option, (button, var) in buttons.items():
            if var.get():
                button.config(
                    bg=selected_bg,
                    fg=selected_fg,
                    activebackground=selected_active,
                    activeforeground=selected_fg,
                    relief=tk.FLAT,
                    bd=0,
                    highlightthickness=0,
                    highlightbackground=selected_bg,
                    highlightcolor=selected_bg,
                    font=('SF Pro Display', 10, 'bold')
                )
            else:
                button.config(
                    bg=normal_bg,
                    fg=normal_fg,
                    activebackground=normal_active,
                    activeforeground=selected_fg,
                    relief=tk.FLAT,
                    bd=0,
                    highlightthickness=0,
                    highlightbackground=normal_bg,
                    highlightcolor=normal_bg,
                    font=('SF Pro Display', 10)
                )
    
    def update_filename_preview(self):
        """Update the filename preview based on current selections"""
        example_parts = ["example_track"]
        
        if self.include_key_in_filename.get():
            example_parts.append("C_Major")
        if self.include_mood_in_filename.get():
            example_parts.append("energetic")
        if self.include_energy_in_filename.get():
            example_parts.append("high")
        if self.include_bpm_in_filename.get():
            example_parts.append("128bpm")
        if self.include_genre_in_filename.get():
            example_parts.append("electronic")
        if self.include_lufs_in_filename.get():
            example_parts.append("-14lufs")
        if self.include_category_in_filename.get():
            example_parts.append("melody")
        
        preview_name = "-".join(example_parts) + ".wav"
        self.filename_preview.config(text=preview_name)
        
    def setup_progress_tab(self, notebook):
        """Setup progress monitoring interface"""
        progress_frame = ttk.Frame(notebook)
        notebook.add(progress_frame, text="Progress")
        
        # Progress log
        ttk.Label(progress_frame, text="Conversion Log", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        self.log_text = scrolledtext.ScrolledText(
            progress_frame,
            height=25,
            width=100,
            bg=self._theme_colors['bg_secondary'],
            fg=self._theme_colors['text_primary'],
            font=('SF Pro Display', 10),
            insertbackground=self._theme_colors['text_primary'],
            borderwidth=0,
            highlightthickness=0
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Clear log button
        ttk.Button(progress_frame, text="Clear Log", command=self.clear_log).pack(pady=5)
        
    def update_pitch_label(self, value):
        """Update pitch shift label"""
        self.pitch_value_label.config(text=f"{float(value):.1f}")
        
    def generate_enhanced_filename(self, original_filename: str, input_file: str) -> str:
        """Generate filename with selected metadata based on user toggles"""
        name_without_ext = os.path.splitext(original_filename)[0]
        
        # Build filename parts based on user selections
        filename_parts = [name_without_ext]
        
        try:
            # Check if we already have analysis results
            if input_file in self.analysis_results:
                result = self.analysis_results[input_file]
            else:
                # No analysis performed yet, only use selected key if available
                if self.include_key_in_filename.get() and self.selected_key.get() != "Original":
                    key_clean = self.selected_key.get().replace('/', '').replace('#', 's').replace('♭', 'b').replace(' ', '_')
                    filename_parts.append(key_clean)
                
                # Return early if no analysis data
                return "-".join(filename_parts)
            
            # Add metadata based on user selections and analysis results
            
            # Key information - SIMPLIFIED: Just use selected key for all files
            if self.include_key_in_filename.get() and self.selected_key.get() != "Original":
                # Always use the selected target key - ignore detected key
                key_clean = self.selected_key.get().replace('/', '').replace('#', 's').replace('♭', 'b').replace(' ', '_')
                filename_parts.append(key_clean)
            
            # BPM information
            if self.include_bpm_in_filename.get() and result and hasattr(result, 'tempo') and result.tempo:
                bpm = f"{int(result.tempo)}bpm"
                filename_parts.append(bpm)
            
            # Genre information
            if self.include_genre_in_filename.get() and result and hasattr(result, 'genre') and result.genre:
                genre_clean = result.genre.replace(' ', '_').lower()
                filename_parts.append(genre_clean)
            
            # Mood information
            if self.include_mood_in_filename.get():
                mood = self.determine_mood(result)
                if mood:
                    filename_parts.append(mood)
            
            # Energy information
            if self.include_energy_in_filename.get():
                energy = self.determine_energy_level(result)
                if energy:
                    filename_parts.append(energy)
            
            # LUFS information
            if self.include_lufs_in_filename.get() and result and hasattr(result, 'lufs') and result.lufs is not None:
                lufs = f"{int(result.lufs)}lufs"
                filename_parts.append(lufs)
            
            # Category information
            if self.include_category_in_filename.get():
                category = self.categorize_file(original_filename)
                if category and category != "uncategorized":
                    filename_parts.append(category)
            
        except Exception as e:
            self.log(f"Error generating enhanced filename: {e}")
            # Fallback to basic filename with selected key if available
            if self.include_key_in_filename.get() and self.selected_key.get() != "Original":
                key_clean = self.selected_key.get().replace('/', '').replace('#', 's').replace('♭', 'b').replace(' ', '_')
                filename_parts.append(key_clean)
        
        # Join parts and return
        enhanced_name = "-".join(filename_parts)
        return enhanced_name
    
    def determine_mood(self, analysis_result) -> str:
        """Determine mood from audio analysis"""
        if not analysis_result:
            return ""
        
        try:
            # Use tempo and spectral features to determine mood
            if hasattr(analysis_result, 'tempo') and analysis_result.tempo:
                tempo = analysis_result.tempo
                
                # Use spectral features if available
                if hasattr(analysis_result, 'mfcc') and analysis_result.mfcc is not None:
                    # Simple mood classification based on tempo and spectral content
                    if tempo < 80:
                        return "calm"
                    elif tempo < 100:
                        return "chill"
                    elif tempo < 130:
                        return "groove"
                    elif tempo < 150:
                        return "energetic"
                    else:
                        return "intense"
                else:
                    # Fallback based on tempo only
                    if tempo < 90:
                        return "slow"
                    elif tempo < 120:
                        return "mid"
                    else:
                        return "fast"
            
            # Try to use key mode for mood (major/minor)
            if hasattr(analysis_result, 'key') and analysis_result.key:
                if 'm' in analysis_result.key.lower():
                    return "minor"
                else:
                    return "major"
                    
        except Exception:
            pass
            
        return ""
    
    def determine_energy_level(self, analysis_result) -> str:
        """Determine energy level from audio analysis"""
        if not analysis_result:
            return ""
        
        try:
            # Use LUFS loudness for energy if available
            if hasattr(analysis_result, 'lufs') and analysis_result.lufs is not None:
                lufs = analysis_result.lufs
                
                if lufs > -10:
                    return "high"
                elif lufs > -18:
                    return "med"
                elif lufs > -25:
                    return "low"
                else:
                    return "quiet"
            
            # Fallback: use tempo for energy estimation
            if hasattr(analysis_result, 'tempo') and analysis_result.tempo:
                tempo = analysis_result.tempo
                
                if tempo > 140:
                    return "high"
                elif tempo > 110:
                    return "med"
                else:
                    return "low"
                    
        except Exception:
            pass
            
        return ""
        
    def update_key_label(self, value):
        """Update key shift label"""
        self.key_value_label.config(text=f"{int(float(value))}")
        
    def calculate_key_transposition(self, selected_key, detected_key=None):
        """Calculate semitones needed to transpose to selected key"""
        if selected_key == "Original":
            return 0
        
        # Key mapping to semitones (C = 0)
        key_to_semitones = {
            "C": 0, "C#/Db": 1, "D": 2, "D#/Eb": 3, "E": 4, "F": 5,
            "F#/Gb": 6, "G": 7, "G#/Ab": 8, "A": 9, "A#/Bb": 10, "B": 11,
            "Cm": 0, "C#m/Dbm": 1, "Dm": 2, "D#m/Ebm": 3, "Em": 4, "Fm": 5,
            "F#m/Gbm": 6, "Gm": 7, "G#m/Abm": 8, "Am": 9, "A#m/Bbm": 10, "Bm": 11
        }
        
        target_semitones = key_to_semitones.get(selected_key, 0)
        
        # If we have detected key information, calculate relative transposition
        if detected_key and detected_key in key_to_semitones:
            source_semitones = key_to_semitones[detected_key]
            return target_semitones - source_semitones
        else:
            # If no detected key, assume original is in C and transpose to target
            return target_semitones
        
    def select_folders(self):
        """Select input folders"""
        folders = filedialog.askdirectory(title="Select folder containing audio files")
        if folders:
            if folders not in self.input_folders:
                self.input_folders.append(folders)
                self.update_input_list()
                
    def select_files(self):
        """Select individual input files"""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(title="Select audio files", filetypes=filetypes)
        for file in files:
            if file not in self.input_files:
                self.input_files.append(file)
        self.update_input_list()
        
    def clear_inputs(self):
        """Clear all input selections"""
        self.input_folders.clear()
        self.input_files.clear()
        self.update_input_list()
        
    def update_input_list(self):
        """Update the input listbox"""
        self.input_listbox.delete(0, tk.END)
        for folder in self.input_folders:
            self.input_listbox.insert(tk.END, f"📁 {folder}")
        for file in self.input_files:
            self.input_listbox.insert(tk.END, f"🎵 {os.path.basename(file)}")
            
    def select_output_folder(self):
        """Select output folder"""
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_folder = folder
            self.output_label.config(
                text=folder,
                foreground=self._theme_colors['text_primary']
            )
            
    def get_audio_files(self) -> List[str]:
        """Get all audio files from selected folders and files"""
        # Comprehensive list of supported input audio extensions
        audio_extensions = {
            # Common formats
            '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma',
            # Professional/Studio formats
            '.aiff', '.aifc', '.au', '.caf', '.w64', '.rf64', '.bwf', '.sd2', '.snd', '.iff', '.svx', '.nist', '.voc', '.ircam', '.xi',
            # Advanced compressed formats
            '.opus', '.ac3', '.mp2', '.dts', '.amr', '.gsm', '.ra', '.rm',
            # Apple formats
            '.m4r', '.m4b', '.m4p', '.caff',
            # Video containers (audio extraction)
            '.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp', '.m4v', '.mka',
            # Lossless compressed
            '.tta', '.tak', '.als', '.ape', '.wv', '.mpc', '.ofr', '.ofs', '.shn',
            # Raw formats
            '.raw', '.pcm'
        }
        files = []
        
        # Add individual files
        files.extend(self.input_files)
        
        # Scan folders
        for folder in self.input_folders:
            for root, dirs, filenames in os.walk(folder):
                for filename in filenames:
                    if Path(filename).suffix.lower() in audio_extensions:
                        files.append(os.path.join(root, filename))
                        
        return files
    
    def categorize_file(self, filename: str, file_path: str = None) -> str:
        """Categorize file based on analysis results or filename"""
        # Use analysis-based categorization if enabled and available
        if self.auto_categorize_by_analysis.get() and file_path and file_path in self.analysis_results:
            analysis_result = self.analysis_results[file_path]
            suggested_category = self.audio_analyzer.categorize_by_analysis(analysis_result)
            if suggested_category != "uncategorized":
                return suggested_category
        
        # Fall back to filename-based categorization
        if not self.auto_categorize.get():
            return "uncategorized"
            
        filename_lower = filename.lower()
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return category
                    
        return "uncategorized"
    
    def show_format_info(self):
        """Show detailed information about all supported audio formats"""
        info_window = tk.Toplevel(self.root)
        info_window.title("Supported Audio Formats")
        info_window.geometry("800x600")
        info_window.configure(bg=self._theme_colors['bg_primary'])

        # Create scrollable text widget
        text_frame = ttk.Frame(info_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_widget = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg=self._theme_colors['bg_secondary'],
            fg=self._theme_colors['text_primary'],
            insertbackground=self._theme_colors['text_primary'],
            borderwidth=0,
            highlightthickness=0
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        format_info = """
🎵 COMPREHENSIVE AUDIO FORMAT SUPPORT 🎵

This converter supports ALL major audio formats and many specialized ones:

══════════════════════════════════════════════════════════════════════════════════
📁 UNCOMPRESSED FORMATS (Best Quality, Large Files)
══════════════════════════════════════════════════════════════════════════════════
• WAV        - Standard uncompressed audio (Windows/Cross-platform)
• AIFF/AIFC  - Audio Interchange File Format (Mac/Professional)
• AU         - Sun/NeXT audio format (Unix systems)
• CAF        - Core Audio Format (Apple professional)
• W64        - Sony Wave64 (>4GB file support)
• RF64       - EBU RF64 (broadcast standard)
• BWF        - Broadcast Wave Format (professional broadcast)

══════════════════════════════════════════════════════════════════════════════════
🗜️ LOSSLESS COMPRESSED (Perfect Quality, Smaller Files)
══════════════════════════════════════════════════════════════════════════════════
• FLAC       - Free Lossless Audio Codec (most popular)
• TTA        - True Audio lossless codec
• TAK        - Tom's Lossless Audio Codec
• ALS        - MPEG-4 Audio Lossless Standard
• APE        - Monkey's Audio (high compression)
• WV         - WavPack lossless codec
• SHN        - Shorten lossless compression

══════════════════════════════════════════════════════════════════════════════════
🎶 LOSSY COMPRESSED (Good Quality, Small Files)
══════════════════════════════════════════════════════════════════════════════════
• MP3        - Most compatible format worldwide
• AAC        - Advanced Audio Codec (better than MP3)
• M4A        - iTunes/Apple AAC container
• OGG        - Ogg Vorbis (open source)
• OPUS       - Modern, high-efficiency codec
• WMA        - Windows Media Audio
• AC3        - Dolby Digital audio
• MP2        - MPEG Layer II audio

══════════════════════════════════════════════════════════════════════════════════
🍎 APPLE ECOSYSTEM FORMATS
══════════════════════════════════════════════════════════════════════════════════
• M4A        - iTunes music format
• M4R        - iPhone ringtone format
• M4B        - iTunes audiobook format
• M4P        - iTunes protected audio
• CAFF       - Core Audio File Format

══════════════════════════════════════════════════════════════════════════════════
🎬 VIDEO CONTAINER FORMATS (Audio Extraction/Embedding)
══════════════════════════════════════════════════════════════════════════════════
• MP4        - MPEG-4 container (most universal)
• MOV        - QuickTime movie format
• AVI        - Audio Video Interleave
• MKV        - Matroska container
• WEBM       - Web Media format
• 3GP        - Mobile video format
• MKA        - Matroska audio-only

══════════════════════════════════════════════════════════════════════════════════
🎛️ PROFESSIONAL/STUDIO FORMATS
══════════════════════════════════════════════════════════════════════════════════
• SD2        - Sound Designer II (Pro Tools)
• SND        - Various sound formats
• IFF        - Interchange File Format
• SVX        - Amiga sound format
• NIST       - NIST SPHERE format
• VOC        - Creative Voice format
• IRCAM      - IRCAM sound format
• XI         - FastTracker instrument format

══════════════════════════════════════════════════════════════════════════════════
📡 RAW & SPECIALIZED FORMATS
══════════════════════════════════════════════════════════════════════════════════
• RAW/PCM    - Raw audio data
• S16LE      - 16-bit signed little-endian
• S24LE      - 24-bit signed little-endian
• S32LE      - 32-bit signed little-endian
• F32LE      - 32-bit float little-endian
• F64LE      - 64-bit float little-endian

══════════════════════════════════════════════════════════════════════════════════
📻 LEGACY/STREAMING FORMATS
══════════════════════════════════════════════════════════════════════════════════
• RA/RM      - RealAudio formats
• AMR        - Adaptive Multi-Rate (mobile)
• GSM        - Global System for Mobile
• DTS        - Digital Theater Systems

══════════════════════════════════════════════════════════════════════════════════
💡 FORMAT RECOMMENDATIONS
══════════════════════════════════════════════════════════════════════════════════

FOR MUSIC PRODUCTION:
• WAV/AIFF   - Recording, mixing, mastering
• FLAC       - Archival storage, distribution
• BWF        - Broadcast and professional work

FOR GENERAL USE:
• MP3        - Maximum compatibility
• AAC/M4A    - Better quality than MP3
• OPUS       - Best modern compression

FOR APPLE DEVICES:
• M4A        - iTunes and iOS
• M4R        - iPhone ringtones
• AIFF       - Professional Apple workflows

FOR ARCHIVAL:
• FLAC       - Best balance of quality and size
• WAV        - Original uncompressed quality
• BWF        - Professional archival standard

══════════════════════════════════════════════════════════════════════════════════
⚙️ TECHNICAL NOTES
══════════════════════════════════════════════════════════════════════════════════
• All formats support custom sample rates and bit depths where applicable
• Metadata preservation varies by format capability
• Some formats require FFmpeg for full functionality
• Video formats will extract/embed audio tracks only
• Raw formats output pure audio data without headers
"""
        
        text_widget.insert(tk.END, format_info)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=10)
    
    def validate_format_compatibility(self, format_name: str) -> tuple[bool, str]:
        """Validate if a format is supported and return compatibility info"""
        format_name = format_name.lower()
        
        # Format categories and their requirements
        format_requirements = {
            # Always supported (via soundfile)
            'wav': (True, "Native support"),
            'flac': (True, "Native support"),
            'aiff': (True, "Native support"),
            'aifc': (True, "Native support"),
            
            # Require FFmpeg (via pydub)
            'mp3': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'aac': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'm4a': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'ogg': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'opus': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'wma': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'ac3': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'mp2': (which("ffmpeg") is not None, "Requires FFmpeg"),
            
            # Apple formats
            'm4r': (which("ffmpeg") is not None, "Requires FFmpeg (Apple format)"),
            'm4b': (which("ffmpeg") is not None, "Requires FFmpeg (Apple format)"),
            'm4p': (which("ffmpeg") is not None, "Requires FFmpeg (Apple format)"),
            'caff': (which("ffmpeg") is not None, "Requires FFmpeg (Apple format)"),
            
            # Professional formats
            'au': (True, "Native support"),
            'caf': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'w64': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'rf64': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'bwf': (which("ffmpeg") is not None, "Requires FFmpeg"),
            
            # Video containers
            'mp4': (which("ffmpeg") is not None, "Requires FFmpeg (video container)"),
            'mov': (which("ffmpeg") is not None, "Requires FFmpeg (video container)"),
            'avi': (which("ffmpeg") is not None, "Requires FFmpeg (video container)"),
            'mkv': (which("ffmpeg") is not None, "Requires FFmpeg (video container)"),
            'webm': (which("ffmpeg") is not None, "Requires FFmpeg (video container)"),
            '3gp': (which("ffmpeg") is not None, "Requires FFmpeg (video container)"),
            'm4v': (which("ffmpeg") is not None, "Requires FFmpeg (video container)"),
            'mka': (which("ffmpeg") is not None, "Requires FFmpeg (video container)"),
            
            # Raw formats
            'raw': (True, "Raw PCM data"),
            'pcm': (True, "Raw PCM data"),
            's16le': (True, "Raw 16-bit signed"),
            's24le': (True, "Raw 24-bit signed"),
            's32le': (True, "Raw 32-bit signed"),
            'f32le': (True, "Raw 32-bit float"),
            'f64le': (True, "Raw 64-bit float"),
            
            # Legacy/specialized (may require additional codecs)
            'ra': (which("ffmpeg") is not None, "Requires FFmpeg (RealAudio)"),
            'rm': (which("ffmpeg") is not None, "Requires FFmpeg (RealMedia)"),
            'amr': (which("ffmpeg") is not None, "Requires FFmpeg (AMR codec)"),
            'gsm': (which("ffmpeg") is not None, "Requires FFmpeg (GSM codec)"),
            
            # Lossless compressed (may require additional libraries)
            'tta': (which("ffmpeg") is not None, "Requires FFmpeg"),
            'tak': (which("ffmpeg") is not None, "Requires FFmpeg (rare format)"),
            'als': (which("ffmpeg") is not None, "Requires FFmpeg (MPEG-4 ALS)"),
            'ape': (which("ffmpeg") is not None, "Requires FFmpeg (Monkey's Audio)"),
            'wv': (which("ffmpeg") is not None, "Requires FFmpeg (WavPack)"),
            'shn': (which("ffmpeg") is not None, "Requires FFmpeg (Shorten)"),
        }
        
        if format_name in format_requirements:
            supported, reason = format_requirements[format_name]
            return supported, reason
        else:
            # Unknown format, try with FFmpeg if available
            if which("ffmpeg"):
                return True, "Experimental support via FFmpeg"
            else:
                return False, "Unknown format, requires FFmpeg"
    
    def check_system_dependencies(self) -> dict:
        """Check which system dependencies are available"""
        dependencies = {
            'ffmpeg': which("ffmpeg") is not None,
            'soundfile': True,  # Always available if we got this far
            'pydub': True,      # Always available if we got this far
            'librosa': True,    # Always available if we got this far
        }
        
        # Check optional dependencies
        try:
            import essentia
            dependencies['essentia'] = True
        except ImportError:
            dependencies['essentia'] = False
            
        try:
            import pyloudnorm
            dependencies['pyloudnorm'] = True
        except ImportError:
            dependencies['pyloudnorm'] = False
            
        try:
            import mutagen
            dependencies['mutagen'] = True
        except ImportError:
            dependencies['mutagen'] = False
            
        return dependencies
    
    def show_system_capabilities(self):
        """Show current system capabilities and format support"""
        capabilities_window = tk.Toplevel(self.root)
        capabilities_window.title("System Capabilities & Format Support")
        capabilities_window.geometry("700x500")
        capabilities_window.configure(bg=self._theme_colors['bg_primary'])

        # Create scrollable text widget
        text_frame = ttk.Frame(capabilities_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_widget = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg=self._theme_colors['bg_secondary'],
            fg=self._theme_colors['text_primary'],
            insertbackground=self._theme_colors['text_primary'],
            borderwidth=0,
            highlightthickness=0
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Check system dependencies
        deps = self.check_system_dependencies()
        
        capability_info = "🔧 SYSTEM CAPABILITIES & FORMAT SUPPORT\n"
        capability_info += "=" * 60 + "\n\n"
        
        # Core dependencies
        capability_info += "📦 CORE LIBRARIES:\n"
        capability_info += f"✅ Librosa (audio processing): Available\n"
        capability_info += f"✅ SoundFile (WAV/FLAC support): Available\n"
        capability_info += f"✅ Pydub (format conversion): Available\n\n"
        
        # System dependencies
        capability_info += "🔧 SYSTEM DEPENDENCIES:\n"
        if deps['ffmpeg']:
            capability_info += "✅ FFmpeg: Installed - Full format support available\n"
        else:
            capability_info += "❌ FFmpeg: Not found - Limited to WAV/FLAC/AIFF only\n"
            capability_info += "   Install with: brew install ffmpeg (macOS)\n\n"
        
        # Optional dependencies
        capability_info += "📋 OPTIONAL LIBRARIES:\n"
        capability_info += f"{'✅' if deps['essentia'] else '❌'} Essentia: {'Available' if deps['essentia'] else 'Not installed'} - Advanced analysis\n"
        capability_info += f"{'✅' if deps['pyloudnorm'] else '❌'} PyLoudnorm: {'Available' if deps['pyloudnorm'] else 'Not installed'} - LUFS loudness\n"
        capability_info += f"{'✅' if deps['mutagen'] else '❌'} Mutagen: {'Available' if deps['mutagen'] else 'Not installed'} - Metadata support\n\n"
        
        # Format support breakdown
        capability_info += "🎵 FORMAT SUPPORT STATUS:\n"
        capability_info += "-" * 40 + "\n"
        
        # Check a sample of formats
        test_formats = ['wav', 'flac', 'aiff', 'mp3', 'aac', 'm4a', 'ogg', 'opus', 'wma', 'raw', 'mp4', 'webm']
        
        for fmt in test_formats:
            supported, reason = self.validate_format_compatibility(fmt)
            status = "✅" if supported else "❌"
            capability_info += f"{status} {fmt.upper()}: {reason}\n"
        
        capability_info += "\n" + "=" * 60 + "\n"
        capability_info += "📖 RECOMMENDATIONS:\n\n"
        
        if not deps['ffmpeg']:
            capability_info += "🚨 INSTALL FFMPEG for full format support:\n"
            capability_info += "   macOS: brew install ffmpeg\n"
            capability_info += "   Ubuntu: sudo apt install ffmpeg\n"
            capability_info += "   Windows: Download from https://ffmpeg.org/\n\n"
        
        if not deps['mutagen']:
            capability_info += "💡 Install mutagen for better metadata support:\n"
            capability_info += "   pip install mutagen\n\n"
        
        capability_info += "🎯 CURRENTLY AVAILABLE FORMATS:\n"
        available_formats = []
        for fmt in self.formats:
            supported, _ = self.validate_format_compatibility(fmt)
            if supported:
                available_formats.append(fmt.upper())
        
        capability_info += f"   {', '.join(available_formats[:15])}\n"
        if len(available_formats) > 15:
            capability_info += f"   ... and {len(available_formats) - 15} more formats\n"
        
        capability_info += f"\nTotal supported formats: {len([f for f in self.formats if self.validate_format_compatibility(f)[0]])}/{len(self.formats)}\n"
        
        text_widget.insert(tk.END, capability_info)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(capabilities_window, text="Close", command=capabilities_window.destroy).pack(pady=10)
    
    def analyze_selected_files(self):
        """Analyze all selected audio files with progress tracking"""
        audio_files = self.get_audio_files()
        if not audio_files:
            messagebox.showwarning("Warning", "No audio files selected")
            return
        
        if self.is_analyzing:
            messagebox.showinfo("Info", "Analysis already in progress")
            return
        
        # Reset stop flag and enable stop button
        self.stop_analysis_flag = False
        self.analyze_button.config(state=tk.DISABLED)
        self.stop_analysis_button.config(state=tk.NORMAL)
        
        # Setup progress bar
        self.analysis_progress.config(maximum=len(audio_files), value=0)
        self.analysis_status_label.config(text=f"Starting analysis of {len(audio_files)} files...")
        
        # Start analysis in separate thread
        self.analysis_thread = threading.Thread(
            target=self.analysis_worker,
            args=(audio_files,),
            daemon=True
        )
        self.analysis_thread.start()
    
    def stop_analysis(self):
        """Stop the current analysis process"""
        if self.is_analyzing:
            self.stop_analysis_flag = True
            self.analysis_status_label.config(text="Stopping analysis...")
            self.log("⏹ Analysis stop requested by user")
    
    def quick_analysis_preview(self):
        """Quick analysis preview of first few selected files"""
        audio_files = self.get_audio_files()
        if not audio_files:
            messagebox.showwarning("Warning", "No audio files selected")
            return
        
        # Analyze first 3 files for quick preview
        preview_files = audio_files[:3]
        
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Quick Analysis Preview")
        preview_window.geometry("600x400")
        preview_window.configure(bg=self._theme_colors['bg_primary'])
        
        ttk.Label(preview_window, text="Quick Analysis Preview", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # Progress bar for preview
        preview_progress = ttk.Progressbar(preview_window, length=400, mode='determinate')
        preview_progress.pack(pady=5)
        
        # Results text
        preview_text = scrolledtext.ScrolledText(
            preview_window,
            height=20,
            width=70,
            bg=self._theme_colors['bg_secondary'],
            fg=self._theme_colors['text_primary'],
            insertbackground=self._theme_colors['text_primary'],
            borderwidth=0,
            highlightthickness=0
        )
        preview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        def preview_worker():
            total = len(preview_files)
            preview_text.insert(tk.END, f"Analyzing {total} files for quick preview...\n\n")
            
            for i, file_path in enumerate(preview_files):
                filename = os.path.basename(file_path)
                preview_text.insert(tk.END, f"Analyzing: {filename}\n")
                preview_text.see(tk.END)
                preview_window.update()
                
                try:
                    # Quick analysis (shorter duration for preview)
                    y, sr = librosa.load(file_path, duration=10, sr=22050)  # 10 seconds max, lower quality
                    
                    # Basic analysis
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    rms_energy = np.mean(librosa.feature.rms(y=y))
                    
                    # Simple key detection
                    chroma_mean = np.mean(chroma, axis=1)
                    key_idx = np.argmax(chroma_mean)
                    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    detected_key = key_names[key_idx]
                    
                    # Quick categorization
                    if tempo < 80:
                        tempo_cat = "slow"
                    elif tempo < 120:
                        tempo_cat = "moderate" 
                    elif tempo < 140:
                        tempo_cat = "fast"
                    else:
                        tempo_cat = "very fast"
                    
                    if rms_energy > 0.5:
                        energy_cat = "high energy"
                    elif rms_energy > 0.2:
                        energy_cat = "medium energy"
                    else:
                        energy_cat = "low energy"
                    
                    preview_text.insert(tk.END, f"  BPM: {tempo:.1f} ({tempo_cat})\n")
                    preview_text.insert(tk.END, f"  Key: {detected_key}\n")
                    preview_text.insert(tk.END, f"  Brightness: {spectral_centroid:.0f} Hz\n")
                    preview_text.insert(tk.END, f"  Energy: {energy_cat}\n")
                    preview_text.insert(tk.END, f"  Suggested: {tempo_cat}, {energy_cat}\n\n")
                    
                except Exception as e:
                    preview_text.insert(tk.END, f"  Error: {str(e)}\n\n")
                
                # Update progress
                progress = ((i + 1) / total) * 100
                preview_progress['value'] = progress
                preview_window.update()
            
            preview_text.insert(tk.END, "Quick preview complete!\n")
            preview_text.insert(tk.END, "Use 'Analyze Selected Files' for full analysis.\n")
            preview_progress['value'] = 100
        
        # Run preview in thread
        import threading
        threading.Thread(target=preview_worker, daemon=True).start()
        
        # Close button
        ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)
    
    def analysis_worker(self, audio_files: List[str]):
        """Worker thread for audio analysis with progress tracking"""
        self.is_analyzing = True
        total_files = len(audio_files)
        self.log(f"Starting analysis of {total_files} files...")
        
        # Clear previous results
        self.analysis_results.clear()
        
        # Clear treeview
        def clear_tree():
            for item in self.analysis_tree.get_children():
                self.analysis_tree.delete(item)
        self.root.after(0, clear_tree)
        
        completed = 0
        for i, file_path in enumerate(audio_files):
            # Check if stop was requested
            if self.stop_analysis_flag:
                def update_stopped():
                    self.analysis_status_label.config(text=f"Analysis stopped. Completed {completed}/{total_files} files")
                    self.analysis_progress.config(value=completed)
                    self.analyze_button.config(state=tk.NORMAL)
                    self.stop_analysis_button.config(state=tk.DISABLED)
                self.root.after(0, update_stopped)
                self.is_analyzing = False
                self.log(f"⏹ Analysis stopped by user after {completed} files")
                return
            
            try:
                filename = os.path.basename(file_path)
                
                # Update progress
                def update_progress():
                    self.analysis_progress.config(value=i)
                    self.analysis_status_label.config(text=f"Analyzing: {filename} ({i+1}/{total_files})")
                self.root.after(0, update_progress)
                
                self.log(f"Analyzing: {filename}")
                
                # Perform analysis
                if self.enable_analysis.get():
                    result = self.audio_analyzer.analyze_audio_file(file_path)
                    self.analysis_results[file_path] = result
                    
                    # Update UI
                    def update_tree():
                        bpm_text = f"{result.bpm:.1f}" if result.bpm else "N/A"
                        key_text = f"{result.key} {result.scale}" if result.key and result.scale else "N/A"
                        lufs_text = f"{result.lufs_integrated:.1f}" if result.lufs_integrated else "N/A"
                        genre_text = result.genre_prediction if result.genre_prediction else "N/A"
                        mood_text = result.mood_prediction if result.mood_prediction else "N/A"
                        energy_text = f"{result.energy_level:.2f}" if result.energy_level else "N/A"
                        category = self.audio_analyzer.categorize_by_analysis(result)
                        duration_text = f"{result.duration:.1f}s"
                        
                        self.analysis_tree.insert("", "end", values=(
                            filename, bpm_text, key_text, lufs_text, genre_text, mood_text, energy_text, category, duration_text
                        ))
                    
                    self.root.after(0, update_tree)
                    
                    if result.analysis_errors:
                        for error in result.analysis_errors:
                            self.log(f"  Warning: {error}")
                    
                    self.log(f"✓ Analysis complete: {filename}")
                else:
                    self.log(f"⚠ Analysis disabled: {filename}")
                
                completed += 1
                    
            except Exception as e:
                self.log(f"✗ Analysis failed: {filename} - {str(e)}")
                completed += 1
        
        # Analysis complete
        def analysis_complete():
            self.analysis_progress.config(value=total_files)
            self.analysis_status_label.config(text=f"✅ Analysis complete! Processed {completed}/{total_files} files")
            self.analyze_button.config(state=tk.NORMAL)
            self.stop_analysis_button.config(state=tk.DISABLED)
        self.root.after(0, analysis_complete)
        
        self.is_analyzing = False
        self.log(f"🎉 Analysis completed: {completed}/{total_files} files processed")
    
    def clear_analysis_data(self):
        """Clear all analysis data"""
        self.analysis_results.clear()
        for item in self.analysis_tree.get_children():
            self.analysis_tree.delete(item)
        self.log("Analysis data cleared")
    
    def export_analysis_report(self):
        """Export analysis results to JSON report"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis data to export")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            title="Save Analysis Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                results_list = list(self.analysis_results.values())
                success = self.audio_analyzer.export_analysis_report(results_list, filename)
                if success:
                    messagebox.showinfo("Success", f"Analysis report exported to {filename}")
                    self.log(f"Analysis report exported: {filename}")
                else:
                    messagebox.showerror("Error", "Failed to export analysis report")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def find_duplicate_files(self):
        """Find and display potential duplicate files"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis data available. Please analyze files first.")
            return
        
        results_list = list(self.analysis_results.values())
        duplicates = self.audio_analyzer.find_duplicates(results_list)
        
        if not duplicates:
            messagebox.showinfo("No Duplicates", "No potential duplicate files found.")
            return
        
        # Show duplicates in a new window
        self.show_duplicates_window(duplicates)
    
    def show_duplicates_window(self, duplicates: List[List[str]]):
        """Show duplicates in a separate window"""
        dup_window = tk.Toplevel(self.root)
        dup_window.title("Potential Duplicate Files")
        dup_window.geometry("600x400")
        dup_window.configure(bg=self._theme_colors['bg_primary'])
        
        ttk.Label(dup_window, text="Potential Duplicate Files", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(dup_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        dup_text = scrolledtext.ScrolledText(
            text_frame,
            height=20,
            width=70,
            bg=self._theme_colors['bg_secondary'],
            fg=self._theme_colors['text_primary'],
            insertbackground=self._theme_colors['text_primary'],
            borderwidth=0,
            highlightthickness=0
        )
        dup_text.pack(fill=tk.BOTH, expand=True)
        
        # Display duplicates
        content = f"Found {len(duplicates)} groups of potential duplicates:\n\n"
        for i, group in enumerate(duplicates, 1):
            content += f"Group {i}:\n"
            for filename in group:
                content += f"  - {filename}\n"
            content += "\n"
        
        dup_text.insert(1.0, content)
        dup_text.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(dup_window, text="Close", command=dup_window.destroy).pack(pady=10)
    
    def show_detailed_analysis(self, event):
        """Show detailed analysis for selected file"""
        selection = self.analysis_tree.selection()
        if not selection:
            return
        
        item = self.analysis_tree.item(selection[0])
        filename = item['values'][0]
        
        # Find the analysis result
        result = None
        for file_path, analysis_result in self.analysis_results.items():
            if os.path.basename(file_path) == filename:
                result = analysis_result
                break
        
        if result:
            self.show_analysis_details_window(result)

    def _prepare_audio_for_export(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio array for exporting, preserving channel layout."""
        cleaned = np.nan_to_num(audio_data, nan=0.0)
        cleaned = np.clip(cleaned, -1.0, 1.0)
        if cleaned.ndim == 2 and cleaned.shape[0] < cleaned.shape[1]:
            cleaned = cleaned.T
        return cleaned

    def _create_audio_segment(self, audio_data: np.ndarray, sample_rate: int) -> AudioSegment:
        """Create an AudioSegment from numpy audio data with correct PCM encoding."""
        prepared = self._prepare_audio_for_export(audio_data)
        buffer = BytesIO()
        subtype_map = {16: 'PCM_16', 24: 'PCM_24', 32: 'PCM_32'}
        subtype = subtype_map.get(self.bit_depth.get(), 'PCM_16')
        sf.write(buffer, prepared, sample_rate, subtype=subtype)
        buffer.seek(0)
        segment = AudioSegment.from_file(buffer, format='wav')
        buffer.close()
        return segment

    def show_analysis_details_window(self, result: AudioAnalysisResult):
        """Show detailed analysis in a separate window"""
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Analysis Details - {result.filename}")
        details_window.geometry("500x600")
        details_window.configure(bg=self._theme_colors['bg_primary'])
        
        # Create scrolled text for details
        text_frame = ttk.Frame(details_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        details_text = scrolledtext.ScrolledText(
            text_frame,
            height=30,
            width=60,
            bg=self._theme_colors['bg_secondary'],
            fg=self._theme_colors['text_primary'],
            insertbackground=self._theme_colors['text_primary'],
            borderwidth=0,
            highlightthickness=0
        )
        details_text.pack(fill=tk.BOTH, expand=True)
        
        # Format detailed information
        content = f"Detailed Analysis Report\n"
        content += f"=" * 50 + "\n\n"
        content += f"File: {result.filename}\n"
        content += f"Duration: {result.duration:.2f} seconds\n"
        content += f"Sample Rate: {result.sample_rate} Hz\n\n"
        
        content += "TEMPO ANALYSIS\n"
        content += "-" * 20 + "\n"
        if result.bpm:
            content += f"BPM: {result.bpm:.2f}\n"
            content += f"Tempo Category: {result.tempo_category}\n"
        else:
            content += "BPM: Unable to detect\n"
        content += "\n"
        
        content += "KEY ANALYSIS\n"
        content += "-" * 20 + "\n"
        if result.key:
            content += f"Key: {result.key}\n"
            content += f"Scale: {result.scale}\n"
            content += f"Confidence: {result.key_confidence:.3f}\n"
        else:
            content += "Key: Unable to detect\n"
        content += "\n"
        
        content += "LOUDNESS ANALYSIS\n"
        content += "-" * 20 + "\n"
        if result.lufs_integrated:
            content += f"LUFS Integrated: {result.lufs_integrated:.2f}\n"
            if result.lufs_short_term:
                content += f"LUFS Short-term: {result.lufs_short_term:.2f}\n"
        if result.peak_db:
            content += f"Peak Level: {result.peak_db:.2f} dB\n"
        if not result.lufs_integrated and not result.peak_db:
            content += "Loudness: Unable to analyze\n"
        content += "\n"
        
        content += "GENRE AND MOOD ANALYSIS\n"
        content += "-" * 30 + "\n"
        if result.genre_prediction:
            content += f"Predicted Genre: {result.genre_prediction}\n"
        if result.mood_prediction:
            content += f"Predicted Mood: {result.mood_prediction}\n"
        if result.energy_level is not None:
            content += f"Energy Level: {result.energy_level:.3f}\n"
        if result.danceability is not None:
            content += f"Danceability: {result.danceability:.3f}\n"
        if result.valence is not None:
            content += f"Valence (Positivity): {result.valence:.3f}\n"
        content += "\n"
        
        content += "RHYTHM ANALYSIS\n"
        content += "-" * 20 + "\n"
        if result.rhythmic_complexity is not None:
            content += f"Rhythmic Complexity: {result.rhythmic_complexity:.3f}\n"
        if result.beat_strength is not None:
            content += f"Beat Strength: {result.beat_strength:.3f}\n"
        if result.onset_density is not None:
            content += f"Onset Density: {result.onset_density:.2f} onsets/sec\n"
        content += "\n"
        content += "SPECTRAL ANALYSIS\n"
        content += "-" * 20 + "\n"
        if result.spectral_centroid:
            content += f"Spectral Centroid: {result.spectral_centroid:.1f} Hz\n"
        if result.spectral_rolloff:
            content += f"Spectral Rolloff: {result.spectral_rolloff:.1f} Hz\n"
        if result.zero_crossing_rate:
            content += f"Zero Crossing Rate: {result.zero_crossing_rate:.4f}\n"
        content += "\n"
        
        content += "FINGERPRINT\n"
        content += "-" * 20 + "\n"
        if result.fingerprint:
            content += f"Fingerprint: {result.fingerprint[:32]}...\n"
            if result.fingerprint_duration:
                content += f"Fingerprint Duration: {result.fingerprint_duration:.1f}s\n"
        else:
            content += "Fingerprint: Not generated\n"
        content += "\n"
        
        content += "SUGGESTED CATEGORIZATION\n"
        content += "-" * 30 + "\n"
        suggested_category = self.audio_analyzer.categorize_by_analysis(result)
        content += f"Suggested Category: {suggested_category}\n\n"
        
        if result.analysis_errors:
            content += "ANALYSIS ERRORS\n"
            content += "-" * 20 + "\n"
            for error in result.analysis_errors:
                content += f"• {error}\n"
        
        details_text.insert(1.0, content)
        details_text.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(details_window, text="Close", command=details_window.destroy).pack(pady=10)
    
    def create_smart_playlists(self):
        """Create smart playlists based on analysis results"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis data available. Please analyze files first.")
            return
        
        results_list = list(self.analysis_results.values())
        playlists = self.audio_analyzer.create_smart_playlists(results_list)
        
        if not playlists:
            messagebox.showinfo("No Playlists", "No smart playlists could be created with current analysis data.")
            return
        
        # Show playlists in a new window
        playlist_window = tk.Toplevel(self.root)
        playlist_window.title("Smart Playlists")
        playlist_window.geometry("700x500")
        
        ttk.Label(playlist_window, text="Smart Playlists", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # Create notebook for playlist tabs
        playlist_notebook = ttk.Notebook(playlist_window)
        playlist_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for playlist_name, tracks in playlists.items():
            # Create tab for each playlist
            tab_frame = ttk.Frame(playlist_notebook)
            playlist_notebook.add(tab_frame, text=f"{playlist_name} ({len(tracks)})")
            
            # Track list
            track_frame = ttk.Frame(tab_frame)
            track_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            track_listbox = tk.Listbox(track_frame, height=15)
            track_scrollbar = ttk.Scrollbar(track_frame, orient=tk.VERTICAL, command=track_listbox.yview)
            track_listbox.configure(yscrollcommand=track_scrollbar.set)
            
            for track in tracks:
                track_listbox.insert(tk.END, track)
            
            track_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            track_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Export button for this playlist
            export_frame = ttk.Frame(tab_frame)
            export_frame.pack(fill=tk.X, padx=5, pady=5)
            
            def make_export_func(name, track_list):
                return lambda: self.export_playlist(name, track_list)
            
            ttk.Button(export_frame, text=f"Export {playlist_name}", 
                      command=make_export_func(playlist_name, tracks)).pack(side=tk.LEFT)
        
        # Close button
        ttk.Button(playlist_window, text="Close", command=playlist_window.destroy).pack(pady=10)
    
    def export_playlist(self, playlist_name: str, tracks: List[str]):
        """Export playlist to M3U file"""
        filename = filedialog.asksaveasfilename(
            title=f"Export {playlist_name} Playlist",
            defaultextension=".m3u",
            filetypes=[("M3U Playlist", "*.m3u"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialvalue=f"{playlist_name.replace(' ', '_')}.m3u"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"#EXTM3U\n")
                    f.write(f"# Smart Playlist: {playlist_name}\n")
                    f.write(f"# Generated by Audio Converter\n\n")
                    
                    for track in tracks:
                        # Find the full path for this track
                        track_path = None
                        for file_path, result in self.analysis_results.items():
                            if os.path.basename(file_path) == track:
                                track_path = file_path
                                break
                        
                        if track_path:
                            f.write(f"#EXTINF:-1,{track}\n")
                            f.write(f"{track_path}\n")
                        else:
                            f.write(f"# {track} (path not found)\n")
                
                messagebox.showinfo("Success", f"Playlist exported to {filename}")
                self.log(f"Exported playlist '{playlist_name}' with {len(tracks)} tracks")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export playlist: {str(e)}")
    
    def find_similar_tracks(self):
        """Find tracks similar to selected track"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis data available. Please analyze files first.")
            return
        
        # Get selected track from analysis tree
        selection = self.analysis_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a track from the analysis results to find similar tracks.")
            return
        
        item = self.analysis_tree.item(selection[0])
        selected_filename = item['values'][0]
        
        # Find the analysis result for selected track
        target_result = None
        for file_path, result in self.analysis_results.items():
            if os.path.basename(file_path) == selected_filename:
                target_result = result
                break
        
        if not target_result:
            messagebox.showerror("Error", "Could not find analysis data for selected track.")
            return
        
        # Find similar tracks
        results_list = list(self.analysis_results.values())
        similar_tracks = self.audio_analyzer.find_similar_tracks(results_list, target_result, similarity_threshold=0.6)
        
        if not similar_tracks:
            messagebox.showinfo("No Similar Tracks", f"No similar tracks found for '{selected_filename}'.")
            return
        
        # Show similar tracks in a new window
        similar_window = tk.Toplevel(self.root)
        similar_window.title(f"Tracks Similar to '{selected_filename}'")
        similar_window.geometry("600x400")
        
        ttk.Label(similar_window, text=f"Tracks Similar to '{selected_filename}'", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Create treeview for similar tracks
        similar_frame = ttk.Frame(similar_window)
        similar_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ("Track", "Similarity")
        similar_tree = ttk.Treeview(similar_frame, columns=columns, show="headings", height=15)
        
        similar_tree.heading("Track", text="Track Name")
        similar_tree.heading("Similarity", text="Similarity %")
        similar_tree.column("Track", width=400)
        similar_tree.column("Similarity", width=100)
        
        # Add similar tracks
        for track_name, similarity in similar_tracks:
            similarity_percent = f"{similarity * 100:.1f}%"
            similar_tree.insert("", "end", values=(track_name, similarity_percent))
        
        # Scrollbar
        similar_scrollbar = ttk.Scrollbar(similar_frame, orient=tk.VERTICAL, command=similar_tree.yview)
        similar_tree.configure(yscrollcommand=similar_scrollbar.set)
        
        similar_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        similar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export similar tracks as playlist
        export_frame = ttk.Frame(similar_window)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def export_similar():
            track_names = [track for track, _ in similar_tracks]
            self.export_playlist(f"Similar_to_{selected_filename}", track_names)
        
        ttk.Button(export_frame, text="Export as Playlist", command=export_similar).pack(side=tk.LEFT)
        
        # Close button
        ttk.Button(similar_window, text="Close", command=similar_window.destroy).pack(pady=10)
    
    def process_audio(self, input_file: str, output_file: str) -> bool:
        """Process a single audio file"""
        try:
            # Validate output format before processing
            selected_fmt = self.selected_format.get().lower()
            is_supported, reason = self.validate_format_compatibility(selected_fmt)
            
            if not is_supported:
                self.log(f"Unsupported format '{selected_fmt}': {reason}")
                return False
            
            # Log format info for user awareness
            if "FFmpeg" in reason:
                self.log(f"Using {selected_fmt} format: {reason}")
            
            # Load audio
            y, sr = librosa.load(input_file, sr=self.sample_rate.get())
            
            # Calculate total pitch shift needed
            total_pitch_shift = 0
            
            # Add manual pitch shift
            if self.pitch_shift.get() != 0:
                total_pitch_shift += self.pitch_shift.get()
            
            # Add manual key shift (for legacy compatibility)
            if self.key_shift.get() != 0:
                total_pitch_shift += self.key_shift.get()
            
            # Add selected key transposition
            selected_key = self.selected_key.get()
            if selected_key != "Original":
                # Try to detect the original key if analysis is available
                detected_key = None
                if input_file in self.analysis_results:
                    detected_key = self.analysis_results[input_file].key
                
                key_transposition = self.calculate_key_transposition(selected_key, detected_key)
                total_pitch_shift += key_transposition
                
                if key_transposition != 0:
                    self.log(f"Transposing to {selected_key}: {key_transposition:+d} semitones")
            
            # Apply all pitch shifting at once for better quality
            if total_pitch_shift != 0:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=total_pitch_shift)
            
            # Normalize if requested
            if self.normalize_audio.get():
                y = librosa.util.normalize(y)
            
            # Remove silence if requested
            if self.remove_silence.get():
                y, _ = librosa.effects.trim(y, top_db=20)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            audio_for_export = self._prepare_audio_for_export(y)
            
            # Categorize formats for optimal processing
            uncompressed_formats = ['wav', 'aiff', 'aifc', 'au', 'caf', 'w64', 'rf64', 'bwf', 'sd2', 'snd', 'iff', 'svx', 'nist', 'voc', 'ircam', 'xi', 'caff']
            lossless_formats = ['flac', 'tta', 'tak', 'als', 'ape', 'wv', 'shn']
            lossy_formats = ['mp3', 'aac', 'm4a', 'ogg', 'opus', 'wma', 'ac3', 'mp2', '3gp', 'webm', 'ra', 'rm', 'amr', 'amr-nb', 'amr-wb', 'gsm']
            raw_formats = ['raw', 'pcm', 's8', 's16le', 's24le', 's32le', 'f32le', 'f64le']
            apple_formats = ['m4r', 'm4b', 'm4p']
            video_formats = ['mka', 'm4v', 'mov', 'avi', 'mkv', 'mp4']
            
            selected_fmt = self.selected_format.get().lower()
            
            # Save audio based on format category
            try:
                if selected_fmt in uncompressed_formats or selected_fmt in lossless_formats:
                    # Use soundfile for high-quality formats
                    if selected_fmt in ['wav', 'flac', 'aiff', 'aifc']:
                        sf.write(output_file, audio_for_export, sr, subtype=f'PCM_{self.bit_depth.get()}')
                    else:
                        # For other lossless/uncompressed, convert via pydub but maintain quality
                        audio_segment = self._create_audio_segment(audio_for_export, sr)
                        # Set high quality parameters for lossless formats
                        if selected_fmt == 'flac':
                            audio_segment.export(output_file, format=selected_fmt, parameters=["-compression_level", "8"])
                        else:
                            audio_segment.export(output_file, format=selected_fmt)

                elif selected_fmt in lossy_formats or selected_fmt in apple_formats:
                    # Use pydub for compressed formats with quality optimization
                    audio_segment = self._create_audio_segment(audio_for_export, sr)

                    # Format-specific quality settings
                    if selected_fmt == 'mp3':
                        audio_segment.export(output_file, format=selected_fmt, bitrate="320k")
                    elif selected_fmt in ['aac', 'm4a', 'm4r', 'm4b', 'm4p']:
                        # M4A and Apple formats need to use mp4 container format
                        audio_segment.export(output_file, format='mp4', codec='aac', bitrate="256k")
                    elif selected_fmt == 'ogg':
                        audio_segment.export(output_file, format=selected_fmt, codec='libvorbis')
                    elif selected_fmt == 'opus':
                        audio_segment.export(output_file, format=selected_fmt, codec='libopus', bitrate="128k")
                    elif selected_fmt == 'wma':
                        audio_segment.export(output_file, format=selected_fmt, codec='wmav2')
                    else:
                        audio_segment.export(output_file, format=selected_fmt)
                        
                elif selected_fmt in video_formats:
                    # Extract audio from video containers or create video with audio
                    audio_segment = self._create_audio_segment(audio_for_export, sr)
                    if selected_fmt in ['mp4', 'm4v']:
                        audio_segment.export(output_file, format='mp4', codec='aac')
                    elif selected_fmt == 'webm':
                        audio_segment.export(output_file, format=selected_fmt, codec='libopus')
                    elif selected_fmt in ['mkv', 'mka']:
                        audio_segment.export(output_file, format='matroska', codec='flac')
                    else:
                        audio_segment.export(output_file, format=selected_fmt)
                        
                elif selected_fmt in raw_formats:
                    # Handle raw audio formats
                    if selected_fmt == 'raw' or selected_fmt == 'pcm':
                        # Save as raw PCM data
                        pcm16 = np.round(audio_for_export * 32767).astype('<i2')
                        if pcm16.ndim == 2:
                            pcm16 = pcm16.reshape(-1, pcm16.shape[1])
                            pcm16 = pcm16.flatten()
                        with open(output_file, 'wb') as f:
                            f.write(pcm16.tobytes())
                    else:
                        # Use pydub for other raw formats
                        audio_segment = self._create_audio_segment(audio_for_export, sr)
                        audio_segment.export(output_file, format=selected_fmt)

                else:
                    # Fallback: try pydub for any other format
                    audio_segment = self._create_audio_segment(audio_for_export, sr)
                    audio_segment.export(output_file, format=selected_fmt)

            except Exception as format_error:
                # If format-specific export fails, try generic export
                self.log(f"Format-specific export failed, trying generic export: {format_error}")
                audio_segment = self._create_audio_segment(audio_for_export, sr)
                audio_segment.export(output_file, format=selected_fmt)

            self.apply_metadata_to_output(input_file, output_file, audio_for_export, sr)

            return True

        except Exception as e:
            self.log(f"Error processing {input_file}: {str(e)}")
            return False
    
    def start_conversion(self):
        """Start the conversion process"""
        if not self.output_folder:
            messagebox.showerror("Error", "Please select an output folder")
            return
            
        audio_files = self.get_audio_files()
        if not audio_files:
            messagebox.showerror("Error", "No audio files found")
            return
            
        self.is_converting = True
        self.convert_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start conversion in separate thread
        self.conversion_thread = threading.Thread(
            target=self.conversion_worker, 
            args=(audio_files,),
            daemon=True
        )
        self.conversion_thread.start()
        
    def stop_conversion(self):
        """Stop the conversion process"""
        self.is_converting = False
        self.convert_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log("Conversion stopped by user")
        
    def conversion_worker(self, audio_files: List[str]):
        """Worker thread for conversion"""
        total_files = len(audio_files)
        successful = 0
        failed = 0
        
        self.log(f"Starting conversion of {total_files} files...")
        
        for i, input_file in enumerate(audio_files):
            if not self.is_converting:
                break
                
            try:
                # Determine output path
                filename = os.path.basename(input_file)
                
                # Generate enhanced filename with key, mood, and energy
                enhanced_name = self.generate_enhanced_filename(filename, input_file)
                
                # Categorize file (now using enhanced method)
                category = self.categorize_file(filename, input_file)
                
                # Build output path
                if self.auto_categorize.get():
                    output_dir = os.path.join(self.output_folder, category)
                else:
                    output_dir = self.output_folder
                    
                if self.preserve_structure.get() and self.input_folders:
                    # Try to preserve relative structure
                    for folder in self.input_folders:
                        if input_file.startswith(folder):
                            rel_path = os.path.relpath(os.path.dirname(input_file), folder)
                            if rel_path != '.':
                                output_dir = os.path.join(output_dir, rel_path)
                            break
                
                output_file = os.path.join(
                    output_dir, 
                    f"{enhanced_name}.{self.selected_format.get()}"
                )
                
                # Process file
                self.log(f"Processing: {filename}")
                if self.process_audio(input_file, output_file):
                    successful += 1
                    self.log(f"✓ Converted: {filename} -> {category}")
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                self.log(f"✗ Failed: {filename} - {str(e)}")
            
            # Update progress
            progress = ((i + 1) / total_files) * 100
            self.progress_var.set(progress)
            
        # Conversion complete
        self.is_converting = False
        self.convert_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var.set(100)
        
        self.log(f"\nConversion complete!")
        self.log(f"Successful: {successful}")
        self.log(f"Failed: {failed}")
        self.log(f"Total: {total_files}")
        
        messagebox.showinfo("Conversion Complete", 
                           f"Conversion finished!\nSuccessful: {successful}\nFailed: {failed}")
    
    def log(self, message: str):
        """Add message to log"""
        def update_log():
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            
        self.root.after(0, update_log)
        
    def clear_log(self):
        """Clear the log"""
        self.log_text.delete(1.0, tk.END)
        
    def load_categories_display(self):
        """Load categories into the text widget"""
        categories_text = "# Audio Categories and Keywords\n"
        categories_text += "# Format: category_name: keyword1, keyword2, keyword3\n\n"
        
        for category, keywords in self.category_keywords.items():
            categories_text += f"{category}: {', '.join(keywords)}\n"
            
        self.categories_text.delete(1.0, tk.END)
        self.categories_text.insert(1.0, categories_text)
        
    def save_categories(self):
        """Save categories from text widget"""
        try:
            text = self.categories_text.get(1.0, tk.END)
            new_categories = {}
            
            for line in text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    if ':' in line:
                        category, keywords = line.split(':', 1)
                        category = category.strip()
                        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                        new_categories[category] = keywords
                        
            self.category_keywords = new_categories
            messagebox.showinfo("Success", "Categories saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save categories: {str(e)}")
            
    def reset_categories(self):
        """Reset categories to default"""
        self.category_keywords = {
            "drums": ["kick", "snare", "hihat", "cymbal", "tom", "drum", "perc", "percussion"],
            "bass": ["bass", "sub", "808", "low"],
            "melody": ["melody", "lead", "main", "hook", "theme"],
            "vocals": ["vocal", "voice", "singing", "rap", "spoken"],
            "fx": ["fx", "effect", "sweep", "riser", "impact", "crash"],
            "loops": ["loop", "pattern", "sequence"],
            "instruments": ["piano", "guitar", "synth", "violin", "flute", "sax"]
        }
        self.load_categories_display()
        
    def save_settings(self):
        """Save settings to file"""
        settings = {
            "sample_rate": self.sample_rate.get(),
            "bit_depth": self.bit_depth.get(),
            "normalize_audio": self.normalize_audio.get(),
            "remove_silence": self.remove_silence.get(),
            "auto_categorize": self.auto_categorize.get(),
            "preserve_structure": self.preserve_structure.get(),
            "selected_format": self.selected_format.get(),
            "selected_key": self.selected_key.get(),
            "category_keywords": self.category_keywords,
            # Analysis settings
            "enable_analysis": self.enable_analysis.get(),
            "analyze_bpm": self.analyze_bpm.get(),
            "analyze_key": self.analyze_key.get(),
            "analyze_loudness": self.analyze_loudness.get(),
            "analyze_fingerprint": self.analyze_fingerprint.get(),
            "auto_categorize_by_analysis": self.auto_categorize_by_analysis.get()
        }
        
        try:
            with open("audio_converter_settings.json", "w") as f:
                json.dump(settings, f, indent=2)
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            
    def load_settings(self):
        """Load settings from file"""
        try:
            with open("audio_converter_settings.json", "r") as f:
                settings = json.load(f)
                
            self.sample_rate.set(settings.get("sample_rate", 44100))
            self.bit_depth.set(settings.get("bit_depth", 16))
            self.normalize_audio.set(settings.get("normalize_audio", True))
            self.remove_silence.set(settings.get("remove_silence", False))
            self.auto_categorize.set(settings.get("auto_categorize", True))
            self.preserve_structure.set(settings.get("preserve_structure", False))
            self.selected_format.set(settings.get("selected_format", "wav"))
            self.selected_key.set(settings.get("selected_key", "Original"))
            
            # Load analysis settings
            self.enable_analysis.set(settings.get("enable_analysis", True))
            self.analyze_bpm.set(settings.get("analyze_bpm", True))
            self.analyze_key.set(settings.get("analyze_key", True))
            self.analyze_loudness.set(settings.get("analyze_loudness", True))
            self.analyze_fingerprint.set(settings.get("analyze_fingerprint", True))
            self.auto_categorize_by_analysis.set(settings.get("auto_categorize_by_analysis", False))
            
            if "category_keywords" in settings:
                self.category_keywords = settings["category_keywords"]
                self.load_categories_display()
                
            messagebox.showinfo("Success", "Settings loaded successfully!")
            
        except FileNotFoundError:
            messagebox.showwarning("Warning", "No settings file found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {str(e)}")

def main():
    """Main function"""
    # Check for required dependencies
    try:
        import librosa
        import soundfile
        import pydub
        from audio_analyzer import IntelligentAudioAnalyzer
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Check for FFmpeg (required by pydub)
    if not which("ffmpeg"):
        print("Warning: FFmpeg not found. Some audio formats may not work.")
        print("Please install FFmpeg for full format support.")
    
    # Initialize and check analysis capabilities
    analyzer = IntelligentAudioAnalyzer()
    print("Audio Analysis Capabilities:")
    print(f"  - LUFS Loudness Metering: {'Available' if analyzer.__class__.__module__ != '__main__' else 'Check pyloudnorm'}")
    print(f"  - Advanced Analysis: {'Available' if analyzer.__class__.__module__ != '__main__' else 'Check essentia'}")
    print(f"  - Audio Fingerprinting: {'Available' if analyzer.__class__.__module__ != '__main__' else 'Check chromaprint'}")
    
    root = tk.Tk()
    app = AudioConverter(root)
    root.mainloop()

if __name__ == "__main__":
    main()