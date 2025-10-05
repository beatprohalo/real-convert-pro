#!/usr/bin/env python3
"""
Audio Apps Launcher
Choose which audio application to open
"""

import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os

class AudioAppsLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Audio Apps Launcher")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        # Main title
        title_label = tk.Label(root, text="🎵 Choose Your Audio Application", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Apps frame
        apps_frame = ttk.Frame(root)
        apps_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # App 1: Audio Converter
        self.create_app_button(
            apps_frame,
            "🔄 Audio Converter",
            "Basic audio format conversion with 62 format support",
            "• Convert between 62 audio formats\n• Batch processing\n• Pitch & key shifting\n• Intelligent analysis",
            "audio_converter.py",
            0
        )
        
        # App 2: Professional Tools (WITH STEM SPLITTER)
        self.create_app_button(
            apps_frame,
            "🎛️ Professional Audio Tools",
            "Advanced processing including STEM SEPARATION",
            "• 🎸 STEM SPLITTER (separate vocals, drums, bass, other)\n• Time-stretching & pitch shifting\n• Audio repair & noise reduction\n• Spectral editing & mastering",
            "professional_tools_gui.py",
            1
        )
        
        # App 3: Advanced Audio App
        self.create_app_button(
            apps_frame,
            "⚡ Advanced Audio Processor",
            "Experimental advanced audio processing",
            "• Advanced processing workflows\n• Machine learning features\n• Experimental tools\n• Research-grade processing",
            "professional_audio_app.py",
            2
        )
        
        # Info label
        info_label = tk.Label(root, 
                            text="💡 The STEM SPLITTER is in the Professional Audio Tools app",
                            font=("Arial", 12, "italic"),
                            fg="blue")
        info_label.pack(pady=(10, 20))
        
    def create_app_button(self, parent, title, description, features, filename, row):
        """Create a button for each app"""
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=row, column=0, sticky="ew", pady=5)
        parent.grid_columnconfigure(0, weight=1)
        
        # Description
        desc_label = tk.Label(frame, text=description, font=("Arial", 10, "bold"))
        desc_label.pack(anchor="w")
        
        # Features
        features_label = tk.Label(frame, text=features, 
                                font=("Arial", 9), justify="left")
        features_label.pack(anchor="w", pady=5)
        
        # Launch button
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=5)
        
        launch_btn = ttk.Button(button_frame, text=f"🚀 Launch {title}", 
                               command=lambda: self.launch_app(filename))
        launch_btn.pack(side="left")
        
        # Test button
        test_btn = ttk.Button(button_frame, text="🔍 Test Import", 
                             command=lambda: self.test_app(filename))
        test_btn.pack(side="right")
    
    def launch_app(self, filename):
        """Launch the selected app"""
        try:
            subprocess.Popen([sys.executable, filename], 
                           cwd=os.path.dirname(os.path.abspath(__file__)))
            self.root.destroy()  # Close launcher
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to launch {filename}:\\n{e}")
    
    def test_app(self, filename):
        """Test if the app can be imported"""
        try:
            module_name = filename.replace('.py', '')
            exec(f"import {module_name}")
            tk.messagebox.showinfo("✅ Test Successful", f"{filename} imports successfully!")
        except Exception as e:
            tk.messagebox.showerror("❌ Test Failed", f"{filename} has errors:\\n{e}")

def main():
    root = tk.Tk()
    app = AudioAppsLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()