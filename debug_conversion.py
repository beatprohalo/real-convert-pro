#!/usr/bin/env python3
"""
Debug script for Real Convert conversion issues
"""

import sys
import os
sys.path.insert(0, '/Users/yvhammg/Desktop/YUSEF APPS')

from audio_converter import AudioConverter
import tkinter as tk
import traceback

def debug_conversion():
    print('🔍 DEBUGGING REAL CONVERT CONVERSION')
    print('='*50)
    
    # Initialize app
    root = tk.Tk()
    root.withdraw()
    app = AudioConverter(root)
    
    # Test file
    test_input = '/Users/yvhammg/Desktop/YUSEF APPS/sample_audio/piano_melody.wav'
    test_output = '/Users/yvhammg/Desktop/YUSEF APPS/debug_output.m4a'
    
    print(f'📂 Input: {test_input}')
    print(f'📂 Output: {test_output}')
    print(f'📏 Input size: {os.path.getsize(test_input)} bytes')
    
    # Set format to m4a
    app.selected_format.set('m4a')
    
    # Override the log method to capture output
    log_messages = []
    original_log = app.log
    def debug_log(message):
        log_messages.append(message)
        print(f'📝 LOG: {message}')
    app.log = debug_log
    
    # Test the conversion
    try:
        print('🚀 Starting conversion...')
        success = app.process_audio(test_input, test_output)
        
        print(f'🎯 Conversion result: {success}')
        
        if os.path.exists(test_output):
            size = os.path.getsize(test_output)
            print(f'📏 Output size: {size} bytes')
            if size > 0:
                print('✅ SUCCESS!')
            else:
                print('❌ ZERO BYTES PROBLEM!')
        else:
            print('❌ FILE NOT CREATED!')
            
        print('\n📝 Log messages:')
        for msg in log_messages:
            print(f'   {msg}')
            
    except Exception as e:
        print(f'❌ Exception: {e}')
        traceback.print_exc()
    
    # Clean up
    if os.path.exists(test_output):
        os.remove(test_output)

if __name__ == '__main__':
    debug_conversion()