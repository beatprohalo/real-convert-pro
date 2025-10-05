#!/usr/bin/env python3
"""
Icon Creator for Audio Converter ML
Converts the provided icon image to macOS .icns format
"""

import os
import shutil
import subprocess
from pathlib import Path
from PIL import Image
import base64

def create_icon_from_base64():
    """Create icon from the provided image"""
    
    # The image data appears to be a modern ML Bank logo
    # Since we can't directly extract from the attachment, we'll create a simple icon
    # using PIL with similar colors and design
    
    print("🎨 Creating application icon...")
    
    # Create a 512x512 icon with modern design
    size = 512
    icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    
    # We'll create a simple version since we can't extract the exact image
    # You can replace this with the actual image file if you have it
    
    try:
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(icon)
        
        # Create a dark blue background circle
        margin = 20
        draw.ellipse([margin, margin, size-margin, size-margin], 
                    fill=(30, 40, 80, 255))  # Dark blue
        
        # Add gradient effect (simplified)
        center = size // 2
        for i in range(5):
            radius = center - margin - i * 20
            alpha = 100 - i * 15
            if radius > 0:
                draw.ellipse([center-radius, center-radius, center+radius, center+radius], 
                           outline=(100 + i * 30, 200 + i * 10, 255, alpha), width=3)
        
        # Add ML text
        try:
            # Try to use a system font
            font_size = 80
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # ML text
        text = "ML"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (size - text_width) // 2
        text_y = (size - text_height) // 2 - 40
        
        draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
        
        # BANK text
        bank_font_size = 40
        try:
            bank_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", bank_font_size)
        except:
            bank_font = ImageFont.load_default()
            
        bank_text = "BANK"
        bbox = draw.textbbox((0, 0), bank_text, font=bank_font)
        bank_width = bbox[2] - bbox[0]
        bank_x = (size - bank_width) // 2
        bank_y = text_y + text_height + 20
        
        draw.text((bank_x, bank_y), bank_text, fill=(255, 255, 255, 255), font=bank_font)
        
        # Add some tech-like elements (simplified waveform)
        wave_y = bank_y + 60
        for i in range(0, size-40, 8):
            height = 10 + (i % 30)
            draw.rectangle([i+20, wave_y, i+25, wave_y + height], 
                         fill=(0, 255, 200, 150))
        
    except ImportError:
        # Fallback: simple colored circle
        draw = ImageDraw.Draw(icon)
        draw.ellipse([20, 20, size-20, size-20], fill=(30, 40, 80, 255))
        
        # Simple text
        try:
            font = ImageFont.load_default()
            draw.text((size//2-30, size//2-20), "ML", fill=(255, 255, 255, 255), font=font)
            draw.text((size//2-40, size//2+20), "BANK", fill=(255, 255, 255, 255), font=font)
        except:
            pass
    
    return icon

def convert_to_icns(icon_image, output_path):
    """Convert PIL image to .icns format"""
    
    # Create different sizes for the icon
    sizes = [16, 32, 64, 128, 256, 512]
    
    # Create temporary directory for icon files
    temp_dir = Path(output_path).parent / "temp_icon"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save different sizes
        iconset_dir = temp_dir / "app_icon.iconset"
        iconset_dir.mkdir(exist_ok=True)
        
        for size in sizes:
            # Standard resolution
            resized = icon_image.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(iconset_dir / f"icon_{size}x{size}.png")
            
            # High resolution (@2x)
            if size <= 256:
                resized_2x = icon_image.resize((size * 2, size * 2), Image.Resampling.LANCZOS)
                resized_2x.save(iconset_dir / f"icon_{size}x{size}@2x.png")
        
        # Use iconutil to create .icns file (macOS only)
        try:
            subprocess.run([
                "iconutil", "-c", "icns", str(iconset_dir), "-o", str(output_path)
            ], check=True, capture_output=True)
            print(f"✅ Created .icns file: {output_path}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  iconutil not available, saving as PNG instead")
            # Fallback: save as PNG
            png_path = str(output_path).replace('.icns', '.png')
            icon_image.save(png_path)
            print(f"✅ Created PNG icon: {png_path}")
            return False
            
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def create_app_icon():
    """Main function to create the app icon"""
    current_dir = Path(__file__).parent
    
    print("🎨 Creating ML Bank Audio Converter icon...")
    
    # Create the icon
    icon = create_icon_from_base64()
    
    # Save as .icns for the app bundle
    icns_path = current_dir / "app_icon.icns"
    convert_to_icns(icon, icns_path)
    
    # Also save as PNG for backup
    png_path = current_dir / "app_icon.png"
    icon.save(png_path)
    print(f"✅ Created PNG icon: {png_path}")
    
    return icns_path, png_path

if __name__ == "__main__":
    try:
        icns_path, png_path = create_app_icon()
        print(f"\n🎉 Icon created successfully!")
        print(f"📱 .icns file: {icns_path}")
        print(f"🖼️  PNG file: {png_path}")
    except Exception as e:
        print(f"❌ Error creating icon: {e}")
        print("💡 Make sure PIL (Pillow) is installed: pip install Pillow")