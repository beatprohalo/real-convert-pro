#!/usr/bin/env python3
"""
Updated Icon Creator for Real Convert
Creates a modern icon showing full format support and functionality
"""

import os
import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

def create_real_convert_icon():
    """Create an updated Real Convert icon showing full functionality"""
    
    print("🎨 Creating Real Convert icon (fully functional version)...")
    
    # Create a 512x512 icon
    size = 512
    icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    
    # Modern gradient background
    center = size // 2
    
    # Create a circular gradient background
    for i in range(center):
        # Create gradient from dark blue to bright blue
        intensity = 1 - (i / center)
        r = int(20 + 80 * intensity)   # Dark blue to bright blue
        g = int(30 + 150 * intensity)  # 
        b = int(80 + 175 * intensity)  # 
        alpha = 255
        
        # Draw concentric circles for gradient effect
        radius = center - i
        if radius > 0:
            draw.ellipse([center-radius, center-radius, center+radius, center+radius], 
                        fill=(r, g, b, alpha))
    
    # Add outer ring to show "complete" status
    ring_width = 8
    draw.ellipse([ring_width, ring_width, size-ring_width, size-ring_width], 
                outline=(0, 255, 150, 255), width=ring_width)  # Green ring for "fully functional"
    
    # Add format indicator icons around the circle
    formats = ['MP3', 'WAV', 'FLAC', 'M4A', 'AAC', 'OGG']
    format_radius = center - 40
    
    try:
        # Try to use a good system font
        format_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        main_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        # Fallback fonts
        format_font = ImageFont.load_default()
        main_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
    
    # Place format labels around the circle
    for i, fmt in enumerate(formats):
        angle = (i * 60) * math.pi / 180  # 60 degrees apart
        x = center + format_radius * math.cos(angle - math.pi/2)  # Start from top
        y = center + format_radius * math.sin(angle - math.pi/2)
        
        # Create a small background for format text
        bbox = draw.textbbox((0, 0), fmt, font=format_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Small rounded background
        bg_padding = 4
        bg_x1 = x - text_width//2 - bg_padding
        bg_y1 = y - text_height//2 - bg_padding
        bg_x2 = x + text_width//2 + bg_padding
        bg_y2 = y + text_height//2 + bg_padding
        
        draw.rounded_rectangle([bg_x1, bg_y1, bg_x2, bg_y2], 
                              radius=6, fill=(255, 255, 255, 200))
        
        # Format text
        draw.text((x - text_width//2, y - text_height//2), fmt, 
                 fill=(30, 30, 30, 255), font=format_font)
    
    # Main "RC" text in center
    main_text = "RC"
    bbox = draw.textbbox((0, 0), main_text, font=main_font)
    main_width = bbox[2] - bbox[0]
    main_height = bbox[3] - bbox[1]
    main_x = (size - main_width) // 2
    main_y = (size - main_height) // 2 - 20
    
    # Text shadow
    shadow_offset = 3
    draw.text((main_x + shadow_offset, main_y + shadow_offset), main_text, 
             fill=(0, 0, 0, 100), font=main_font)
    
    # Main text
    draw.text((main_x, main_y), main_text, fill=(255, 255, 255, 255), font=main_font)
    
    # Subtitle
    subtitle = "CONVERT"
    bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    sub_width = bbox[2] - bbox[0]
    sub_x = (size - sub_width) // 2
    sub_y = main_y + main_height + 5
    
    draw.text((sub_x, sub_y), subtitle, fill=(255, 255, 255, 220), font=subtitle_font)
    
    # Add checkmark to show "fully functional"
    check_size = 30
    check_x = center + 80
    check_y = center + 80
    
    # Green circle background for checkmark
    check_radius = 25
    draw.ellipse([check_x - check_radius, check_y - check_radius, 
                 check_x + check_radius, check_y + check_radius], 
                fill=(0, 200, 100, 255))
    
    # White checkmark
    check_points = [
        (check_x - 12, check_y),
        (check_x - 4, check_y + 8),
        (check_x + 12, check_y - 8)
    ]
    draw.line(check_points[:2], fill=(255, 255, 255, 255), width=4)
    draw.line(check_points[1:], fill=(255, 255, 255, 255), width=4)
    
    # Add subtle waveform at bottom
    wave_y_start = size - 60
    wave_height = 20
    for i in range(0, size, 12):
        height = wave_height * (0.3 + 0.7 * abs(math.sin(i * 0.1)))
        draw.rectangle([i, wave_y_start + wave_height - height, i + 6, wave_y_start + wave_height], 
                      fill=(255, 255, 255, 80))
    
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
        iconset_dir = temp_dir / "real_convert_icon.iconset"
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

def update_app_icon():
    """Update the Real Convert app icon"""
    current_dir = Path(__file__).parent
    
    print("🎨 Creating updated Real Convert icon (fully functional)...")
    
    # Create the new icon
    icon = create_real_convert_icon()
    
    # Save new icons
    icns_path = current_dir / "real_convert_icon_updated.icns"
    png_path = current_dir / "real_convert_icon_updated.png"
    
    # Create .icns file
    convert_to_icns(icon, icns_path)
    
    # Save PNG version
    icon.save(png_path)
    print(f"✅ Created PNG icon: {png_path}")
    
    # Update the app bundle icon
    app_icon_path = current_dir / "Real Convert.app" / "Contents" / "Resources" / "real_convert_icon.icns"
    if icns_path.exists() and app_icon_path.parent.exists():
        shutil.copy2(icns_path, app_icon_path)
        print(f"✅ Updated app bundle icon: {app_icon_path}")
    
    return icns_path, png_path

if __name__ == "__main__":
    try:
        icns_path, png_path = update_app_icon()
        print(f"\n🎉 Real Convert icon updated successfully!")
        print(f"📱 .icns file: {icns_path}")
        print(f"🖼️  PNG file: {png_path}")
        print(f"✅ App bundle icon updated - shows full format support!")
    except Exception as e:
        print(f"❌ Error creating icon: {e}")
        print("💡 Make sure PIL (Pillow) is installed in the virtual environment")