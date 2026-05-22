from PIL import Image, ImageEnhance

def make_large_white_menubar_icon(input_path, output_path):
    """Create a larger white menu bar icon using more of the available space"""
    img = Image.open(input_path)
    img = img.convert("RGBA")
    
    # Convert to white
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[3] > 0:
            newData.append((255, 255, 255, 255))
        else:
            newData.append((255, 255, 255, 0))
    
    img.putdata(newData)
    
    # Remove empty space
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # Make it bigger - use almost all of the 44x44 space
    # Standard macOS menu bar icons are about 22pt (44px @2x)
    target_size = (42, 42)  # Almost full size
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Create canvas
    new_img = Image.new("RGBA", (44, 44), (0, 0, 0, 0))
    offset = ((44 - img.width) // 2, (44 - img.height) // 2)
    new_img.paste(img, offset)
    
    new_img.save(output_path, "PNG")
    print(f"Saved large menu bar icon to {output_path}")

def make_high_quality_dock_icon(input_path, output_path, size=512):
    """Create a high-quality dock icon"""
    img = Image.open(input_path)
    img = img.convert("RGBA")
    
    # Remove white background
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    
    img.putdata(newData)
    
    # Crop to content
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # Enhance colors for dock visibility - MAXIMUM VIBRANCE!
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(2.5)  # Extremely saturated
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.8)  # Maximum contrast
    
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.3)  # Brighter
    
    # Resize to standard dock icon size with padding
    img.thumbnail((int(size * 0.85), int(size * 0.85)), Image.Resampling.LANCZOS)
    
    # Create canvas
    new_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    new_img.paste(img, offset)
    
    new_img.save(output_path, "PNG")
    print(f"Saved high-quality dock icon to {output_path}")

if __name__ == "__main__":
    make_large_white_menubar_icon("logo_transparent.png", "logo_menu_white.png")
    make_high_quality_dock_icon("logo.png", "logo_dock.png", 512)
