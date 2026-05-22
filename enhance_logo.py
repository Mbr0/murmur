from PIL import Image, ImageEnhance

def enhance_logo(input_path, output_path):
    img = Image.open(input_path)
    
    # 1. Crop to bounding box (remove empty space)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # 2. Resize to maximize usage of 44x44 space
    # Target size is 44x44, but let's leave a tiny padding so it's not touching edges
    target_size = (40, 40) 
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Create a new 44x44 canvas
    new_img = Image.new("RGBA", (44, 44), (0, 0, 0, 0))
    
    # Center the cropped/resized image
    offset = ((44 - img.width) // 2, (44 - img.height) // 2)
    new_img.paste(img, offset)
    
    # 3. Enhance Contrast and Brightness
    # Make it pop more
    enhancer = ImageEnhance.Contrast(new_img)
    new_img = enhancer.enhance(1.5) # 50% more contrast
    
    enhancer = ImageEnhance.Brightness(new_img)
    new_img = enhancer.enhance(1.2) # 20% brighter
    
    new_img.save(output_path, "PNG")
    print(f"Saved enhanced logo to {output_path}")

if __name__ == "__main__":
    enhance_logo("logo_transparent.png", "logo_menu.png")
