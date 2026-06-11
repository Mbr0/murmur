from PIL import Image, ImageDraw

def create_small_state_icon(icon_type, output_path, size=18):
    """Create small state icons for menu bar (recording, loading, error)"""
    img = Image.new("RGBA", (44, 44), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Center position
    center_x, center_y = 22, 22
    
    if icon_type == "recording":
        # Red circle
        radius = size // 2
        draw.ellipse(
            [(center_x - radius, center_y - radius), 
             (center_x + radius, center_y + radius)],
            fill=(255, 60, 60, 255)
        )
    elif icon_type == "processing":
        radius = size // 2
        draw.ellipse(
            [(center_x - radius, center_y - radius),
             (center_x + radius, center_y + radius)],
            outline=(0, 0, 0, 255),
            width=2
        )
        draw.arc(
            [(center_x - radius, center_y - radius),
             (center_x + radius, center_y + radius)],
            start=0, end=90,
            fill=(0, 0, 0, 255),
            width=3
        )
    elif icon_type == "error":
        # X mark
        offset = size // 2
        draw.line(
            [(center_x - offset, center_y - offset), 
             (center_x + offset, center_y + offset)],
            fill=(255, 80, 80, 255),
            width=3
        )
        draw.line(
            [(center_x - offset, center_y + offset), 
             (center_x + offset, center_y - offset)],
            fill=(255, 80, 80, 255),
            width=3
        )
    
    img.save(output_path, "PNG")
    print(f"Saved {icon_type} icon to {output_path}")

if __name__ == "__main__":
    import os

    icons_dir = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "icons")
    create_small_state_icon("recording", os.path.join(icons_dir, "icon_recording.png"), 18)
    create_small_state_icon("processing", os.path.join(icons_dir, "icon_processing.png"), 16)
    create_small_state_icon("error", os.path.join(icons_dir, "icon_error.png"), 16)
