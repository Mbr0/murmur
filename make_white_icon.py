from PIL import Image

def make_white_icon(input_path, output_path):
    img = Image.open(input_path)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        # If pixel is not transparent, make it white
        if item[3] > 0:
            newData.append((255, 255, 255, 255)) # Pure white, full opacity
        else:
            newData.append((255, 255, 255, 0)) # Transparent

    img.putdata(newData)
    
    # Resize to standard menu bar size (22pt -> 44px for retina)
    # Ensure it fits well
    img.thumbnail((44, 44), Image.Resampling.LANCZOS)
    
    img.save(output_path, "PNG")
    print(f"Saved white icon to {output_path}")

if __name__ == "__main__":
    # Use the transparent one as source
    make_white_icon("logo_transparent.png", "logo_menu_white.png")
