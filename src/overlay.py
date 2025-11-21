from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

def overlay_caption(image_path: str, text: str, output_path: str, font_path: str=None, font_size:int=42, margin:int=20):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # Load font
    if font_path and os.path.isfile(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Wrap text
    max_width = 22 if W < 700 else 32
    lines = textwrap.wrap(text, width=max_width)

    # Compute text block size
    line_heights = []
    max_line_w = 0
    for line in lines:
        w, h = draw.textbbox((0,0), line, font=font)[2:]
        max_line_w = max(max_line_w, w)
        line_heights.append(h)
    total_h = sum(line_heights) + (len(lines)-1)*8

    # Position (top or bottom). Here: bottom with a semi-transparent bar
    x = (W - max_line_w)//2
    y = H - total_h - margin

    # Background rectangle
    pad = 10
    rect_xy = (x - pad, y - pad, x + max_line_w + pad, y + total_h + pad)
    overlay = Image.new('RGBA', (W, H), (0,0,0,0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(rect_xy, fill=(0,0,0,160), outline=(255,255,255,180), width=2)

    # Draw text
    ty = y
    for line, lh in zip(lines, line_heights):
        tw = draw.textlength(line, font=font)
        tx = (W - tw)//2
        overlay_draw.text((tx, ty), line, font=font, fill=(255,255,255,255))
        ty += lh + 8

    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    img = img.convert("RGB")
    img.save(output_path)
    return output_path
