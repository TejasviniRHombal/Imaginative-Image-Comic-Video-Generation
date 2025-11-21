"""
Simple Imaginative Image Generation Project
Objective 4: Generate imaginative images from user text prompts.

This code can be directly uploaded to GitHub as app.py
Works using placeholder images WITHOUT any API key.
You can later plug in any AI image API.

Folders created:
outputs/images/

To run:
python app.py --prompt "A magical forest with glowing trees"

Requirements:
Pillow

Install:
pip install Pillow
"""

import os
import argparse
import textwrap
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "outputs/images"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_placeholder(prompt: str, size=(1024, 1024)):
    """Creates a placeholder imaginative image with prompt text.
    This works WITHOUT internet or API key.
    """
    img = Image.new("RGB", size, (230, 230, 250))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    lines = textwrap.wrap(prompt, width=25)
    y = 40
    for line in lines:
        w, h = draw.textsize(line, font=font)
        draw.text(((size[0] - w) // 2, y), line, fill=(20, 20, 40), font=font)
        y += h + 10

    return img


def main():
    parser = argparse.ArgumentParser(description="Imaginative Image Generator")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate imaginative image")
    args = parser.parse_args()

    print("Generating image for prompt:", args.prompt)

    img = generate_placeholder(args.prompt)

    filename = os.path.join(OUT_DIR, "generated_image.png")
    img.save(filename)

    print("Image saved at:", filename)


if __name__ == "__main__":
    main()
