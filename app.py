import os
import sys
import time
import json
import math
import textwrap
import hashlib
import requests
from datetime import datetime
from io import BytesIO
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

# moviepy for video creation
from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip

# Optional import for OpenAI - only used if available and API key present
try:
    import openai
except Exception:
    openai = None

# ----------------------------- Configuration ------------------------------
# You can change defaults here
OUTPUT_DIR = "outputs"
DEFAULT_IMAGE_SIZE = (1024, 1024)
FONT_PATH = None  # let PIL pick a default font; set path to ttf if you prefer

# Create outputs dir if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------- Helper Utilities ----------------------------

def sanitize_filename(s: str, maxlen: int = 80) -> str:
    """Create a safe filename fragment from arbitrary text."""
    s = s.strip().lower()
    # keep alphanumerics, hyphen and underscore
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    if len(safe) > maxlen:
        # shorten but keep a hash suffix to avoid collisions
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
        safe = safe[: maxlen - 9] + "_" + h
    return safe or "prompt"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_image_pil(img: Image.Image, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path)
    return path

# --------------------------- Image Backends --------------------------------

# 1) OpenAI Images example (DALL-E style). Uses `openai` package if available.
#    This is an example; behavior may require changes depending on the installed SDK.

def generate_image_openai(prompt: str, n: int = 1, size: tuple = DEFAULT_IMAGE_SIZE) -> List[Image.Image]:
    """Generate images using OpenAI Images API (example).

    Requires OPENAI_API_KEY environment variable and `openai` pip package.
    Returns a list of PIL.Image objects.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or openai is None:
        raise EnvironmentError("OpenAI API key missing or openai package not installed.")

    openai.api_key = api_key
    size_str = f"{size[0]}x{size[1]}"

    print(f"[openai] requesting {n} image(s) of size {size_str} for prompt: {prompt[:80]}...")

    # NOTE: adapt for the installed openai package version. This example shows
    # the common 'images.generate' or 'Image.create' pattern but you may need
    # to change the call per your SDK.
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=size_str,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI image creation failed: {e}")

    images = []
    for d in response["data"]:
        b64 = d.get("b64_json") or d.get("b64") or None
        if not b64:
            # some responses return a URL
            url = d.get("url")
            if not url:
                continue
            r = requests.get(url)
            images.append(Image.open(BytesIO(r.content)).convert("RGBA"))
        else:
            img_data = BytesIO(base64.b64decode(b64))
            images.append(Image.open(img_data).convert("RGBA"))

    return images


# 2) Generic HTTP Stable Diffusion-style endpoint example

def generate_image_stable(prompt: str, n: int = 1, size: tuple = DEFAULT_IMAGE_SIZE) -> List[Image.Image]:
    """Generate images by calling a Stable Diffusion-style HTTP API.

    Expects environment variables:
      STABILITY_API_KEY (optional)
      STABILITY_API_URL (a POST endpoint that returns raw image bytes or JSON with image URLs)

    This function attempts a few common response formats and falls back to raising
    a helpful exception so you can adapt it to your hosted endpoint.
    """
    url = os.getenv("STABILITY_API_URL")
    api_key = os.getenv("STABILITY_API_KEY")

    if not url:
        raise EnvironmentError("STABILITY_API_URL not set in environment")

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "prompt": prompt,
        "width": size[0],
        "height": size[1],
        "samples": n,
    }

    print(f"[stable] POST {url} (prompt len {len(prompt)})")
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Stable endpoint returned {r.status_code}: {r.text}")

    # Try to handle a few common response types
    content_type = r.headers.get("Content-Type", "")
    images = []

    if "application/json" in content_type:
        data = r.json()
        # look for base64 fields or urls
        if isinstance(data, dict) and "images" in data:
            for imgobj in data["images"]:
                if isinstance(imgobj, str) and imgobj.startswith("data:image"):
                    # data URL
                    head, b64 = imgobj.split(",", 1)
                    images.append(Image.open(BytesIO(base64.b64decode(b64))).convert("RGBA"))
                elif isinstance(imgobj, str) and imgobj.startswith("http"):
                    rr = requests.get(imgobj)
                    images.append(Image.open(BytesIO(rr.content)).convert("RGBA"))
                elif isinstance(imgobj, dict) and "b64" in imgobj:
                    images.append(Image.open(BytesIO(base64.b64decode(imgobj["b64"]))).convert("RGBA"))
        else:
            raise RuntimeError("Unrecognized JSON response from Stable endpoint; please adapt this function.")
    else:
        # Assume raw image bytes (single image)
        images.append(Image.open(BytesIO(r.content)).convert("RGBA"))

    return images


# 3) Local placeholder generator (fallback)

def generate_image_placeholder(prompt: str, n: int = 1, size: tuple = DEFAULT_IMAGE_SIZE) -> List[Image.Image]:
    """Create simple placeholder images using Pillow so the rest of the pipeline
    can be developed/tested without an API key.
    """
    images = []
    for i in range(n):
        img = Image.new("RGBA", size, (30 + i * 10, 30 + i * 20, 60 + i * 15))
        draw = ImageDraw.Draw(img)
        # Use a reasonably readable font; fallback to default if none.
        try:
            font = ImageFont.truetype(FONT_PATH if FONT_PATH else "DejaVuSans-Bold.ttf", 28)
        except Exception:
            font = ImageFont.load_default()

        text = f"{prompt[:60]}\n(panel {i+1})"
        lines = textwrap.fill(text, width=30)
        draw.multiline_text((20, 20), lines, font=font, fill=(255, 255, 255))
        images.append(img)
    return images


# Universal wrapper to pick a backend

def generate_images(prompt: str, n: int = 1, size: tuple = DEFAULT_IMAGE_SIZE, backend_preference: Optional[str] = None) -> List[Image.Image]:
    """High-level function that selects a backend based on environment and
    optional user preference.

    backend_preference: 'openai', 'stable', or 'auto' (default). If no working
    backend is available, falls back to placeholder images.
    """
    pref = backend_preference or os.getenv("AI_BACKEND", "auto")

    backends = []
    if pref == "openai":
        backends = [generate_image_openai]
    elif pref == "stable":
        backends = [generate_image_stable]
    else:
        # auto: prefer OpenAI if key present, else Stability if configured, else placeholder
        if os.getenv("OPENAI_API_KEY") and openai is not None:
            backends.append(generate_image_openai)
        if os.getenv("STABILITY_API_URL"):
            backends.append(generate_image_stable)
        backends.append(generate_image_placeholder)

    last_err = None
    for fn in backends:
        try:
            return fn(prompt, n=n, size=size)
        except Exception as e:
            last_err = e
            print(f"[warning] backend {fn.__name__} failed: {e}")

    raise RuntimeError(f"All image backends failed. Last error: {last_err}")

# ------------------------- Comic & Video Helpers ---------------------------

def compose_comic(images: List[Image.Image], captions: List[str], columns: int = 2, panel_size: tuple = (512, 512)) -> Image.Image:
    """Arrange images into a grid and overlay captions below each panel.

    images: list of PIL Images (will be resized to panel_size)
    captions: list of strings for each panel (len must match images)
    """
    assert len(images) == len(captions)

    rows = math.ceil(len(images) / columns)
    w, h = panel_size
    padding = 10
    caption_height = 60

    out_w = columns * w + (columns + 1) * padding
    out_h = rows * (h + caption_height) + (rows + 1) * padding

    canvas = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(FONT_PATH if FONT_PATH else "DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for idx, (img, cap) in enumerate(zip(images, captions)):
        r = idx // columns
        c = idx % columns
        x = padding + c * (w + padding)
        y = padding + r * (h + caption_height + padding)

        panel = img.copy().convert("RGBA").resize((w, h))
        canvas.paste(panel, (x, y))

        # draw caption centered below panel
        cap_box = (x, y + h + 4, x + w, y + h + caption_height - 4)
        wrapped = textwrap.fill(cap, width=30)
        tw, th = draw.multiline_textsize(wrapped, font=font)
        tx = x + (w - tw) // 2
        ty = y + h + (caption_height - th) // 2
        draw.multiline_text((tx, ty), wrapped, font=font, fill=(0, 0, 0))

    return canvas


def create_simple_video_from_images(images: List[Image.Image], captions: Optional[List[str]] = None, duration_per_image: float = 2.0, output_path: Optional[str] = None) -> str:
    """Make a simple slideshow video from images and optional captions using moviepy.

    Returns path to video file saved locally.
    """
    if captions is None:
        captions = [""] * len(images)
    assert len(images) == len(captions)

    clips = []
    for img, cap in zip(images, captions):
        # moviepy works with numpy arrays or file paths; convert PIL->RGB bytes
        img_rgb = img.convert("RGB")
        clip = ImageClip(np.array(img_rgb)).set_duration(duration_per_image)

        if cap:
            txt = TextClip(cap, fontsize=24, font="DejaVu-Sans", size=(clip.w, None), method="caption")
            txt = txt.set_pos(("center", clip.h - 80)).set_duration(duration_per_image)
            clip = CompositeVideoClip([clip, txt])

        clips.append(clip)

    # simple crossfade
    final = concatenate_videoclips(clips, method="compose")
    outname = output_path or f"comic_video_{timestamp()}.mp4"
    outpath = os.path.join(OUTPUT_DIR, outname)
    final.write_videofile(outpath, fps=24, codec="libx264", audio=False)
    return outpath

# --------------------------- High-level Flows -----------------------------


def prompt_to_panels(prompt: str, panels: int = 4) -> List[str]:
    """Create short panel prompts from a longer instruction. This is a naive
    helper — you can replace it with a smarter LLM call to split a story into
    beats if you want.
    """
    # naive split: split by sentences, then group into panels
    parts = [p.strip() for p in prompt.replace("?", ".").split(".") if p.strip()]
    if not parts:
        parts = [prompt]

    # distribute parts across panels
    result = []
    for i in range(panels):
        piece = (
            " ".join(parts[i::panels])
            if len(parts) >= panels
            else (parts[i] if i < len(parts) else parts[-1])
        )
        # make each panel prompt explicit
        result.append(f"Panel {i+1}: {piece}")
    return result


def make_comic_from_prompt(prompt: str, panels: int = 4, panel_size=(512, 512), backend: Optional[str] = None) -> dict:
    """Generate images for each comic panel and produce a combined comic image
    plus a short video slideshow.

    Returns a dict with paths: { 'comic_image': ..., 'video': ..., 'panel_paths': [...] }
    """
    panel_prompts = prompt_to_panels(prompt, panels)
    print(f"Generating {panels} images for comic...")

    images = []
    for i, p in enumerate(panel_prompts):
        imgs = generate_images(p, n=1, size=panel_size, backend_preference=backend)
        # take the first image
        images.append(imgs[0])

    # create captions (simple) — you can improve this by running an LLM to create witty captions
    captions = [f"{i+1}. {p}" for i, p in enumerate(panel_prompts)]

    # compose comic
    comic_img = compose_comic(images, captions, columns=min(2, panels), panel_size=panel_size)
    fname_base = sanitize_filename(prompt)
    comic_name = f"comic_{fname_base}_{timestamp()}.png"
    comic_path = save_image_pil(comic_img.convert("RGB"), comic_name)

    # save panels individually
    panel_paths = []
    for i, img in enumerate(images):
        ppath = save_image_pil(img.convert("RGB"), f"panel_{i+1}_{fname_base}_{timestamp()}.png")
        panel_paths.append(ppath)

    # create a short video slideshow
    try:
        video_path = create_simple_video_from_images(images, captions=captions, duration_per_image=2.0, output_path=f"video_{fname_base}_{timestamp()}.mp4")
    except Exception as e:
        print(f"[warning] video creation failed: {e}")
        video_path = ""

    return {"comic_image": comic_path, "video": video_path, "panel_paths": panel_paths}


def generate_imaginative_images(prompt: str, n: int = 1, size: tuple = DEFAULT_IMAGE_SIZE, backend: Optional[str] = None) -> List[str]:
    """Generate imaginative images and save them locally. Returns list of paths."""
    images = generate_images(prompt, n=n, size=size, backend_preference=backend)
    base = sanitize_filename(prompt)
    paths = []
    for i, img in enumerate(images):
        name = f"img_{base}_{i+1}_{timestamp()}.png"
        p = save_image_pil(img.convert("RGB"), name)
        paths.append(p)
    return paths

# ------------------------------ CLI --------------------------------------

import numpy as np


def print_menu():
    print("\nAI Comic & Image Generator")
    print("1) Create a comic from a prompt")
    print("2) Generate imaginative images")
    print("3) Exit")


def main_loop():
    print("Welcome! This program will generate images and simple comics/videos using AI backends or a local placeholder.")
    while True:
        print_menu()
        choice = input("Choose an option (1-3): ").strip()
        if choice == "1":
            prompt = input("Enter a comic prompt (e.g. 'a robot and a cat exploring space'): ").strip()
            try:
                panels = int(input("Number of panels (default 4): ").strip() or "4")
            except ValueError:
                panels = 4
            backend = input("Preferred backend (openai/stable/auto) [auto]: ").strip() or "auto"
            result = make_comic_from_prompt(prompt, panels=panels, panel_size=(512, 512), backend=backend)
            print("Comic saved:", result.get("comic_image"))
            if result.get("video"):
                print("Video saved:", result.get("video"))
            print("Panels:")
            for p in result.get("panel_paths", []):
                print(" -", p)

        elif choice == "2":
            prompt = input("Enter an image prompt: ").strip()
            try:
                n = int(input("How many images? (default 1): ").strip() or "1")
            except ValueError:
                n = 1
            size_choice = input("Size (e.g. 512, 768, 1024) [1024]: ").strip() or "1024"
            try:
                size_i = int(size_choice)
                size = (size_i, size_i)
            except Exception:
                size = DEFAULT_IMAGE_SIZE
            backend = input("Preferred backend (openai/stable/auto) [auto]: ").strip() or "auto"
            paths = generate_imaginative_images(prompt, n=n, size=size, backend=backend)
            print("Saved images:")
            for p in paths:
                print(" -", p)

        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nInterrupted — exiting.")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
