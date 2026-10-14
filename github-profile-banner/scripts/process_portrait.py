"""
scripts/process_portrait.py
===========================
Processes the original portrait photograph of Rakesh Babu Gajula for both
Dark Mode (#0A161F bg, #A78BFA lit dots) and Light Mode (neutral bg, #7C3AED dots).

Pipeline:
1. Load original portrait (without modifying the source).
2. Crop to head-and-shoulders composition.
3. Resize to 300 x 340 working grid.
4. Apply 1.3x contrast enhancement, ImageOps.autocontrast(cutoff=1), UnsharpMask(radius=3, percent=140).
5. Segment background for dark mode with color-distance thresholding, closing, hole filling,
   largest component selection, and boundary error-diffusion bleed suppression.
6. 1-bit Floyd-Steinberg dithering.
7. Convert dot runs into crisp SVG <path> elements (shape-rendering="crispEdges").
8. Save portrait-dithered-dark.svg, portrait-dithered-light.svg, and coordinate arrays.
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
from scipy import ndimage

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
SOURCE_PORTRAIT = ASSETS_DIR / "portrait-original.jpg"

GRID_W = 300
GRID_H = 340

# Crop bounding box within original 768x1024 image
# (9, 40, 759, 890) -> keeps hair, face, neck, collar, jacket lapels and full shoulders visible
CROP_BOX = (9, 40, 759, 890)


def extract_subject_mask(crop_rgb):
    """
    Extracts a clean, halo-free subject mask from the cropped RGB image.
    Uses natural color differences, physical color bounds, and morphological operations.
    """
    arr = np.array(crop_rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    h, w = arr.shape[:2]
    y_coords, x_coords = np.ogrid[:h, :w]

    # 1. Warm tones (skin + mustard jacket): r - b > 16
    # 2. Dark hair, eyebrows, mustache: r, g, b < 110
    # 3. White t-shirt in inner chest collar (bounded by jacket lapels):
    is_tshirt = (x_coords >= 110) & (x_coords <= 185) & (y_coords >= 215) & (b > 210) & (g > 200)

    is_subject = ((r - b) > 16) | ((r < 110) & (g < 110) & (b < 110)) | is_tshirt

    # Morphological closing to bridge small gaps between hair strands and collar seams
    mask = ndimage.binary_closing(is_subject, structure=np.ones((7, 7)))
    mask = ndimage.binary_fill_holes(mask)

    # Keep largest connected component (the person)
    labeled, num_features = ndimage.label(mask)
    if num_features > 0:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        mask = (labeled == (np.argmax(sizes) + 1))

    # Gentle cleanup: opening then closing then hole-filling
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)

    return mask


def dither_floyd_steinberg(gray_arr, invert=False, mask=None):
    """
    Applies 1-bit Floyd-Steinberg error-diffusion dithering.
    If mask is provided, error diffusion across the mask boundary is explicitly suppressed
    to prevent haloing, noise, and boundary bleed.
    """
    h, w = gray_arr.shape
    buf = gray_arr.copy().astype(np.float32)
    out = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            old_val = buf[y, x]

            # If pixel is outside mask in dark mode, set to background (no dot) and zero error
            if mask is not None and not mask[y, x]:
                out[y, x] = 0
                continue

            if invert:
                # Dark mode: lit dots on dark panel (bright areas get dots)
                new_val = 255.0 if old_val >= 128.0 else 0.0
                out[y, x] = 1 if new_val == 255.0 else 0
            else:
                # Light mode: dark dots on light panel (dark areas get dots)
                new_val = 255.0 if old_val >= 128.0 else 0.0
                out[y, x] = 1 if new_val == 0.0 else 0

            err = old_val - new_val

            # Distribute error only to neighbors (suppress bleed across mask boundary)
            if x + 1 < w:
                if mask is None or mask[y, x + 1]:
                    buf[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < h:
                if x > 0 and (mask is None or mask[y + 1, x - 1]):
                    buf[y + 1, x - 1] += err * (3.0 / 16.0)
                if mask is None or mask[y + 1, x]:
                    buf[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < w and (mask is None or mask[y + 1, x + 1]):
                    buf[y + 1, x + 1] += err * (1.0 / 16.0)

    return out


def dots_to_svg_runs(binary_grid, color_hex):
    """
    Converts a 2D binary dot grid (H x W) into optimized SVG horizontal run paths.
    Contiguous dots in a row are merged into a single horizontal rectangle:
    M x y h run_len v 1 h -run_len Z
    """
    h, w = binary_grid.shape
    d_parts = []
    total_dots = int(binary_grid.sum())

    for y in range(h):
        in_run = False
        start_x = 0
        for x in range(w):
            val = binary_grid[y, x]
            if val and not in_run:
                in_run = True
                start_x = x
            elif not val and in_run:
                run_len = x - start_x
                d_parts.append(f"M{start_x},{y}h{run_len}v1h-{run_len}Z")
                in_run = False
        if in_run:
            run_len = w - start_x
            d_parts.append(f"M{start_x},{y}h{run_len}v1h-{run_len}Z")

    d_attr = "".join(d_parts)
    svg = f'<path d="{d_attr}" fill="{color_hex}" shape-rendering="crispEdges"/>'
    return svg, total_dots


def get_dot_coordinates(binary_grid):
    """Returns list of (x, y) coordinates of all active dots."""
    ys, xs = np.where(binary_grid == 1)
    return list(zip(xs.tolist(), ys.tolist()))


def process_portrait():
    print(f"Loading source portrait from {SOURCE_PORTRAIT}...")
    if not SOURCE_PORTRAIT.exists():
        raise FileNotFoundError(f"Source portrait missing: {SOURCE_PORTRAIT}")

    orig = Image.open(SOURCE_PORTRAIT)
    print(f"Original image size: {orig.size}, mode: {orig.mode}")

    # 1. Crop to head-and-shoulders composition
    crop = orig.crop(CROP_BOX).resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)
    print(f"Cropped & resized working grid: {crop.size}")

    # 2. Contrast enhancement (~1.3x)
    enhancer = ImageEnhance.Contrast(crop)
    crop_enh = enhancer.enhance(1.3)

    # 3. ImageOps.autocontrast with cutoff 1%
    crop_ac = ImageOps.autocontrast(crop_enh, cutoff=1)

    # 4. UnsharpMask(radius=3, percent=140)
    crop_sharp = crop_ac.filter(ImageFilter.UnsharpMask(radius=3, percent=140))

    # Convert to grayscale
    gray = crop_sharp.convert("L")
    gray_arr = np.array(gray, dtype=np.float32)

    # 5. Extract clean subject mask for dark mode
    print("Extracting dark-mode subject mask...")
    subject_mask = extract_subject_mask(crop)
    mask_coverage = float(subject_mask.mean() * 100)
    print(f"Subject mask coverage: {mask_coverage:.2f}%")

    # 6. Floyd-Steinberg dithering
    print("Applying Floyd-Steinberg dithering for dark mode...")
    dark_grid = dither_floyd_steinberg(gray_arr, invert=True, mask=subject_mask)
    dark_dots = int(dark_grid.sum())
    print(f"Dark mode total dot count: {dark_dots}")

    print("Applying Floyd-Steinberg dithering for light mode...")
    light_grid = dither_floyd_steinberg(gray_arr, invert=False, mask=None)
    light_dots = int(light_grid.sum())
    print(f"Light mode total dot count: {light_dots}")

    # 7. Convert to standalone SVGs
    dark_path_svg, _ = dots_to_svg_runs(dark_grid, "#A78BFA")
    dark_svg_full = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {GRID_W} {GRID_H}" width="{GRID_W}" height="{GRID_H}">
  <rect width="{GRID_W}" height="{GRID_H}" fill="#0A161F"/>
  {dark_path_svg}
</svg>"""

    light_path_svg, _ = dots_to_svg_runs(light_grid, "#7C3AED")
    light_svg_full = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {GRID_W} {GRID_H}" width="{GRID_W}" height="{GRID_H}">
  <rect width="{GRID_W}" height="{GRID_H}" fill="#F8FAFC"/>
  {light_path_svg}
</svg>"""

    dark_out_file = ASSETS_DIR / "portrait-dithered-dark.svg"
    light_out_file = ASSETS_DIR / "portrait-dithered-light.svg"
    with open(dark_out_file, "w", encoding="utf-8") as f:
        f.write(dark_svg_full)
    with open(light_out_file, "w", encoding="utf-8") as f:
        f.write(light_svg_full)
    print(f"Saved: {dark_out_file}")
    print(f"Saved: {light_out_file}")

    # 8. Save coordinates to JSON for downstream banner animation generation
    coords_data = {
        "grid_size": [GRID_W, GRID_H],
        "dark_mode": {
            "dot_count": dark_dots,
            "color": "#A78BFA",
            "coords": get_dot_coordinates(dark_grid),
        },
        "light_mode": {
            "dot_count": light_dots,
            "color": "#7C3AED",
            "coords": get_dot_coordinates(light_grid),
        },
    }
    coords_file = ASSETS_DIR / "portrait-dots.json"
    with open(coords_file, "w", encoding="utf-8") as f:
        json.dump(coords_data, f)
    print(f"Saved dot coordinates to: {coords_file}")

    return coords_data


if __name__ == "__main__":
    process_portrait()
