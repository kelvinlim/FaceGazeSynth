"""High-level gaze redirection pipeline for real photos."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS
from ..redirection.detection import detect_eyes, FaceDetection
from ..redirection.compositing import redirect_both_eyes

# Default gaze angles for sweep
DEFAULT_ANGLES = [-20, -15, -10, -5, 0, 5, 10, 15, 20]


def redirect_gaze(
    image_path: str,
    angle_deg: float,
    params: EyeParameters = DEFAULT_PARAMS,
    detection: Optional[FaceDetection] = None,
) -> Image.Image:
    """Redirect gaze in a real photo to the specified angle.

    Args:
        image_path: Path to input photo (frontal, center gaze).
        angle_deg: Target horizontal gaze angle in degrees.
        params: Eye parameters for physics model.
        detection: Pre-computed detection (avoids re-running MediaPipe).

    Returns:
        PIL Image with redirected gaze.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    if detection is None:
        detection = detect_eyes(img)
    result = redirect_both_eyes(img, detection, angle_deg, params)
    return Image.fromarray(result)


def _angle_label(angle: float) -> str:
    """Format angle as 'Center', '10 L', '10 R', etc."""
    if angle == 0:
        return "Center"
    direction = "L" if angle < 0 else "R"
    return f"{abs(int(angle))}\u00b0 {direction}"


def redirect_gaze_sweep(
    image_path: str,
    angles: list = None,
    params: EyeParameters = DEFAULT_PARAMS,
) -> Image.Image:
    """Generate a labeled grid of gaze-redirected images.

    Args:
        image_path: Path to input photo.
        angles: List of target angles. Default: [-20..20] in 5° steps.
        params: Eye parameters.

    Returns:
        PIL Image grid with angle labels.
    """
    if angles is None:
        angles = DEFAULT_ANGLES

    img = np.array(Image.open(image_path).convert("RGB"))
    detection = detect_eyes(img)

    images = []
    for angle in angles:
        result = redirect_both_eyes(img, detection, float(angle), params)
        images.append(Image.fromarray(result))

    # Build labeled grid
    w, h = images[0].size
    label_h = max(24, h // 12)
    grid_w = w * len(angles)
    grid = Image.new("RGB", (grid_w, h + label_h), (38, 38, 46))
    draw = ImageDraw.Draw(grid)

    font_size = max(12, h // 20)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (angle, im) in enumerate(zip(angles, images)):
        grid.paste(im, (i * w, label_h))
        label = _angle_label(angle)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (i * w + w // 2 - tw // 2, (label_h - font_size) // 2),
            label, fill=(220, 220, 220), font=font,
        )

    return grid


def redirect_batch(
    input_dir: str = "samples",
    output_dir: str = "output/redirected",
    angles: list = None,
    params: EyeParameters = DEFAULT_PARAMS,
) -> list:
    """Process all photos in a directory, generating gaze sweeps.

    Produces per-person directories with individual angle images plus
    a sweep grid, and a JSON manifest.

    Args:
        input_dir: Directory containing center-gaze photos.
        output_dir: Output directory.
        angles: Target angles. Default: [-20..20] in 5° steps.
        params: Eye parameters.

    Returns:
        List of manifest entries (dicts).
    """
    if angles is None:
        angles = DEFAULT_ANGLES

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = []
    image_files = sorted(in_path.glob("*.png")) + sorted(in_path.glob("*.jpg"))

    for img_file in image_files:
        name = img_file.stem.replace("_Center", "")
        person_dir = out_path / name
        person_dir.mkdir(exist_ok=True)

        print(f"Processing {name}...")
        img = np.array(Image.open(img_file).convert("RGB"))
        detection = detect_eyes(img)

        for angle in angles:
            result = redirect_both_eyes(img, detection, float(angle), params)
            label = "Center" if angle == 0 else f"{abs(angle)}{'L' if angle < 0 else 'R'}"
            out_file = person_dir / f"{name}_{label}.png"
            Image.fromarray(result).save(out_file)

            manifest.append({
                "source": str(img_file.name),
                "identity": name,
                "angle_deg": angle,
                "output": str(out_file.relative_to(out_path)),
            })

        # Also save sweep grid
        sweep = redirect_gaze_sweep(str(img_file), angles, params)
        sweep.save(person_dir / f"{name}_sweep.png")
        print(f"  Saved {len(angles)} angles + sweep grid")

    # Write manifest
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")

    return manifest
