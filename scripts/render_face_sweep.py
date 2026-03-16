#!/usr/bin/env python3
"""Render a grid of faces at different gaze angles."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from facegazesynth.pipeline.face_render import render_face_sweep


def _angle_label(angle: float) -> str:
    """Format angle as 'Center', '10° L', '10° R', etc."""
    if angle == 0:
        return "Center"
    direction = "L" if angle < 0 else "R"
    return f"{abs(int(angle))}\u00b0 {direction}"


def _add_labels(grid: Image.Image, angles: list, resolution: int) -> Image.Image:
    """Add angle labels above each frame in the grid."""
    label_height = max(24, resolution // 6)
    labeled = Image.new("RGB", (grid.width, grid.height + label_height), (38, 38, 46))
    labeled.paste(grid, (0, label_height))

    draw = ImageDraw.Draw(labeled)
    # Try to load a reasonable font size
    font_size = max(12, resolution // 10)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, angle in enumerate(angles):
        label = _angle_label(angle)
        x_center = i * resolution + resolution // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text(
            (x_center - text_w // 2, (label_height - font_size) // 2),
            label,
            fill=(220, 220, 220),
            font=font,
        )

    return labeled


def main():
    parser = argparse.ArgumentParser(description="Render face gaze sweep")
    parser.add_argument("--resolution", type=int, default=128, help="Width per face in pixels")
    parser.add_argument("--angles", type=str, default=None, help="Gaze angles as JSON array (degrees)")
    parser.add_argument("--betas", type=str, default=None, help="Identity coefficients as JSON array")
    parser.add_argument("--expression", type=str, default=None, help="Expression coefficients as JSON array")
    parser.add_argument("--model-path", type=str, default=None, help="Path to FLAME model directory")
    parser.add_argument("--output", type=str, default="output/face_sweep.png", help="Output path")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    angles = json.loads(args.angles) if args.angles else None
    betas = np.array(json.loads(args.betas)) if args.betas else None
    expression = np.array(json.loads(args.expression)) if args.expression else None

    effective_angles = angles if angles is not None else [0, 5, 10, 15, 20, 30]

    img = render_face_sweep(
        angles=angles,
        resolution=args.resolution,
        betas=betas,
        expression=expression,
        model_path=args.model_path,
    )
    img = _add_labels(img, effective_angles, args.resolution)
    img.save(args.output)
    print(f"Saved to {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
