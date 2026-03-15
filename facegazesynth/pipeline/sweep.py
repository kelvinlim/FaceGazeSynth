"""Gaze angle sweep: render stereo pairs at all target angles."""

import numpy as np
from PIL import Image
from pathlib import Path

from .stereo_pair import render_stereo_pair
from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS
from ..rendering.lighting import PointLight


DEFAULT_ANGLES = [0, 5, 10, 15, 20, 30]


def render_sweep(
    angles: list[float] = None,
    resolution: int = 256,
    output_dir: str = "output/sweep",
    params: EyeParameters = DEFAULT_PARAMS,
    light: PointLight = None,
    include_negative: bool = True,
) -> Image.Image:
    """Render stereo pairs at all target gaze angles.

    Saves individual images and a composite grid.

    Args:
        angles: List of horizontal gaze angles (degrees). Default: [0,5,10,15,20,30].
        resolution: Resolution per eye (height).
        output_dir: Directory for individual images.
        params: Eye parameters.
        light: Light source.
        include_negative: If True, also render negative (leftward) angles.

    Returns:
        Composite grid PIL Image.
    """
    if angles is None:
        angles = DEFAULT_ANGLES

    # Build full angle list
    all_angles = []
    for a in sorted(angles):
        if include_negative and a > 0:
            all_angles.append(-a)
        all_angles.append(a)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    images = []
    for angle in all_angles:
        print(f"  Rendering {angle:+d}°...")
        img = render_stereo_pair(
            theta_h_deg=angle,
            resolution=resolution,
            params=params,
            light=light,
        )
        img.save(out_path / f"stereo_{angle:+03d}deg.png")
        images.append((angle, img))

    # Build composite grid
    n_images = len(images)
    cols = min(n_images, 3)
    rows = (n_images + cols - 1) // cols

    pair_w, pair_h = images[0][1].size
    padding = 4
    label_height = 20

    grid_w = cols * pair_w + (cols + 1) * padding
    grid_h = rows * (pair_h + label_height) + (rows + 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), (38, 38, 46))

    for i, (angle, img) in enumerate(images):
        row = i // cols
        col = i % cols
        x = padding + col * (pair_w + padding)
        y = padding + row * (pair_h + label_height + padding)
        grid.paste(img, (x, y))

    grid.save(out_path / "composite_grid.png")
    print(f"Saved composite grid to {out_path / 'composite_grid.png'}")

    return grid
