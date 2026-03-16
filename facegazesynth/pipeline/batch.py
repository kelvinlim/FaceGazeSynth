"""Batch generation pipeline for diverse face + gaze + emotion combinations."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..face_model.expressions import (
    get_emotion_params,
    list_emotions,
    random_identity,
)
from .face_render import render_face


@dataclass
class SampleSpec:
    """Specification for a single generated sample."""

    identity_seed: int
    emotion: str
    emotion_intensity: float
    theta_h_deg: float
    theta_v_deg: float
    albedo_seed: int


def generate_batch(
    n_identities: int = 5,
    emotions: Optional[list[str]] = None,
    gaze_angles: Optional[list[float]] = None,
    resolution: int = 256,
    output_dir: str = "output/batch",
    use_albedo: bool = True,
    identity_scale: float = 1.5,
    base_seed: int = 0,
    perspective: bool = False,
) -> list[dict]:
    """Generate a batch of diverse face renders.

    Produces identity × emotion × gaze_angle combinations.

    Args:
        n_identities: Number of random identities to generate.
        emotions: List of emotions. Default: ["neutral", "happy", "sad", "angry", "surprised"].
        gaze_angles: List of horizontal gaze angles. Default: [0, 10, 20].
        resolution: Image width in pixels.
        output_dir: Directory for output images and manifest.
        use_albedo: Use data-driven albedo textures.
        identity_scale: Standard deviation for identity sampling.
        base_seed: Base random seed for reproducibility.
        perspective: Use perspective camera.

    Returns:
        List of dicts with sample metadata (filepath, parameters).
    """
    if emotions is None:
        emotions = ["neutral", "happy", "sad", "angry", "surprised"]
    if gaze_angles is None:
        gaze_angles = [0.0, 10.0, 20.0]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = []
    total = n_identities * len(emotions) * len(gaze_angles)
    count = 0

    for id_idx in range(n_identities):
        identity_seed = base_seed + id_idx
        albedo_seed = base_seed + id_idx + 10000
        betas = random_identity(seed=identity_seed, scale=identity_scale)

        for emotion in emotions:
            for angle in gaze_angles:
                count += 1
                fname = f"id{id_idx:03d}_{emotion}_gaze{angle:+.0f}.png"
                fpath = out_path / fname

                print(f"  [{count}/{total}] {fname}")

                img = render_face(
                    theta_h_deg=angle,
                    resolution=resolution,
                    betas=betas,
                    emotion=emotion,
                    use_albedo=use_albedo,
                    albedo_seed=albedo_seed,
                    perspective=perspective,
                )
                img.save(str(fpath))

                entry = {
                    "file": fname,
                    "identity_seed": identity_seed,
                    "albedo_seed": albedo_seed,
                    "emotion": emotion,
                    "theta_h_deg": angle,
                    "theta_v_deg": 0.0,
                    "resolution": resolution,
                    "betas": betas.tolist(),
                }
                manifest.append(entry)

    # Save manifest
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Generated {len(manifest)} images in {output_dir}/")

    return manifest
