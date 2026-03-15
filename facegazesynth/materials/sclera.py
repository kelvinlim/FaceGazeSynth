"""Sclera material: off-white with subtle vascular tinting."""

import numpy as np


def sclera_color_at(
    hit_points: np.ndarray,
    normals: np.ndarray,
    sclera_center: np.ndarray,
    base_color: np.ndarray = None,
) -> np.ndarray:
    """Compute sclera color with subtle variation.

    Args:
        hit_points: (N, 3) surface points on sclera.
        normals: (N, 3) surface normals.
        sclera_center: (3,) sclera center.
        base_color: (3,) RGB base color.

    Returns:
        (N, 3) RGB colors in [0, 1].
    """
    if base_color is None:
        base_color = np.array([0.94, 0.92, 0.88])

    offset = hit_points - sclera_center
    theta = np.arctan2(offset[:, 1], offset[:, 0])

    n = len(hit_points)
    colors = np.tile(base_color, (n, 1))

    # Subtle vascular tinting: slight pinkish variation
    vascular = 0.02 * np.sin(5 * theta) + 0.01 * np.sin(11 * theta + 1.3)
    colors[:, 0] += vascular  # more in red channel
    colors[:, 1] -= 0.3 * np.abs(vascular)  # slight green reduction

    return np.clip(colors, 0.0, 1.0)
