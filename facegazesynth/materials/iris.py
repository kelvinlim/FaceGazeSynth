"""Procedural iris texture.

Generates a realistic iris pattern in polar coordinates with:
- Radial collagen fibers
- Collarette ring (boundary between pupillary and ciliary zones)
- Crypts (dark spots)
- Limbal darkening
- Organic noise for natural variation
"""

import numpy as np


def iris_color_at(
    hit_points: np.ndarray,
    iris_center: np.ndarray,
    iris_outer_radius: float,
    pupil_radius: float,
    base_color: np.ndarray = None,
    seed: int = 42,
) -> np.ndarray:
    """Compute procedural iris color for hit points on the iris plane.

    Args:
        hit_points: (N, 3) points on the iris plane.
        iris_center: (3,) center of iris disc.
        iris_outer_radius: Outer radius of iris.
        pupil_radius: Inner radius (pupil edge).
        base_color: (3,) RGB base color. Default: brown.
        seed: Random seed for reproducible texture.

    Returns:
        (N, 3) RGB colors in [0, 1].
    """
    if base_color is None:
        base_color = np.array([0.45, 0.28, 0.12])

    rng = np.random.RandomState(seed)

    # Convert to polar coordinates relative to iris center
    offset = hit_points - iris_center
    r = np.sqrt(offset[:, 0]**2 + offset[:, 1]**2)
    theta = np.arctan2(offset[:, 1], offset[:, 0])

    # Normalize radius: 0 at pupil edge, 1 at iris outer edge
    r_norm = (r - pupil_radius) / (iris_outer_radius - pupil_radius)
    r_norm = np.clip(r_norm, 0.0, 1.0)

    n = len(hit_points)
    intensity = np.ones(n)

    # --- Radial fibers ---
    n_fibers = 90
    fiber = 0.5 + 0.5 * np.sin(n_fibers * theta + 0.3 * np.sin(7 * theta))
    # Fibers are more visible in the ciliary zone (outer)
    fiber_strength = 0.15 * r_norm
    intensity += fiber * fiber_strength

    # --- Collarette ring ---
    # At about 40% of iris width from pupil (r_norm ~ 0.4)
    # Broad, subtle brightening — not a sharp ring
    collarette_pos = 0.4
    collarette = np.exp(-((r_norm - collarette_pos) ** 2) / (2 * 0.06**2))
    intensity += 0.06 * collarette

    # --- Pupillary zone (inner) is slightly darker ---
    inner_dark = np.exp(-(r_norm**2) / (2 * 0.15**2))
    intensity -= 0.1 * inner_dark

    # --- Limbal darkening (outer edge) ---
    # Gradual fade, not a sharp cutoff
    limbal = np.clip((r_norm - 0.7) / 0.3, 0.0, 1.0)
    intensity -= 0.2 * limbal ** 2

    # --- Crypts (scattered dark spots) ---
    # Use deterministic hash for reproducibility
    n_crypts = 30
    crypt_angles = rng.uniform(-np.pi, np.pi, n_crypts)
    crypt_radii = rng.uniform(0.2, 0.8, n_crypts)
    crypt_sizes = rng.uniform(0.03, 0.06, n_crypts)

    for i in range(n_crypts):
        # Distance in (r_norm, theta) space
        dr = r_norm - crypt_radii[i]
        # Wrap theta difference
        dtheta = theta - crypt_angles[i]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        # Scale theta by r to get arc distance
        d2 = dr**2 + (dtheta * r_norm * 0.3)**2
        crypt = np.exp(-d2 / (2 * crypt_sizes[i]**2))
        intensity -= 0.2 * crypt

    # --- Fine noise for organic variation ---
    # Simple high-frequency angular noise
    noise = 0.03 * np.sin(31 * theta + 5 * r_norm) + 0.02 * np.sin(53 * theta - 3 * r_norm)
    intensity += noise

    # Clamp and apply to base color
    intensity = np.clip(intensity, 0.3, 1.3)
    colors = base_color[np.newaxis, :] * intensity[:, np.newaxis]

    return np.clip(colors, 0.0, 1.0)
