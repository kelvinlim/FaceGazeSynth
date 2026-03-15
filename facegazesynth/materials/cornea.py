"""Cornea material: Fresnel specular reflection (Purkinje image)."""

import numpy as np
from ..optics.refraction import fresnel_reflectance


def corneal_specular(
    hit_points: np.ndarray,
    normals: np.ndarray,
    ray_dirs: np.ndarray,
    light_position: np.ndarray,
    n_air: float = 1.0,
    n_cornea: float = 1.376,
    shininess: float = 120.0,
) -> np.ndarray:
    """Compute corneal specular highlight (Purkinje image).

    Uses Blinn-Phong (half-vector) model which works well with
    orthographic cameras.

    Args:
        hit_points: (N, 3) points on cornea surface.
        normals: (N, 3) cornea surface normals.
        ray_dirs: (N, 3) incoming ray directions.
        light_position: (3,) light position.
        n_air: Refractive index of air.
        n_cornea: Refractive index of cornea.
        shininess: Specular exponent.

    Returns:
        (N,) specular intensity for each point.
    """
    # Light direction
    light_dir = light_position - hit_points
    light_dir = light_dir / np.linalg.norm(light_dir, axis=1, keepdims=True)

    # View direction (toward camera)
    view_dir = -ray_dirs

    # Blinn-Phong: half vector between light and view
    half_vec = light_dir + view_dir
    half_vec = half_vec / np.linalg.norm(half_vec, axis=1, keepdims=True)

    ndoth = np.sum(normals * half_vec, axis=1)
    spec = np.maximum(ndoth, 0.0) ** shininess

    # Scale by Fresnel reflectance
    cos_i = np.maximum(np.sum(normals * light_dir, axis=1), 0.0)
    fresnel = fresnel_reflectance(cos_i, n_air, n_cornea)

    return spec * fresnel * 5.0
