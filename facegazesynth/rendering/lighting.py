"""Lighting model: point light with Lambertian diffuse and ambient."""

import numpy as np
from dataclasses import dataclass


@dataclass
class PointLight:
    """A point light source."""
    position: np.ndarray  # (3,) world position in mm
    intensity: float = 1.0  # overall brightness multiplier


def lambertian_diffuse(
    hit_points: np.ndarray,
    normals: np.ndarray,
    light: PointLight,
    ambient: float = 0.2,
) -> np.ndarray:
    """Compute Lambertian diffuse shading.

    Args:
        hit_points: (N, 3) surface points.
        normals: (N, 3) surface normals (unit vectors).
        light: Point light source.
        ambient: Ambient light intensity [0, 1].

    Returns:
        (N,) shading intensity in [0, 1].
    """
    light_dir = light.position - hit_points  # (N, 3)
    dist = np.linalg.norm(light_dir, axis=1, keepdims=True)
    light_dir = light_dir / np.maximum(dist, 1e-8)

    # Lambertian: max(0, N·L)
    ndotl = np.sum(normals * light_dir, axis=1)
    diffuse = np.maximum(ndotl, 0.0) * light.intensity

    return np.clip(ambient + diffuse, 0.0, 1.0)


def specular_highlight(
    hit_points: np.ndarray,
    normals: np.ndarray,
    ray_dirs: np.ndarray,
    light: PointLight,
    shininess: float = 80.0,
    strength: float = 0.8,
) -> np.ndarray:
    """Compute specular highlight (Blinn-Phong).

    Args:
        hit_points: (N, 3) surface points.
        normals: (N, 3) surface normals (unit vectors).
        ray_dirs: (N, 3) incoming ray directions (toward surface).
        light: Point light source.
        shininess: Specular exponent (higher = tighter highlight).
        strength: Specular intensity multiplier.

    Returns:
        (N,) specular intensity.
    """
    light_dir = light.position - hit_points
    light_dir = light_dir / np.linalg.norm(light_dir, axis=1, keepdims=True)

    view_dir = -ray_dirs  # points toward camera
    half_vec = light_dir + view_dir
    half_vec = half_vec / np.linalg.norm(half_vec, axis=1, keepdims=True)

    ndoth = np.sum(normals * half_vec, axis=1)
    spec = np.maximum(ndoth, 0.0) ** shininess * strength

    return spec
