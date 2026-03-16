"""Limbus darkening: smooth ring at the cornea-sclera boundary."""

import numpy as np


def limbus_darkening(
    hit_points: np.ndarray,
    cornea_center: np.ndarray,
    cornea_radius: float,
    limbus_half_angle: float,
    width_mm: float = 0.5,
) -> np.ndarray:
    """Compute limbus darkening factor for sclera points near the limbus.

    Args:
        hit_points: (N, 3) surface points.
        cornea_center: (3,) cornea sphere center.
        cornea_radius: Cornea sphere radius.
        limbus_half_angle: Half-angle of limbus from cornea center.
        width_mm: Width of darkening zone in mm.

    Returns:
        (N,) multiplier in [0.6, 1.0] — lower near limbus.
    """
    # Compute angular distance from the limbus circle
    to_hit = hit_points - cornea_center
    dist = np.linalg.norm(to_hit, axis=1)
    # Project onto cornea axis (which is +Z in unrotated, but use actual geometry)
    # Use the vector from cornea center to the hit point
    cos_angle = to_hit[:, 2] / np.maximum(dist, 1e-8)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Angular distance from limbus
    angular_dist = np.abs(angle - limbus_half_angle)
    linear_dist = angular_dist * cornea_radius  # approximate arc length in mm

    # Gentle darkening with wider falloff
    darkening = 0.15 * np.exp(-(linear_dist**2) / (2 * (width_mm * 0.8)**2))

    return 1.0 - darkening
