"""Ray-geometry intersection routines.

All functions operate on batched rays (N, 3) for vectorized rendering.
Return NaN for misses so results stay as contiguous arrays.
"""

import numpy as np


def intersect_ray_sphere(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Intersect batch of rays with a sphere.

    Args:
        ray_origin: (N, 3) ray origins.
        ray_dir: (N, 3) ray directions (unit vectors).
        sphere_center: (3,) sphere center.
        sphere_radius: Sphere radius.

    Returns:
        t: (N,) intersection distance (NaN if miss).
        hit_point: (N, 3) intersection point (NaN if miss).
        normal: (N, 3) outward surface normal at hit (NaN if miss).
        hit_mask: (N,) boolean, True for rays that hit.
    """
    # Vector from ray origin to sphere center
    oc = ray_origin - sphere_center  # (N, 3)

    # Quadratic coefficients: |d|^2 t^2 + 2(oc·d)t + (|oc|^2 - r^2) = 0
    # Since d is unit vector, a = 1
    b = np.sum(oc * ray_dir, axis=1)  # half of actual b
    c = np.sum(oc * oc, axis=1) - sphere_radius**2

    discriminant = b**2 - c

    hit_mask = discriminant >= 0

    # Compute t for hits (nearest positive intersection)
    sqrt_disc = np.where(hit_mask, np.sqrt(np.maximum(discriminant, 0)), 0.0)
    t1 = -b - sqrt_disc  # nearer
    t2 = -b + sqrt_disc  # farther

    # Pick nearest positive t
    t = np.where(t1 > 1e-6, t1, t2)
    hit_mask = hit_mask & (t > 1e-6)

    # Compute hit points and normals
    t_safe = np.where(hit_mask, t, 0.0)
    hit_point = ray_origin + t_safe[:, np.newaxis] * ray_dir
    normal = (hit_point - sphere_center) / sphere_radius

    # Set misses to NaN
    t = np.where(hit_mask, t, np.nan)
    hit_point = np.where(hit_mask[:, np.newaxis], hit_point, np.nan)
    normal = np.where(hit_mask[:, np.newaxis], normal, np.nan)

    return t, hit_point, normal, hit_mask


def intersect_ray_plane(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect batch of rays with an infinite plane.

    Args:
        ray_origin: (N, 3) ray origins.
        ray_dir: (N, 3) ray directions.
        plane_point: (3,) a point on the plane.
        plane_normal: (3,) plane normal (unit vector).

    Returns:
        t: (N,) intersection distance (NaN if parallel/behind).
        hit_point: (N, 3) intersection point.
        hit_mask: (N,) boolean.
    """
    denom = np.sum(ray_dir * plane_normal, axis=1)  # (N,)

    # Parallel check (|denom| near zero)
    hit_mask = np.abs(denom) > 1e-8

    diff = plane_point - ray_origin  # (N, 3)
    t = np.sum(diff * plane_normal, axis=1) / np.where(hit_mask, denom, 1.0)

    # Must be in front of ray
    hit_mask = hit_mask & (t > 1e-6)

    t_safe = np.where(hit_mask, t, 0.0)
    hit_point = ray_origin + t_safe[:, np.newaxis] * ray_dir

    t = np.where(hit_mask, t, np.nan)
    hit_point = np.where(hit_mask[:, np.newaxis], hit_point, np.nan)

    return t, hit_point, hit_mask


def is_within_cornea_cap(
    hit_point: np.ndarray,
    cornea_center: np.ndarray,
    cornea_radius: float,
    limbus_half_angle: float,
) -> np.ndarray:
    """Check if points on the cornea sphere are within the cornea cap.

    The cornea cap is the portion of the cornea sphere that protrudes
    forward (toward +Z) from the limbus circle.

    Args:
        hit_point: (N, 3) points on the cornea sphere surface.
        cornea_center: (3,) cornea sphere center.
        cornea_radius: Cornea sphere radius.
        limbus_half_angle: Half-angle subtended by limbus from cornea center.

    Returns:
        (N,) boolean, True if point is within the cap.
    """
    # Vector from cornea center to hit point
    to_hit = hit_point - cornea_center
    # The cap axis is +Z (optical axis)
    cos_angle = to_hit[:, 2] / cornea_radius
    return cos_angle >= np.cos(limbus_half_angle)
