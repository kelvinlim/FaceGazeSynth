"""Snell's law refraction in 3D.

Implements vectorized refraction at curved surfaces using the standard
3D vector formulation. Handles total internal reflection.
"""

import numpy as np


def refract(
    incident: np.ndarray,
    normal: np.ndarray,
    n1: float,
    n2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Refract incident rays at a surface using Snell's law.

    The normal should point from the n2 medium toward n1 (i.e., outward
    from the refracting surface, toward the incoming ray side).

    Args:
        incident: (N, 3) incident ray directions (unit vectors, pointing toward surface).
        normal: (N, 3) outward surface normals (unit vectors).
        n1: Refractive index of the medium the ray is coming from.
        n2: Refractive index of the medium the ray is entering.

    Returns:
        refracted: (N, 3) refracted direction vectors (NaN for TIR).
        valid_mask: (N,) boolean, False where total internal reflection occurs.
    """
    eta = n1 / n2

    # cos(theta_i) = -dot(normal, incident) -- incident points toward surface
    cos_i = -np.sum(normal * incident, axis=1)  # (N,)

    # Handle rays hitting from the wrong side: flip normal
    flip = cos_i < 0
    if np.any(flip):
        normal = normal.copy()
        normal[flip] *= -1
        cos_i[flip] *= -1

    sin2_t = eta**2 * (1.0 - cos_i**2)

    # Total internal reflection where sin2_t > 1
    valid_mask = sin2_t <= 1.0
    cos_t = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))

    # Refracted direction: eta * incident + (eta * cos_i - cos_t) * normal
    refracted = (
        eta * incident
        + (eta * cos_i - cos_t)[:, np.newaxis] * normal
    )

    # Normalize (should already be unit, but floating point)
    length = np.linalg.norm(refracted, axis=1, keepdims=True)
    length = np.where(length > 1e-10, length, 1.0)
    refracted = refracted / length

    # Set TIR to NaN
    refracted = np.where(valid_mask[:, np.newaxis], refracted, np.nan)

    return refracted, valid_mask


def fresnel_reflectance(
    cos_i: np.ndarray,
    n1: float,
    n2: float,
) -> np.ndarray:
    """Schlick's approximation for Fresnel reflectance.

    Args:
        cos_i: (N,) cosine of incidence angle.
        n1: Refractive index of incident medium.
        n2: Refractive index of transmitted medium.

    Returns:
        (N,) reflectance values in [0, 1].
    """
    r0 = ((n1 - n2) / (n1 + n2))**2
    return r0 + (1 - r0) * (1 - np.abs(cos_i))**5
