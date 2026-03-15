"""Theoretical iris displacement vs. gaze angle curves.

Provides predictions with and without corneal refraction for validation.
"""

import numpy as np
from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS


def naive_displacement(angle_deg: float, params: EyeParameters = DEFAULT_PARAMS) -> float:
    """Predicted apparent iris displacement WITHOUT corneal refraction.

    Simple geometric projection: the iris center moves in 3D by
    R_rot * sin(theta), viewed orthographically.

    Args:
        angle_deg: Gaze angle in degrees.
        params: Eye parameters.

    Returns:
        Apparent iris displacement in mm (horizontal).
    """
    # Distance from rotation center to iris center along optical axis
    # Iris is at corneal_apex_z - iris_setback
    # Rotation center is at corneal_apex_z - rotation_center_depth
    iris_to_rot = params.rotation_center_depth - params.iris_setback
    return iris_to_rot * np.sin(np.radians(angle_deg))


def refraction_corrected_displacement(
    angle_deg: float,
    params: EyeParameters = DEFAULT_PARAMS,
) -> float:
    """Predicted apparent iris displacement WITH corneal refraction.

    Uses the single-surface refraction model (Gullstrand simplified eye).
    The cornea acts as a refracting surface that shifts the apparent iris
    position. For a single refracting spherical surface:

        Apparent lateral shift = physical_shift * (n1/n2) * (R / (R - d))

    where R = corneal radius, d = distance from cornea to iris,
    n1 = air, n2 = aqueous. This is derived from the thin-lens-like
    imaging equation for a single refracting surface.

    More precisely, for a spherical refracting surface:
        1/v - n1/(n2*u) = (n2 - n1)/(n2 * R)
    where u = object distance (iris behind cornea), v = image distance.
    The magnification m = (n1 * v) / (n2 * u).

    Args:
        angle_deg: Gaze angle in degrees.
        params: Eye parameters.

    Returns:
        Apparent iris displacement in mm (horizontal).
    """
    R = params.cornea_radius_of_curvature
    d = params.iris_setback  # iris distance behind corneal surface
    n1 = params.ior_air
    n2 = params.ior_aqueous

    # Object distance (iris behind cornea, measured from corneal surface)
    # Convention: u is negative for object on same side as incoming light
    u = -d  # iris is behind the refracting surface

    # Single refracting surface equation: n2/v - n1/u = (n2 - n1)/R
    # Solve for v: n2/v = (n2 - n1)/R + n1/u
    rhs = (n2 - n1) / R + n1 / u
    v = n2 / rhs  # image distance

    # Lateral magnification: m = (n1 * v) / (n2 * u)
    m = (n1 * v) / (n2 * u)

    # Physical displacement (same as naive)
    phys_disp = naive_displacement(angle_deg, params)

    # Apparent displacement is magnified by |m|
    return phys_disp * abs(m)


def displacement_curves(
    angles_deg: np.ndarray = None,
    params: EyeParameters = DEFAULT_PARAMS,
) -> dict:
    """Compute both displacement curves for a range of angles.

    Args:
        angles_deg: Array of gaze angles. Default: 0 to 35 in 1° steps.
        params: Eye parameters.

    Returns:
        Dict with 'angles', 'naive', 'refracted' arrays.
    """
    if angles_deg is None:
        angles_deg = np.arange(0, 36, 1.0)

    naive = np.array([naive_displacement(a, params) for a in angles_deg])
    refracted = np.array([refraction_corrected_displacement(a, params) for a in angles_deg])

    return {
        "angles": angles_deg,
        "naive": naive,
        "refracted": refracted,
    }
