"""Map physics-based gaze displacement (mm) to pixel displacement."""

from dataclasses import dataclass

import numpy as np

from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS
from ..validation.expected_curves import refraction_corrected_displacement
from .detection import EyeDetection


@dataclass
class GazeMapping:
    """Calibration between physical and pixel space for one eye."""
    mm_per_pixel: float
    iris_radius_px: float
    iris_radius_mm: float  # apparent (magnified) iris radius


def calibrate_eye(
    detection: EyeDetection,
    params: EyeParameters = DEFAULT_PARAMS,
) -> GazeMapping:
    """Establish mm-to-pixel mapping from detected iris size.

    The visible iris radius through the cornea is magnified by the
    single-surface refraction. We compute that magnification to get
    the correct physical scale.

    Args:
        detection: Detected eye landmarks.
        params: Physical eye parameters.

    Returns:
        GazeMapping with calibrated scale.
    """
    R = params.cornea_radius_of_curvature
    d = params.iris_setback
    n1 = params.ior_air
    n2 = params.ior_aqueous

    # Single refracting surface magnification (same as in expected_curves.py)
    u = -d
    rhs = (n2 - n1) / R + n1 / u
    v = n2 / rhs
    m = (n1 * v) / (n2 * u)

    # Apparent (magnified) iris radius in mm
    apparent_iris_radius_mm = params.iris_outer_radius * abs(m)

    # Scale: detected pixels correspond to this physical size
    mm_per_pixel = apparent_iris_radius_mm / detection.iris_radius

    return GazeMapping(
        mm_per_pixel=mm_per_pixel,
        iris_radius_px=detection.iris_radius,
        iris_radius_mm=apparent_iris_radius_mm,
    )


def target_displacement_px(
    angle_deg: float,
    mapping: GazeMapping,
    params: EyeParameters = DEFAULT_PARAMS,
) -> float:
    """Compute pixel displacement for a target gaze angle.

    Uses refraction_corrected_displacement() from the physics engine
    to get the displacement in mm, then converts to pixels.

    Args:
        angle_deg: Target horizontal gaze angle in degrees.
            Positive = rightward (from subject's perspective).
        mapping: Calibrated pixel scale for this eye.
        params: Physical eye parameters.

    Returns:
        Horizontal pixel displacement (positive = rightward in image).
    """
    disp_mm = refraction_corrected_displacement(angle_deg, params)
    return disp_mm / mapping.mm_per_pixel
