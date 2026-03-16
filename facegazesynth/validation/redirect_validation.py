"""Validation for gaze redirection: round-trip displacement measurement."""

import numpy as np
from PIL import Image

from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS
from ..redirection.detection import detect_eyes
from ..redirection.compositing import redirect_both_eyes
from ..redirection.physics_mapping import calibrate_eye
from ..validation.expected_curves import refraction_corrected_displacement


def validate_redirection(
    image_path: str,
    angles: list = None,
    params: EyeParameters = DEFAULT_PARAMS,
) -> dict:
    """Round-trip validation: redirect photo, re-detect iris, measure displacement.

    For each target angle, redirects the gaze, re-runs detection on the result,
    and measures the iris displacement. Compares against physics predictions.

    Args:
        image_path: Path to center-gaze input photo.
        angles: Angles to test. Default: [5, 10, 15, 20].
        params: Eye parameters.

    Returns:
        Dict with 'angles', 'measured_mm', 'predicted_mm', 'rms_error_mm'.
    """
    if angles is None:
        angles = [5, 10, 15, 20]

    img = np.array(Image.open(image_path).convert("RGB"))
    original_det = detect_eyes(img)

    # Calibrate using right eye (viewer's left)
    mapping = calibrate_eye(original_det.right_eye, params)

    measured_mm = []
    predicted_mm = []

    for angle in angles:
        # Redirect gaze
        result = redirect_both_eyes(img, original_det, float(angle), params)

        # Re-detect iris in redirected image
        try:
            new_det = detect_eyes(result)
        except ValueError:
            measured_mm.append(float("nan"))
            predicted_mm.append(refraction_corrected_displacement(angle, params))
            continue

        # Measure displacement of right eye iris center (pixels)
        orig_cx = original_det.right_eye.iris_center[0]
        new_cx = new_det.right_eye.iris_center[0]
        displacement_px = new_cx - orig_cx

        # Convert to mm
        displacement_mm = displacement_px * mapping.mm_per_pixel
        measured_mm.append(displacement_mm)
        predicted_mm.append(refraction_corrected_displacement(angle, params))

    measured = np.array(measured_mm)
    predicted = np.array(predicted_mm)
    valid = ~np.isnan(measured)
    rms = np.sqrt(np.mean((measured[valid] - predicted[valid]) ** 2)) if valid.any() else float("nan")

    return {
        "angles": angles,
        "measured_mm": measured.tolist(),
        "predicted_mm": predicted.tolist(),
        "rms_error_mm": rms,
    }
