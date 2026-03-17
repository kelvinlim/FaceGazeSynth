"""Seamless compositing of redirected eye regions into original photo."""

import cv2
import numpy as np

from .detection import EyeDetection, FaceDetection
from .physics_mapping import GazeMapping, calibrate_eye, target_displacement_px
from .warping import warp_eye_region, eyelid_mask
from .specular import reposition_specular
from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS


def redirect_single_eye(
    image: np.ndarray,
    detection: EyeDetection,
    angle_deg: float,
    params: EyeParameters = DEFAULT_PARAMS,
) -> np.ndarray:
    """Redirect gaze for a single eye.

    Uses extract-inpaint-paste: the warping step internally inpaints the
    old iris region with sclera, then composites the iris at the new
    position. Specular highlight is repositioned afterwards.

    Args:
        image: (H, W, 3) uint8 RGB image.
        detection: Eye detection result.
        angle_deg: Target gaze angle in degrees.
        params: Physical eye parameters.

    Returns:
        (H, W, 3) uint8 image with this eye redirected.
    """
    if angle_deg == 0:
        return image.copy()

    mapping = calibrate_eye(detection, params)
    disp_px = target_displacement_px(angle_deg, mapping, params)

    # Extract-inpaint-paste (warping handles sclera inpainting internally)
    result, eye_m, old_iris_m, new_iris_m = warp_eye_region(
        image, detection, disp_px, angle_deg
    )

    # Reposition specular highlight
    result = reposition_specular(result, detection, angle_deg, mapping, eye_m, params)

    return result


def redirect_both_eyes(
    image: np.ndarray,
    detection: FaceDetection,
    angle_deg: float,
    params: EyeParameters = DEFAULT_PARAMS,
) -> np.ndarray:
    """Redirect gaze for both eyes (conjugate gaze).

    Args:
        image: (H, W, 3) uint8 RGB image.
        detection: Full face detection with both eyes.
        angle_deg: Target horizontal gaze angle in degrees.
        params: Physical eye parameters.

    Returns:
        (H, W, 3) uint8 image with both eyes redirected.
    """
    # Process right eye first (viewer's left), then left eye
    # Both eyes get the same horizontal angle (conjugate gaze)
    result = redirect_single_eye(image, detection.right_eye, angle_deg, params)
    result = redirect_single_eye(result, detection.left_eye, angle_deg, params)
    return result
