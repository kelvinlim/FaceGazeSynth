"""Sclera inpainting for regions exposed when iris moves."""

import cv2
import numpy as np


def inpaint_sclera(
    image: np.ndarray,
    old_iris_mask: np.ndarray,
    new_iris_mask: np.ndarray,
    eye_mask: np.ndarray,
) -> np.ndarray:
    """Fill exposed sclera region where iris used to be.

    When the iris moves, the area it vacated needs to be filled with
    sclera texture. We use OpenCV's Telea inpainting, which works well
    for the smooth sclera surface.

    Args:
        image: (H, W, 3) uint8 image with iris already warped.
        old_iris_mask: (H, W) uint8, 255 where iris was originally.
        new_iris_mask: (H, W) uint8, 255 where iris is now.
        eye_mask: (H, W) uint8, 255 inside eye aperture.

    Returns:
        (H, W, 3) uint8 image with exposed sclera filled.
    """
    # Exposed region: was iris, is not iris now, is within eye aperture
    exposed = (old_iris_mask > 0) & (new_iris_mask == 0) & (eye_mask > 0)
    inpaint_mask = exposed.astype(np.uint8) * 255

    # Dilate slightly to cover transition artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

    if inpaint_mask.sum() == 0:
        return image

    # Convert to BGR for OpenCV inpainting
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
