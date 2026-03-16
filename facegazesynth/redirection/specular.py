"""Corneal specular highlight (Purkinje image) repositioning."""

import cv2
import numpy as np

from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS
from .detection import EyeDetection
from .physics_mapping import GazeMapping


def _find_specular(image: np.ndarray, detection: EyeDetection,
                   eye_mask: np.ndarray) -> tuple:
    """Find the corneal specular highlight in the eye region.

    Returns:
        (center_xy, radius, brightness) or None if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Restrict search to eye aperture near iris
    cx, cy = int(detection.iris_center[0]), int(detection.iris_center[1])
    r = int(detection.iris_radius * 1.3)
    search_mask = np.zeros_like(gray)
    cv2.circle(search_mask, (cx, cy), r, 255, -1)
    search_mask = search_mask & eye_mask

    masked = gray.copy()
    masked[search_mask == 0] = 0

    # Find brightest region
    threshold = max(200, np.percentile(gray[search_mask > 0], 98))
    bright = (masked >= threshold).astype(np.uint8) * 255

    # Find connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright)
    if n_labels < 2:  # only background
        return None

    # Pick the brightest, smallest component (specular is small and bright)
    best = None
    best_score = -1
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1 or area > detection.iris_radius ** 2:
            continue
        # Score: brightness / area (we want bright and small)
        region_brightness = np.mean(gray[labels == i])
        score = region_brightness / max(area, 1)
        if score > best_score:
            best_score = score
            best = i

    if best is None:
        return None

    center = centroids[best]
    area = stats[best, cv2.CC_STAT_AREA]
    radius = max(1, int(np.sqrt(area / np.pi)))
    brightness = np.mean(gray[labels == best])

    return (np.array(center), radius, brightness)


def reposition_specular(
    image: np.ndarray,
    detection: EyeDetection,
    angle_deg: float,
    mapping: GazeMapping,
    eye_mask: np.ndarray,
    params: EyeParameters = DEFAULT_PARAMS,
) -> np.ndarray:
    """Move the corneal specular highlight to match redirected gaze.

    The Purkinje image shifts approximately by cornea_radius * sin(theta) / 2
    in the direction of gaze (reflection geometry).

    Args:
        image: (H, W, 3) uint8 RGB image with iris already warped.
        detection: Original eye detection.
        angle_deg: Target gaze angle in degrees.
        mapping: Calibrated pixel mapping.
        eye_mask: (H, W) eye aperture mask.
        params: Eye parameters.

    Returns:
        (H, W, 3) uint8 image with repositioned specular highlight.
    """
    spec = _find_specular(image, detection, eye_mask)
    if spec is None:
        return image  # no highlight found, skip

    old_center, radius, brightness = spec
    result = image.copy()

    # Compute highlight shift: approximately cornea_radius * sin(theta) / 2
    # converted to pixels via mm_per_pixel
    shift_mm = params.cornea_radius_of_curvature * np.sin(np.radians(angle_deg)) / 2
    shift_px = shift_mm / mapping.mm_per_pixel

    new_center = old_center.copy()
    new_center[0] += shift_px

    # Remove old highlight: inpaint the small region
    remove_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(remove_mask, (int(old_center[0]), int(old_center[1])),
               radius + 2, 255, -1)
    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    bgr = cv2.inpaint(bgr, remove_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    result = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Add new highlight at computed position
    ncx, ncy = int(new_center[0]), int(new_center[1])
    if 0 <= ncx < image.shape[1] and 0 <= ncy < image.shape[0]:
        # Draw a small bright Gaussian spot
        for c in range(3):
            highlight = np.zeros(image.shape[:2], dtype=np.float32)
            cv2.circle(highlight, (ncx, ncy), max(1, radius), 1.0, -1)
            highlight = cv2.GaussianBlur(highlight, (0, 0), max(0.5, radius * 0.4))
            highlight *= min(255, brightness * 1.1)
            channel = result[:, :, c].astype(np.float32)
            channel = np.maximum(channel, highlight)
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    return result
