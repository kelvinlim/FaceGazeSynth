"""Iris region warping for gaze redirection."""

import cv2
import numpy as np

from .detection import EyeDetection


def eyelid_mask(detection: EyeDetection, shape: tuple) -> np.ndarray:
    """Create a binary mask of the eye aperture from eyelid contours.

    Args:
        detection: Eye detection with eyelid landmarks.
        shape: (H, W) of the image.

    Returns:
        (H, W) uint8 mask, 255 inside eye aperture.
    """
    upper = detection.eyelid_upper
    lower = detection.eyelid_lower[::-1]
    polygon = np.vstack([upper, lower]).astype(np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def iris_mask(center: np.ndarray, radius: float, shape: tuple) -> np.ndarray:
    """Create a circular mask for the iris region."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
    return mask


def warp_eye_region(
    image: np.ndarray,
    detection: EyeDetection,
    displacement_px: float,
    angle_deg: float,
) -> tuple:
    """Warp the iris region to redirect gaze.

    For each output pixel near the iris, we compute where to sample from
    the original image by undoing the displacement and foreshortening.

    The iris moves from old_center to new_center = old_center + displacement.
    Foreshortening compresses the iris horizontally by cos(angle) around
    its new center.

    Args:
        image: (H, W, 3) uint8 RGB image.
        detection: Eye detection result.
        displacement_px: Horizontal pixel shift for iris center.
        angle_deg: Target gaze angle (for foreshortening).

    Returns:
        (warped_image, eye_mask_out, old_iris_mask, new_iris_mask)
    """
    h, w = image.shape[:2]
    result = image.copy()

    old_cx, old_cy = detection.iris_center
    r = detection.iris_radius
    cos_a = np.cos(np.radians(abs(angle_deg)))
    new_cx = old_cx + displacement_px

    # Work region: generous bounding box around both old and new iris positions
    margin = 1.5
    patch_r = int(r * margin)
    region_cx = (old_cx + new_cx) / 2
    half_w = patch_r + abs(displacement_px) / 2 + r * 0.5
    x0 = max(0, int(region_cx - half_w))
    y0 = max(0, int(old_cy - patch_r))
    x1 = min(w, int(region_cx + half_w))
    y1 = min(h, int(old_cy + patch_r))
    patch_h, patch_w = y1 - y0, x1 - x0

    # Build sampling map: for each destination pixel, find source in original.
    # Destination pixel (dx, dy) in image coords.
    # Position relative to new iris center: rx = dx - new_cx, ry = dy - new_cy
    # Undo foreshortening (stretch x by 1/cos) to get position relative to old center:
    #   src_x = rx / cos_a + old_cx
    #   src_y = ry + old_cy
    yy = np.arange(y0, y1, dtype=np.float32)
    xx = np.arange(x0, x1, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    # Relative to new iris center
    rx = grid_x - new_cx
    ry = grid_y - old_cy  # cy doesn't change

    # Undo foreshortening to get source coordinates
    inv_cos = 1.0 / max(cos_a, 0.5)
    src_x = (rx * inv_cos + old_cx).astype(np.float32)
    src_y = (ry + old_cy).astype(np.float32)

    # Remap
    warped_patch = cv2.remap(
        image, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Soft blending mask: 1.0 near new iris center, fading out
    dist_from_new = np.sqrt(rx ** 2 + ry ** 2)
    inner_r = r * 0.85
    outer_r = r * margin
    alpha = np.clip((outer_r - dist_from_new) / (outer_r - inner_r), 0, 1)
    alpha = (alpha ** 2).astype(np.float32)

    # Blend into result
    alpha_3 = alpha[:, :, np.newaxis]
    region = result[y0:y1, x0:x1].astype(np.float32)
    blended = warped_patch.astype(np.float32) * alpha_3 + region * (1 - alpha_3)
    result[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

    # Masks for downstream steps
    eye_m = eyelid_mask(detection, (h, w))
    old_iris_m = iris_mask(detection.iris_center, r, (h, w))
    new_center = np.array([new_cx, old_cy])
    new_iris_m = iris_mask(new_center, r * cos_a, (h, w))

    return result, eye_m, old_iris_m, new_iris_m
