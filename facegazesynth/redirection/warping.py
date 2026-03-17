"""Iris region warping for gaze redirection.

Uses an extract-inpaint-paste approach:
1. Inpaint the old iris region with sclera (clean slate)
2. Extract the iris patch from the original image
3. Paste the iris at the new position with foreshortening
All operations are clipped to the eyelid aperture.
"""

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


def _soft_iris_mask(center: np.ndarray, radius: float, shape: tuple,
                    feather: float = 0.15) -> np.ndarray:
    """Create a soft-edged circular mask for iris blending.

    Args:
        center: (2,) iris center (x, y).
        radius: Iris radius in pixels.
        shape: (H, W).
        feather: Fraction of radius for the soft edge (0-1).

    Returns:
        (H, W) float32 mask in [0, 1].
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    inner = radius * (1.0 - feather)
    alpha = np.clip((radius - dist) / max(radius * feather, 0.5), 0, 1)
    return alpha.astype(np.float32)


def warp_eye_region(
    image: np.ndarray,
    detection: EyeDetection,
    displacement_px: float,
    angle_deg: float,
) -> tuple:
    """Redirect gaze for one eye using extract-inpaint-paste.

    Steps:
    1. Inpaint the old iris region to create a clean sclera background
    2. Extract the iris+pupil patch from the original image
    3. Apply foreshortening (horizontal cos scaling)
    4. Paste at the new position, clipped to the eyelid aperture

    Args:
        image: (H, W, 3) uint8 RGB image.
        detection: Eye detection result.
        displacement_px: Horizontal pixel shift for iris center.
        angle_deg: Target gaze angle (for foreshortening).

    Returns:
        (warped_image, eye_mask_out, old_iris_mask, new_iris_mask)
    """
    h, w = image.shape[:2]
    old_cx, old_cy = detection.iris_center
    r = detection.iris_radius
    cos_a = np.cos(np.radians(abs(angle_deg)))
    new_cx = old_cx + displacement_px
    new_center = np.array([new_cx, old_cy])

    eye_m = eyelid_mask(detection, (h, w))
    old_iris_m = iris_mask(detection.iris_center, r, (h, w))
    new_iris_m = iris_mask(new_center, r * cos_a, (h, w))

    # --- Step 1: Inpaint old iris region to get clean sclera ---
    # Dilate slightly to cover limbus darkening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inpaint_region = cv2.dilate(old_iris_m, kernel, iterations=1)
    # Only inpaint within the eye aperture
    inpaint_region = inpaint_region & eye_m

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    clean_sclera = cv2.inpaint(bgr, inpaint_region, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    clean_sclera = cv2.cvtColor(clean_sclera, cv2.COLOR_BGR2RGB)

    # --- Step 2: Extract iris patch from original ---
    # Use a generous radius to capture the full iris + limbus
    extract_r = r * 1.15
    ex0 = max(0, int(old_cx - extract_r))
    ey0 = max(0, int(old_cy - extract_r))
    ex1 = min(w, int(old_cx + extract_r))
    ey1 = min(h, int(old_cy + extract_r))
    iris_patch = image[ey0:ey1, ex0:ex1].copy()
    patch_h, patch_w = iris_patch.shape[:2]

    # Create soft circular alpha for the extracted patch
    local_cx = old_cx - ex0
    local_cy = old_cy - ey0
    yy, xx = np.mgrid[0:patch_h, 0:patch_w]
    dist = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)
    inner = r * 0.92
    outer = extract_r
    patch_alpha = np.clip((outer - dist) / max(outer - inner, 0.5), 0, 1)
    patch_alpha = patch_alpha.astype(np.float32)

    # --- Step 3: Apply foreshortening ---
    if cos_a < 0.999:
        # Scale horizontally by cos(angle) around patch center
        M = np.array([
            [cos_a, 0, local_cx * (1 - cos_a)],
            [0, 1, 0],
        ], dtype=np.float32)
        iris_patch = cv2.warpAffine(
            iris_patch, M, (patch_w, patch_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # Also foreshorten the alpha mask
        patch_alpha = cv2.warpAffine(
            patch_alpha, M, (patch_w, patch_h),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

    # --- Step 4: Paste at new position, clipped to eyelid ---
    # Compute destination rectangle
    dx = displacement_px
    nx0 = max(0, int(ex0 + dx))
    ny0 = ey0
    nx1 = min(w, int(ex1 + dx))
    ny1 = ey1

    # Compute corresponding source region in the patch
    # (handles clipping at image edges)
    sx0 = nx0 - int(ex0 + dx)
    sy0 = 0
    sx1 = sx0 + (nx1 - nx0)
    sy1 = sy0 + (ny1 - ny0)

    if sx1 <= sx0 or sy1 <= sy0 or nx1 <= nx0 or ny1 <= ny0:
        return clean_sclera, eye_m, old_iris_m, new_iris_m

    src_patch = iris_patch[sy0:sy1, sx0:sx1]
    src_alpha = patch_alpha[sy0:sy1, sx0:sx1]

    # Clip alpha to eyelid aperture
    eyelid_region = eye_m[ny0:ny1, nx0:nx1].astype(np.float32) / 255.0
    src_alpha = src_alpha * eyelid_region

    # Composite onto clean sclera background
    result = clean_sclera.copy()
    alpha_3 = src_alpha[:, :, np.newaxis]
    dst_region = result[ny0:ny1, nx0:nx1].astype(np.float32)
    composited = src_patch.astype(np.float32) * alpha_3 + dst_region * (1 - alpha_3)
    result[ny0:ny1, nx0:nx1] = np.clip(composited, 0, 255).astype(np.uint8)

    return result, eye_m, old_iris_m, new_iris_m
