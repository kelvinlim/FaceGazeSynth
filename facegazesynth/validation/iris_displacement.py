"""Measure apparent iris center from rendered images.

Detects the pupil (darkest region within the eye) and computes its
centroid to determine the apparent iris displacement.
"""

import numpy as np
from PIL import Image


def measure_iris_displacement(
    image: Image.Image,
    viewport_width_mm: float = 30.0,
) -> tuple[float, float]:
    """Measure the apparent iris/pupil center displacement from image center.

    Finds the pupil as the darkest region that is surrounded by brighter
    pixels (to distinguish from the dark background). Returns displacement
    from image center in mm.

    Args:
        image: Rendered eye image (single eye, square).
        viewport_width_mm: Physical viewport width in mm.

    Returns:
        (dx_mm, dy_mm): Horizontal and vertical displacement of pupil
        center from image center, in mm.
    """
    img = np.array(image.convert("L"), dtype=float)
    h, w = img.shape

    # Step 1: Find the eye region (sclera + iris + pupil, all brighter than background)
    # Background is ~38 in flat shading mode. The eye surface is always > 50.
    # Use a flood-fill-like approach: the eye is the large bright blob in the center.
    eye_mask = img > 50  # sclera and iris are brighter than 50

    if not np.any(eye_mask):
        return 0.0, 0.0

    # Step 2: The pupil is the dark region INSIDE the eye.
    # Find pixels that are very dark (< 10) and are surrounded by eye pixels.
    pupil_candidate = img < 10

    # Filter: a true pupil pixel should have bright neighbors (iris/sclera)
    # Use the eye_mask to confirm the pupil is inside the eye.
    # Dilate the pupil candidate and check overlap with eye_mask.
    # Simple approach: find dark pixels whose neighbors include eye pixels.
    from scipy import ndimage

    # Dilate pupil candidates to get their neighborhood
    dilated = ndimage.binary_dilation(pupil_candidate, iterations=3)
    # Pupil pixels are those that are dark AND whose dilated region overlaps the eye
    pupil_near_eye = pupil_candidate & dilated & ndimage.binary_dilation(eye_mask, iterations=3)

    # Label connected components and keep the one nearest image center
    labeled, n_labels = ndimage.label(pupil_near_eye)
    if n_labels == 0:
        return 0.0, 0.0

    # Find the component closest to the eye center (centroid of eye_mask)
    eye_cy, eye_cx = ndimage.center_of_mass(eye_mask)
    best_label = 1
    best_dist = float("inf")
    for label_id in range(1, n_labels + 1):
        cy, cx = ndimage.center_of_mass(labeled == label_id)
        dist = (cx - eye_cx)**2 + (cy - eye_cy)**2
        if dist < best_dist:
            best_dist = dist
            best_label = label_id

    pupil_mask = labeled == best_label
    ys, xs = np.where(pupil_mask)
    cx = np.mean(xs)
    cy = np.mean(ys)

    # Convert from pixel coords to mm
    mm_per_pixel = viewport_width_mm / w
    dx_mm = (cx - w / 2) * mm_per_pixel
    dy_mm = (cy - h / 2) * mm_per_pixel

    return dx_mm, dy_mm


def measure_iris_diameter(
    image: Image.Image,
    viewport_width_mm: float = 30.0,
    sclera_threshold: float = 180,
) -> float:
    """Measure the apparent iris diameter in mm.

    Finds the iris region (between pupil dark and sclera bright)
    and measures its horizontal extent.

    Args:
        image: Rendered eye image (single eye, square, 0° gaze).
        viewport_width_mm: Physical viewport width in mm.
        sclera_threshold: Min brightness to consider as sclera.

    Returns:
        Apparent iris diameter in mm.
    """
    img = np.array(image.convert("L"), dtype=float)
    h, w = img.shape
    mm_per_pixel = viewport_width_mm / w

    # Take horizontal slice through center
    center_row = img[h // 2, :]

    # Iris region: brighter than background (>50), darker than sclera
    iris_pixels = (center_row > 50) & (center_row < sclera_threshold)

    if not np.any(iris_pixels):
        return 0.0

    xs = np.where(iris_pixels)[0]
    diameter_pixels = xs[-1] - xs[0]
    return diameter_pixels * mm_per_pixel
