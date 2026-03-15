"""Ray representation and camera ray generation."""

import numpy as np
from dataclasses import dataclass


@dataclass
class Ray:
    """A ray defined by origin and direction.

    For batch operations, origin and direction can be (N, 3) arrays.
    """
    origin: np.ndarray     # (3,) or (N, 3)
    direction: np.ndarray  # (3,) unit vector, or (N, 3) unit vectors


def generate_ortho_rays(
    viewport_width: float,
    viewport_height: float,
    resolution_x: int,
    resolution_y: int,
    camera_z: float = 100.0,
) -> Ray:
    """Generate orthographic camera rays looking along -Z.

    Args:
        viewport_width: Physical width of viewport in mm.
        viewport_height: Physical height of viewport in mm.
        resolution_x: Image width in pixels.
        resolution_y: Image height in pixels.
        camera_z: Z position of the camera plane (far in front of eye).

    Returns:
        Ray with origin (N, 3) and direction (N, 3) where N = res_x * res_y.
        Pixel ordering is row-major (top-left to bottom-right).
    """
    # Pixel centers in normalized [0, 1] coordinates
    u = (np.arange(resolution_x) + 0.5) / resolution_x
    v = (np.arange(resolution_y) + 0.5) / resolution_y

    # Map to physical coordinates centered at (0, 0)
    # u goes left-to-right → x; v goes top-to-bottom → -y
    x = (u - 0.5) * viewport_width
    y = (0.5 - v) * viewport_height

    # Create grid
    xx, yy = np.meshgrid(x, y)  # both (res_y, res_x)
    xx = xx.ravel()
    yy = yy.ravel()
    n = len(xx)

    origins = np.column_stack([xx, yy, np.full(n, camera_z)])
    directions = np.column_stack([np.zeros(n), np.zeros(n), np.full(n, -1.0)])

    return Ray(origin=origins, direction=directions)
