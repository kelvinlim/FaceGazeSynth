"""Camera for ray generation."""

import numpy as np
from ..optics.rays import Ray, generate_ortho_rays


class OrthographicCamera:
    """Orthographic camera looking along -Z, centered at origin."""

    def __init__(
        self,
        viewport_width: float = 30.0,
        viewport_height: float = 30.0,
        resolution_x: int = 512,
        resolution_y: int = 512,
        camera_z: float = 100.0,
    ):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.camera_z = camera_z

    def generate_rays(self) -> Ray:
        return generate_ortho_rays(
            self.viewport_width,
            self.viewport_height,
            self.resolution_x,
            self.resolution_y,
            self.camera_z,
        )

    def mm_per_pixel(self) -> float:
        """Physical size of one pixel in mm."""
        return self.viewport_width / self.resolution_x
