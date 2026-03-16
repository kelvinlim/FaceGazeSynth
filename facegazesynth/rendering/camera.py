"""Camera for ray generation: orthographic and perspective."""

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


class PerspectiveCamera:
    """Perspective camera looking along -Z from a point source.

    Rays diverge from the camera position through a virtual sensor plane.
    More realistic for face photography than orthographic projection.
    """

    def __init__(
        self,
        focal_length: float = 50.0,
        sensor_width: float = 36.0,
        sensor_height: float = 36.0,
        resolution_x: int = 512,
        resolution_y: int = 512,
        camera_distance: float = 500.0,
    ):
        """
        Args:
            focal_length: Lens focal length in mm.
            sensor_width: Sensor width in mm (36mm = full-frame).
            sensor_height: Sensor height in mm.
            resolution_x: Image width in pixels.
            resolution_y: Image height in pixels.
            camera_distance: Distance from camera to world origin in mm.
        """
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.camera_distance = camera_distance

    def generate_rays(self) -> Ray:
        """Generate perspective rays diverging from camera position."""
        # Camera sits at (0, 0, camera_distance), looking along -Z
        cam_pos = np.array([0.0, 0.0, self.camera_distance])

        # Pixel centers on sensor plane (at z = camera_distance - focal_length)
        u = (np.arange(self.resolution_x) + 0.5) / self.resolution_x
        v = (np.arange(self.resolution_y) + 0.5) / self.resolution_y

        # Sensor coordinates centered at optical axis
        sx = (u - 0.5) * self.sensor_width
        sy = (0.5 - v) * self.sensor_height

        xx, yy = np.meshgrid(sx, sy)
        xx = xx.ravel()
        yy = yy.ravel()
        n = len(xx)

        # Points on the sensor plane
        sensor_z = self.camera_distance - self.focal_length
        sensor_points = np.column_stack([xx, yy, np.full(n, sensor_z)])

        # Ray directions: from camera through sensor points, normalized
        directions = sensor_points - cam_pos
        lengths = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= lengths

        origins = np.tile(cam_pos, (n, 1))
        return Ray(origin=origins, direction=directions)

    def mm_per_pixel(self) -> float:
        """Approximate physical pixel size at the world origin (z=0)."""
        # Field of view at z=0: sensor projects to (distance/focal_length) * sensor_size
        fov_width = (self.camera_distance / self.focal_length) * self.sensor_width
        return fov_width / self.resolution_x
