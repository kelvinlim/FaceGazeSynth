"""Tests for gaze rotation."""

import numpy as np
import pytest
from facegazesynth.eye_model.geometry import build_geometry
from facegazesynth.eye_model.rotation import rotate_eye, rotation_matrix


class TestRotationMatrix:
    def test_identity_at_zero(self):
        """Zero rotation should give identity."""
        R = rotation_matrix(np.array([0, 1, 0]), 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_deg_around_z(self):
        """90° around Z should map X→Y."""
        R = rotation_matrix(np.array([0, 0, 1]), np.pi / 2)
        result = R @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)

    def test_orthogonal(self):
        """Rotation matrix should be orthogonal."""
        R = rotation_matrix(np.array([1, 1, 1]), 0.7)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestRotateEye:
    def setup_method(self):
        self.geom = build_geometry()

    def test_zero_rotation_preserves_geometry(self):
        """No rotation should return identical geometry."""
        rotated = rotate_eye(self.geom, 0.0, 0.0)
        np.testing.assert_allclose(rotated.corneal_apex, self.geom.corneal_apex, atol=1e-10)
        np.testing.assert_allclose(rotated.iris_center, self.geom.iris_center, atol=1e-10)

    def test_rotation_center_unchanged(self):
        """Rotation center should not move when eye rotates."""
        rotated = rotate_eye(self.geom, 15.0, 0.0)
        np.testing.assert_allclose(rotated.rotation_center, self.geom.rotation_center, atol=1e-10)

    def test_horizontal_rotation_shifts_iris_x(self):
        """Horizontal gaze should shift the iris center in X."""
        rotated = rotate_eye(self.geom, 20.0, 0.0)
        # Iris should move in +X for rightward gaze
        assert rotated.iris_center[0] > self.geom.iris_center[0] + 0.1

    def test_distances_preserved(self):
        """Rotation should preserve distances between components."""
        rotated = rotate_eye(self.geom, 25.0, 10.0)

        dist_orig = np.linalg.norm(self.geom.corneal_apex - self.geom.sclera_center)
        dist_rot = np.linalg.norm(rotated.corneal_apex - rotated.sclera_center)
        assert pytest.approx(dist_orig, abs=1e-6) == dist_rot

    def test_sclera_radius_unchanged(self):
        rotated = rotate_eye(self.geom, 30.0, 0.0)
        assert rotated.sclera_radius == self.geom.sclera_radius
