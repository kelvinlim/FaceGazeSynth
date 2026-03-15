"""Tests for eyeball geometry construction."""

import numpy as np
import pytest
from facegazesynth.eye_model.geometry import build_geometry
from facegazesynth.eye_model.parameters import EyeParameters


class TestBuildGeometry:
    def setup_method(self):
        self.geom = build_geometry()
        self.params = EyeParameters()

    def test_sclera_at_origin(self):
        np.testing.assert_array_equal(self.geom.sclera_center, [0, 0, 0])

    def test_corneal_apex_position(self):
        """Corneal apex should be at sclera_radius + protrusion along +Z."""
        expected_z = self.params.sclera_radius + self.params.cornea_protrusion
        assert pytest.approx(self.geom.corneal_apex[2], abs=1e-6) == expected_z
        assert self.geom.corneal_apex[0] == 0.0
        assert self.geom.corneal_apex[1] == 0.0

    def test_cornea_center_behind_apex(self):
        """Cornea center is R_c behind the apex."""
        expected_z = self.geom.corneal_apex[2] - self.params.cornea_radius_of_curvature
        assert pytest.approx(self.geom.cornea_center[2], abs=1e-6) == expected_z

    def test_cornea_sphere_passes_through_apex(self):
        """The corneal apex should lie on the cornea sphere."""
        dist = np.linalg.norm(self.geom.corneal_apex - self.geom.cornea_center)
        assert pytest.approx(dist, abs=1e-6) == self.geom.cornea_radius

    def test_limbus_circle_on_both_spheres(self):
        """The limbus circle should lie on both the sclera and cornea spheres."""
        # A point on the limbus: at limbus_half_angle from cornea axis
        r_limbus = self.geom.cornea_radius * np.sin(self.geom.limbus_half_angle)
        z_limbus = self.geom.cornea_center[2] + self.geom.cornea_radius * np.cos(self.geom.limbus_half_angle)
        limbus_point = np.array([r_limbus, 0, z_limbus])

        # Should be on sclera sphere
        dist_sclera = np.linalg.norm(limbus_point - self.geom.sclera_center)
        assert pytest.approx(dist_sclera, abs=1e-4) == self.geom.sclera_radius

        # Should be on cornea sphere
        dist_cornea = np.linalg.norm(limbus_point - self.geom.cornea_center)
        assert pytest.approx(dist_cornea, abs=1e-4) == self.geom.cornea_radius

    def test_iris_behind_cornea(self):
        """Iris plane should be behind the corneal apex."""
        assert self.geom.iris_center[2] < self.geom.corneal_apex[2]
        expected_z = self.geom.corneal_apex[2] - self.params.iris_setback
        assert pytest.approx(self.geom.iris_center[2], abs=1e-6) == expected_z

    def test_iris_inside_sclera(self):
        """Iris should be inside the sclera sphere."""
        dist = np.linalg.norm(self.geom.iris_center - self.geom.sclera_center)
        assert dist < self.geom.sclera_radius

    def test_rotation_center_behind_apex(self):
        """Rotation center should be 13.5mm behind the corneal apex."""
        expected_z = self.geom.corneal_apex[2] - self.params.rotation_center_depth
        assert pytest.approx(self.geom.rotation_center[2], abs=1e-6) == expected_z
