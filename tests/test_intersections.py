"""Tests for ray-geometry intersection routines."""

import numpy as np
import pytest
from facegazesynth.optics.intersections import (
    intersect_ray_sphere,
    intersect_ray_plane,
    is_within_cornea_cap,
)


class TestRaySphere:
    def _single_ray(self, origin, direction):
        """Helper: wrap a single ray as (1, 3) arrays."""
        return np.array([origin]), np.array([direction])

    def test_hit_along_z_axis(self):
        """Ray along -Z hits a sphere at origin."""
        origin, direction = self._single_ray([0, 0, 50], [0, 0, -1])
        t, hit, normal, mask = intersect_ray_sphere(origin, direction, np.zeros(3), 10.0)

        assert mask[0]
        assert pytest.approx(t[0], abs=1e-6) == 40.0  # 50 - 10
        assert pytest.approx(hit[0, 2], abs=1e-6) == 10.0
        # Normal should point toward +Z at the front of sphere
        assert pytest.approx(normal[0, 2], abs=1e-6) == 1.0

    def test_miss(self):
        """Ray that misses the sphere returns NaN and False mask."""
        origin, direction = self._single_ray([100, 0, 50], [0, 0, -1])
        t, hit, normal, mask = intersect_ray_sphere(origin, direction, np.zeros(3), 10.0)

        assert not mask[0]
        assert np.isnan(t[0])

    def test_tangent_ray(self):
        """Ray tangent to sphere should barely hit (discriminant ≈ 0)."""
        origin, direction = self._single_ray([10, 0, 50], [0, 0, -1])
        t, hit, normal, mask = intersect_ray_sphere(origin, direction, np.zeros(3), 10.0)

        assert mask[0]
        # Tangent hit is at z=0
        assert pytest.approx(hit[0, 2], abs=0.01) == 0.0

    def test_ray_inside_sphere(self):
        """Ray originating inside sphere hits the far surface."""
        origin, direction = self._single_ray([0, 0, 0], [0, 0, -1])
        t, hit, normal, mask = intersect_ray_sphere(origin, direction, np.zeros(3), 10.0)

        assert mask[0]
        assert t[0] > 0  # hits far side
        assert pytest.approx(hit[0, 2], abs=1e-6) == -10.0

    def test_batch(self):
        """Multiple rays processed correctly."""
        origins = np.array([[0, 0, 50], [100, 0, 50], [0, 5, 50]])
        dirs = np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1]])
        t, hit, normal, mask = intersect_ray_sphere(origins, dirs, np.zeros(3), 10.0)

        assert mask[0] and not mask[1] and mask[2]


class TestRayPlane:
    def test_hit(self):
        """Ray hits XY plane at z=5."""
        origins = np.array([[0.0, 0.0, 50.0]])
        dirs = np.array([[0.0, 0.0, -1.0]])
        t, hit, mask = intersect_ray_plane(origins, dirs, np.array([0, 0, 5.0]), np.array([0, 0, 1.0]))

        assert mask[0]
        assert pytest.approx(t[0], abs=1e-6) == 45.0
        assert pytest.approx(hit[0, 2], abs=1e-6) == 5.0

    def test_parallel_miss(self):
        """Ray parallel to plane misses."""
        origins = np.array([[0.0, 0.0, 50.0]])
        dirs = np.array([[1.0, 0.0, 0.0]])
        t, hit, mask = intersect_ray_plane(origins, dirs, np.array([0, 0, 5.0]), np.array([0, 0, 1.0]))

        assert not mask[0]

    def test_behind_ray(self):
        """Plane behind ray origin: no hit."""
        origins = np.array([[0.0, 0.0, 2.0]])
        dirs = np.array([[0.0, 0.0, -1.0]])
        t, hit, mask = intersect_ray_plane(origins, dirs, np.array([0, 0, 5.0]), np.array([0, 0, 1.0]))

        # Ray at z=2 going -Z, plane at z=5 is behind → no hit
        assert not mask[0]


class TestCorneaCap:
    def test_apex_is_in_cap(self):
        """The corneal apex (front of cap) should be within the cap."""
        from facegazesynth.eye_model.geometry import build_geometry

        geom = build_geometry()
        # Apex is on the cornea sphere, at the front
        apex = geom.corneal_apex.reshape(1, 3)
        result = is_within_cornea_cap(
            apex, geom.cornea_center, geom.cornea_radius, geom.limbus_half_angle
        )
        assert result[0]

    def test_back_of_cornea_sphere_is_outside_cap(self):
        """A point on the back of the cornea sphere should NOT be in the cap."""
        from facegazesynth.eye_model.geometry import build_geometry

        geom = build_geometry()
        # Back of cornea sphere is at cornea_center - (0, 0, R_c)
        back = (geom.cornea_center - np.array([0, 0, geom.cornea_radius])).reshape(1, 3)
        result = is_within_cornea_cap(
            back, geom.cornea_center, geom.cornea_radius, geom.limbus_half_angle
        )
        assert not result[0]
