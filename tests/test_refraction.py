"""Tests for Snell's law refraction."""

import numpy as np
import pytest
from facegazesynth.optics.refraction import refract, fresnel_reflectance


class TestRefract:
    def test_normal_incidence_no_bending(self):
        """Ray hitting surface head-on should pass through undeviated."""
        incident = np.array([[0, 0, -1.0]])
        normal = np.array([[0, 0, 1.0]])
        refracted, valid = refract(incident, normal, 1.0, 1.376)

        assert valid[0]
        np.testing.assert_allclose(refracted[0], [0, 0, -1], atol=1e-10)

    def test_snell_angle(self):
        """Verify Snell's law: n1*sin(θ1) = n2*sin(θ2)."""
        theta1 = np.radians(30)
        incident = np.array([[np.sin(theta1), 0, -np.cos(theta1)]])
        normal = np.array([[0, 0, 1.0]])
        n1, n2 = 1.0, 1.376

        refracted, valid = refract(incident, normal, n1, n2)
        assert valid[0]

        # Compute transmitted angle
        sin_theta2 = np.sqrt(refracted[0, 0]**2 + refracted[0, 1]**2)
        # Snell: n1 * sin(theta1) = n2 * sin(theta2)
        assert pytest.approx(n1 * np.sin(theta1), abs=1e-6) == n2 * sin_theta2

    def test_total_internal_reflection(self):
        """Ray at steep angle from dense medium should TIR."""
        # Glass to air at 60° (beyond critical angle for n=1.5)
        theta = np.radians(60)
        incident = np.array([[np.sin(theta), 0, -np.cos(theta)]])
        normal = np.array([[0, 0, 1.0]])

        refracted, valid = refract(incident, normal, 1.5, 1.0)
        assert not valid[0]
        assert np.all(np.isnan(refracted[0]))

    def test_symmetric_refraction(self):
        """Refracting in then out should approximately recover original direction."""
        theta1 = np.radians(20)
        incident = np.array([[np.sin(theta1), 0, -np.cos(theta1)]])
        normal_in = np.array([[0, 0, 1.0]])

        # Air → glass
        refracted, _ = refract(incident, normal_in, 1.0, 1.5)
        # Glass → air (normal flipped for exit surface)
        normal_out = np.array([[0, 0, -1.0]])
        recovered, valid = refract(refracted, normal_out, 1.5, 1.0)

        assert valid[0]
        np.testing.assert_allclose(recovered[0], incident[0], atol=1e-6)

    def test_batch(self):
        """Multiple rays processed correctly."""
        incidents = np.array([
            [0, 0, -1],
            [0.3, 0, -np.sqrt(1 - 0.3**2)],
        ], dtype=float)
        normals = np.array([[0, 0, 1.0], [0, 0, 1.0]])

        refracted, valid = refract(incidents, normals, 1.0, 1.376)
        assert np.all(valid)
        # First ray undeviated
        np.testing.assert_allclose(refracted[0], [0, 0, -1], atol=1e-10)
        # Second ray should bend toward normal (smaller angle)
        assert abs(refracted[1, 0]) < abs(incidents[1, 0])


class TestFresnelReflectance:
    def test_normal_incidence(self):
        """At normal incidence, reflectance = ((n1-n2)/(n1+n2))^2."""
        r = fresnel_reflectance(np.array([1.0]), 1.0, 1.376)
        expected = ((1.0 - 1.376) / (1.0 + 1.376))**2
        assert pytest.approx(r[0], abs=1e-6) == expected

    def test_grazing_incidence(self):
        """At grazing incidence (cos_i → 0), reflectance → 1."""
        r = fresnel_reflectance(np.array([0.01]), 1.0, 1.5)
        assert r[0] > 0.9
