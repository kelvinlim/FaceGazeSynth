"""Tests for the composite face+eye renderer."""

import numpy as np
import pytest

from facegazesynth.face_model.flame_mesh import build_face_mesh
from facegazesynth.face_model.composition import compose_face_with_eyes
from facegazesynth.rendering.camera import OrthographicCamera
from facegazesynth.rendering.composite_renderer import render_composite
from facegazesynth.rendering.lighting import PointLight


@pytest.fixture(scope="module")
def scene():
    face_mesh = build_face_mesh()
    return compose_face_with_eyes(face_mesh, theta_h_deg=0.0, theta_v_deg=0.0)


@pytest.fixture(scope="module")
def light():
    return PointLight(position=np.array([50.0, 80.0, 150.0]), intensity=0.75)


@pytest.fixture(scope="module")
def rendered(scene, light):
    """Render at low resolution for testing."""
    camera = OrthographicCamera(
        viewport_width=200.0,
        viewport_height=250.0,
        resolution_x=64,
        resolution_y=80,
    )
    rays = camera.generate_rays()
    colors = render_composite(
        scene, rays.origin, rays.direction, light=light
    )
    return colors.reshape(80, 64, 3)


class TestCompositeRenderer:
    def test_output_shape(self, rendered):
        """Output should be correct shape."""
        assert rendered.shape == (80, 64, 3)

    def test_output_range(self, rendered):
        """Colors should be in [0, 1]."""
        assert rendered.min() >= 0.0
        assert rendered.max() <= 1.0

    def test_not_all_background(self, rendered):
        """Should contain non-background pixels (face or eye content)."""
        bg = np.array([0.15, 0.15, 0.18])
        is_bg = np.all(np.abs(rendered - bg) < 0.02, axis=2)
        # At least 10% of pixels should be non-background
        assert is_bg.mean() < 0.90

    def test_has_skin_pixels(self, rendered):
        """Should contain warm-toned skin pixels."""
        # Skin is warm (R > G > B)
        warm = (rendered[:, :, 0] > 0.3) & (rendered[:, :, 0] > rendered[:, :, 2])
        assert warm.sum() > 10

    def test_different_gaze_produces_different_image(self, light):
        """Different gaze angles should produce different renders."""
        face_mesh = build_face_mesh()
        camera = OrthographicCamera(
            viewport_width=200.0, viewport_height=250.0,
            resolution_x=32, resolution_y=40,
        )
        rays = camera.generate_rays()

        scene_0 = compose_face_with_eyes(face_mesh, 0.0, 0.0)
        scene_20 = compose_face_with_eyes(face_mesh, 20.0, 0.0)

        img_0 = render_composite(scene_0, rays.origin, rays.direction, light=light)
        img_20 = render_composite(scene_20, rays.origin, rays.direction, light=light)

        assert not np.allclose(img_0, img_20, atol=0.01)
