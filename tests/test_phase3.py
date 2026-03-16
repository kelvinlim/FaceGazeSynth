"""Tests for Phase 3: diverse faces with emotions."""

import numpy as np
import pytest

from facegazesynth.materials.albedo import (
    load_albedo_model,
    sample_albedo_texture,
    lookup_albedo_at_triangles,
)
from facegazesynth.face_model.expressions import (
    get_emotion_params,
    list_emotions,
    random_identity,
    EMOTION_PRESETS,
)
from facegazesynth.face_model.flame_mesh import build_face_mesh
from facegazesynth.face_model.composition import compose_face_with_eyes
from facegazesynth.rendering.camera import OrthographicCamera
from facegazesynth.rendering.composite_renderer import render_composite
from facegazesynth.rendering.lighting import PointLight


class TestAlbedoModel:
    @pytest.fixture(scope="class")
    def albedo_model(self):
        return load_albedo_model()

    def test_load(self, albedo_model):
        assert albedo_model.mean.shape == (512, 512, 3)
        assert albedo_model.components.shape[:3] == (512, 512, 3)
        assert albedo_model.n_components == 145

    def test_uv_coords(self, albedo_model):
        assert albedo_model.uv_coords.shape[1] == 2
        assert albedo_model.uv_coords.min() >= 0.0
        assert albedo_model.uv_coords.max() <= 1.0

    def test_sample_default(self, albedo_model):
        tex = sample_albedo_texture(seed=42, albedo_model=albedo_model)
        assert tex.shape == (512, 512, 3)
        assert tex.min() >= 0.0
        assert tex.max() <= 1.0

    def test_sample_reproducible(self, albedo_model):
        t1 = sample_albedo_texture(seed=42, albedo_model=albedo_model)
        t2 = sample_albedo_texture(seed=42, albedo_model=albedo_model)
        np.testing.assert_array_equal(t1, t2)

    def test_sample_different_seeds(self, albedo_model):
        t1 = sample_albedo_texture(seed=0, albedo_model=albedo_model)
        t2 = sample_albedo_texture(seed=99, albedo_model=albedo_model)
        assert not np.allclose(t1, t2)

    def test_lookup_at_triangles(self, albedo_model):
        tex = sample_albedo_texture(seed=42, albedo_model=albedo_model)
        tri_idx = np.array([0, 100, 500])
        bary = np.array([[0.5, 0.3, 0.2], [0.3, 0.3, 0.4], [0.1, 0.8, 0.1]])
        colors = lookup_albedo_at_triangles(tex, tri_idx, bary, albedo_model)
        assert colors.shape == (3, 3)
        assert np.all(colors >= 0.0) and np.all(colors <= 1.0)


class TestEmotionPresets:
    def test_list_emotions(self):
        emotions = list_emotions()
        assert "neutral" in emotions
        assert "happy" in emotions
        assert len(emotions) >= 7

    def test_get_neutral(self):
        expr, jaw = get_emotion_params("neutral")
        np.testing.assert_array_equal(expr, np.zeros(10))
        np.testing.assert_array_equal(jaw, np.zeros(3))

    def test_get_happy(self):
        expr, jaw = get_emotion_params("happy")
        assert expr.shape == (10,)
        assert jaw.shape == (3,)
        assert np.any(expr != 0)

    def test_intensity_scaling(self):
        expr_full, _ = get_emotion_params("happy", intensity=1.0)
        expr_half, _ = get_emotion_params("happy", intensity=0.5)
        np.testing.assert_allclose(expr_half, expr_full * 0.5)

    def test_unknown_emotion_raises(self):
        with pytest.raises(ValueError, match="Unknown emotion"):
            get_emotion_params("nonexistent_emotion")

    def test_random_identity(self):
        b1 = random_identity(seed=0)
        b2 = random_identity(seed=0)
        b3 = random_identity(seed=1)
        np.testing.assert_array_equal(b1, b2)
        assert not np.allclose(b1, b3)

    def test_all_presets_valid_shapes(self):
        for emotion in EMOTION_PRESETS:
            expr, jaw = get_emotion_params(emotion)
            assert expr.shape == (10,), f"{emotion} expression wrong shape"
            assert jaw.shape == (3,), f"{emotion} jaw wrong shape"


class TestAlbedoRendering:
    def test_render_with_albedo(self):
        """Render with albedo texture produces different result than procedural."""
        face_mesh = build_face_mesh()
        scene = compose_face_with_eyes(face_mesh, 0.0, 0.0)
        light = PointLight(position=np.array([50.0, 80.0, 150.0]), intensity=0.75)
        camera = OrthographicCamera(
            viewport_width=200.0, viewport_height=250.0,
            resolution_x=32, resolution_y=40,
        )
        rays = camera.generate_rays()

        # Procedural
        img_proc = render_composite(
            scene, rays.origin, rays.direction, light=light,
        )
        # Albedo
        albedo_model = load_albedo_model()
        tex = sample_albedo_texture(seed=42, albedo_model=albedo_model)
        img_alb = render_composite(
            scene, rays.origin, rays.direction, light=light,
            albedo_texture=tex, albedo_model=albedo_model,
        )

        assert not np.allclose(img_proc, img_alb, atol=0.01)

    def test_expression_changes_mesh(self):
        """Different emotions produce different face geometry."""
        mesh_neutral = build_face_mesh()
        expr_happy, jaw_happy = get_emotion_params("happy")
        mesh_happy = build_face_mesh(expression=expr_happy, jaw_pose=jaw_happy)
        assert not np.allclose(
            mesh_neutral.vertices, mesh_happy.vertices, atol=0.1
        )
