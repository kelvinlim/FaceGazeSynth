"""Tests for FLAME face model loading and eye-face composition."""

import numpy as np
import pytest

from facegazesynth.face_model.flame_mesh import build_face_mesh
from facegazesynth.face_model.composition import compose_face_with_eyes


class TestFlameMeshLoader:
    """Test FLAME mesh loading and processing."""

    @pytest.fixture(scope="class")
    def face_mesh(self):
        return build_face_mesh()

    def test_vertex_count_after_eyeball_removal(self, face_mesh):
        """Face mesh should have ~3931 vertices (eyeball submeshes removed)."""
        assert 3800 <= face_mesh.vertices.shape[0] <= 4200
        assert face_mesh.vertices.shape[1] == 3

    def test_face_count(self, face_mesh):
        """Face mesh should have fewer faces than original (~9976)."""
        assert face_mesh.faces.shape[0] < 9976
        assert face_mesh.faces.shape[1] == 3

    def test_vertex_normals_shape(self, face_mesh):
        """Normals should match vertex count."""
        assert face_mesh.vertex_normals.shape == face_mesh.vertices.shape

    def test_vertex_normals_unit_length(self, face_mesh):
        """Normals should be approximately unit length."""
        lengths = np.linalg.norm(face_mesh.vertex_normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=0.01)

    def test_coordinates_in_mm(self, face_mesh):
        """Vertices should be in mm (face ~200mm wide)."""
        x_range = face_mesh.vertices[:, 0].max() - face_mesh.vertices[:, 0].min()
        assert 100 < x_range < 300  # face width in mm

    def test_eye_joints_extracted(self, face_mesh):
        """Eye joints should be valid 3D points."""
        assert face_mesh.left_eye_joint.shape == (3,)
        assert face_mesh.right_eye_joint.shape == (3,)

    def test_eye_joints_symmetric(self, face_mesh):
        """Left and right eye joints should be roughly symmetric in X."""
        # Right eye should have positive X, left eye negative X
        assert face_mesh.right_eye_joint[0] > 0
        assert face_mesh.left_eye_joint[0] < 0
        # Y and Z should be similar
        np.testing.assert_allclose(
            face_mesh.left_eye_joint[1], face_mesh.right_eye_joint[1], atol=2.0
        )

    def test_eye_joints_reasonable_position(self, face_mesh):
        """Eye joints should be in the upper face region."""
        # IPD should be roughly 50-80mm
        ipd = abs(face_mesh.right_eye_joint[0] - face_mesh.left_eye_joint[0])
        assert 40 < ipd < 90

    def test_identity_variation(self):
        """Different betas should produce different meshes."""
        mesh_neutral = build_face_mesh()
        mesh_varied = build_face_mesh(betas=np.array([3.0, -2.0, 1.0]))
        # Vertices should differ
        assert not np.allclose(mesh_neutral.vertices, mesh_varied.vertices, atol=0.1)

    def test_expression_variation(self):
        """Different expressions should produce different meshes."""
        mesh_neutral = build_face_mesh()
        mesh_smile = build_face_mesh(expression=np.array([2.0, 0.0, 0.0, 1.0]))
        assert not np.allclose(mesh_neutral.vertices, mesh_smile.vertices, atol=0.1)


class TestComposition:
    """Test eye-face composition."""

    @pytest.fixture(scope="class")
    def scene(self):
        face_mesh = build_face_mesh()
        return compose_face_with_eyes(face_mesh, theta_h_deg=0.0, theta_v_deg=0.0)

    def test_scene_has_trimesh(self, scene):
        """Scene should contain a trimesh for ray intersection."""
        assert scene.face_trimesh is not None
        assert len(scene.face_trimesh.vertices) > 0

    def test_scene_has_both_eyes(self, scene):
        """Scene should have left and right eye geometries."""
        assert scene.left_eye is not None
        assert scene.right_eye is not None

    def test_eyes_positioned_at_joints(self, scene):
        """Eyes should be positioned near the FLAME eye joint locations."""
        face_mesh = build_face_mesh()
        # Eye sclera centers should be near the joint positions
        np.testing.assert_allclose(
            scene.right_eye.sclera_center, face_mesh.right_eye_joint, atol=15.0
        )
        np.testing.assert_allclose(
            scene.left_eye.sclera_center, face_mesh.left_eye_joint, atol=15.0
        )

    def test_left_eye_mirrored(self, scene):
        """Left eye iris normal should point roughly opposite X vs right."""
        # Both eyes should look forward (positive Z component in iris normal)
        # but left eye's cornea should be mirrored
        assert scene.left_eye.cornea_center[0] < 0  # left side
        assert scene.right_eye.cornea_center[0] > 0  # right side

    def test_gaze_rotation(self):
        """Rotated gaze should change iris normals."""
        face_mesh = build_face_mesh()
        scene_straight = compose_face_with_eyes(face_mesh, 0.0, 0.0)
        scene_rotated = compose_face_with_eyes(face_mesh, 20.0, 0.0)
        # Iris normals should differ
        assert not np.allclose(
            scene_straight.right_eye.iris_normal,
            scene_rotated.right_eye.iris_normal,
            atol=0.01,
        )
