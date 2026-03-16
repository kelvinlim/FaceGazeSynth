"""Eye-face composition: position physics-based eyeballs in a FLAME face mesh."""

from dataclasses import dataclass

import numpy as np
import trimesh

from ..eye_model.geometry import EyeballGeometry, build_geometry
from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS
from ..eye_model.rotation import rotate_eye
from .flame_mesh import FaceMesh


@dataclass
class CompositeScene:
    """A face mesh with two positioned eyeball geometries."""

    face_trimesh: trimesh.Trimesh  # face mesh for ray intersection
    left_eye: EyeballGeometry
    right_eye: EyeballGeometry
    face_vertex_normals: np.ndarray  # (N, 3) for smooth shading
    original_face_indices: np.ndarray  # (F,) maps stripped faces to original FLAME


def _mirror_geometry(geom: EyeballGeometry) -> EyeballGeometry:
    """Mirror geometry across the YZ plane (flip X) for the left eye."""
    def flip(p):
        result = p.copy()
        result[0] = -result[0]
        return result

    return EyeballGeometry(
        sclera_center=flip(geom.sclera_center),
        sclera_radius=geom.sclera_radius,
        cornea_center=flip(geom.cornea_center),
        cornea_radius=geom.cornea_radius,
        limbus_half_angle=geom.limbus_half_angle,
        iris_center=flip(geom.iris_center),
        iris_normal=flip(geom.iris_normal),
        iris_outer_radius=geom.iris_outer_radius,
        pupil_radius=geom.pupil_radius,
        rotation_center=flip(geom.rotation_center),
        corneal_apex=flip(geom.corneal_apex),
    )


def _shift_geometry(geom: EyeballGeometry, offset: np.ndarray) -> EyeballGeometry:
    """Translate all geometry points by offset."""
    return EyeballGeometry(
        sclera_center=geom.sclera_center + offset,
        sclera_radius=geom.sclera_radius,
        cornea_center=geom.cornea_center + offset,
        cornea_radius=geom.cornea_radius,
        limbus_half_angle=geom.limbus_half_angle,
        iris_center=geom.iris_center + offset,
        iris_normal=geom.iris_normal.copy(),
        iris_outer_radius=geom.iris_outer_radius,
        pupil_radius=geom.pupil_radius,
        rotation_center=geom.rotation_center + offset,
        corneal_apex=geom.corneal_apex + offset,
    )


def compose_face_with_eyes(
    face_mesh: FaceMesh,
    theta_h_deg: float = 0.0,
    theta_v_deg: float = 0.0,
    eye_params: EyeParameters = DEFAULT_PARAMS,
) -> CompositeScene:
    """Position physics-based eyeballs at FLAME eye joint locations.

    Args:
        face_mesh: FLAME face mesh (eyeball vertices already removed).
        theta_h_deg: Horizontal gaze angle in degrees.
        theta_v_deg: Vertical gaze angle in degrees.
        eye_params: Physical eye parameters.

    Returns:
        CompositeScene with face trimesh and two positioned eyeball geometries.
    """
    # Symmetrize eye joint positions — FLAME has slight L/R asymmetry that
    # causes visually different corneal refraction between the two eyes.
    # Average Y and Z so both eyes sit at the same depth and height;
    # keep X (interpupillary distance) from FLAME.
    right_joint = face_mesh.right_eye_joint.copy()
    left_joint = face_mesh.left_eye_joint.copy()
    avg_y = (right_joint[1] + left_joint[1]) / 2.0
    avg_z = (right_joint[2] + left_joint[2]) / 2.0
    right_joint[1] = avg_y
    right_joint[2] = avg_z
    left_joint[1] = avg_y
    left_joint[2] = avg_z

    # Build right eye: rotate, then translate to FLAME's right eye joint
    right_geom = build_geometry(eye_params)
    if theta_h_deg != 0.0 or theta_v_deg != 0.0:
        right_geom = rotate_eye(right_geom, theta_h_deg, theta_v_deg)
    right_geom = _shift_geometry(right_geom, right_joint)

    # Build left eye: mirror first (left-eye anatomy), then rotate for conjugate gaze
    left_geom = build_geometry(eye_params)
    left_geom = _mirror_geometry(left_geom)
    if theta_h_deg != 0.0 or theta_v_deg != 0.0:
        left_geom = rotate_eye(left_geom, theta_h_deg, theta_v_deg)
    left_geom = _shift_geometry(left_geom, left_joint)

    # Build trimesh for ray intersection
    face_tm = trimesh.Trimesh(
        vertices=face_mesh.vertices,
        faces=face_mesh.faces,
        process=False,
    )

    return CompositeScene(
        face_trimesh=face_tm,
        left_eye=left_geom,
        right_eye=right_geom,
        face_vertex_normals=face_mesh.vertex_normals,
        original_face_indices=face_mesh.original_face_indices,
    )
