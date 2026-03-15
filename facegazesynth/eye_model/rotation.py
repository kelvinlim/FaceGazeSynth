"""Gaze rotation mechanics.

The eye rotates around an anatomical rotation center (13.5mm behind the
corneal apex), NOT the geometric center of the sclera sphere.
"""

import numpy as np
from .geometry import EyeballGeometry


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula: rotation matrix for angle around axis.

    Args:
        axis: (3,) unit vector rotation axis.
        angle_rad: Rotation angle in radians.

    Returns:
        (3, 3) rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def rotate_eye(
    geom: EyeballGeometry,
    theta_h_deg: float = 0.0,
    theta_v_deg: float = 0.0,
) -> EyeballGeometry:
    """Rotate eyeball geometry around the anatomical rotation center.

    Args:
        geom: Original eyeball geometry (typically at 0° gaze).
        theta_h_deg: Horizontal gaze angle in degrees (positive = rightward).
        theta_v_deg: Vertical gaze angle in degrees (positive = upward).

    Returns:
        New EyeballGeometry with all positions rotated.
    """
    pivot = geom.rotation_center

    # Build combined rotation: horizontal around Y, vertical around X
    R = np.eye(3)
    if theta_h_deg != 0.0:
        R = rotation_matrix(np.array([0.0, 1.0, 0.0]), np.radians(theta_h_deg)) @ R
    if theta_v_deg != 0.0:
        R = rotation_matrix(np.array([1.0, 0.0, 0.0]), np.radians(theta_v_deg)) @ R

    def rot_point(p: np.ndarray) -> np.ndarray:
        return R @ (p - pivot) + pivot

    def rot_vec(v: np.ndarray) -> np.ndarray:
        return R @ v

    return EyeballGeometry(
        sclera_center=rot_point(geom.sclera_center),
        sclera_radius=geom.sclera_radius,
        cornea_center=rot_point(geom.cornea_center),
        cornea_radius=geom.cornea_radius,
        limbus_half_angle=geom.limbus_half_angle,
        iris_center=rot_point(geom.iris_center),
        iris_normal=rot_vec(geom.iris_normal),
        iris_outer_radius=geom.iris_outer_radius,
        pupil_radius=geom.pupil_radius,
        rotation_center=geom.rotation_center.copy(),
        corneal_apex=rot_point(geom.corneal_apex),
    )
