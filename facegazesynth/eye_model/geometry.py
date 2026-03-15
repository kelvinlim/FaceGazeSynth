"""Eyeball composite geometry.

Coordinate system: eyeball (sclera) center at origin, forward-looking along +Z.
The cornea protrudes forward from the sclera along +Z.
"""

import numpy as np
from dataclasses import dataclass
from .parameters import EyeParameters, DEFAULT_PARAMS


@dataclass
class EyeballGeometry:
    """Computed geometry for a single eyeball.

    All positions are 3D numpy arrays in mm.
    """
    # Sclera sphere
    sclera_center: np.ndarray
    sclera_radius: float

    # Cornea sphere (the sphere whose cap forms the cornea dome)
    cornea_center: np.ndarray
    cornea_radius: float

    # Limbus: the circle where cornea cap meets sclera
    limbus_half_angle: float  # half-angle from cornea center axis to limbus circle

    # Iris plane
    iris_center: np.ndarray  # center of iris disc (on optical axis)
    iris_normal: np.ndarray  # plane normal (points toward viewer, +Z initially)
    iris_outer_radius: float
    pupil_radius: float

    # Rotation center (for gaze rotation)
    rotation_center: np.ndarray

    # Corneal apex (frontmost point of cornea)
    corneal_apex: np.ndarray


def build_geometry(params: EyeParameters = DEFAULT_PARAMS) -> EyeballGeometry:
    """Construct eyeball geometry from physical parameters.

    The cornea is modeled as a spherical cap that protrudes from the sclera.
    The two spheres (sclera and cornea) intersect at the limbus circle.
    """
    R_s = params.sclera_radius
    R_c = params.cornea_radius_of_curvature
    protrusion = params.cornea_protrusion

    # Sclera center at origin
    sclera_center = np.array([0.0, 0.0, 0.0])

    # Corneal apex is at the front of the eye: sclera surface + protrusion
    corneal_apex = np.array([0.0, 0.0, R_s + protrusion])

    # Cornea sphere center: the apex is on the cornea sphere surface,
    # so cornea_center is R_c behind the apex along +Z
    cornea_center = np.array([0.0, 0.0, corneal_apex[2] - R_c])

    # Limbus circle: where the two spheres intersect.
    # A point on both spheres satisfies:
    #   x^2 + y^2 + z^2 = R_s^2            (sclera)
    #   x^2 + y^2 + (z - cz)^2 = R_c^2     (cornea, cz = cornea_center[2])
    # Subtracting: z^2 - (z - cz)^2 = R_s^2 - R_c^2
    #   2*cz*z - cz^2 = R_s^2 - R_c^2
    #   z = (R_s^2 - R_c^2 + cz^2) / (2 * cz)
    cz = cornea_center[2]
    z_limbus = (R_s**2 - R_c**2 + cz**2) / (2 * cz)

    # The limbus half-angle (from cornea center's perspective):
    # cos(half_angle) = (z_limbus - cz) / R_c
    cos_half = (z_limbus - cz) / R_c
    limbus_half_angle = np.arccos(np.clip(cos_half, -1.0, 1.0))

    # Iris plane: perpendicular to optical axis, set back from corneal apex
    iris_z = corneal_apex[2] - params.iris_setback
    iris_center = np.array([0.0, 0.0, iris_z])
    iris_normal = np.array([0.0, 0.0, 1.0])

    # Rotation center: behind the corneal apex along -Z
    rotation_center = np.array([0.0, 0.0, corneal_apex[2] - params.rotation_center_depth])

    return EyeballGeometry(
        sclera_center=sclera_center,
        sclera_radius=R_s,
        cornea_center=cornea_center,
        cornea_radius=R_c,
        limbus_half_angle=limbus_half_angle,
        iris_center=iris_center,
        iris_normal=iris_normal,
        iris_outer_radius=params.iris_outer_radius,
        pupil_radius=params.pupil_radius,
        rotation_center=rotation_center,
        corneal_apex=corneal_apex,
    )
