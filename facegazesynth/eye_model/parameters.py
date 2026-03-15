"""Physical constants for the human eyeball model.

All dimensions in millimeters. Based on the Gullstrand simplified eye model
with anatomical measurements from standard ophthalmic references.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EyeParameters:
    # Sclera (main sphere)
    sclera_radius: float = 12.0  # mm

    # Cornea (protruding dome)
    cornea_radius_of_curvature: float = 7.8  # mm
    cornea_protrusion: float = 2.5  # mm beyond sclera surface

    # Iris (flat annular disc behind cornea)
    iris_outer_radius: float = 6.0  # mm (12mm diameter)
    iris_setback: float = 3.6  # mm behind corneal apex

    # Pupil (circular aperture in iris)
    pupil_radius: float = 1.75  # mm (3.5mm diameter, normal lighting)

    # Rotation
    rotation_center_depth: float = 13.5  # mm behind corneal apex

    # Optics (Gullstrand simplified: single-surface model, air -> aqueous)
    ior_air: float = 1.0
    ior_aqueous: float = 1.336

    # Limbus (cornea-sclera boundary)
    limbus_width: float = 0.5  # mm, for darkening ring

    # Sclera color (linear RGB, 0-1)
    sclera_color: tuple = (0.94, 0.92, 0.88)


DEFAULT_PARAMS = EyeParameters()
