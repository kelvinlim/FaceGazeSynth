"""Render a stereo pair of eyeballs at a given gaze angle."""

import numpy as np
from PIL import Image

from ..eye_model.geometry import build_geometry, EyeballGeometry
from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS
from ..eye_model.rotation import rotate_eye
from ..rendering.camera import OrthographicCamera
from ..rendering.renderer import render_eye
from ..rendering.lighting import PointLight


def _mirror_geometry(geom: EyeballGeometry) -> EyeballGeometry:
    """Mirror geometry across the YZ plane (flip X) for the left eye."""
    def flip(p):
        result = p.copy()
        result[0] = -result[0]
        return result

    def flip_normal(n):
        result = n.copy()
        result[0] = -result[0]
        return result

    return EyeballGeometry(
        sclera_center=flip(geom.sclera_center),
        sclera_radius=geom.sclera_radius,
        cornea_center=flip(geom.cornea_center),
        cornea_radius=geom.cornea_radius,
        limbus_half_angle=geom.limbus_half_angle,
        iris_center=flip(geom.iris_center),
        iris_normal=flip_normal(geom.iris_normal),
        iris_outer_radius=geom.iris_outer_radius,
        pupil_radius=geom.pupil_radius,
        rotation_center=flip(geom.rotation_center),
        corneal_apex=flip(geom.corneal_apex),
    )


def render_stereo_pair(
    theta_h_deg: float = 0.0,
    theta_v_deg: float = 0.0,
    resolution: int = 512,
    ipd: float = 63.0,
    params: EyeParameters = DEFAULT_PARAMS,
    light: PointLight = None,
) -> Image.Image:
    """Render two eyes side by side with conjugate gaze.

    Args:
        theta_h_deg: Horizontal gaze angle in degrees.
        theta_v_deg: Vertical gaze angle in degrees.
        resolution: Resolution per eye (height). Width is 2x.
        ipd: Interpupillary distance in mm.
        params: Eye parameters.
        light: Light source.

    Returns:
        PIL Image (RGB), width = 2 * resolution.
    """
    if light is None:
        light = PointLight(
            position=np.array([15.0, 20.0, 60.0]),
            intensity=0.75,
        )

    half_ipd = ipd / 2.0
    viewport_per_eye = 35.0  # mm, enough to show full eye + margin

    # Build right eye geometry (centered at +half_ipd on X)
    geom_right = build_geometry(params)
    if theta_h_deg != 0.0 or theta_v_deg != 0.0:
        geom_right = rotate_eye(geom_right, theta_h_deg, theta_v_deg)

    # Left eye is mirror image, then shifted
    geom_left = build_geometry(params)
    if theta_h_deg != 0.0 or theta_v_deg != 0.0:
        # Mirror the gaze angle for conjugate gaze (both eyes look same direction)
        geom_left = rotate_eye(geom_left, theta_h_deg, theta_v_deg)
    geom_left = _mirror_geometry(geom_left)

    # Shift eyes to their positions
    def shift_geom(geom, dx):
        offset = np.array([dx, 0.0, 0.0])
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

    geom_left_shifted = shift_geom(geom_left, -half_ipd)
    geom_right_shifted = shift_geom(geom_right, half_ipd)

    # Create camera covering both eyes
    total_width = ipd + viewport_per_eye
    camera = OrthographicCamera(
        viewport_width=total_width,
        viewport_height=viewport_per_eye,
        resolution_x=resolution * 2,
        resolution_y=resolution,
    )

    rays = camera.generate_rays()

    # Render each eye and composite (take nearest hit)
    colors_left = render_eye(
        geom_left_shifted, rays.origin, rays.direction,
        n_air=params.ior_air, n_aqueous=params.ior_aqueous,
        flat_shading=False, light=light,
    )
    colors_right = render_eye(
        geom_right_shifted, rays.origin, rays.direction,
        n_air=params.ior_air, n_aqueous=params.ior_aqueous,
        flat_shading=False, light=light,
    )

    # Simple compositing: take non-background pixels from each eye
    bg = np.array([0.15, 0.15, 0.18])
    is_bg_left = np.all(np.abs(colors_left - bg) < 0.01, axis=1)
    is_bg_right = np.all(np.abs(colors_right - bg) < 0.01, axis=1)

    colors = colors_left.copy()
    colors[is_bg_left & ~is_bg_right] = colors_right[is_bg_left & ~is_bg_right]
    # Where both have content, prefer whichever is not background
    colors[~is_bg_left & ~is_bg_right] = colors_left[~is_bg_left & ~is_bg_right]

    img_array = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    img_array = img_array.reshape(resolution, resolution * 2, 3)
    return Image.fromarray(img_array)
