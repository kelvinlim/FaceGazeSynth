"""Render a single eyeball at a given gaze angle."""

import numpy as np
from PIL import Image

from ..eye_model.geometry import build_geometry
from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS
from ..eye_model.rotation import rotate_eye
from ..rendering.camera import OrthographicCamera
from ..rendering.renderer import render_eye
from ..rendering.lighting import PointLight


def render_single_eye(
    theta_h_deg: float = 0.0,
    theta_v_deg: float = 0.0,
    resolution: int = 512,
    viewport_size: float = 30.0,
    params: EyeParameters = DEFAULT_PARAMS,
    flat_shading: bool = False,
    light: PointLight = None,
) -> Image.Image:
    """Render one eyeball and return as a PIL Image.

    Args:
        theta_h_deg: Horizontal gaze angle in degrees.
        theta_v_deg: Vertical gaze angle in degrees.
        resolution: Image resolution (square).
        viewport_size: Physical viewport size in mm.
        params: Eye parameters.
        flat_shading: If True, use flat colors (no lighting/materials).
        light: Light source. If None, uses a default upper-right light.

    Returns:
        PIL Image (RGB).
    """
    geom = build_geometry(params)

    if theta_h_deg != 0.0 or theta_v_deg != 0.0:
        geom = rotate_eye(geom, theta_h_deg, theta_v_deg)

    if light is None:
        light = PointLight(
            position=np.array([15.0, 20.0, 60.0]),
            intensity=0.75,
        )

    camera = OrthographicCamera(
        viewport_width=viewport_size,
        viewport_height=viewport_size,
        resolution_x=resolution,
        resolution_y=resolution,
    )

    rays = camera.generate_rays()
    colors = render_eye(
        geom, rays.origin, rays.direction,
        n_air=params.ior_air,
        n_aqueous=params.ior_aqueous,
        flat_shading=flat_shading,
        light=light,
    )

    # Reshape to image and convert to uint8
    img_array = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    img_array = img_array.reshape(resolution, resolution, 3)
    return Image.fromarray(img_array)
