"""Render a full face with physics-based eyeballs at a given gaze angle."""

from typing import Optional

import numpy as np
from PIL import Image

from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS
from ..face_model.flame_mesh import build_face_mesh
from ..face_model.composition import compose_face_with_eyes
from ..rendering.camera import OrthographicCamera
from ..rendering.composite_renderer import render_composite
from ..rendering.lighting import PointLight


def render_face(
    theta_h_deg: float = 0.0,
    theta_v_deg: float = 0.0,
    resolution: int = 256,
    betas: Optional[np.ndarray] = None,
    expression: Optional[np.ndarray] = None,
    jaw_pose: Optional[np.ndarray] = None,
    eye_params: EyeParameters = DEFAULT_PARAMS,
    light: PointLight = None,
    model_path: Optional[str] = None,
    viewport_width: float = 200.0,
    viewport_height: float = 250.0,
    bg_color: tuple = (0.15, 0.15, 0.18),
    skin_base_color: Optional[np.ndarray] = None,
) -> Image.Image:
    """Render a full face with physics-based eyeballs.

    Args:
        theta_h_deg: Horizontal gaze angle in degrees.
        theta_v_deg: Vertical gaze angle in degrees.
        resolution: Image width in pixels. Height scales proportionally.
        betas: FLAME identity coefficients.
        expression: FLAME expression coefficients.
        jaw_pose: FLAME jaw rotation (axis-angle).
        eye_params: Physical eye parameters.
        light: Light source. Uses default if None.
        model_path: Path to FLAME model. Uses default if None.
        viewport_width: Physical viewport width in mm.
        viewport_height: Physical viewport height in mm.
        bg_color: Background color RGB (0-1).
        skin_base_color: (3,) skin color override.

    Returns:
        PIL Image (RGB).
    """
    # Build face mesh
    face_mesh = build_face_mesh(
        betas=betas,
        expression=expression,
        jaw_pose=jaw_pose,
        model_path=model_path,
    )

    # Compose face with eyes
    scene = compose_face_with_eyes(
        face_mesh, theta_h_deg, theta_v_deg, eye_params
    )

    if light is None:
        light = PointLight(
            position=np.array([50.0, 80.0, 150.0]),
            intensity=0.75,
        )

    # Camera
    res_y = int(resolution * viewport_height / viewport_width)
    camera = OrthographicCamera(
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        resolution_x=resolution,
        resolution_y=res_y,
    )

    rays = camera.generate_rays()

    # Render with eye-region supersampling
    colors = render_composite(
        scene,
        rays.origin,
        rays.direction,
        light=light,
        bg_color=bg_color,
        n_air=eye_params.ior_air,
        n_aqueous=eye_params.ior_aqueous,
        skin_base_color=skin_base_color,
        eye_supersample=4,
        pixel_size_mm=camera.mm_per_pixel(),
    )

    img_array = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    img_array = img_array.reshape(res_y, resolution, 3)
    return Image.fromarray(img_array)


def render_face_sweep(
    angles: Optional[list] = None,
    resolution: int = 128,
    betas: Optional[np.ndarray] = None,
    expression: Optional[np.ndarray] = None,
    jaw_pose: Optional[np.ndarray] = None,
    eye_params: EyeParameters = DEFAULT_PARAMS,
    model_path: Optional[str] = None,
) -> Image.Image:
    """Render a grid of faces at different gaze angles.

    Args:
        angles: List of horizontal gaze angles in degrees.
            Default: [0, 5, 10, 15, 20, 30].
        resolution: Width per face image in pixels.
        betas: FLAME identity coefficients.
        expression: FLAME expression coefficients.
        jaw_pose: FLAME jaw rotation.
        eye_params: Physical eye parameters.
        model_path: Path to FLAME model.

    Returns:
        PIL Image grid (RGB).
    """
    if angles is None:
        angles = [0, 5, 10, 15, 20, 30]

    images = []
    for angle in angles:
        img = render_face(
            theta_h_deg=angle,
            resolution=resolution,
            betas=betas,
            expression=expression,
            jaw_pose=jaw_pose,
            eye_params=eye_params,
            model_path=model_path,
        )
        images.append(img)

    # Create horizontal grid
    widths = [img.width for img in images]
    max_height = max(img.height for img in images)
    grid = Image.new("RGB", (sum(widths), max_height), (38, 38, 46))

    x_offset = 0
    for img in images:
        grid.paste(img, (x_offset, 0))
        x_offset += img.width

    return grid
