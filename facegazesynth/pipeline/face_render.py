"""Render a full face with physics-based eyeballs at a given gaze angle."""

from typing import Optional

import numpy as np
from PIL import Image

from ..eye_model.parameters import EyeParameters, DEFAULT_PARAMS
from ..face_model.flame_mesh import build_face_mesh
from ..face_model.composition import compose_face_with_eyes
from ..face_model.expressions import get_emotion_params, random_identity
from ..materials.albedo import load_albedo_model, sample_albedo_texture
from ..rendering.camera import OrthographicCamera, PerspectiveCamera
from ..rendering.composite_renderer import render_composite
from ..rendering.lighting import PointLight


def render_face(
    theta_h_deg: float = 0.0,
    theta_v_deg: float = 0.0,
    resolution: int = 256,
    betas: Optional[np.ndarray] = None,
    expression: Optional[np.ndarray] = None,
    jaw_pose: Optional[np.ndarray] = None,
    emotion: Optional[str] = None,
    emotion_intensity: float = 1.0,
    eye_params: EyeParameters = DEFAULT_PARAMS,
    light: PointLight = None,
    model_path: Optional[str] = None,
    viewport_width: float = 200.0,
    viewport_height: float = 250.0,
    bg_color: tuple = (0.15, 0.15, 0.18),
    skin_base_color: Optional[np.ndarray] = None,
    perspective: bool = False,
    focal_length: float = 50.0,
    camera_distance: float = 500.0,
    use_albedo: bool = False,
    albedo_coefficients: Optional[np.ndarray] = None,
    albedo_seed: Optional[int] = None,
) -> Image.Image:
    """Render a full face with physics-based eyeballs.

    Args:
        theta_h_deg: Horizontal gaze angle in degrees.
        theta_v_deg: Vertical gaze angle in degrees.
        resolution: Image width in pixels. Height scales proportionally.
        betas: FLAME identity coefficients.
        expression: FLAME expression coefficients (overridden by emotion).
        jaw_pose: FLAME jaw rotation (overridden by emotion).
        emotion: Emotion name (e.g. "happy", "sad"). Overrides expression/jaw_pose.
        emotion_intensity: Scale factor for emotion expression (0-1).
        eye_params: Physical eye parameters.
        light: Light source. Uses default if None.
        model_path: Path to FLAME model. Uses default if None.
        viewport_width: Physical viewport width in mm (orthographic only).
        viewport_height: Physical viewport height in mm (orthographic only).
        bg_color: Background color RGB (0-1).
        skin_base_color: (3,) skin color override (procedural mode).
        perspective: If True, use perspective camera.
        focal_length: Lens focal length in mm (perspective only).
        camera_distance: Distance from camera to origin in mm (perspective only).
        use_albedo: If True, use data-driven albedo texture for skin.
        albedo_coefficients: PCA coefficients for albedo sampling.
        albedo_seed: Random seed for albedo sampling.

    Returns:
        PIL Image (RGB).
    """
    # Emotion overrides expression/jaw_pose
    if emotion is not None:
        expression, jaw_pose = get_emotion_params(emotion, emotion_intensity)

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

    # Albedo texture
    albedo_texture = None
    albedo_model_data = None
    if use_albedo:
        albedo_model_data = load_albedo_model()
        albedo_texture = sample_albedo_texture(
            coefficients=albedo_coefficients,
            albedo_model=albedo_model_data,
            seed=albedo_seed,
        )

    # Camera
    if perspective:
        aspect = viewport_height / viewport_width
        sensor_w = 36.0
        sensor_h = sensor_w * aspect
        res_y = int(resolution * aspect)
        camera = PerspectiveCamera(
            focal_length=focal_length,
            sensor_width=sensor_w,
            sensor_height=sensor_h,
            resolution_x=resolution,
            resolution_y=res_y,
            camera_distance=camera_distance,
        )
    else:
        res_y = int(resolution * viewport_height / viewport_width)
        camera = OrthographicCamera(
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            resolution_x=resolution,
            resolution_y=res_y,
        )

    rays = camera.generate_rays()

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
        albedo_texture=albedo_texture,
        albedo_model=albedo_model_data,
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
    emotion: Optional[str] = None,
    eye_params: EyeParameters = DEFAULT_PARAMS,
    model_path: Optional[str] = None,
    use_albedo: bool = False,
    albedo_seed: Optional[int] = None,
) -> Image.Image:
    """Render a grid of faces at different gaze angles.

    Args:
        angles: List of horizontal gaze angles in degrees.
        resolution: Width per face image in pixels.
        betas: FLAME identity coefficients.
        expression: FLAME expression coefficients.
        jaw_pose: FLAME jaw rotation.
        emotion: Emotion name (overrides expression/jaw_pose).
        eye_params: Physical eye parameters.
        model_path: Path to FLAME model.
        use_albedo: Use data-driven albedo texture.
        albedo_seed: Seed for albedo sampling.

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
            emotion=emotion,
            eye_params=eye_params,
            model_path=model_path,
            use_albedo=use_albedo,
            albedo_seed=albedo_seed,
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
