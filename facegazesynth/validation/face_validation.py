"""Validate iris displacement in face-context renders.

Renders eyes within a FLAME face mesh, crops the right eye region,
and measures iris displacement to verify corneal refraction still
matches theoretical predictions with face geometry present.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from ..eye_model.parameters import DEFAULT_PARAMS
from ..face_model.flame_mesh import build_face_mesh
from ..face_model.composition import compose_face_with_eyes
from ..rendering.camera import OrthographicCamera
from ..rendering.composite_renderer import render_composite
from ..rendering.lighting import PointLight
from .iris_displacement import measure_iris_displacement
from .expected_curves import naive_displacement, refraction_corrected_displacement


def _render_and_crop_right_eye(
    theta_h_deg: float,
    resolution: int = 512,
    face_mesh=None,
) -> tuple[Image.Image, float]:
    """Render face and crop the right eye region.

    Returns:
        (cropped_eye_image, crop_viewport_mm): the cropped image and its
        physical width in mm, needed for displacement measurement.
    """
    if face_mesh is None:
        face_mesh = build_face_mesh()

    scene = compose_face_with_eyes(face_mesh, theta_h_deg, 0.0)

    light = PointLight(
        position=np.array([50.0, 80.0, 150.0]),
        intensity=0.75,
    )

    viewport_w = 200.0
    viewport_h = 250.0
    res_x = resolution
    res_y = int(resolution * viewport_h / viewport_w)

    camera = OrthographicCamera(
        viewport_width=viewport_w,
        viewport_height=viewport_h,
        resolution_x=res_x,
        resolution_y=res_y,
    )

    rays = camera.generate_rays()
    colors = render_composite(
        scene, rays.origin, rays.direction, light=light,
        eye_supersample=1,  # no supersampling for faster validation
    )

    img_array = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    img_array = img_array.reshape(res_y, res_x, 3)
    full_img = Image.fromarray(img_array)

    # Crop around right eye joint (positive X side)
    mm_per_px = viewport_w / res_x
    eye_pos = face_mesh.right_eye_joint

    # Convert eye position to pixel coordinates
    # Camera: x=0 maps to center of image, +x = right
    eye_px_x = int(res_x / 2 + eye_pos[0] / mm_per_px)
    eye_px_y = int(res_y / 2 - eye_pos[1] / mm_per_px)

    # Crop a 30mm x 30mm region around the eye
    crop_mm = 30.0
    crop_px = int(crop_mm / mm_per_px)
    half = crop_px // 2

    x1 = max(0, eye_px_x - half)
    y1 = max(0, eye_px_y - half)
    x2 = min(res_x, x1 + crop_px)
    y2 = min(res_y, y1 + crop_px)

    cropped = full_img.crop((x1, y1, x2, y2))
    actual_crop_mm = (x2 - x1) * mm_per_px

    return cropped, actual_crop_mm


def run_face_validation(
    angles: list[float] = None,
    resolution: int = 512,
    output_path: str = "output/face_validation_curve.png",
) -> dict:
    """Run iris displacement validation using face-context renders.

    Args:
        angles: Gaze angles to test.
        resolution: Render resolution (image width).
        output_path: Path to save validation plot.

    Returns:
        Dict with 'angles', 'measured', 'naive', 'refracted', 'rms_error'.
    """
    if angles is None:
        angles = [0, 5, 10, 15, 20, 25, 30]

    # Build face mesh once
    face_mesh = build_face_mesh()

    measured = []
    for angle in angles:
        print(f"  Face context: measuring at {angle}°...")
        cropped, crop_mm = _render_and_crop_right_eye(
            angle, resolution=resolution, face_mesh=face_mesh,
        )
        dx, _ = measure_iris_displacement(cropped, viewport_width_mm=crop_mm)
        measured.append(dx)

    measured = np.array(measured)
    angles_arr = np.array(angles, dtype=float)

    naive = np.array([naive_displacement(a) for a in angles])
    refracted = np.array([refraction_corrected_displacement(a) for a in angles])

    rms = np.sqrt(np.mean((measured - refracted) ** 2))

    # Plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    smooth_angles = np.linspace(0, 32, 100)
    smooth_naive = np.array([naive_displacement(a) for a in smooth_angles])
    smooth_refr = np.array([refraction_corrected_displacement(a) for a in smooth_angles])

    ax.plot(smooth_angles, smooth_naive, "b--", label="Naive (no refraction)", linewidth=1.5)
    ax.plot(smooth_angles, smooth_refr, "g-", label="Refraction-corrected theory", linewidth=1.5)
    ax.plot(angles, measured, "ro", markersize=8,
            label=f"Face-context (measured), RMS={rms:.3f}mm")
    ax.set_xlabel("Gaze angle (degrees)")
    ax.set_ylabel("Apparent iris displacement (mm)")
    ax.set_title("Iris Displacement — Face Context Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFace Context Validation Results:")
    print(f"  RMS error vs refraction-corrected theory: {rms:.4f} mm")
    print(f"  Plot saved to {output_path}")

    print(f"\n  {'Angle':>6s}  {'Naive':>8s}  {'Theory':>8s}  {'Measured':>8s}")
    for i, a in enumerate(angles):
        print(f"  {a:>5.0f}°  {naive[i]:>8.3f}  {refracted[i]:>8.3f}  {measured[i]:>8.3f}")

    return {
        "angles": angles_arr,
        "measured": measured,
        "naive": naive,
        "refracted": refracted,
        "rms_error": rms,
    }
