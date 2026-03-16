"""Compare measured iris displacement against theoretical predictions."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..pipeline.single_eye import render_single_eye
from ..eye_model.parameters import DEFAULT_PARAMS
from .iris_displacement import measure_iris_displacement
from .expected_curves import displacement_curves, naive_displacement, refraction_corrected_displacement
from .face_validation import run_face_validation


def run_validation(
    angles: list[float] = None,
    resolution: int = 512,
    output_path: str = "output/validation_curve.png",
) -> dict:
    """Run the iris displacement validation suite.

    Renders eyes at each gaze angle, measures the apparent iris
    displacement, and compares against theoretical predictions.

    Args:
        angles: Gaze angles to test. Default: [0, 5, 10, 15, 20, 25, 30].
        resolution: Render resolution.
        output_path: Path to save the validation plot.

    Returns:
        Dict with 'angles', 'measured', 'naive', 'refracted', 'rms_error'.
    """
    if angles is None:
        angles = [0, 5, 10, 15, 20, 25, 30]

    measured = []
    for angle in angles:
        print(f"  Measuring at {angle}°...")
        img = render_single_eye(
            theta_h_deg=angle,
            resolution=resolution,
            flat_shading=True,  # flat shading for cleaner detection
        )
        dx, _ = measure_iris_displacement(img)
        measured.append(dx)

    measured = np.array(measured)
    angles_arr = np.array(angles, dtype=float)

    # Theoretical curves
    naive = np.array([naive_displacement(a) for a in angles])
    refracted = np.array([refraction_corrected_displacement(a) for a in angles])

    # RMS error vs refracted prediction
    rms = np.sqrt(np.mean((measured - refracted)**2))

    # Compute magnification ratio (measured / naive) to show refraction effect
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_ratio = np.where(naive > 0.01, measured / naive, 1.0)

    # --- Plot ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Displacement curves
    smooth_angles = np.linspace(0, 32, 100)
    smooth_naive = np.array([naive_displacement(a) for a in smooth_angles])
    smooth_refr = np.array([refraction_corrected_displacement(a) for a in smooth_angles])

    ax1.plot(smooth_angles, smooth_naive, "b--", label="Naive (no refraction)", linewidth=1.5)
    ax1.plot(smooth_angles, smooth_refr, "g-", label="Refraction-corrected theory", linewidth=1.5)
    ax1.plot(angles, measured, "ro", markersize=8, label=f"Rendered (measured), RMS={rms:.3f}mm")
    ax1.set_xlabel("Gaze angle (degrees)")
    ax1.set_ylabel("Apparent iris displacement (mm)")
    ax1.set_title("Iris Displacement vs. Gaze Angle")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Magnification ratio
    smooth_mag = smooth_refr / np.maximum(smooth_naive, 0.01)
    ax2.axhline(y=1.0, color="b", linestyle="--", label="No refraction (ratio=1)", alpha=0.7)
    ax2.plot(smooth_angles[1:], smooth_mag[1:], "g-", label="Theoretical magnification", linewidth=1.5)
    nonzero = angles_arr > 0
    if np.any(nonzero):
        ax2.plot(angles_arr[nonzero], mag_ratio[nonzero], "ro", markersize=8, label="Measured magnification")
    ax2.set_xlabel("Gaze angle (degrees)")
    ax2.set_ylabel("Displacement ratio (with / without refraction)")
    ax2.set_title("Corneal Refraction Magnification Effect")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.9, 1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nValidation Results:")
    print(f"  RMS error vs refraction-corrected theory: {rms:.4f} mm")
    print(f"  Plot saved to {output_path}")

    print(f"\n  {'Angle':>6s}  {'Naive':>8s}  {'Theory':>8s}  {'Measured':>8s}  {'Ratio':>6s}")
    for i, a in enumerate(angles):
        ratio = f"{mag_ratio[i]:.3f}" if naive[i] > 0.01 else "N/A"
        print(f"  {a:>5.0f}°  {naive[i]:>8.3f}  {refracted[i]:>8.3f}  {measured[i]:>8.3f}  {ratio:>6s}")

    return {
        "angles": angles_arr,
        "measured": measured,
        "naive": naive,
        "refracted": refracted,
        "rms_error": rms,
    }
