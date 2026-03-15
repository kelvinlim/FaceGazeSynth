"""Diagnostic visualizations for debugging the ray tracer."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..eye_model.geometry import build_geometry
from ..eye_model.parameters import DEFAULT_PARAMS


def plot_cross_section(output_path: str = "output/cross_section.png"):
    """Plot a 2D cross-section of the eyeball geometry.

    Shows sclera sphere, cornea cap, iris plane, pupil, and rotation center.
    """
    geom = build_geometry()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Sclera circle (XZ cross-section)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        geom.sclera_center[0] + geom.sclera_radius * np.cos(theta),
        geom.sclera_center[2] + geom.sclera_radius * np.sin(theta),
        "b-", linewidth=1.5, label="Sclera"
    )

    # Cornea cap (arc of the cornea sphere within cap region)
    cap_angles = np.linspace(-geom.limbus_half_angle, geom.limbus_half_angle, 100)
    cornea_x = geom.cornea_center[0] + geom.cornea_radius * np.sin(cap_angles)
    cornea_z = geom.cornea_center[2] + geom.cornea_radius * np.cos(cap_angles)
    ax.plot(cornea_x, cornea_z, "c-", linewidth=2.5, label="Cornea cap")

    # Iris plane (horizontal line)
    iris_z = geom.iris_center[2]
    ax.plot(
        [-geom.iris_outer_radius, -geom.pupil_radius],
        [iris_z, iris_z],
        "brown", linewidth=3, label="Iris"
    )
    ax.plot(
        [geom.pupil_radius, geom.iris_outer_radius],
        [iris_z, iris_z],
        "brown", linewidth=3
    )

    # Pupil gap
    ax.plot(
        [-geom.pupil_radius, geom.pupil_radius],
        [iris_z, iris_z],
        "k-", linewidth=1, alpha=0.3, label="Pupil"
    )

    # Rotation center
    ax.plot(
        geom.rotation_center[0], geom.rotation_center[2],
        "r+", markersize=15, markeredgewidth=2, label="Rotation center"
    )

    # Corneal apex
    ax.plot(
        geom.corneal_apex[0], geom.corneal_apex[2],
        "c^", markersize=10, label="Corneal apex"
    )

    # Cornea center
    ax.plot(
        geom.cornea_center[0], geom.cornea_center[2],
        "cx", markersize=10, label="Cornea center"
    )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title("Eyeball Cross-Section (XZ plane)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Cross-section saved to {output_path}")


def plot_ray_fan(
    n_rays: int = 15,
    output_path: str = "output/ray_fan.png",
):
    """Plot a fan of parallel rays refracting through the cornea.

    Shows rays entering the cornea and bending toward the iris plane,
    demonstrating the corneal lens effect.
    """
    from ..optics.intersections import intersect_ray_sphere, is_within_cornea_cap, intersect_ray_plane
    from ..optics.refraction import refract

    geom = build_geometry()
    params = DEFAULT_PARAMS

    # Generate horizontal fan of rays in XZ plane
    x_range = np.linspace(-7, 7, n_rays)
    ray_origins = np.column_stack([x_range, np.zeros(n_rays), np.full(n_rays, 50.0)])
    ray_dirs = np.column_stack([np.zeros(n_rays), np.zeros(n_rays), np.full(n_rays, -1.0)])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Draw eyeball geometry
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        geom.sclera_radius * np.cos(theta),
        geom.sclera_radius * np.sin(theta),
        "b-", alpha=0.3, linewidth=1
    )
    cap_angles = np.linspace(-geom.limbus_half_angle, geom.limbus_half_angle, 100)
    ax.plot(
        geom.cornea_radius * np.sin(cap_angles) + geom.cornea_center[0],
        geom.cornea_center[2] + geom.cornea_radius * np.cos(cap_angles),
        "c-", linewidth=2
    )
    iris_z = geom.iris_center[2]
    ax.plot([-geom.iris_outer_radius, geom.iris_outer_radius], [iris_z, iris_z], "brown", linewidth=2)

    # Trace each ray
    t_cornea, hp_cornea, n_cornea, mask_cornea = intersect_ray_sphere(
        ray_origins, ray_dirs, geom.cornea_center, geom.cornea_radius
    )
    cap_mask = np.zeros(n_rays, dtype=bool)
    if np.any(mask_cornea):
        cap_mask = mask_cornea & is_within_cornea_cap(
            hp_cornea, geom.cornea_center, geom.cornea_radius, geom.limbus_half_angle
        )

    for i in range(n_rays):
        x0, z0 = ray_origins[i, 0], ray_origins[i, 2]

        if cap_mask[i]:
            # Draw incoming ray to cornea
            hx, hz = hp_cornea[i, 0], hp_cornea[i, 2]
            ax.plot([x0, hx], [z0, hz], "r-", alpha=0.5, linewidth=0.8)

            # Refract
            incident = ray_dirs[i:i+1]
            normal = n_cornea[i:i+1]
            refracted, valid = refract(incident, normal, params.ior_air, params.ior_aqueous)

            if valid[0]:
                # Trace refracted ray to iris plane
                t_iris, hp_iris, mask_iris = intersect_ray_plane(
                    hp_cornea[i:i+1], refracted,
                    geom.iris_center, geom.iris_normal
                )
                if mask_iris[0]:
                    ix, iz = hp_iris[0, 0], hp_iris[0, 2]
                    ax.plot([hx, ix], [hz, iz], "r-", alpha=0.7, linewidth=0.8)
                else:
                    # Extend refracted ray
                    end = hp_cornea[i] + refracted[0] * 20
                    ax.plot([hx, end[0]], [hz, end[2]], "r--", alpha=0.3, linewidth=0.8)
        else:
            # Miss cornea — draw to sclera or through
            ax.plot([x0, x0], [z0, -15], "gray", alpha=0.2, linewidth=0.5)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title("Ray Fan: Corneal Refraction")
    ax.set_aspect("equal")
    ax.set_ylim(-5, 20)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ray fan diagram saved to {output_path}")
