"""Main ray tracing renderer for the eyeball model.

Traces rays through the scene: tests cornea cap first (with refraction
to the iris plane), then sclera. Supports both flat shading and
material-based shading.
"""

import numpy as np
from ..eye_model.geometry import EyeballGeometry
from ..optics.intersections import (
    intersect_ray_sphere,
    intersect_ray_plane,
    is_within_cornea_cap,
)
from ..optics.refraction import refract


# Surface IDs for hit tracking
SURFACE_NONE = 0
SURFACE_SCLERA = 1
SURFACE_IRIS = 2
SURFACE_PUPIL = 3
SURFACE_CORNEA_FRONT = 4  # cornea surface (for specular highlight)


def render_eye(
    geom: EyeballGeometry,
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
    n_air: float = 1.0,
    n_aqueous: float = 1.336,
    flat_shading: bool = True,
    light=None,
    sclera_color: tuple = (0.94, 0.92, 0.88),
    iris_color: tuple = (0.45, 0.28, 0.12),
    bg_color: tuple = (0.15, 0.15, 0.18),
    return_depth: bool = False,
) -> np.ndarray:
    """Render an eyeball for a batch of rays.

    Args:
        geom: Eyeball geometry.
        ray_origins: (N, 3) ray origins.
        ray_dirs: (N, 3) ray directions (unit vectors).
        n_air: Refractive index of air.
        n_aqueous: Refractive index of aqueous humor.
        flat_shading: If True, use flat colors. If False, use material shading.
        light: PointLight instance (required if flat_shading=False).
        sclera_color: RGB tuple (0-1) for sclera (flat shading only).
        iris_color: RGB tuple (0-1) for iris (flat shading only).
        bg_color: RGB tuple (0-1) for background.
        return_depth: If True, return (colors, depths) tuple.

    Returns:
        (N, 3) RGB colors in [0, 1]. If return_depth=True, returns
        (colors, depths) where depths is (N,) with np.inf for misses.
    """
    n = len(ray_origins)
    colors = np.tile(np.array(bg_color), (n, 1))

    # Track which surface each ray hits and the nearest t
    surface_id = np.full(n, SURFACE_NONE, dtype=int)
    best_t = np.full(n, np.inf)
    hit_normals = np.zeros((n, 3))
    hit_points = np.zeros((n, 3))

    # --- Step 1: Test cornea sphere intersection ---
    t_cornea, hp_cornea, n_cornea, mask_cornea_sphere = intersect_ray_sphere(
        ray_origins, ray_dirs, geom.cornea_center, geom.cornea_radius
    )

    # Filter to just the cap region
    cap_mask = np.zeros(n, dtype=bool)
    if np.any(mask_cornea_sphere):
        cap_test = is_within_cornea_cap(
            hp_cornea, geom.cornea_center, geom.cornea_radius,
            geom.limbus_half_angle
        )
        cap_mask = mask_cornea_sphere & cap_test

    # --- Step 2: For cornea cap hits, refract and trace to iris plane ---
    iris_hit_mask = np.zeros(n, dtype=bool)
    pupil_hit_mask = np.zeros(n, dtype=bool)

    if np.any(cap_mask):
        # Refract at cornea surface (air → aqueous, single-surface model)
        cornea_normals = n_cornea[cap_mask]
        incident_dirs = ray_dirs[cap_mask]

        refracted_dirs, refract_valid = refract(
            incident_dirs, cornea_normals, n_air, n_aqueous
        )

        # Trace refracted rays to iris plane
        valid_refract = cap_mask.copy()
        valid_refract[cap_mask] &= refract_valid

        if np.any(valid_refract):
            refr_origins = hp_cornea[valid_refract]

            # Build full refracted direction array
            refr_dirs_full = np.zeros((n, 3))
            refr_dirs_full[cap_mask] = refracted_dirs
            refr_dirs = refr_dirs_full[valid_refract]

            t_iris, hp_iris, mask_iris_plane = intersect_ray_plane(
                refr_origins, refr_dirs, geom.iris_center, geom.iris_normal
            )

            if np.any(mask_iris_plane):
                # Check if hit is within iris disc
                iris_hp = hp_iris[mask_iris_plane]
                iris_offset = iris_hp - geom.iris_center
                iris_r2 = np.sum(iris_offset * iris_offset, axis=1)

                within_iris = iris_r2 <= geom.iris_outer_radius**2
                within_pupil = iris_r2 <= geom.pupil_radius**2

                # Map back to global indices
                global_valid = np.where(valid_refract)[0]
                global_iris_plane = global_valid[mask_iris_plane]

                iris_global = global_iris_plane[within_iris & ~within_pupil]
                pupil_global = global_iris_plane[within_pupil]

                # Use the cornea t for depth sorting (the iris is seen through cornea)
                iris_hit_mask[iris_global] = True
                pupil_hit_mask[pupil_global] = True

                # For iris/pupil hits, the effective t is the cornea t
                t_eff = t_cornea[iris_global]
                closer = t_eff < best_t[iris_global]
                idx = iris_global[closer]
                surface_id[idx] = SURFACE_IRIS
                best_t[idx] = t_eff[closer]
                hit_points[idx] = hp_iris[mask_iris_plane][within_iris & ~within_pupil][closer]
                hit_normals[idx] = geom.iris_normal

                t_eff_p = t_cornea[pupil_global]
                closer_p = t_eff_p < best_t[pupil_global]
                idx_p = pupil_global[closer_p]
                surface_id[idx_p] = SURFACE_PUPIL
                best_t[idx_p] = t_eff_p[closer_p]

        # Cornea cap hits that didn't reach iris/pupil → show sclera behind
        # (ray passed through cornea but missed the iris)
        # Use the cornea t for depth ordering, but defer shading to the
        # sclera intersection below so normals/hit points are consistent
        # with the rest of the sclera surface (avoids a lighting seam at
        # the limbus boundary).
        cornea_no_iris = cap_mask & ~iris_hit_mask & ~pupil_hit_mask

    # --- Step 3: Test sclera sphere (excluding cornea cap region) ---
    t_sclera, hp_sclera, n_sclera, mask_sclera = intersect_ray_sphere(
        ray_origins, ray_dirs, geom.sclera_center, geom.sclera_radius
    )

    # Exclude sclera hits where the ray hit the cornea cap AND reached the
    # iris/pupil (those pixels are already resolved). But allow sclera hits
    # for cornea-cap rays that missed the iris — these show the sclera
    # visible through the cornea, and should use sclera normals for
    # consistent lighting across the limbus boundary.
    if np.any(mask_sclera):
        already_resolved = cap_mask & (iris_hit_mask | pupil_hit_mask)
        sclera_occluded = already_resolved & mask_sclera & (t_cornea <= t_sclera)
        valid_sclera = mask_sclera & ~sclera_occluded

        if np.any(valid_sclera):
            t_s = t_sclera[valid_sclera]
            closer = t_s < best_t[valid_sclera]
            idx = np.where(valid_sclera)[0][closer]
            surface_id[idx] = SURFACE_SCLERA
            best_t[idx] = t_s[closer]
            hit_points[idx] = hp_sclera[valid_sclera][closer]
            hit_normals[idx] = n_sclera[valid_sclera][closer]

    # --- Step 4: Assign colors ---
    if flat_shading:
        colors[surface_id == SURFACE_SCLERA] = np.array(sclera_color)
        colors[surface_id == SURFACE_IRIS] = np.array(iris_color)
        colors[surface_id == SURFACE_PUPIL] = np.array([0.0, 0.0, 0.0])
    else:
        from .shading import shade_pixels
        # Prepare cornea hit info for specular highlights
        cornea_hp = np.where(cap_mask[:, np.newaxis], hp_cornea, 0.0)
        cornea_normals = np.where(cap_mask[:, np.newaxis], n_cornea, 0.0)

        colors = shade_pixels(
            surface_id, hit_points, hit_normals, ray_dirs,
            cornea_hp, cornea_normals, cap_mask,
            geom, light, bg_color=np.array(bg_color),
        )

    if return_depth:
        return colors, best_t
    return colors
