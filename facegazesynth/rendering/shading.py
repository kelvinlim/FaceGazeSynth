"""Combine materials with lighting for final pixel colors."""

import numpy as np
from ..rendering.lighting import PointLight, lambertian_diffuse, specular_highlight
from ..materials.iris import iris_color_at
from ..materials.sclera import sclera_color_at
from ..materials.cornea import corneal_specular
from ..materials.limbus import limbus_darkening
from ..eye_model.geometry import EyeballGeometry


def shade_pixels(
    surface_id: np.ndarray,
    hit_points: np.ndarray,
    hit_normals: np.ndarray,
    ray_dirs: np.ndarray,
    cornea_hit_points: np.ndarray,
    cornea_hit_normals: np.ndarray,
    cornea_hit_mask: np.ndarray,
    geom: EyeballGeometry,
    light: PointLight,
    bg_color: np.ndarray = None,
    iris_base_color: np.ndarray = None,
) -> np.ndarray:
    """Apply material-based shading to all pixels.

    Args:
        surface_id: (N,) int surface ID per pixel.
        hit_points: (N, 3) surface hit points.
        hit_normals: (N, 3) surface normals.
        ray_dirs: (N, 3) ray directions.
        cornea_hit_points: (N, 3) cornea surface hit points (for specular).
        cornea_hit_normals: (N, 3) cornea surface normals.
        cornea_hit_mask: (N,) bool mask for rays that hit cornea.
        geom: Eyeball geometry.
        light: Light source.
        bg_color: (3,) background color.
        iris_base_color: (3,) iris base color.

    Returns:
        (N, 3) RGB colors in [0, 1].
    """
    from ..rendering.renderer import SURFACE_SCLERA, SURFACE_IRIS, SURFACE_PUPIL

    if bg_color is None:
        bg_color = np.array([0.15, 0.15, 0.18])
    if iris_base_color is None:
        iris_base_color = np.array([0.45, 0.28, 0.12])

    n = len(surface_id)
    colors = np.tile(bg_color, (n, 1))

    # --- Sclera ---
    sclera_mask = surface_id == SURFACE_SCLERA
    if np.any(sclera_mask):
        sc_points = hit_points[sclera_mask]
        sc_normals = hit_normals[sclera_mask]

        # Base color with variation
        sc_colors = sclera_color_at(sc_points, sc_normals, geom.sclera_center)

        # Diffuse lighting
        diffuse = lambertian_diffuse(sc_points, sc_normals, light, ambient=0.25)
        sc_colors = sc_colors * diffuse[:, np.newaxis]

        # Limbus darkening
        limbus_mult = limbus_darkening(
            sc_points, geom.cornea_center, geom.cornea_radius,
            geom.limbus_half_angle
        )
        sc_colors = sc_colors * limbus_mult[:, np.newaxis]

        colors[sclera_mask] = sc_colors

    # --- Iris ---
    iris_mask = surface_id == SURFACE_IRIS
    if np.any(iris_mask):
        ir_points = hit_points[iris_mask]

        # Procedural iris texture
        ir_colors = iris_color_at(
            ir_points, geom.iris_center, geom.iris_outer_radius,
            geom.pupil_radius, base_color=iris_base_color
        )

        # Simple diffuse for iris (use iris normal)
        ir_normals = np.tile(geom.iris_normal, (np.sum(iris_mask), 1))
        diffuse = lambertian_diffuse(ir_points, ir_normals, light, ambient=0.3)
        ir_colors = ir_colors * diffuse[:, np.newaxis]

        colors[iris_mask] = ir_colors

    # --- Pupil ---
    pupil_mask = surface_id == SURFACE_PUPIL
    colors[pupil_mask] = np.array([0.02, 0.02, 0.02])

    # --- Corneal specular highlight (Purkinje image) ---
    # Apply on top of whatever color is beneath
    if np.any(cornea_hit_mask):
        spec = corneal_specular(
            cornea_hit_points[cornea_hit_mask],
            cornea_hit_normals[cornea_hit_mask],
            ray_dirs[cornea_hit_mask],
            light.position,
        )
        # Add specular as white highlight
        colors[cornea_hit_mask] += spec[:, np.newaxis] * np.array([1.0, 1.0, 1.0])

    return np.clip(colors, 0.0, 1.0)
