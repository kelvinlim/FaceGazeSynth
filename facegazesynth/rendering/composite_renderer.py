"""Composite renderer: traces rays against face mesh + two eyeball geometries.

Depth-buffer compositing: for each ray, test the face mesh (via trimesh)
and both eyes (via render_eye), keep the nearest hit. Eye regions use
subpixel supersampling for anti-aliased iris/sclera detail.
"""

import numpy as np

from ..face_model.composition import CompositeScene
from ..materials.skin import skin_color_at
from ..rendering.renderer import render_eye
from ..rendering.lighting import PointLight


def render_composite(
    scene: CompositeScene,
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
    light: PointLight,
    bg_color: tuple = (0.15, 0.15, 0.18),
    n_air: float = 1.0,
    n_aqueous: float = 1.336,
    skin_base_color: np.ndarray = None,
    eye_supersample: int = 4,
    pixel_size_mm: float = None,
    albedo_texture: np.ndarray = None,
    albedo_model=None,
) -> np.ndarray:
    """Render composite scene with face mesh and two eyes.

    Args:
        scene: CompositeScene with face trimesh and eye geometries.
        ray_origins: (N, 3) ray origins.
        ray_dirs: (N, 3) ray directions.
        light: Point light source.
        bg_color: Background color RGB (0-1).
        n_air: Refractive index of air.
        n_aqueous: Refractive index of aqueous humor.
        skin_base_color: (3,) skin color override (procedural mode).
        eye_supersample: NxN subpixel samples per eye pixel (1 = off).
        pixel_size_mm: Physical pixel size in mm (needed for supersampling).
        albedo_texture: (512, 512, 3) sampled albedo texture. If provided,
            uses UV-mapped albedo instead of procedural skin color.
        albedo_model: Pre-loaded AlbedoModel for UV lookup.

    Returns:
        (N, 3) RGB image in [0, 1].
    """
    n = len(ray_origins)
    bg = np.array(bg_color)
    colors = np.tile(bg, (n, 1))
    best_depth = np.full(n, np.inf)

    ss = eye_supersample
    do_supersample = ss > 1 and pixel_size_mm is not None

    # --- 1. Render both eyes ---
    for eye_geom in [scene.left_eye, scene.right_eye]:
        if not do_supersample:
            eye_colors, eye_depth = render_eye(
                eye_geom, ray_origins, ray_dirs,
                n_air=n_air, n_aqueous=n_aqueous,
                flat_shading=False, light=light,
                bg_color=bg_color, return_depth=True,
            )
            eye_hit = eye_depth < best_depth
            colors[eye_hit] = eye_colors[eye_hit]
            best_depth[eye_hit] = eye_depth[eye_hit]
        else:
            eye_colors, eye_depth = _render_eye_supersampled(
                eye_geom, ray_origins, ray_dirs,
                n_air, n_aqueous, light, bg_color,
                ss, pixel_size_mm,
            )
            eye_hit = eye_depth < best_depth
            colors[eye_hit] = eye_colors[eye_hit]
            best_depth[eye_hit] = eye_depth[eye_hit]

    # --- 2. Trace face mesh ---
    face_tm = scene.face_trimesh
    hit_locs, hit_ray_idx, hit_tri_idx = face_tm.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_dirs,
        multiple_hits=False,
    )

    if len(hit_ray_idx) > 0:
        face_vecs = hit_locs - ray_origins[hit_ray_idx]
        face_depths = np.linalg.norm(face_vecs, axis=1)

        face_closer = face_depths < best_depth[hit_ray_idx]
        winning_rays = hit_ray_idx[face_closer]
        winning_locs = hit_locs[face_closer]
        winning_tris = hit_tri_idx[face_closer]
        winning_depths = face_depths[face_closer]

        if len(winning_rays) > 0:
            bary = _compute_barycentric(
                winning_locs, face_tm.vertices, face_tm.faces, winning_tris,
            )
            face_normals = face_tm.face_normals[winning_tris]

            # Look up albedo colors if texture is provided
            pixel_albedo = None
            if albedo_texture is not None:
                from ..materials.albedo import lookup_albedo_at_triangles
                # Map stripped-mesh triangle indices back to original FLAME
                # face indices for correct UV lookup
                orig_tris = scene.original_face_indices[winning_tris]
                pixel_albedo = lookup_albedo_at_triangles(
                    albedo_texture, orig_tris, bary, albedo_model,
                )

            skin_colors = skin_color_at(
                hit_points=winning_locs,
                normals=face_normals,
                barycentric=bary,
                face_vertex_normals=scene.face_vertex_normals,
                triangle_indices=winning_tris,
                faces=face_tm.faces,
                light_position=light.position,
                light_intensity=light.intensity,
                base_color=skin_base_color,
                albedo_colors=pixel_albedo,
            )

            colors[winning_rays] = skin_colors
            best_depth[winning_rays] = winning_depths

    return colors


def _render_eye_supersampled(
    eye_geom,
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
    n_air: float,
    n_aqueous: float,
    light: PointLight,
    bg_color: tuple,
    ss: int,
    pixel_size_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one eye with NxN supersampling in the eye region only.

    First does a coarse pass to find which pixels hit the eye, then
    re-renders those pixels with jittered subpixel rays and averages.

    Returns:
        (colors (N, 3), depths (N,)) — supersampled colors for eye pixels,
        inf depth for non-eye pixels.
    """
    n = len(ray_origins)

    # Coarse pass: find which rays hit this eye
    coarse_colors, coarse_depth = render_eye(
        eye_geom, ray_origins, ray_dirs,
        n_air=n_air, n_aqueous=n_aqueous,
        flat_shading=False, light=light,
        bg_color=bg_color, return_depth=True,
    )

    hit_mask = np.isfinite(coarse_depth) & (coarse_depth < 1e6)
    if not np.any(hit_mask):
        return coarse_colors, coarse_depth

    # Expand the hit region by 1 pixel in each direction for edge AA.
    # We work in flat index space, so we need the image dimensions.
    # Since we don't have them here, dilate by checking neighboring origins.
    hit_indices = np.where(hit_mask)[0]

    # Build NxN subpixel grid offsets within each pixel
    # Offsets in mm, centered on pixel center
    step = pixel_size_mm / ss
    half = pixel_size_mm / 2.0 - step / 2.0
    offsets = []
    for iy in range(ss):
        for ix in range(ss):
            dx = -half + ix * step
            dy = -half + iy * step
            offsets.append((dx, dy))

    # For orthographic camera, rays are parallel along -Z.
    # Subpixel jitter = shift ray origin in X, Y.
    n_hit = len(hit_indices)
    accum_colors = np.zeros((n_hit, 3))

    base_origins = ray_origins[hit_indices]  # (n_hit, 3)
    base_dirs = ray_dirs[hit_indices]  # (n_hit, 3)

    for dx, dy in offsets:
        jittered_origins = base_origins.copy()
        jittered_origins[:, 0] += dx
        jittered_origins[:, 1] += dy

        sample_colors = render_eye(
            eye_geom, jittered_origins, base_dirs,
            n_air=n_air, n_aqueous=n_aqueous,
            flat_shading=False, light=light,
            bg_color=bg_color, return_depth=False,
        )
        accum_colors += sample_colors

    accum_colors /= (ss * ss)

    # Write supersampled colors back
    result_colors = coarse_colors.copy()
    result_colors[hit_indices] = accum_colors

    return result_colors, coarse_depth


def _compute_barycentric(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    tri_indices: np.ndarray,
) -> np.ndarray:
    """Compute barycentric coordinates for points on triangles."""
    v0 = vertices[faces[tri_indices, 0]]
    v1 = vertices[faces[tri_indices, 1]]
    v2 = vertices[faces[tri_indices, 2]]

    e0 = v1 - v0
    e1 = v2 - v0
    v = points - v0

    d00 = np.sum(e0 * e0, axis=1)
    d01 = np.sum(e0 * e1, axis=1)
    d11 = np.sum(e1 * e1, axis=1)
    d20 = np.sum(v * e0, axis=1)
    d21 = np.sum(v * e1, axis=1)

    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2

    return np.stack([w0, w1, w2], axis=1)
