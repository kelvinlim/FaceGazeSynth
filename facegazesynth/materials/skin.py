"""Face skin shader: Lambertian diffuse with position-based color variation."""

import numpy as np


def skin_color_at(
    hit_points: np.ndarray,
    normals: np.ndarray,
    barycentric: np.ndarray,
    face_vertex_normals: np.ndarray,
    triangle_indices: np.ndarray,
    faces: np.ndarray,
    light_position: np.ndarray,
    light_intensity: float = 0.75,
    base_color: np.ndarray = None,
    ambient: float = 0.25,
) -> np.ndarray:
    """Shade face skin with smooth Lambertian diffuse lighting.

    Args:
        hit_points: (N, 3) intersection points in mm.
        normals: (N, 3) face normals at hit points (from trimesh).
        barycentric: (N, 3) barycentric coordinates for interpolation.
        face_vertex_normals: (V, 3) per-vertex normals of the face mesh.
        triangle_indices: (N,) which triangle was hit per ray.
        faces: (F, 3) face mesh triangle indices.
        light_position: (3,) light position in mm.
        light_intensity: Light intensity scalar.
        base_color: (3,) RGB base skin color. Default warm skin tone.
        ambient: Ambient light level.

    Returns:
        (N, 3) RGB colors in [0, 1].
    """
    if base_color is None:
        base_color = np.array([0.76, 0.60, 0.48])

    n = len(hit_points)
    colors = np.zeros((n, 3))

    # Smooth normals via barycentric interpolation
    tri_verts = faces[triangle_indices]  # (N, 3) vertex indices
    n0 = face_vertex_normals[tri_verts[:, 0]]
    n1 = face_vertex_normals[tri_verts[:, 1]]
    n2 = face_vertex_normals[tri_verts[:, 2]]
    smooth_normals = (
        barycentric[:, 0:1] * n0
        + barycentric[:, 1:2] * n1
        + barycentric[:, 2:3] * n2
    )
    lengths = np.linalg.norm(smooth_normals, axis=1, keepdims=True)
    smooth_normals /= np.maximum(lengths, 1e-10)

    # Subtle color variation by position (forehead lighter, cheeks rosier)
    # Normalize Y position to [0, 1] range across the face
    y_vals = hit_points[:, 1]
    y_min, y_max = y_vals.min(), y_vals.max()
    y_range = max(y_max - y_min, 1e-6)
    y_norm = (y_vals - y_min) / y_range  # 0=bottom, 1=top

    # Forehead (top): slightly lighter
    # Cheeks (mid): slightly rosier
    # Chin (bottom): slightly darker
    variation = np.ones((n, 3))
    variation[:, 0] += 0.04 * np.sin(y_norm * np.pi)  # red boost mid-face
    variation[:, 1] -= 0.02 * (1.0 - y_norm)  # less green at bottom
    variation[:, 2] -= 0.03 * (1.0 - y_norm)  # less blue at bottom

    local_color = base_color * variation

    # Lambertian diffuse
    to_light = light_position - hit_points
    dist = np.linalg.norm(to_light, axis=1, keepdims=True)
    to_light /= np.maximum(dist, 1e-10)
    ndotl = np.sum(smooth_normals * to_light, axis=1)
    ndotl = np.clip(ndotl, 0.0, 1.0)

    diffuse = ambient + (1.0 - ambient) * ndotl * light_intensity
    colors = local_color * diffuse[:, np.newaxis]

    return np.clip(colors, 0.0, 1.0)
