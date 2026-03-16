"""Albedo texture model: data-driven skin appearance from AlbedoMM PCA model.

Samples diverse skin textures by combining a mean albedo with PCA components.
Maps textures onto FLAME mesh faces via UV coordinates.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Default path to albedo model
DEFAULT_ALBEDO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "flame2023",
    "albedoModel2020_FLAME_albedoPart.npz",
)

_cached_albedo = None
_cached_albedo_path = None


@dataclass
class AlbedoModel:
    """Loaded albedo PCA model."""

    mean: np.ndarray  # (512, 512, 3) mean diffuse texture
    components: np.ndarray  # (512, 512, 3, n_components)
    spec_mean: np.ndarray  # (512, 512, 3) mean specular
    spec_components: np.ndarray  # (512, 512, 3, n_components)
    uv_coords: np.ndarray  # (n_uv, 2) UV texture coordinates
    uv_faces: np.ndarray  # (n_faces, 3) texture face indices
    n_components: int


def load_albedo_model(path: Optional[str] = None) -> AlbedoModel:
    """Load the AlbedoMM PCA model with caching."""
    global _cached_albedo, _cached_albedo_path
    if path is None:
        path = os.path.abspath(DEFAULT_ALBEDO_PATH)
    if _cached_albedo is not None and _cached_albedo_path == path:
        return _cached_albedo

    data = np.load(path, allow_pickle=True)
    # AlbedoMM stores textures in BGR order — convert to RGB
    model = AlbedoModel(
        mean=data["MU"][:, :, ::-1].astype(np.float32),
        components=data["PC"][:, :, ::-1, :].astype(np.float32),
        spec_mean=data["specMU"][:, :, ::-1].astype(np.float32),
        spec_components=data["specPC"][:, :, ::-1, :].astype(np.float32),
        uv_coords=data["vt"].astype(np.float32),
        uv_faces=data["ft"].astype(np.int32),
        n_components=data["PC"].shape[3],
    )
    _cached_albedo = model
    _cached_albedo_path = path
    return model


def sample_albedo_texture(
    coefficients: Optional[np.ndarray] = None,
    n_components: int = 30,
    albedo_model: Optional[AlbedoModel] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample a diffuse albedo texture from the PCA model.

    Args:
        coefficients: (n_components,) PCA coefficients. If None, samples
            randomly from a normal distribution scaled to match the model.
        n_components: Number of PCA components to use (max 145).
        albedo_model: Pre-loaded model. Loads default if None.
        seed: Random seed for reproducible sampling.

    Returns:
        (512, 512, 3) RGB albedo texture in [0, 1].
    """
    if albedo_model is None:
        albedo_model = load_albedo_model()

    n_components = min(n_components, albedo_model.n_components)

    if coefficients is None:
        rng = np.random.RandomState(seed)
        # PCA components have large norms (~80 for PC0, decreasing).
        # Scale coefficients so each component contributes ~5% variation
        # relative to the mean (~0.35).
        pc = albedo_model.components[:, :, :, :n_components]
        component_norms = np.array([
            np.linalg.norm(pc[:, :, :, i]) for i in range(n_components)
        ])
        target_variation = 0.03  # ~3% of mean per component
        scales = target_variation / np.maximum(component_norms, 1e-10)
        coefficients = rng.randn(n_components).astype(np.float32) * scales
    else:
        coefficients = np.asarray(coefficients, dtype=np.float32)[:n_components]

    # texture = mean + sum(coeff_i * PC_i)
    pc = albedo_model.components[:, :, :, :n_components]  # (512, 512, 3, n)
    texture = albedo_model.mean + np.tensordot(pc, coefficients, axes=(3, 0))

    return np.clip(texture, 0.0, 1.0)


def lookup_albedo_at_triangles(
    albedo_texture: np.ndarray,
    triangle_indices: np.ndarray,
    barycentric: np.ndarray,
    albedo_model: Optional[AlbedoModel] = None,
) -> np.ndarray:
    """Look up albedo color for hit points using UV mapping.

    Args:
        albedo_texture: (512, 512, 3) sampled texture.
        triangle_indices: (N,) which mesh triangle was hit.
        barycentric: (N, 3) barycentric coordinates within each triangle.
        albedo_model: Pre-loaded model with UV data.

    Returns:
        (N, 3) RGB albedo colors per hit point.
    """
    if albedo_model is None:
        albedo_model = load_albedo_model()

    h, w = albedo_texture.shape[:2]

    # Get UV coordinates for each triangle's vertices
    uv_faces = albedo_model.uv_faces  # (F, 3) indices into uv_coords
    uv_coords = albedo_model.uv_coords  # (n_uv, 2)

    tri_uv_idx = uv_faces[triangle_indices]  # (N, 3)
    uv0 = uv_coords[tri_uv_idx[:, 0]]  # (N, 2)
    uv1 = uv_coords[tri_uv_idx[:, 1]]
    uv2 = uv_coords[tri_uv_idx[:, 2]]

    # Interpolate UV using barycentric coordinates
    uv = (
        barycentric[:, 0:1] * uv0
        + barycentric[:, 1:2] * uv1
        + barycentric[:, 2:3] * uv2
    )

    # Convert UV to pixel coordinates (nearest-neighbor sampling)
    px = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    py = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)

    return albedo_texture[py, px]
