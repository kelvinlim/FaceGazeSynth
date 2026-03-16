"""FLAME mesh loader and face mesh generation.

Wraps smplx.create() to load the FLAME 2023 Open model, build a face mesh
with configurable identity/expression, remove FLAME's built-in eyeball
submeshes, and extract eye joint positions for our physics-based eyeballs.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch

# Default model path
DEFAULT_FLAME_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "flame2023"
)


@dataclass
class FaceMesh:
    """Face mesh with eyeball vertices removed."""

    vertices: np.ndarray  # (N, 3) in mm
    faces: np.ndarray  # (F, 3) triangle indices (re-indexed)
    vertex_normals: np.ndarray  # (N, 3)
    original_face_indices: np.ndarray  # (F,) maps to original FLAME face indices
    left_eye_joint: np.ndarray  # (3,) in mm
    right_eye_joint: np.ndarray  # (3,) in mm
    n_shape: int  # number of shape params used
    n_expression: int  # number of expression params used


def _compute_vertex_normals(
    vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Compute per-vertex normals by averaging adjacent face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    # Accumulate face normals to vertices
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)

    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)
    normals /= lengths
    return normals


def _remove_eyeball_vertices(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove disconnected eyeball submeshes from the FLAME mesh.

    FLAME has 3 connected components: main face + 2 eyeball submeshes.
    The main face is the largest component.

    Returns:
        (face_vertices, face_faces, kept_mask, original_face_idx) — vertices
        and faces for the main face component only, a boolean mask of which
        original vertices were kept, and indices into the original face array.
    """
    n_verts = vertices.shape[0]
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adj = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_verts, n_verts)
    )
    n_components, labels = connected_components(adj, directed=False)

    # Main face is the largest component
    component_sizes = np.bincount(labels)
    main_component = np.argmax(component_sizes)

    kept_mask = labels == main_component
    old_to_new = np.full(n_verts, -1, dtype=int)
    old_to_new[kept_mask] = np.arange(kept_mask.sum())

    # Filter faces: keep only faces where all 3 vertices are in main component
    face_kept = kept_mask[faces[:, 0]] & kept_mask[faces[:, 1]] & kept_mask[faces[:, 2]]
    original_face_idx = np.where(face_kept)[0]
    new_faces = old_to_new[faces[face_kept]]

    return vertices[kept_mask], new_faces, kept_mask, original_face_idx


_cached_model = None
_cached_model_path = None


def _get_flame_model(model_path: str):
    """Load FLAME model with caching (avoids repeated deserialization)."""
    global _cached_model, _cached_model_path
    if _cached_model is not None and _cached_model_path == model_path:
        return _cached_model

    import smplx

    model = smplx.create(
        model_path=model_path,
        model_type="flame",
    )
    model.eval()
    _cached_model = model
    _cached_model_path = model_path
    return model


def build_face_mesh(
    betas: Optional[np.ndarray] = None,
    expression: Optional[np.ndarray] = None,
    jaw_pose: Optional[np.ndarray] = None,
    model_path: Optional[str] = None,
) -> FaceMesh:
    """Build a FLAME face mesh with optional identity and expression.

    Args:
        betas: (n_shape,) identity coefficients. Default: zeros (neutral).
        expression: (n_expression,) expression coefficients. Default: zeros.
        jaw_pose: (3,) axis-angle jaw rotation. Default: zeros (closed mouth).
        model_path: Path to FLAME model directory. Default: models/flame2023/.

    Returns:
        FaceMesh with eyeball vertices removed, in millimeters.
    """
    if model_path is None:
        model_path = os.path.abspath(DEFAULT_FLAME_PATH)

    model = _get_flame_model(model_path)

    # Prepare inputs
    n_shape = 10
    n_expression = 10
    betas_t = torch.zeros(1, n_shape)
    expression_t = torch.zeros(1, n_expression)
    jaw_pose_t = torch.zeros(1, 3)

    if betas is not None:
        n = min(len(betas), n_shape)
        betas_t[0, :n] = torch.from_numpy(np.asarray(betas[:n], dtype=np.float32))
    if expression is not None:
        n = min(len(expression), n_expression)
        expression_t[0, :n] = torch.from_numpy(
            np.asarray(expression[:n], dtype=np.float32)
        )
    if jaw_pose is not None:
        jaw_pose_t[0] = torch.from_numpy(
            np.asarray(jaw_pose, dtype=np.float32)
        )

    with torch.no_grad():
        output = model(
            betas=betas_t,
            expression=expression_t,
            jaw_pose=jaw_pose_t,
        )

    # Convert to numpy, meters → mm
    verts_m = output.vertices[0].numpy()  # (5023, 3)
    joints_m = output.joints[0].numpy()  # (110, 3)
    faces_all = model.faces_tensor.numpy()  # (9976, 3)

    verts_mm = verts_m * 1000.0
    joints_mm = joints_m * 1000.0

    # Eye joints: joint 3 = right eye, joint 4 = left eye
    right_eye_joint = joints_mm[3]
    left_eye_joint = joints_mm[4]

    # Remove eyeball submeshes
    face_verts, face_faces, _, orig_face_idx = _remove_eyeball_vertices(verts_mm, faces_all)

    # Compute normals
    normals = _compute_vertex_normals(face_verts, face_faces)

    return FaceMesh(
        vertices=face_verts,
        faces=face_faces,
        vertex_normals=normals,
        original_face_indices=orig_face_idx,
        left_eye_joint=left_eye_joint,
        right_eye_joint=right_eye_joint,
        n_shape=n_shape,
        n_expression=n_expression,
    )
