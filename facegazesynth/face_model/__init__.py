"""Face model module: FLAME mesh loading and eye-face composition."""

from .flame_mesh import FaceMesh, build_face_mesh
from .composition import CompositeScene, compose_face_with_eyes

__all__ = [
    "FaceMesh",
    "build_face_mesh",
    "CompositeScene",
    "compose_face_with_eyes",
]
