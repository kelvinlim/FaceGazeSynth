"""Emotion presets: map emotion labels to FLAME expression coefficients.

FLAME uses 10 expression blendshape coefficients. These presets provide
approximate mappings from common emotion labels to those coefficients,
loosely based on FACS (Facial Action Coding System) action unit activations.
"""

import numpy as np
from typing import Optional

# FLAME expression coefficients are abstract PCA components, not direct
# FACS action units. These presets were tuned to produce visually
# recognizable expressions with the FLAME 2023 model.
EMOTION_PRESETS: dict[str, np.ndarray] = {
    "neutral": np.zeros(10),
    "happy": np.array([1.5, 0.0, 0.0, 0.8, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),
    "sad": np.array([-0.5, 0.3, 0.0, -0.3, 0.5, 0.0, -0.2, 0.0, 0.3, 0.0]),
    "angry": np.array([-0.8, -0.5, 0.0, 0.0, 0.8, 0.0, 0.0, 0.3, 0.0, 0.0]),
    "surprised": np.array([0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5]),
    "disgusted": np.array([-0.3, -0.3, 0.5, 0.0, 0.5, 0.3, -0.3, 0.0, 0.0, 0.0]),
    "fearful": np.array([0.5, 0.5, 0.0, -0.3, 0.3, 0.0, -0.2, 0.0, -0.3, 0.3]),
    "contempt": np.array([0.3, 0.0, 0.3, 0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]),
}

# Jaw pose presets (axis-angle rotation, typically around X for open/close)
JAW_PRESETS: dict[str, np.ndarray] = {
    "neutral": np.zeros(3),
    "happy": np.array([0.1, 0.0, 0.0]),  # slightly open
    "sad": np.zeros(3),
    "angry": np.array([0.05, 0.0, 0.0]),
    "surprised": np.array([0.4, 0.0, 0.0]),  # mouth open
    "disgusted": np.array([0.15, 0.0, 0.0]),
    "fearful": np.array([0.3, 0.0, 0.0]),
    "contempt": np.zeros(3),
}


def get_emotion_params(
    emotion: str,
    intensity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get FLAME expression and jaw_pose for an emotion label.

    Args:
        emotion: Emotion name (e.g., "happy", "sad", "angry").
        intensity: Scale factor for the expression (0.0 = neutral, 1.0 = full).

    Returns:
        (expression, jaw_pose): FLAME-compatible parameter arrays.

    Raises:
        ValueError: If emotion name is not recognized.
    """
    emotion = emotion.lower().strip()
    if emotion not in EMOTION_PRESETS:
        available = ", ".join(sorted(EMOTION_PRESETS.keys()))
        raise ValueError(f"Unknown emotion '{emotion}'. Available: {available}")

    expression = EMOTION_PRESETS[emotion] * intensity
    jaw_pose = JAW_PRESETS[emotion] * intensity

    return expression, jaw_pose


def list_emotions() -> list[str]:
    """Return list of available emotion names."""
    return sorted(EMOTION_PRESETS.keys())


def random_identity(
    n_components: int = 10,
    scale: float = 1.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random FLAME identity (beta) coefficients.

    Args:
        n_components: Number of shape components.
        scale: Standard deviation for sampling.
        seed: Random seed.

    Returns:
        (n_components,) beta coefficients.
    """
    rng = np.random.RandomState(seed)
    return rng.randn(n_components).astype(np.float32) * scale
