"""Eye and iris detection using MediaPipe FaceLandmarker (478 landmarks)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Default model path
_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"

# MediaPipe Face Mesh landmark indices for iris and eye contours.
# Iris: 5 landmarks per eye — center + 4 cardinal points.
LEFT_IRIS_CENTER = 468
LEFT_IRIS_CARDINALS = [469, 470, 471, 472]
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_CARDINALS = [474, 475, 476, 477]

# Eye contour landmarks.
LEFT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]
LEFT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LEFT_EYE_INNER_CORNER = 133
LEFT_EYE_OUTER_CORNER = 33

RIGHT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
RIGHT_EYE_INNER_CORNER = 362
RIGHT_EYE_OUTER_CORNER = 263


@dataclass
class EyeDetection:
    """Detection result for a single eye."""
    iris_center: np.ndarray    # (2,) pixel coordinates (x, y)
    iris_radius: float         # pixels
    eye_corners: tuple         # (inner_corner, outer_corner) as (2,) arrays
    eyelid_upper: np.ndarray   # (K, 2) upper eyelid contour points
    eyelid_lower: np.ndarray   # (K, 2) lower eyelid contour points
    eye_width: float           # pixels, corner-to-corner distance


@dataclass
class FaceDetection:
    """Detection result for a full face."""
    left_eye: EyeDetection     # subject's left eye (viewer's right)
    right_eye: EyeDetection    # subject's right eye (viewer's left)
    image_shape: tuple          # (H, W)


def _lm_to_px(landmarks, indices, h, w):
    """Convert normalized landmarks at given indices to pixel coords."""
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])


def _extract_eye(landmarks, h, w, iris_center_idx, iris_cardinal_idx,
                 upper_idx, lower_idx, inner_idx, outer_idx):
    """Extract EyeDetection from landmark list."""
    iris_center = np.array([
        landmarks[iris_center_idx].x * w,
        landmarks[iris_center_idx].y * h,
    ])
    cardinal_pts = _lm_to_px(landmarks, iris_cardinal_idx, h, w)
    iris_radius = np.mean(np.linalg.norm(cardinal_pts - iris_center, axis=1))

    inner = np.array([landmarks[inner_idx].x * w, landmarks[inner_idx].y * h])
    outer = np.array([landmarks[outer_idx].x * w, landmarks[outer_idx].y * h])
    eye_width = np.linalg.norm(outer - inner)

    upper = _lm_to_px(landmarks, upper_idx, h, w)
    lower = _lm_to_px(landmarks, lower_idx, h, w)

    return EyeDetection(
        iris_center=iris_center,
        iris_radius=iris_radius,
        eye_corners=(inner, outer),
        eyelid_upper=upper,
        eyelid_lower=lower,
        eye_width=eye_width,
    )


def detect_eyes(image: np.ndarray, model_path: str = None) -> FaceDetection:
    """Detect eyes and iris in a face image using MediaPipe FaceLandmarker.

    Args:
        image: (H, W, 3) uint8 RGB image.
        model_path: Path to face_landmarker.task model file.

    Returns:
        FaceDetection with left and right eye data.

    Raises:
        ValueError: If no face is detected.
    """
    h, w = image.shape[:2]
    model = str(model_path or _DEFAULT_MODEL)

    base_options = mp.tasks.BaseOptions(model_asset_path=model)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face detected in image")

    landmarks = result.face_landmarks[0]

    left_eye = _extract_eye(
        landmarks, h, w,
        LEFT_IRIS_CENTER, LEFT_IRIS_CARDINALS,
        LEFT_EYE_UPPER, LEFT_EYE_LOWER,
        LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER,
    )
    right_eye = _extract_eye(
        landmarks, h, w,
        RIGHT_IRIS_CENTER, RIGHT_IRIS_CARDINALS,
        RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
        RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER,
    )

    return FaceDetection(
        left_eye=left_eye,
        right_eye=right_eye,
        image_shape=(h, w),
    )


def draw_debug_overlay(image: np.ndarray, detection: FaceDetection) -> np.ndarray:
    """Draw detection landmarks on the image for debugging."""
    out = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    for eye, color in [(detection.left_eye, (0, 255, 0)),
                       (detection.right_eye, (255, 0, 0))]:
        cx, cy = int(eye.iris_center[0]), int(eye.iris_center[1])
        cv2.circle(out, (cx, cy), int(eye.iris_radius), color, 1)
        cv2.circle(out, (cx, cy), 2, color, -1)

        for corner in eye.eye_corners:
            cv2.circle(out, (int(corner[0]), int(corner[1])), 3, (0, 255, 255), -1)

        for contour in [eye.eyelid_upper, eye.eyelid_lower]:
            pts = contour.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, color, 1)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
