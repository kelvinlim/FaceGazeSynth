"""Tests for Phase 4: physics-guided gaze redirection on real photos."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Skip all tests if sample photos are not available
SAMPLES_DIR = Path("samples")
SAMPLE_FILES = sorted(SAMPLES_DIR.glob("*_Center.png"))
HAS_SAMPLES = len(SAMPLE_FILES) > 0

pytestmark = pytest.mark.skipif(not HAS_SAMPLES, reason="Sample photos not found")


@pytest.fixture(scope="module")
def dean_image():
    return np.array(Image.open("samples/Dean_Center.png").convert("RGB"))


@pytest.fixture(scope="module")
def dean_detection(dean_image):
    from facegazesynth.redirection.detection import detect_eyes
    return detect_eyes(dean_image)


# --- Detection tests ---


class TestEyeDetection:
    def test_detect_both_eyes(self, dean_detection):
        """Both eyes detected."""
        assert dean_detection.left_eye is not None
        assert dean_detection.right_eye is not None

    def test_iris_radius_reasonable(self, dean_detection):
        """Iris radius is in expected range for ~512px images."""
        for eye in [dean_detection.left_eye, dean_detection.right_eye]:
            assert 8 < eye.iris_radius < 50, f"Iris radius {eye.iris_radius} outside range"

    def test_iris_center_within_image(self, dean_detection):
        """Iris centers are within image bounds."""
        h, w = dean_detection.image_shape
        for eye in [dean_detection.left_eye, dean_detection.right_eye]:
            assert 0 < eye.iris_center[0] < w
            assert 0 < eye.iris_center[1] < h

    def test_eye_width_positive(self, dean_detection):
        """Eye width (corner-to-corner) is positive."""
        for eye in [dean_detection.left_eye, dean_detection.right_eye]:
            assert eye.eye_width > 10

    def test_all_samples_detected(self):
        """Detection succeeds on all sample photos."""
        from facegazesynth.redirection.detection import detect_eyes

        for f in SAMPLE_FILES:
            img = np.array(Image.open(f).convert("RGB"))
            det = detect_eyes(img)
            assert det.left_eye is not None
            assert det.right_eye is not None


# --- Physics mapping tests ---


class TestPhysicsMapping:
    def test_zero_angle_zero_displacement(self, dean_detection):
        from facegazesynth.redirection.physics_mapping import (
            calibrate_eye, target_displacement_px,
        )
        mapping = calibrate_eye(dean_detection.right_eye)
        disp = target_displacement_px(0.0, mapping)
        assert abs(disp) < 1e-6

    def test_displacement_increases_with_angle(self, dean_detection):
        from facegazesynth.redirection.physics_mapping import (
            calibrate_eye, target_displacement_px,
        )
        mapping = calibrate_eye(dean_detection.right_eye)
        prev = 0.0
        for angle in [5, 10, 15, 20]:
            disp = abs(target_displacement_px(float(angle), mapping))
            assert disp > prev, f"Displacement not increasing at {angle} deg"
            prev = disp

    def test_mm_per_pixel_reasonable(self, dean_detection):
        from facegazesynth.redirection.physics_mapping import calibrate_eye
        mapping = calibrate_eye(dean_detection.right_eye)
        # For ~512px face images, expect ~0.3-1.0 mm/px
        assert 0.1 < mapping.mm_per_pixel < 2.0


# --- Warping tests ---


class TestWarping:
    def test_zero_displacement_identity(self, dean_image, dean_detection):
        from facegazesynth.redirection.warping import warp_eye_region
        result, _, _, _ = warp_eye_region(
            dean_image, dean_detection.right_eye, 0.0, 0.0,
        )
        assert result.shape == dean_image.shape
        # With zero displacement, result should be very close to original
        diff = np.abs(result.astype(float) - dean_image.astype(float))
        assert diff.mean() < 1.0, "Zero-displacement warp should preserve image"

    def test_warp_preserves_size(self, dean_image, dean_detection):
        from facegazesynth.redirection.warping import warp_eye_region
        result, _, _, _ = warp_eye_region(
            dean_image, dean_detection.right_eye, 5.0, 10.0,
        )
        assert result.shape == dean_image.shape


# --- End-to-end tests ---


class TestEndToEnd:
    def test_redirect_single(self, dean_image, dean_detection):
        from facegazesynth.redirection.compositing import redirect_both_eyes
        result = redirect_both_eyes(dean_image, dean_detection, 10.0)
        assert result.shape == dean_image.shape
        assert result.dtype == np.uint8

    def test_redirect_produces_different_image(self, dean_image, dean_detection):
        from facegazesynth.redirection.compositing import redirect_both_eyes
        result = redirect_both_eyes(dean_image, dean_detection, 20.0)
        assert not np.array_equal(result, dean_image), "20° redirect should differ"

    def test_left_right_symmetry(self, dean_image, dean_detection):
        from facegazesynth.redirection.compositing import redirect_both_eyes
        left = redirect_both_eyes(dean_image, dean_detection, -10.0)
        right = redirect_both_eyes(dean_image, dean_detection, 10.0)
        # The two images should be different from each other
        assert not np.array_equal(left, right)

    def test_sweep_generates_grid(self):
        from facegazesynth.pipeline.redirect import redirect_gaze_sweep
        grid = redirect_gaze_sweep("samples/Dean_Center.png", angles=[-10, 0, 10])
        assert grid.width > grid.height  # horizontal grid
