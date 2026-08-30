from __future__ import annotations

import numpy as np
import pytest

from measurepilot.calibration import CalibrationError
from measurepilot.quality import evaluate_capture_quality, to_grayscale


def test_sharp_checkerboard_has_no_warning() -> None:
    image = ((np.indices((128, 128)).sum(axis=0) % 2) * 255).astype(np.uint8)
    quality = evaluate_capture_quality(image, sharpness_threshold=80.0)
    assert quality.width_px == 128
    assert quality.height_px == 128
    assert quality.sharpness > 80.0
    assert quality.warnings == ()


def test_flat_capture_emits_bounded_low_sharpness_warning() -> None:
    image = np.full((80, 120, 3), 127, dtype=np.uint8)
    quality = evaluate_capture_quality(image, sharpness_threshold=1.0)
    assert quality.sharpness == 0.0
    assert quality.warnings == (
        "LOW_SHARPNESS: capture may be unsuitable for later contour extraction",
    )
    assert quality.as_dict()["warnings"] == list(quality.warnings)


def test_float_image_is_converted_to_uint8_grayscale() -> None:
    image = np.zeros((10, 12, 3), dtype=np.float32)
    image[:, :, 1] = 1.0
    gray = to_grayscale(image)
    assert gray.shape == (10, 12)
    assert gray.dtype == np.uint8
    assert int(gray.max()) > 0


def test_invalid_quality_inputs_are_rejected() -> None:
    with pytest.raises(CalibrationError, match="empty"):
        evaluate_capture_quality(np.empty((0, 0), dtype=np.uint8))
    with pytest.raises(CalibrationError, match="sharpness_threshold"):
        evaluate_capture_quality(np.zeros((10, 10), dtype=np.uint8), sharpness_threshold=-1.0)
