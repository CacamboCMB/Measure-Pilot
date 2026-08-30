from __future__ import annotations

import cv2
import numpy as np
import pytest

from measurepilot.calibration import CalibrationError, render_calibration_image
from measurepilot.quality import QualityThresholds, quality_warnings, sharpness_score, to_grayscale


def test_sharpness_score_separates_sharp_and_blurred_images() -> None:
    sharp = render_calibration_image(px_per_mm=4.0)
    blurred = cv2.GaussianBlur(sharp, (31, 31), 8.0)

    assert sharpness_score(sharp) > sharpness_score(blurred) * 10.0


def test_warning_generation_is_bounded_and_ordered() -> None:
    thresholds = QualityThresholds(
        min_sharpness_score=50.0,
        min_input_px_per_mm=2.0,
        max_reprojection_error_px=2.0,
        reprojection_warning_ratio=0.5,
    )

    warnings = quality_warnings(
        sharpness=20.0,
        input_px_per_mm=1.0,
        reprojection_error_px=1.25,
        thresholds=thresholds,
    )

    assert [warning.split(":", 1)[0] for warning in warnings] == [
        "LOW_SHARPNESS",
        "LOW_METRIC_RESOLUTION",
        "ELEVATED_REPROJECTION_ERROR",
    ]


def test_values_at_warning_threshold_do_not_warn() -> None:
    thresholds = QualityThresholds(
        min_sharpness_score=50.0,
        min_input_px_per_mm=2.0,
        max_reprojection_error_px=2.0,
        reprojection_warning_ratio=0.5,
    )

    assert quality_warnings(
        sharpness=50.0,
        input_px_per_mm=2.0,
        reprojection_error_px=1.0,
        thresholds=thresholds,
    ) == []


def test_grayscale_accepts_bgr_and_rejects_unsupported_shape() -> None:
    bgr = np.zeros((10, 20, 3), dtype=np.uint8)
    assert to_grayscale(bgr).shape == (10, 20)

    with pytest.raises(CalibrationError, match="unsupported image shape"):
        to_grayscale(np.zeros((10, 20, 2), dtype=np.uint8))


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(CalibrationError, match="between zero and one"):
        QualityThresholds(reprojection_warning_ratio=1.0).validate()
