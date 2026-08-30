from __future__ import annotations

import cv2
import numpy as np
import pytest

from measurepilot.features import FeatureConfig, extract_features
from measurepilot.segmentation import AnalysisError

PX_PER_MM = 4.0


def _plate_mask() -> np.ndarray:
    mask = np.zeros((1188, 840), dtype=np.uint8)
    cv2.rectangle(mask, (240, 740), (600, 940), 255, -1)
    cv2.circle(mask, (320, 840), 20, 0, -1)
    cv2.circle(mask, (520, 840), 20, 0, -1)
    return mask


def test_rectangle_and_two_circular_holes_are_classified() -> None:
    result = extract_features(_plate_mask(), px_per_mm=PX_PER_MM)
    report = result.report
    assert len(report["features"]["line_segments"]) == 4
    assert len(report["features"]["circular_holes"]) == 2
    assert report["unresolved"] == []
    centers = [
        hole["center_mm"]
        for hole in report["features"]["circular_holes"]
    ]
    assert centers[0][0] == pytest.approx(80.0, abs=0.3)
    assert centers[1][0] == pytest.approx(130.0, abs=0.3)
    for hole in report["features"]["circular_holes"]:
        assert hole["diameter_mm"] == pytest.approx(10.0, abs=0.6)
        assert hole["circularity"] >= 0.84
        assert 0.0 <= hole["confidence"] <= 1.0


def test_non_circular_hole_is_retained_as_unresolved() -> None:
    mask = np.zeros((1188, 840), dtype=np.uint8)
    cv2.rectangle(mask, (240, 740), (600, 940), 255, -1)
    cv2.rectangle(mask, (300, 810), (420, 850), 0, -1)
    result = extract_features(mask, px_per_mm=PX_PER_MM)
    assert result.report["features"]["circular_holes"] == []
    assert any(
        item["code"] == "NON_CIRCULAR_HOLE"
        for item in result.report["unresolved"]
    )
    assert len(result.report["contours"]["holes"]) == 1


def test_contour_output_is_closed_and_deterministic() -> None:
    first = extract_features(_plate_mask(), px_per_mm=PX_PER_MM)
    second = extract_features(_plate_mask(), px_per_mm=PX_PER_MM)
    assert first.report == second.report
    points = first.report["contours"]["outer"]["points_mm"]
    simplified = first.report["contours"]["outer"]["simplified_points_mm"]
    assert points[0] == points[-1]
    assert simplified[0] == simplified[-1]
    assert points[0] == min(points[:-1])


def test_empty_mask_is_rejected() -> None:
    with pytest.raises(AnalysisError, match="empty"):
        extract_features(
            np.zeros((100, 100), dtype=np.uint8),
            px_per_mm=PX_PER_MM,
        )


def test_feature_configuration_is_validated() -> None:
    with pytest.raises(AnalysisError, match="simplify_tolerance_mm"):
        extract_features(
            _plate_mask(),
            px_per_mm=PX_PER_MM,
            config=FeatureConfig(simplify_tolerance_mm=0.0),
        )
