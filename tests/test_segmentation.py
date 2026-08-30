from __future__ import annotations

import cv2
import numpy as np
import pytest

from measurepilot.calibration import render_calibration_image
from measurepilot.segmentation import AnalysisError, SegmentationConfig, segment_foreground

PX_PER_MM = 4.0


def _dark_plate(*, holes: bool = True) -> np.ndarray:
    image = render_calibration_image(PX_PER_MM)
    cv2.rectangle(image, (240, 740), (600, 940), 35, -1)
    if holes:
        cv2.circle(image, (320, 840), 20, 255, -1)
        cv2.circle(image, (520, 840), 20, 255, -1)
    return image


def test_explicit_dark_polarity_selects_one_plate() -> None:
    result = segment_foreground(
        _dark_plate(),
        SegmentationConfig(px_per_mm=PX_PER_MM, polarity="dark"),
    )
    assert result.polarity == "dark"
    assert result.area_mm2 > 4_000.0
    assert result.bbox_mm[0] == pytest.approx(60.0, abs=1.0)
    assert result.bbox_mm[1] == pytest.approx(185.0, abs=1.0)
    assert result.dominance == pytest.approx(1.0)


def test_auto_polarity_selects_only_valid_interpretation() -> None:
    dark = segment_foreground(
        _dark_plate(),
        SegmentationConfig(px_per_mm=PX_PER_MM, polarity="auto"),
    )
    assert dark.polarity == "dark"

    light_image = np.full((1188, 840), 20, dtype=np.uint8)
    cv2.rectangle(light_image, (240, 740), (600, 940), 230, -1)
    light = segment_foreground(
        light_image,
        SegmentationConfig(px_per_mm=PX_PER_MM, polarity="auto"),
    )
    assert light.polarity == "light"


def test_marker_regions_do_not_become_a_part() -> None:
    with pytest.raises(AnalysisError, match="no qualifying dark foreground"):
        segment_foreground(
            render_calibration_image(PX_PER_MM),
            SegmentationConfig(px_per_mm=PX_PER_MM, polarity="dark"),
        )


def test_two_similarly_sized_parts_are_rejected_as_ambiguous() -> None:
    image = np.full((1188, 840), 255, dtype=np.uint8)
    cv2.rectangle(image, (200, 400), (360, 560), 20, -1)
    cv2.rectangle(image, (480, 400), (640, 560), 20, -1)
    with pytest.raises(AnalysisError, match="ambiguous dark foreground"):
        segment_foreground(
            image,
            SegmentationConfig(px_per_mm=PX_PER_MM, polarity="dark"),
        )


def test_component_touching_excluded_page_boundary_is_rejected() -> None:
    image = np.full((1188, 840), 255, dtype=np.uint8)
    cv2.rectangle(image, (0, 400), (220, 650), 20, -1)
    with pytest.raises(AnalysisError, match="touches an excluded boundary"):
        segment_foreground(
            image,
            SegmentationConfig(px_per_mm=PX_PER_MM, polarity="dark"),
        )


def test_wrong_metric_dimensions_are_rejected() -> None:
    with pytest.raises(AnalysisError, match="dimensions do not match"):
        segment_foreground(
            np.full((500, 500), 255, dtype=np.uint8),
            SegmentationConfig(px_per_mm=PX_PER_MM, polarity="dark"),
        )
