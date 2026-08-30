from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from measurepilot.calibration import MARKER_IDS, CalibrationError, render_calibration_image
from measurepilot.rectification import (
    _validate_detected_markers,
    detect_required_markers,
    rectify_file,
    rectify_image,
)


def _perspective_capture() -> tuple[np.ndarray, np.ndarray]:
    reference = render_calibration_image(4.0)
    height, width = reference.shape
    source = np.array(
        ((0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)),
        dtype=np.float32,
    )
    destination = np.array(
        ((95.0, 75.0), (905.0, 42.0), (944.0, 1244.0), (52.0, 1282.0)),
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    capture = cv2.warpPerspective(
        reference,
        transform,
        (1000, 1330),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return reference, capture


def _square(cx: float, cy: float, half_size: float = 10.0) -> np.ndarray:
    return np.array(
        (
            (cx - half_size, cy - half_size),
            (cx + half_size, cy - half_size),
            (cx + half_size, cy + half_size),
            (cx - half_size, cy + half_size),
        ),
        dtype=np.float64,
    )


def test_perspective_capture_is_rectified_to_exact_metric_shape() -> None:
    reference, capture = _perspective_capture()
    detected = detect_required_markers(capture)
    assert tuple(detected) == MARKER_IDS

    result = rectify_image(capture, px_per_mm=4.0)
    assert result.image.shape == reference.shape == (1188, 840)
    assert result.report["marker_ids"] == [0, 1, 2, 3]
    assert result.report["output_image"] == {"height_px": 1188, "width_px": 840}
    assert result.report["px_per_mm"] == 4.0
    assert result.report["reprojection_error_px"] < 0.5
    assert len(result.report["homography"]) == 3


def test_missing_required_marker_is_explicit_failure() -> None:
    image = render_calibration_image(4.0)
    image[55:162, 55:162] = 255
    with pytest.raises(CalibrationError, match="missing required marker IDs: 0"):
        detect_required_markers(image)


def test_duplicate_required_marker_is_rejected() -> None:
    detections = {0: [_square(20.0, 20.0), _square(50.0, 20.0)]}
    with pytest.raises(CalibrationError, match="duplicate required marker IDs: 0"):
        _validate_detected_markers(detections)


def test_inconsistent_marker_order_is_rejected() -> None:
    detections = {
        0: [_square(20.0, 20.0)],
        1: [_square(180.0, 20.0)],
        2: [_square(80.0, 80.0)],
        3: [_square(20.0, 180.0)],
    }
    with pytest.raises(CalibrationError, match="inconsistent page ordering"):
        _validate_detected_markers(detections)


def test_rectify_file_writes_png_and_machine_readable_report(tmp_path) -> None:
    _, capture = _perspective_capture()
    source = tmp_path / "capture.png"
    output = tmp_path / "rectified.png"
    report_path = tmp_path / "report.json"
    assert cv2.imwrite(str(source), capture)

    report = rectify_file(source, output, report_path, px_per_mm=4.0)
    written_image = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    written_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert written_image is not None
    assert written_image.shape == (1188, 840)
    assert written_report == report
    assert written_report["dictionary"] == "DICT_4X4_50"
    assert written_report["layout_version"] == 1
    assert set(written_report) >= {
        "marker_ids",
        "input_image",
        "output_image",
        "px_per_mm",
        "sharpness",
        "homography",
        "reprojection_error_px",
        "warnings",
    }


def test_overlapping_paths_do_not_modify_the_source_or_publish_report(tmp_path) -> None:
    _, capture = _perspective_capture()
    source = tmp_path / "capture.png"
    report_path = tmp_path / "report.json"
    assert cv2.imwrite(str(source), capture)
    before = source.read_bytes()

    with pytest.raises(CalibrationError, match="paths must be distinct"):
        rectify_file(source, source, report_path)

    assert source.read_bytes() == before
    assert not report_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_output_scale_is_rejected() -> None:
    image = render_calibration_image(4.0)
    with pytest.raises(CalibrationError, match="px_per_mm"):
        rectify_image(image, px_per_mm=float("nan"))
