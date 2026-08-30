from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from measurepilot.calibration import DEFAULT_LAYOUT, REQUIRED_MARKER_IDS, CalibrationError, render_calibration_image
from measurepilot.cli import main
from measurepilot.rectification import _build_marker_map, detect_required_markers, rectify_image


def _perspective_capture(*, px_per_mm: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    page = render_calibration_image(px_per_mm=px_per_mm)
    height, width = page.shape
    page_corners = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    capture_corners = np.asarray(
        [[120.0, 80.0], [930.0, 145.0], [880.0, 1240.0], [75.0, 1165.0]],
        dtype=np.float32,
    )
    page_to_capture = cv2.getPerspectiveTransform(page_corners, capture_corners)
    capture = cv2.warpPerspective(
        page,
        page_to_capture,
        (1040, 1320),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return page, capture


def test_perspective_capture_is_rectified_to_metric_a4() -> None:
    _page, capture = _perspective_capture()

    rectified, report = rectify_image(capture, output_px_per_mm=4.0)

    assert rectified.shape == (1188, 840)
    assert report.marker_ids == REQUIRED_MARKER_IDS
    assert report.source_width_px == 1040
    assert report.source_height_px == 1320
    assert report.output_px_per_mm == 4.0
    assert report.input_px_per_mm_estimate > 3.0
    assert report.reprojection_error_px < 1.5
    assert np.isfinite(np.asarray(report.homography)).all()

    detected = detect_required_markers(rectified)
    for marker_id, corners in detected.items():
        expected = DEFAULT_LAYOUT.placement(marker_id).corners_mm() * 4.0
        assert np.max(np.abs(corners - expected)) <= 2.0


def test_missing_marker_causes_explicit_failure() -> None:
    page = render_calibration_image(px_per_mm=4.0)
    placement = DEFAULT_LAYOUT.placement(2)
    x0 = int(placement.x_mm * 4.0) - 4
    y0 = int(placement.y_mm * 4.0) - 4
    side = int(placement.size_mm * 4.0) + 8
    page[y0 : y0 + side, x0 : x0 + side] = 255

    with pytest.raises(CalibrationError, match=r"missing required marker IDs: \[2\]"):
        rectify_image(page)


def test_duplicate_marker_ids_are_rejected() -> None:
    square = np.asarray([[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]])

    with pytest.raises(CalibrationError, match="duplicate marker IDs"):
        _build_marker_map([square, square + 40.0], np.asarray([[0], [0]]))


def test_geometrically_inconsistent_marker_is_rejected() -> None:
    page = render_calibration_image(px_per_mm=4.0)
    placement = DEFAULT_LAYOUT.placement(2)
    x0 = int(placement.x_mm * 4.0)
    y0 = int(placement.y_mm * 4.0)
    side = int(placement.size_mm * 4.0)
    marker = page[y0 : y0 + side, x0 : x0 + side].copy()
    page[y0 - 8 : y0 + side + 8, x0 - 8 : x0 + side + 8] = 255
    page[y0 - 40 : y0 - 40 + side, x0 - 55 : x0 - 55 + side] = marker

    with pytest.raises(CalibrationError, match="marker geometry is inconsistent"):
        rectify_image(page)


def test_rectification_cli_writes_png_and_json_report(tmp_path: Path, capsys: object) -> None:
    _page, capture = _perspective_capture()
    source = tmp_path / "capture.png"
    output = tmp_path / "rectified.png"
    report_path = tmp_path / "quality.json"
    assert cv2.imwrite(str(source), capture)

    assert main(
        [
            "calibration",
            "rectify",
            str(source),
            str(output),
            "--report",
            str(report_path),
            "--px-per-mm",
            "3",
        ]
    ) == 0

    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert output.exists()
    assert cv2.imread(str(output), cv2.IMREAD_GRAYSCALE).shape == (891, 630)
    assert stdout_report == file_report
    assert file_report["marker_ids"] == [0, 1, 2, 3]
    assert file_report["rectified_image"]["px_per_mm"] == 3.0


def test_failed_file_rectification_emits_no_candidate_files(tmp_path: Path) -> None:
    source = tmp_path / "incomplete.png"
    output = tmp_path / "rectified.png"
    report = tmp_path / "rectified.json"
    image = render_calibration_image(px_per_mm=4.0)
    placement = DEFAULT_LAYOUT.placement(1)
    x0 = int(placement.x_mm * 4.0) - 4
    y0 = int(placement.y_mm * 4.0) - 4
    side = int(placement.size_mm * 4.0) + 8
    image[y0 : y0 + side, x0 : x0 + side] = 255
    assert cv2.imwrite(str(source), image)

    from measurepilot.rectification import rectify_image_file

    with pytest.raises(CalibrationError, match=r"missing required marker IDs: \[1\]"):
        rectify_image_file(source, output, report_destination=report)

    assert not output.exists()
    assert not report.exists()


@pytest.mark.parametrize("collision", ["source-output", "source-report", "output-report"])
def test_file_outputs_cannot_overwrite_each_other_or_the_source(
    tmp_path: Path, collision: str
) -> None:
    from measurepilot.rectification import rectify_image_file

    source = tmp_path / "capture.png"
    output = tmp_path / "rectified.png"
    report = tmp_path / "rectified.json"
    assert cv2.imwrite(str(source), render_calibration_image(px_per_mm=4.0))
    original_source = source.read_bytes()

    if collision == "source-output":
        output = source
    elif collision == "source-report":
        report = source
    else:
        report = output

    with pytest.raises(CalibrationError, match="overwrite|different paths"):
        rectify_image_file(source, output, report_destination=report)

    assert source.read_bytes() == original_source
    if output != source:
        assert not output.exists()
    if report not in (source, output):
        assert not report.exists()


def test_invalid_output_resolution_is_rejected() -> None:
    page = render_calibration_image(px_per_mm=4.0)

    with pytest.raises(CalibrationError, match="safety limit"):
        rectify_image(page, output_px_per_mm=21.0)
