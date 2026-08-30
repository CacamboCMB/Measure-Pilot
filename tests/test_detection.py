from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from measurepilot.calibration import render_calibration_image
from measurepilot.cli import main
from measurepilot.rectification import rectify_image
from measurepilot.detection import (
    DetectionError,
    canonical_detection_bytes,
    detect_image_file,
    detect_planar_part,
    render_detection_overlay,
)

PX_PER_MM = 4.0


def _page_with_part(*, second_part: bool = False, touches_boundary: bool = False) -> np.ndarray:
    image = render_calibration_image(px_per_mm=PX_PER_MM)

    def point(x_mm: float, y_mm: float) -> tuple[int, int]:
        return (round(x_mm * PX_PER_MM), round(y_mm * PX_PER_MM))

    left = 54.0 if touches_boundary else 70.0
    cv2.rectangle(image, point(left, 96.0), point(142.0, 202.0), 0, thickness=-1)
    cv2.circle(image, point(84.0, 120.0), round(5.0 * PX_PER_MM), 255, thickness=-1)
    cv2.circle(image, point(128.0, 177.0), round(4.0 * PX_PER_MM), 255, thickness=-1)
    cv2.rectangle(image, point(102.0, 145.0), point(114.0, 153.0), 255, thickness=-1)
    if second_part:
        cv2.rectangle(image, point(72.0, 207.0), point(115.0, 228.0), 0, thickness=-1)
    return image


def _bounds(points: tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    return min(x), min(y), max(x), max(y)


def test_detects_profile_two_circles_and_polygon_cutout() -> None:
    image = _page_with_part()

    result = detect_planar_part(image, px_per_mm=PX_PER_MM)

    min_x, min_y, max_x, max_y = _bounds(result.profile_points_mm)
    assert (min_x, min_y, max_x, max_y) == pytest.approx((70.0, 96.0, 142.0, 202.0), abs=0.4)
    assert result.profile_status == "estimated"
    assert result.profile_uncertainty_mm == pytest.approx(0.125)
    assert len(result.circles) == 2
    assert len(result.cutouts) == 1

    first, second = result.circles
    assert first.center_mm == pytest.approx((84.0, 120.0), abs=0.35)
    assert first.radius_mm == pytest.approx(5.0, abs=0.35)
    assert second.center_mm == pytest.approx((128.0, 177.0), abs=0.35)
    assert second.radius_mm == pytest.approx(4.0, abs=0.35)
    assert result.cutouts[0].feature_id == "cutout-001"
    assert len(result.source_sha256) == 64


def test_detection_is_deterministic_and_overlay_preserves_dimensions() -> None:
    image = _page_with_part()

    first = detect_planar_part(image, px_per_mm=PX_PER_MM)
    second = detect_planar_part(image.copy(), px_per_mm=PX_PER_MM)
    overlay = render_detection_overlay(image, first)

    assert canonical_detection_bytes(first) == canonical_detection_bytes(second)
    assert overlay.shape == (*image.shape, 3)
    assert not np.array_equal(overlay[..., 0], image)



def test_m1_perspective_rectification_feeds_m2_detection() -> None:
    page = _page_with_part()
    height, width = page.shape
    source_corners = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    capture_corners = np.asarray(
        [[90.0, 70.0], [925.0, 130.0], [880.0, 1260.0], [55.0, 1170.0]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_corners, capture_corners)
    capture = cv2.warpPerspective(page, transform, (1020, 1320), borderValue=255)

    rectified, report = rectify_image(capture, output_px_per_mm=PX_PER_MM)
    result = detect_planar_part(rectified, px_per_mm=report.output_px_per_mm)

    assert len(result.circles) == 2
    assert len(result.cutouts) == 1
    assert len(result.profile_points_mm) <= 8
    assert len(result.cutouts[0].points_mm) <= 12
    assert _bounds(result.profile_points_mm) == pytest.approx((70.0, 96.0, 142.0, 202.0), abs=0.75)

def test_no_part_is_rejected() -> None:
    image = render_calibration_image(px_per_mm=PX_PER_MM)

    with pytest.raises(DetectionError, match="no supported planar part"):
        detect_planar_part(image, px_per_mm=PX_PER_MM)


def test_ambiguous_capture_is_rejected() -> None:
    image = render_calibration_image(px_per_mm=PX_PER_MM)
    cv2.rectangle(image, (280, 400), (430, 620), 0, thickness=-1)
    cv2.rectangle(image, (450, 400), (600, 620), 0, thickness=-1)

    with pytest.raises(DetectionError, match="multiple plausible parts"):
        detect_planar_part(image, px_per_mm=PX_PER_MM)


def test_work_area_boundary_contact_is_rejected() -> None:
    with pytest.raises(DetectionError, match="touches the work-area boundary"):
        detect_planar_part(_page_with_part(touches_boundary=True), px_per_mm=PX_PER_MM)


def test_wrong_metric_dimensions_are_rejected() -> None:
    image = np.full((500, 500), 255, dtype=np.uint8)

    with pytest.raises(DetectionError, match="dimensions do not match"):
        detect_planar_part(image, px_per_mm=PX_PER_MM)


def test_file_detection_writes_deterministic_json_and_overlay(tmp_path: Path) -> None:
    source = tmp_path / "rectified.png"
    output = tmp_path / "detection.json"
    overlay = tmp_path / "overlay.png"
    assert cv2.imwrite(str(source), _page_with_part())

    detect_image_file(source, output, px_per_mm=PX_PER_MM, overlay_destination=overlay)
    first_bytes = output.read_bytes()
    detect_image_file(source, output, px_per_mm=PX_PER_MM, overlay_destination=overlay)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.read_bytes() == first_bytes
    assert overlay.exists()
    assert payload["model_version"] == "measurepilot-detection-v1"
    assert payload["source_sha256"]


def test_detection_cli_writes_json_and_overlay(tmp_path: Path, capsys: object) -> None:
    source = tmp_path / "rectified.png"
    output = tmp_path / "detection.json"
    overlay = tmp_path / "overlay.png"
    assert cv2.imwrite(str(source), _page_with_part())

    assert main(
        [
            "analysis",
            "detect",
            str(source),
            str(output),
            "--px-per-mm",
            "4",
            "--overlay",
            str(overlay),
        ]
    ) == 0

    stdout = json.loads(capsys.readouterr().out)
    assert stdout == json.loads(output.read_text(encoding="utf-8"))
    assert overlay.exists()


def test_output_paths_cannot_collide_with_source(tmp_path: Path) -> None:
    source = tmp_path / "rectified.png"
    assert cv2.imwrite(str(source), _page_with_part())
    original = source.read_bytes()

    with pytest.raises(DetectionError, match="different paths"):
        detect_image_file(source, source, px_per_mm=PX_PER_MM)

    assert source.read_bytes() == original
