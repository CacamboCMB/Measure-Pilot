from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from measurepilot.calibration import (
    ARUCO_DICTIONARY_NAME,
    CALIBRATION_LAYOUT_VERSION,
    DEFAULT_LAYOUT,
    REQUIRED_MARKER_IDS,
    CalibrationError,
    generate_calibration_pdf,
    render_calibration_image,
)
from measurepilot.cli import main
from measurepilot.rectification import detect_required_markers


def test_versioned_layout_has_exact_a4_marker_geometry() -> None:
    layout = DEFAULT_LAYOUT

    assert layout.version == CALIBRATION_LAYOUT_VERSION
    assert (layout.page_width_mm, layout.page_height_mm) == (210.0, 297.0)
    assert layout.marker_size_mm == 30.0
    assert layout.ruler_length_mm == 100.0
    assert tuple(item.marker_id for item in layout.marker_placements) == REQUIRED_MARKER_IDS
    assert np.array_equal(
        layout.placement(0).corners_mm(),
        np.asarray([[15.0, 15.0], [45.0, 15.0], [45.0, 45.0], [15.0, 45.0]]),
    )
    assert np.array_equal(
        layout.placement(2).corners_mm(),
        np.asarray([[165.0, 252.0], [195.0, 252.0], [195.0, 282.0], [165.0, 282.0]]),
    )


def test_metric_reference_raster_contains_all_required_markers() -> None:
    image = render_calibration_image(px_per_mm=4.0)
    detected = detect_required_markers(image)

    assert image.shape == (1188, 840)
    assert tuple(detected) == REQUIRED_MARKER_IDS
    for marker_id, corners in detected.items():
        expected = DEFAULT_LAYOUT.placement(marker_id).corners_mm() * 4.0
        assert np.max(np.abs(corners - expected)) <= 1.1


def test_invalid_raster_resolution_is_rejected() -> None:
    with pytest.raises(CalibrationError, match="positive finite"):
        render_calibration_image(px_per_mm=0.0)


def test_calibration_pdf_is_deterministic_and_nonempty(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"

    generate_calibration_pdf(first)
    generate_calibration_pdf(second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().startswith(b"%PDF-")
    assert first.stat().st_size > 3_000


def test_calibration_sheet_cli_creates_pdf(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "sheet.pdf"

    assert main(["calibration", "sheet", str(output)]) == 0

    assert output.exists()
    assert capsys.readouterr().out == f"CREATED {output}\n"


def test_opencv_dictionary_name_is_explicit() -> None:
    assert ARUCO_DICTIONARY_NAME == "DICT_4X4_50"
    assert hasattr(cv2, "aruco")
